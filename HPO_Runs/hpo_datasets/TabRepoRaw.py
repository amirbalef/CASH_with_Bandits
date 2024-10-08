from __future__ import annotations
import traceback
import tabrepo
import numpy as np
import copy
import importlib
import openml
import warnings

import sklearn

import ConfigSpace

from autogluon.common import space as ag_space
from autogluon.tabular.models import (
    XGBoostModel,
    RFModel,
    XTModel,
    CatBoostModel,
    LGBModel,
    TabularNeuralNetTorchModel,
    NNFastAiTabularModel,
)
from autogluon.core.metrics import make_scorer
from autogluon.core.utils.utils import CVSplitter
from autogluon.core.data import LabelCleaner

from amltk.pipeline import Choice, Component
from amltk.optimization import Trial
from amltk.pipeline import Sequential
from amltk.store import PathBucket

import ast


def rmse_func(y_true, y_pred, **kwargs):
    if kwargs:
        return sklearn.metrics.mean_squared_error(
            y_true, y_pred, squared=False, **kwargs
        )
    else:
        return np.sqrt(((y_true - y_pred) ** 2).mean())


def get_metric_obj(metric_name):
    metric_scorers = {}
    metric_scorers["roc_auc"] = make_scorer(
        name="roc_auc",
        score_func=sklearn.metrics.roc_auc_score,
        optimum=1.0,
        greater_is_better=True,
        needs_threshold=True,
    )
    metric_scorers["log_loss"] = make_scorer(
        name="log_loss",
        score_func=sklearn.metrics.log_loss,
        optimum=0.0,
        needs_proba=True,
        greater_is_better=False,
    )
    metric_scorers["root_mean_squared_error"] = make_scorer(
        name="root_mean_squared_error",
        score_func=rmse_func,
        optimum=0.0,
        greater_is_better=False,
    )
    return metric_scorers[metric_name]


models = {
    "CatBoost": CatBoostModel,
    "ExtraTrees": XTModel,
    "NeuralNetFastAI": NNFastAiTabularModel,
    "LightGBM": LGBModel,
    "NeuralNetTorch": TabularNeuralNetTorchModel,
    "RandomForest": RFModel,
    "XGBoost": XGBoostModel,
}


class Classifier:
    def __init__(self, model_name, time_limit, task_infos, output_path, **config):
        task_type, n_classes, metric_name = task_infos
        self.metric = get_metric_obj(metric_name)
        hyperparameters = {}
        for k_old in config:
            k = k_old[len(model_name) + 1 :]
            if model_name == "NeuralNetFastAI" and k == "layers":
                hyperparameters[k] = ast.literal_eval(config[k_old])
            else:
                hyperparameters[k] = config[k_old]
        self.model = models[model_name](
            path=str(output_path) + "/autogluon_output",
            problem_type=task_type,
            eval_metric=self.metric,
            hyperparameters=hyperparameters,
        )
        self.time_limit = time_limit
        self.task_type = task_type

    def fit(self, X, y):
        X_train, y_train = X
        X_val, y_val = y
        self.model.fit(
            X=X_train,
            y=y_train,
            X_val=X_val,
            y_val=y_val,
            time_limit=self.time_limit,
            num_cpus=1,
            num_gpus=0,
            verbosity=0,
        )

    def predict(self, X):
        if self.task_type == "multiclass":
            pred = self.model.predict_proba(X[0])
            labels = np.arange(pred.shape[1], dtype=np.int32)
            return self.metric.convert_score_to_error(
                self.metric(X[1], pred, labels=labels)
            )
        else:
            pred = self.model.predict_proba(X[0])
            return self.metric.convert_score_to_error(self.metric(X[1], pred))


class TabRepoRaw:
    def __init__(self, context_name="D244_F3_C1416_30"):
        self.context = tabrepo.get_context(name=context_name)
        self.config_hyperparameters = self.context.load_configs_hyperparameters()
        self.repo: tabrepo.EvaluationRepository = tabrepo.load_repository(
            context_name, cache=True
        )
        self.methods = self.get_methods()
        self.module_names = {
            "CatBoost": "catboost",
            "ExtraTrees": "extra_trees",
            "NeuralNetFastAI": "fastai",
            "LightGBM": "lightgbm",
            "NeuralNetTorch": "nn_torch",
            "RandomForest": "random_forest",
            "XGBoost": "xgboost",
        }
        self.num_models = len(self.module_names)
        self.task_infos = None

        self.tabrepo_configs = []
        self.valid_configs = []

    def get_instances_list(self):
        return self.repo.datasets()

    def get_methods(self):
        configs = self.repo.configs()
        methods = {}
        for item in configs:
            method, key = item.split("_")[0:2]
            if method not in methods.keys():
                methods[method] = [item]
            else:
                methods[method].append(item)
        return methods

    def get_subspace(self, method_id):
        subspace = ConfigSpace.configuration_space.ConfigurationSpace(name=method_id)
        generate = importlib.import_module(
            "tabrepo.models." + self.module_names[method_id] + ".generate"
        )
        for key, value in generate.search_space.items():
            name = key
            if isinstance(value, ag_space.Real):
                hyp = ConfigSpace.api.types.float.Float(
                    name=method_id + "_" + key,
                    bounds=[value.lower, value.upper],
                    default=value.default,
                    log=value.log,
                )
            elif isinstance(value, ag_space.Int):
                hyp = ConfigSpace.api.types.integer.Integer(
                    name=method_id + "_" + key, bounds=[value.lower, value.upper]
                )
            elif isinstance(value, ag_space.Categorical):
                datas = value.data
                if all(isinstance(x, bool) for x in value.data):
                    is_ordinal = False
                elif all(isinstance(x, (int, float)) for x in value.data):
                    is_ordinal = True
                else:
                    if method_id + "_" + key == "NeuralNetFastAI_layers":
                        datas = [str(data) for data in value.data]
                    is_ordinal = False
                hyp = ConfigSpace.api.types.categorical.Categorical(
                    name=method_id + "_" + key,
                    items=datas,
                    default=datas[0],
                    ordered=is_ordinal,
                )
            else:
                print("oh no", name, value)
                NotImplementedError
            subspace.add_hyperparameter(hyp)
        return subspace

    def get_task_infos(self, instance):
        """
        root mean-squared error (RMSE) for regression, the area under the
        receiver operating characteristic curve (AUC) for binary classification and log loss for multi-class
        classification.
        """
        task_id = self.repo.dataset_to_tid(instance)
        warnings.filterwarnings("ignore")
        task = openml.tasks.get_task(task_id, download_splits=True)
        n_classes = None
        if task.task_type == "Supervised Classification":
            n_classes = len(task.class_labels)
            if n_classes == 2:
                task_type = "binary"
                metric_name = "roc_auc"
            else:
                task_type = "multiclass"
                metric_name = "log_loss"
        elif task.task_type == "Supervised Regression":
            metric_name = "root_mean_squared_error"
            task_type = "regression"
        else:
            NotImplementedError
        self.task_infos = task_type, n_classes, metric_name
        return self.task_infos

    def get_data(self, instance, fold=0, seed=0, n_splits=4):
        task_type, n_classes, metric_name = self.get_task_infos(instance)

        task_id = self.repo.dataset_to_tid(instance)
        task = openml.tasks.get_task(task_id, download_splits=True)
        dataset = task.get_dataset()
        # n_repeats, n_folds, _ = task.get_split_dimensions()
        target_name = dataset.default_target_attribute
        train_indices, test_indices = task.get_train_test_split_indices(fold=fold)
        X, y, _, _ = dataset.get_data(dataset_format="dataframe", target=target_name)

        label_cleaner = LabelCleaner.construct(problem_type=task_type, y=y)
        y = label_cleaner.transform(y)

        X_train = X.iloc[train_indices]
        y_train = y.iloc[train_indices]
        X_test = X.iloc[test_indices]
        y_test = y.iloc[test_indices]

        stratified = (task_type == "binary") or (task_type == "multiclass")
        splitter = CVSplitter(
            n_splits=n_splits, random_state=seed, stratified=stratified
        )
        k_fold_split_indx = splitter.split(X_train, y_train)

        return ((X_train, y_train), (X_test, y_test), k_fold_split_indx)

    def get_subspace_configs_and_scores(
        self, subspace, method_id, dataset_name, fold=0
    ):
        configs_for_subpace = []
        configs_for_metric = []
        list_configs = self.methods[method_id]
        for item in list_configs:
            method_id, key_id = item.split("_")[0:2]
            config_key = method_id + "_" + key_id
            item_hyperparameters = copy.deepcopy(
                self.config_hyperparameters[config_key]["hyperparameters"]
            )
            if "c" in key_id:  # default configurations
                default_config = subspace.get_default_configuration()
                for k, v in default_config.items():
                    hyp_name = k[len(method_id + "_") :]
                    if hyp_name not in item_hyperparameters:
                        item_hyperparameters[hyp_name] = v

            if "ag_args" in item_hyperparameters:
                del item_hyperparameters["ag_args"]
            new_config = {}
            for k_old in item_hyperparameters:
                name = method_id + "_" + k_old
                value = item_hyperparameters[k_old]
                if name == "NeuralNetFastAI_layers":
                    value = str(value)
                new_config[name] = value
            configs_for_subpace.append(new_config)
            configs_for_metric.append(item)
        scores = np.asarray(
            self.repo.metrics(
                datasets=[dataset_name], configs=configs_for_metric, folds=[fold]
            )["metric_error_val"].tolist()
        )
        return configs_for_subpace, scores, configs_for_metric

    def get_tabrepo_scores(self, configs, dataset_name, fold=0):
        configs_for_metric = []
        for config in configs:
            indx = self.valid_configs.index(config)
            configs_for_metric.append(self.tabrepo_configs[indx])
        return self.repo.metrics(
            datasets=[dataset_name], configs=configs_for_metric, folds=[fold]
        )

    def get_valid_configs(self, space, configs):
        valid_configs = []
        for i, method in enumerate(self.methods.keys()):
            for config in configs[method]:
                new_config = {"methods:__choice__": method}
                for k_old in config:
                    name = "methods:" + method + ":" + k_old
                    new_config[name] = config[k_old]
                valid_configs.append(
                    ConfigSpace.configuration_space.Configuration(
                        space, values=new_config
                    )
                )
        return valid_configs

    def get_metric(self, instance):
        _, _, metric_name = self.get_task_infos(instance)
        return get_metric_obj(metric_name)

    def get_pipeline(
        self,
        instance,
        fold=0,
        time_limit=3600,
        return_valid_configs=True,
        output_path=None,
    ):
        task_infos = self.get_task_infos(instance)
        routes = []
        configs = {}
        self.tabrepo_configs = []
        self.valid_configs = []
        for i, method in enumerate(self.methods.keys()):
            sub_space = self.get_subspace(method)
            its_configs, _, tabrepo_configs = self.get_subspace_configs_and_scores(
                sub_space, method, dataset_name=instance, fold=fold
            )
            configs[method] = its_configs
            self.tabrepo_configs.extend(tabrepo_configs)
            item = Component(
                Classifier,
                config={
                    "model_name": method,
                    "task_infos": task_infos,
                    "time_limit": time_limit,
                    "output_path": output_path,
                },
                space=sub_space,
                name=method,
            )
            routes.append(item)
        pipeline = Choice(*routes, name="methods")
        if return_valid_configs:
            valid_configs = self.get_valid_configs(
                pipeline.search_space("configspace"), configs
            )
            self.valid_configs = valid_configs
            return pipeline, valid_configs
        else:
            return pipeline

    @staticmethod
    def target_function(
        trial: Trial,
        bucket: PathBucket,
        _pipeline: Sequential,
    ) -> Trial.Report:
        train_data_unfold, test_data, k_fold_split_indx = bucket[
            "experiment_data.pkl"
        ].load()

        try:
            val_error = []
            test_error = []
            for i, (train_idx, val_idx) in enumerate(k_fold_split_indx):
                train_data = (
                    train_data_unfold[0].iloc[train_idx],
                    train_data_unfold[1].iloc[train_idx],
                )
                val_data = (
                    train_data_unfold[0].iloc[val_idx],
                    train_data_unfold[1].iloc[val_idx],
                )

                configured_pipeline = _pipeline.configure(trial.config)
                the_pipeline = configured_pipeline.build("sklearn")
                with trial.profile("fit"):
                    the_pipeline.fit(train_data, val_data)
                with trial.profile("predictions"):
                    val_error.append(the_pipeline.predict(val_data))
                    test_error.append(the_pipeline.predict(test_data))

        except Exception as e:
            trial.store(
                {
                    "exception.txt": f"{e}\n traceback: {str(traceback.format_exc())}",
                    "config.json": dict(trial.config),
                }
            )
            return trial.fail(e)

        with trial.profile("scoring"):
            model_error = np.mean(val_error)
            model_error_test = np.mean(test_error)
            trial.summary["model_error"] = model_error
            trial.summary["model_error_test"] = model_error_test

        # Save all of this to the file system
        # trial.store(
        #    {
        #        "config.json": dict(trial.config),
        #        "scores.json": trial.summary,
        #    },
        # )
        # Finally report the success
        return trial.success(model_error=model_error)


"""  
def test():
    from autogluon.core.utils.utils import CVSplitter
    seed = 0
    dataset = TabRepo(context_name = "D244_F3_C1416_30")
    instance_names = dataset.get_instances_list()
    instance = instance_names[0]
    print(instance)
    pipeline, valid_configs  = dataset.get_pipeline(instance, fold= 0, val_fold=None)
    #X_train,y_train, X_val, y_val, X_test, y_test = dataset.get_data(instance=instance, val_fold=0)
    X_train,y_train, X_test, y_test = dataset.get_data(instance=instance)
    metric = dataset.get_metric(instance)


    #config = pipeline.search_space("configspace").get_default_configuration()
    #print(config)
    config = pipeline.search_space("configspace").sample_configuration()
    configured_pipeline = pipeline.configure(config)
    sklearn_pipeline = configured_pipeline.build("sklearn")
    train_data = X_train,y_train



    splitter = CVSplitter(n_splits=8, random_state = seed)
    val_error = []
    test_error = []
    for i, (train_idx, val_idx) in enumerate(splitter.split(X_train,y_train)):
        train_data = X_train.iloc[train_idx], y_train.iloc[train_idx]
        val_data = X_train.iloc[val_idx], y_train.iloc[val_idx]

        sklearn_pipeline.fit(train_data, val_data)
        val_error.append(metric.convert_score_to_error(metric(sklearn_pipeline.predict(val_data[0]), val_data[1])))
        test_error.append(metric.convert_score_to_error(metric(sklearn_pipeline.predict(X_test), y_test)))

    print("error val", np.mean(val_error))
    print("error test", np.mean(test_error))

    end
    for i in range(100):
        #config = valid_configs[np.random.randint(0, len(valid_configs))]
        config = pipeline.search_space("configspace").sample_configuration()
        configured_pipeline = pipeline.configure(config)
        sklearn_pipeline = configured_pipeline.build("sklearn")
        sklearn_pipeline.fit(train_data, val_data)
        print("error val", metric.convert_score_to_error(metric(sklearn_pipeline.predict(X_val), y_val)))
        print("error test", metric.convert_score_to_error(metric(sklearn_pipeline.predict(X_test), y_test)))
test()
"""
