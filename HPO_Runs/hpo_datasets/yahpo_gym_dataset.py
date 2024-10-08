# fust firsdt time, it writes files very dangerous in parallel
# from yahpo_gym import local_config
# local_config.init_config()
# data_path = "/mnt/qb/work/eggensperger/eww131/others/yahpo_data"
# local_config.set_data_path(data_path)

from yahpo_gym import benchmark_set
from amltk.pipeline import Component
import ConfigSpace

from amltk.optimization import Trial
from amltk.pipeline import Sequential
from amltk.store import PathBucket


class Model:
    def __init__(self, parameters, **config):
        self.parameters = parameters
        self.config = config

    def fit(self, X, y):
        pass

    def predict(self, X):
        global benchset
        self.config.update(self.parameters)
        # print(self.config, flush=True)
        cost = 1 - benchset.objective_function(self.config, multithread=False)[0]["auc"]
        # print("predict",cost , self.config, flush=True)
        return cost


class yahpo_gym_dataset:
    def __init__(self, scenario="rbv2_super"):
        global benchset
        benchset = benchmark_set.BenchmarkSet(
            scenario, active_session=False, multithread=False
        )

    def get_instances_list(self):
        return benchset.instances

    def get_pipeline(self, instance="375"):
        benchset.set_instance(instance)
        space = benchset.get_opt_space()
        parameters_list = ["num.impute.selected.cpo", "repl", "task_id", "trainsize"]
        parameters = {}
        space_dict = dict(space)
        for k, v in space.items():
            if k in parameters_list:
                parameters[k] = v
                del space_dict[k]
        cleaned_space = ConfigSpace.configuration_space.ConfigurationSpace(
            name=space.name,
            space=space_dict,
        )
        cleaned_space.add_conditions(space.get_conditions())
        parameters = ConfigSpace.configuration_space.ConfigurationSpace(
            space=parameters
        )
        parameters = dict(parameters.sample_configuration())
        parameters["repl"] = 6
        parameters["num.impute.selected.cpo"] = "impute.mean"
        parameters["trainsize"] = 0.525
        # print(cleaned_space)
        return Component(Model, config={"parameters": parameters}, space=cleaned_space)


    @staticmethod
    def target_function(
        trial: Trial,
        bucket: PathBucket,
        _pipeline: Sequential,
    ) -> Trial.Report:
        # Configure the pipeline with the trial config before building it.
        configured_pipeline = _pipeline.configure(trial.config)
        sklearn_pipeline = configured_pipeline.build("sklearn")
        # Fit the pipeline, indicating when you want to start the trial timing and error
        with trial.begin():
            sklearn_pipeline.fit(None, None)

        if trial.exception:
            trial.store(
                {
                    "exception.txt": f"{trial.exception}\n traceback: {trial.traceback}",
                    "config.json": dict(trial.config),
                }
            )
            return trial.fail()

        # Make our predictions with the model
        result = sklearn_pipeline.predict(None)

        if isinstance(result, tuple):
            model_error, infos = result
            trial.summary.update(
                {
                    "model_error": model_error,
                    "model_infos": infos,
                },
            )
        else:
            model_error = result
            trial.summary.update(
                {
                    "model_error": model_error,
                },
            )
        # Save all of this to the file system
        # trial.store(
        #    {
        #        "config.json": dict(trial.config),
        #        "scores.json": trial.summary,
        #    },
        # )
        # Finally report the success
        return trial.success(model_error=model_error)
