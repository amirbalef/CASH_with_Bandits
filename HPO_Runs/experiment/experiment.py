import os
from optimizers import optimizers
from amltk.store import PathBucket
from amltk.optimization import History
import pickle


def experiment_task(
    base_output_path,
    optimizer_name,
    iterations,
    dataset,
    dataset_params,
    save_history_freq,
    seed,
):
    trial_name = str(seed)
    output_path = base_output_path.joinpath(trial_name)

    bucket = PathBucket(output_path, clean=False, create=True)
    pipeline = get_pipeline(dataset, dataset_params, bucket, seed)
    limit_to_configs = None
    if isinstance(pipeline, tuple):
        pipeline, limit_to_configs = pipeline

    optimizer = optimizers.get_optimizer(
        optimizer_name, pipeline, bucket, seed, limit_to_configs, iterations
    )

    remaining_iterations = iterations
    history = History()
    if os.path.exists(output_path.joinpath("history.pkl")):
        print("Resuming!")
        with open(output_path.joinpath("history.pkl"), "rb") as handle:
            history = pickle.load(handle)
        

        if iterations <= len(history):
            print("all results are ready")
            return output_path  # exit()

        for report in history:
            #print(report)
            optimizer.tell(report)
            remaining_iterations -= 1

    
    print(str(remaining_iterations) + " iterations is left")
    for iteration in range(1, remaining_iterations + 1):
        trial = optimizer.ask()
        report = dataset.target_function(
            trial, bucket=bucket, _pipeline=pipeline
        )
        #print(report)
        history.add(report)
        optimizer.tell(report)
        if iteration % save_history_freq == 0:
            with open(output_path.joinpath("history.pkl"), "wb") as handle:
                pickle.dump(history, handle, protocol=pickle.HIGHEST_PROTOCOL)
        
    history_df = history.df()

    

    history_df.to_pickle(output_path.joinpath("result.pkl"))
    return output_path


def get_pipeline(dataset, dataset_params, bucket, seed):
    if dataset_params["name"] == "TabRepoRaw":
        experiment_data = dataset.get_data(
            instance=dataset_params["instance"],
            fold=dataset_params["fold"],
            seed=seed,
            n_splits=dataset_params["n_splits"],
        )
        bucket.store({"experiment_data.pkl": experiment_data})
        pipeline = dataset.get_pipeline(
            dataset_params["instance"],
            fold=dataset_params["fold"],
            output_path=bucket.path,
            time_limit=dataset_params["time_limit"],
            return_valid_configs=False,
        )
    else:
        pipeline = dataset.get_pipeline(instance=dataset_params["instance"])

    return pipeline