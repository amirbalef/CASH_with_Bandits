from __future__ import annotations

from functools import partial
from pathlib import Path
import argparse

from amltk.scheduling import Scheduler
from experiment import experiment

# python main_run_hpo.py --dataset TabRepo --instance arcene --optimizer RandomSearch_Arm_1

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", nargs="?", default="synth", help="dataset name")
    parser.add_argument(
        "--instance", nargs="?", default="get", help="instance of the dataset names"
    )
    parser.add_argument("--fold", nargs="?", default=0, type=int, help="fold")
    parser.add_argument("--n_splits", nargs="?", default=4, type=int, help="CV")
    parser.add_argument(
        "--time_limit", nargs="?", default=600, type=int, help="for each fold"
    )
    parser.add_argument(
        "--optimizer", nargs="?", default="Random", help="Random, SMAC, ..."
    )
    parser.add_argument("--output_root_dir", nargs="?", default="results/")
    parser.add_argument(
        "--iterations", nargs="?", default=200, type=int, help="number_of_iterations"
    )
    parser.add_argument("--trial_number", nargs="?", default=0, type=int, help="seed")
    parser.add_argument(
        "--trials", nargs="?", default=32, type=int, help="number of trials"
    )
    parser.add_argument(
        "--n_worker_scheduler",
        nargs="?",
        default=8,
        type=int,
        help="",
    )
    parser.add_argument(
        "--save_history_freq", nargs="?", default=10, type=int, help="save_history_freq"
    )
    args = parser.parse_args()

    dataset_params = {}
    dataset_params["name"] = args.dataset
    if args.dataset == "TabRepo":
        from hpo_datasets.TabRepo import TabRepo

        dataset = TabRepo(context_name="D244_F3_C1416_200")
    elif args.dataset == "TabRepoRaw":
        from hpo_datasets.TabRepoRaw import TabRepoRaw
        dataset = TabRepoRaw(context_name = "D244_F3_C1416_200")
        dataset_params["fold"] = args.fold
        dataset_params["n_splits"] = args.n_splits
        dataset_params["time_limit"] = args.time_limit

    elif args.dataset == "yahpo_gym":
        from hpo_datasets.yahpo_gym_dataset import yahpo_gym_dataset

        dataset = yahpo_gym_dataset()

    elif args.dataset == "synth":
        from hpo_datasets.synthetic_dataset import Synthetic_dataset

        dataset = Synthetic_dataset()

    else:
        exit()

    instance_names = dataset.get_instances_list()

    if args.instance in instance_names:
        dataset_params["instance"] = args.instance
    else:
        print("The instance name is wrong and not in", instance_names)
        exit()

    optimizer_name = args.optimizer
    iterations = args.iterations
    save_history_freq = args.save_history_freq
    base_output_path = Path(
        args.output_root_dir
        + "/"
        + dataset_params["name"]
        + "/"
        + dataset_params["instance"]
        + "/"
        + optimizer_name
        + "/"
    )

    experiment_task_per_seed = partial(
        experiment.experiment_task,
        base_output_path,
        optimizer_name,
        iterations,
        dataset,
        dataset_params,
        save_history_freq,
    )

    scheduler = Scheduler.with_processes(args.n_worker_scheduler)
    task = scheduler.task(experiment_task_per_seed)

    seed_numbers = iter(range(args.trial_number, args.trials))
    results = []

    # When the scheduler starts, submit #n_worker_scheduler tasks to the processes
    @scheduler.on_start(repeat=args.n_worker_scheduler)
    def on_start():
        n = next(seed_numbers)
        task.submit(n)

    # When the task is done, store the result
    @task.on_result
    def on_result(_, result: float):
        results.append(result)

    # Easy to incrementently add more functionallity
    @task.on_result
    def launch_next(_, result: float):
        if (n := next(seed_numbers, None)) is not None:
            task.submit(n)

    # React to issues when they happen
    @task.on_exception
    def stop_something_went_wrong(_, exception: Exception):
        scheduler.stop()

    # Start the scheduler and run it as you like
    scheduler.run(timeout=None)

    print("Done, results are avaible in ", results)
