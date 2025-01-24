import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


import plotting_utils
import pandas as pd
import exp_utils
import matplotlib.pyplot as plt
import pylab
import numpy as np
import analysis_utils
import algorithms_data




dataset_names = ["SubSupernet"]


def get_plot(order, dataset_name):
    dataset = pd.read_csv("../datasets/" + dataset_name + ".csv")

    instances = sorted(dataset["instance"].unique())
    all_arm_index_list = dataset["arm_index"].unique()
    valid_arm_index_list = [item for item in all_arm_index_list if item >= 0]
    number_of_arms = len(valid_arm_index_list)
    number_of_trails = len(dataset["repetition"].unique())
    horizon_time = len(dataset["iteration"].unique())
    classes = dataset["classifier"].unique()
    combined_search_algorithms = list(dataset[dataset["arm_index"] < 0]["optimizer"].unique())

    policy_algorithms = {}
    policy_algorithms["MaxUCB"] = 1
    #policy_algorithms["Max_KG"] = 1
    policy_algorithms["Q_BayesUCB"] = 1
    policy_algorithms["ER_UCB_S"] = 1
    policy_algorithms["Rising_Bandit"] = 1
    # policy_algorithms["QoMax_ETC"] = 1
    policy_algorithms["QoMax_SDA"] = 1
    # policy_algorithms["ER_UCB_N"] = 1
    policy_algorithms["Max_Median"] = 1
    # policy_algorithms["Exp3_OG"] = 1
    # policy_algorithms["TS_Poisson"] = 1
    # policy_algorithms["TS_Gaussian"] = 1
    policy_algorithms["UCB"] = 1
    #policy_algorithms["MaxSearch_Gaussian"] = 1
    #policy_algorithms["MaxSearch_SubGaussian"] = 1
    #policy_algorithms["Threshold_Ascent"] = 1
    policy_algorithms["Successive_Halving"] = 1
    #policy_algorithms["Random"] = 1
    if(len(combined_search_algorithms)>0):
        combined_search_algorithms.append(combined_search_algorithms.pop(0))
        for algorithm in combined_search_algorithms:
            #if algorithm != "SMAC_NoInit":
                policy_algorithms[algorithm] = 1
    policy_algorithms["Oracle_Arm"] = 1


    result_directory = "../results/"
    all_result = exp_utils.fetch_results(
        policy_algorithms, result_directory, dataset_name
    )

    from cycler import cycler

    colors = plotting_utils.CB_color_cycle[
        :8
    ]  # ['red', 'orange', 'blue', 'green', 'cyan', 'brown', 'olive'] #, 'purple ']
    all_cyclers = cycler(color=colors) * cycler(linestyle=["-"])  # ,"--"

    if dataset_name != "Reshuffling":
        colorcycler = cycler(color=["black"])
        lines = ["-", ":"]
        if dataset_name == "TabRepo":
            lines = ["-"]
        if  dataset_name == "SubSupernet":
            lines = ["-"]
            all_result["RandomSearch_FSS"] = all_result.pop("RandomSearch")
            all_result["Oracle_Arm"] = all_result.pop("Oracle_Arm")
        if "SMAC_NoInit" in policy_algorithms:
            lines = ["-", "--", ":"]
        linecycler = cycler(linestyle=lines)
        all_cyclers = all_cyclers.concat(colorcycler * linecycler)

    if "Oracle_Arm" in policy_algorithms:
        colorcycler = cycler(color=["grey"])
        lines = ["-"]
        linecycler = cycler(linestyle=lines)
        all_cyclers = all_cyclers.concat(colorcycler * linecycler)

    path = result_directory + "plots_for_paper/SubSupernet"
    if not os.path.exists(path):
        os.makedirs(path)

    data = {}
    data["all_result"] = all_result
    data["dataset"] = dataset
    data["horizon_time"] = horizon_time
    data["number_of_arms"] = number_of_arms
    data["instances"] = instances
    data["number_of_trails"] = number_of_trails
    data["saving_path"] = path
    data["dataset_name"] = dataset_name
    data["cyclers"] = all_cyclers
    data["test_data"] = -1
    data["fig_size"] = (6, 5)
    data["legend"] = False
    data["tilte"] = False
    data["ylabel"] = None
    

    data["set_ylim"] = None
    if data["dataset_name"] == "TabRepo":
        data["set_ylim"] = (1, 0.05)
    if data["dataset_name"] == "Reshuffling":
        data["set_ylim"] = (1, 0.02)
    if data["dataset_name"] == "TabRepoRaw":
        data["set_ylim"] = (1, 0.1)
        data["legend"] = "seperate"
    if data["dataset_name"] == "YaHPOGym":
        data["set_ylim"] = (1, 0.05)
        data["legend"] = "seperate"
    if data["dataset_name"] == "SubSupernet":
        data["set_ylim"] = (1, 0.03)
        data["legend"] = "seperate"


    if order == 0:
        data["ylabel"] = "Normalized loss"
    data["plot_type"] = "Normalized loss"
    data["saving_name"] = "norm_loss"
    data["plot_confidence_type"] = "mean_std"
    plotting_utils.plot_averaged_on_datasets(data)

    if order == 0:
        data["ylabel"] = "Ranking"
    data["plot_type"] = "Ranking"
    data["saving_name"] = "ranking"
    data["plot_confidence_type"] = "mean_std"
    plotting_utils.plot_averaged_on_datasets(data)


for i, dataset_name in enumerate(dataset_names):
    get_plot(i, dataset_name)
