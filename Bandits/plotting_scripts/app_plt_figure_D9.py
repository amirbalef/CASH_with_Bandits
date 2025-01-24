import os
import pandas as pd
import exp_utils
import analysis_utils
import matplotlib.pyplot as plt
from autorank import autorank, plot_stats
import algorithms_data


dataset_names = ["TabRepo", "TabRepoRaw", "YaHPOGym", "Reshuffling"]


def get_plot(order, dataset_name):
    dataset = pd.read_csv("../datasets/" + dataset_name + ".csv")

    instances = sorted(dataset["instance"].unique())
    all_arm_index_list = dataset["arm_index"].unique()
    valid_arm_index_list = [item for item in all_arm_index_list if item >= 0]
    number_of_arms = len(valid_arm_index_list)
    number_of_trails = len(dataset["repetition"].unique())
    horizon_time = len(dataset["iteration"].unique())
    classes = dataset["classifier"].unique()
    combined_search_algorithms = dataset[dataset["arm_index"] < 0]["optimizer"].unique()

    policy_algorithms = {}
    policy_algorithms["MaxUCB"] = 3
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
    # policy_algorithms["MaxSearch_Gaussian"] = 1
    # policy_algorithms["MaxSearch_SubGaussian"] = 1
    # policy_algorithms["Threshold_Ascent"] = 1
    # policy_algorithms["Random"] = 1
    for algorithm in reversed(combined_search_algorithms):
        policy_algorithms[algorithm] = 2
    policy_algorithms["Oracle_Arm"] = 2

    result_directory = "../results/"
    all_result = exp_utils.fetch_results(
        policy_algorithms, result_directory, dataset_name
    )

    res = analysis_utils.get_error_per_instance_time(
        all_result, number_of_arms, instances, number_of_trails, horizon_time
    )

    path = result_directory + "/plots_for_paper/app/fig_autoranks/"
    if not os.path.exists(path):
        os.makedirs(path)


    plt.rcParams.update({"font.size": 18})
    fig = plt.figure()
    data = pd.DataFrame()
    for key, value in policy_algorithms.items():
        if value == 1 or value == 3:
            k = algorithms_data.printing_name_dict[key]
            if key == "MaxUCB":
                k = "MaxUCB"
            # data[k] = [item for i in range(len(instances)) for item in res[i].T[key] ]

            data[k] = [item for i in range(len(instances)) for item in res[i].T[key]]

    result = autorank(
        data, alpha=0.05, verbose=False, order="ascending", force_mode="nonparametric"
    )
    plot_stats(result, width=4, allow_insignificant=True)
    plt.savefig(
        path + "/" + dataset_name + "_autorank_bandits.pdf", dpi=600, bbox_inches="tight"
    )
    plt.close(fig)

    fig = plt.figure()
    data2 = pd.DataFrame()
    for key, value in policy_algorithms.items():
        if value == 2 or value == 3:
            k = algorithms_data.printing_name_dict[key]
            if key == "MaxUCB":
                k = "MaxUCB"
            data2[k] = [item for i in range(len(instances)) for item in res[i].T[key]]
    result2 = autorank(
        data2, alpha=0.05, verbose=False, order="ascending", force_mode="nonparametric"
    )
    plot_stats(result2, width=4, allow_insignificant=True)
    plt.savefig(
        path + "/" + dataset_name + "_autorank_hpo.pdf", dpi=600, bbox_inches="tight"
    )
    plt.close(fig)


for i, dataset_name in enumerate(dataset_names):
    get_plot(i, dataset_name)