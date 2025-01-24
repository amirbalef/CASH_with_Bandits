import os
import plotting_utils
import pandas as pd
import exp_utils
import matplotlib.pyplot as plt
import pylab
import numpy as np
import analysis_utils
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
    combined_search_algorithms = list(dataset[dataset["arm_index"] < 0]["optimizer"].unique())

    policy_algorithms = {}
    policy_algorithms["MaxUCB"] = 1
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
    # policy_algorithms["Successive_Halving"] = 1
    policy_algorithms["Random"] = 1
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
            lines = [":"]
        if "SMAC_NoInit" in policy_algorithms:
            lines = ["-", "--", ":"]
        linecycler = cycler(linestyle=lines)
        all_cyclers = all_cyclers.concat(colorcycler * linecycler)

    if "Oracle_Arm" in policy_algorithms:
        colorcycler = cycler(color=["grey"])
        lines = ["-"]
        linecycler = cycler(linestyle=lines)
        all_cyclers = all_cyclers.concat(colorcycler * linecycler)

    path = result_directory + "plots_for_paper/app/fig_rankings/"
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
    data["fig_size"] = (4, 5)
    data["legend"] = False
    data["tilte"] = False
    data["ylabel"] = None
    if order == 0:
        data["ylabel"] = "Ranking"

    data["set_ylim"] = None
    if data["dataset_name"] == "TabRepo":
        data["set_ylim"] = (1, 0.05)
    if data["dataset_name"] == "Reshuffling":
        data["set_ylim"] = (1, 0.02)
    if data["dataset_name"] == "TabRepoRaw":
        data["set_ylim"] = (1, 0.1)
    if data["dataset_name"] == "YaHPOGym":
        data["set_ylim"] = (1, 0.05)
        data["legend"] = "seperate"

    data["plot_type"] = "Ranking"
    data["saving_name"] = "ranking"
    data["plot_confidence_type"] = "mean_std"
    plotting_utils.plot_averaged_on_datasets(data)




def plot_averaged_on_datasets(data):
    linewidth = 3
    if "linewidth" in data:
        linewidth = data["linewidth"]

    if data["plot_type"] == "Ranking":
        mean, std = analysis_utils.get_ranks_per_instance_MC(
            data["all_result"],
            data["horizon_time"],
            data["number_of_arms"],
            data["instances"],
            data["number_of_trails"],
            num_samples=100,
        )
    else:
        print("Only for ranking!")
        exit()
    index = np.arange(data["horizon_time"])
    # setting font sizeto 30
    plt.rcParams.update({"font.size": 26})
    if "fig_size" in data.keys():
        fig, ax = plt.subplots(figsize=data["fig_size"])
    else:
        fig, ax = plt.subplots(figsize=(8, 6))
    ax.set_prop_cycle(data["cyclers"])

    for i, item in enumerate(data["all_result"].keys()):
        zorder = None
        if i == 0:
            zorder = 100
        ax.plot(
            index,
            mean[i],
            label=algorithms_data.printing_name_dict[item],
            linewidth=linewidth,
            zorder=zorder,
        )  # , marker=i,markevery=10)
        ax.fill_between(index, mean[i] - std[i], mean[i] + std[i], alpha=0.3)

    ax.set(xlabel="Iteration", ylabel=data["ylabel"])
    # if data["plot_type"] == "Normalized Error":
    #     ax.ticklabel_format(style="plain")
    #     ax.set_yscale("log")
    #     ax.yaxis.set_minor_formatter(plticker.NullFormatter())
    #     ax.yaxis.set_ticks(ax.get_yticks())
    #     ax.set_yticklabels([str(x) for x in ax.get_yticks()])
    #     ax.set_ylim(data["set_ylim"])

    if data["legend"] == "inside":  # dataset_name != "TabRepo":
        loc = "center right"
        if "fig_size" in data:
            bbox_to_anchor = (0.65 * 8 / data["fig_size"][0], 0.5, 1, 0.1)
        else:
            bbox_to_anchor = (0.65, 0.5, 1, 0.1)
        ax.legend(
            loc=loc,
            ncol=1,
            fontsize=22,
            bbox_to_anchor=bbox_to_anchor,
            handletextpad=0.15,
            handlelength=1.5,
            frameon=False,
        )
    if data["tilte"]:
        plt.title(algorithms_data.printing_name_dict[data["dataset_name"]])
    # plt.savefig(data["saving_path"] +"/"+data["dataset_name"]+"_" + data["saving_name"] +".png", dpi=600, bbox_inches='tight')
    # plt.tight_layout()
    plt.savefig(
        data["saving_path"]
        + "/"
        + data["dataset_name"]
        + "_"
        + data["saving_name"]
        + ".pdf",
        dpi=600,
        bbox_inches="tight",
    )
    plt.close()
    if data["legend"] == "seperate":
        figlegend = pylab.figure(figsize=(3, 8))
        pylab.figlegend(*ax.get_legend_handles_labels(), loc="upper left")
        figlegend.savefig(
            data["saving_path"] + "/" + "ranking_legend.pdf",
            dpi=600,
            bbox_inches="tight",
        )


for i, dataset_name in enumerate(dataset_names):
    get_plot(i, dataset_name)
plt.show()
