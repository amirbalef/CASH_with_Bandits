import os
import pickle
import plotting_utils
import algorithms_data
import pandas as pd
import exp_utils
import matplotlib.pyplot as plt
import pylab
import numpy as np
import seaborn as sns


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

    df = dataset[(dataset["arm_index"]>=0)]
    df = df[["instance", "arm_index", "repetition", "iteration", "loss"]]
    raw_dataset = df.sort_values(
        by=["instance", "arm_index", "repetition", "iteration"]
    )["loss"].values.reshape(
        len(instances), number_of_arms, number_of_trails, horizon_time
    )


    policy_algorithms = {}
    policy_algorithms["MaxUCB"] = 1
    policy_algorithms["Q_BayesUCB"] = 1
    policy_algorithms["ER_UCB_S"] = 1
    policy_algorithms["Rising_Bandit"] = 1
    policy_algorithms["QoMax_SDA"] = 1
    policy_algorithms["Max_Median"] = 1
    policy_algorithms["UCB"] = 1
    

    result_directory = "../results/"
    all_pulls_result = exp_utils.fetch_pullings(
        policy_algorithms, result_directory, dataset_name
    )

    pull_s = []
    for intance_num, intance in enumerate(instances):
        pull_list = []
        for trail in range(number_of_trails):
            data = raw_dataset[intance_num][:, trail]
            data = data[:, :horizon_time]
            data_min = np.min(data, axis=1)
            selected_arms = np.argsort(data_min)

            list_pulls = []
            for key, alg_pulls in all_pulls_result.items():
                pulled_arms = np.bincount(
                    alg_pulls[intance_num][trail][:horizon_time],
                    minlength=number_of_arms,
                )
                list_pulls.append(pulled_arms[selected_arms])

            pull_list.append(list_pulls)

        pull_s.append(pull_list)
    pulls = np.asarray(pull_s)


    dfs = []
    for alg_num, alg_name in enumerate(all_pulls_result.keys()):
        data = {}
        for i in range(0, number_of_arms):
            data[str(i)] = pulls[:, :, alg_num, i].flatten()
        df = pd.DataFrame(data)
        df["Algorithm"] = algorithms_data.printing_name_dict[alg_name]
        dfs.append(df)

    ###########################
    path = result_directory + "/plots_for_paper/app/fig_pullings/"
    if not os.path.exists(path):
        os.makedirs(path)


    combined_df = pd.concat(dfs)
    # Melt the combined DataFrame
    melted_df = pd.melt(
        combined_df, id_vars=["Algorithm"], var_name="arms", value_name="Value"
    )

    plt.rcParams["text.usetex"] = True
    # Fig size
    plt.rcParams["figure.figsize"] = 10, 6
    plt.rcParams.update({"font.size": 26})

    fig, ax = plt.subplots()
    my_pal = plotting_utils.CB_color_cycle[: len(all_pulls_result.keys())]

    g1 = sns.boxplot(
        x="arms",
        y="Value",
        hue="Algorithm",
        data=melted_df,
        linewidth=0.4,
        showfliers=False,
        medianprops={"color": "black", "linewidth": 2},
        showcaps=False,
        # linecolor="#137",
        palette=my_pal,
        gap=0.2,
        boxprops=dict(facecolor="white", alpha=0.3, edgecolor="black"),
    )
    g1.get_legend().remove()

    g2 = sns.barplot(
        x="arms",
        y="Value",
        hue="Algorithm",
        errorbar=None,
        data=melted_df,
        palette=my_pal,
        saturation=0.75,
        gap=0.0,
    )

    # plt.yscale('log')
    # axins.set_xlim(0.95, 1.05)
    plt.xticks(range(number_of_arms))
    plt.ylabel("Number of pulls")
    plt.xlabel("arms")

    # extract the existing handles and labels
    h, l = g2.get_legend_handles_labels()

    ax.legend(
        h[len(all_pulls_result.keys()) :],
        l[len(all_pulls_result.keys()) :],
        loc="center left",
        bbox_to_anchor=(1, 0.5),
    )

    plt.title(algorithms_data.printing_name_dict[dataset_name])

    plt.savefig(path + dataset_name + "_pulls_arms.pdf", dpi=600, bbox_inches="tight")


    dfs = []
    for alg_num, alg_name in enumerate(all_pulls_result.keys()):
        data = {}
        data[algorithms_data.printing_name_dict[alg_name]] = pulls[
            :, :, alg_num, 0
        ].flatten()
        df = pd.DataFrame(data)
        dfs.append(df)
    combined_df = pd.concat(dfs)
    # Melt the combined DataFrame
    melted_df = pd.melt(combined_df)
    print(melted_df)
    melted_df = melted_df.rename(
        columns={"variable": "Algorithm", "value": "Optimal Arm Pull Count"}
    )
 
    ############################
    plt.rcParams["text.usetex"] = True
    # Fig size
    plt.rcParams["figure.figsize"] = 6, 6
    plt.rcParams.update({"font.size": 20})
    fig = plt.figure()

    sns.set_style("white")
    my_pal = plotting_utils.CB_color_cycle[: len(all_pulls_result.keys())]

    palette = my_pal
    ax = sns.violinplot(
        x="Algorithm",
        y="Optimal Arm Pull Count",
        data=melted_df,
        hue="Algorithm",
        dodge=False,
        palette=palette,
        scale="width",
        inner=None,
        rasterized=True,
        bw_adjust=0.5,
        cut=0,
    )
    xlim = ax.get_xlim()
    ylim = ax.get_ylim()
    for violin in ax.collections:
        bbox = violin.get_paths()[0].get_extents()
        x0, y0, width, height = bbox.bounds
        violin.set_clip_path(
            plt.Rectangle((x0, y0), width / 2, height, transform=ax.transData)
        )

    sns.boxplot(
        x="Algorithm",
        y="Optimal Arm Pull Count",
        data=melted_df,
        saturation=1,
        showfliers=False,
        width=0.3,
        boxprops={"zorder": 3, "facecolor": "none"},
        ax=ax,
    )
    old_len_collections = len(ax.collections)
    sns.stripplot(
        x="Algorithm",
        y="Optimal Arm Pull Count",
        data=melted_df,
        hue="Algorithm",
        palette=palette,
        dodge=False,
        ax=ax,
        s=np.sqrt(200 / len(instances)),
        alpha=0.5,
        rasterized=True,
    )
    for dots in ax.collections[old_len_collections:]:
        dots.set_offsets(dots.get_offsets() + np.array([0.25, 0]))
    ax.set_xlim(xlim)
    ax.set_ylim(ylim)
    ax.tick_params(axis="x", rotation=90)
    # ax.legend_.remove()
    plt.savefig(
        path + dataset_name + "_pulls_arms_violinplot.pdf", dpi=600, bbox_inches="tight"
    )


for i, dataset_name in enumerate(dataset_names):
    print(dataset_name)
    get_plot(i, dataset_name)