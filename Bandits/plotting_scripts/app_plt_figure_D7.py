import os
import algorithms_data
import analysis_utils
import pandas as pd
import exp_utils
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import matplotlib

dataset_names = ["TabRepo", "TabRepoRaw", "YaHPOGym", "Reshuffling"]


df_ranks = []
for dataset_name in dataset_names:
    dataset = pd.read_csv("../datasets/" + dataset_name + ".csv")

    instances = sorted(dataset["instance"].unique())
    all_arm_index_list = dataset["arm_index"].unique()
    valid_arm_index_list = [item for item in all_arm_index_list if item >= 0]
    number_of_arms = len(valid_arm_index_list)
    number_of_trails = len(dataset["repetition"].unique())
    horizon_time = len(dataset["iteration"].unique())
    classes = dataset["classifier"].unique()
    combined_search_algorithms = list(dataset[dataset["arm_index"] < 0]["optimizer"].unique())

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

    res_per_instance = analysis_utils.get_normallized_error_per_instance_time(
        all_result,
        number_of_arms,
        instances,
        number_of_trails,
        horizon_time
    )

    df = pd.DataFrame()
    for j, key in enumerate(all_result.keys()):
        k = algorithms_data.printing_name_dict[key]
        if key == "MaxUCB":
            k = "MaxUCB"
        df[k] = [np.mean(res_per_instance[i][j]) for i, item in enumerate(instances)]


    
    for instance_num, instance_name in enumerate(instances):
        data_list = []
        for trail in range(number_of_trails):
            data = raw_dataset[instance_num][:, trail]
            # print(dataset[intance].shape)
            top_value = np.min(raw_dataset[instance_num][:, :, :])
            baseline_value = np.max(raw_dataset[instance_num][:, :, :])
            denominator = max(1e-5, baseline_value - top_value)
            data = (data - top_value) / denominator
            data_min = np.min(data, axis=1)
            selected_arms = np.argsort(data_min)
            data_list.append(data[selected_arms, :])
        data_list = np.asarray(data_list)

    indexes = np.argsort(df["Oracle Arm"])
    df_rank = df.T
    df_rank = df_rank.reindex(columns=indexes)
    df_ranks.append(df_rank)

result = pd.concat(df_ranks, axis=1)

list_lenghth = [len(df_ranks[i].T) for i in range(len(df_ranks))]

path = result_directory + "/plots_for_paper/app/fig_norm_loss/"

if not os.path.exists(path):
    os.mkdir(path)


figsize = (16, 4)
title_size = 18

fig, axs = plt.subplots(
    1, 4, figsize=figsize, sharey=False, gridspec_kw={"width_ratios": [150, 50, 100, 50]}
)
plt.subplots_adjust(wspace=0.1, hspace=0.1)

cbar_ax = fig.add_axes([0.91, 0.2, 0.03, 0.6])

for i, dataset_name in enumerate(dataset_names):
    ax = axs[i]
    cmap = matplotlib.colormaps.get_cmap("RdYlGn_r")
    cmap.set_bad("white")
    print(sum(list_lenghth[:i]), sum(list_lenghth[:i]) + list_lenghth[i])
    sax = sns.heatmap(
        result.iloc[:, sum(list_lenghth[:i]) : sum(list_lenghth[:i]) + list_lenghth[i]],
        cmap=cmap,
        ax=ax,
        cbar=i == 0,
        cbar_ax=None if i else cbar_ax,
    )
    for _, spine in sax.spines.items():
        spine.set_visible(True)
    if i != 0:
        sax.set_yticks([])
    else:
        sax.tick_params(labelsize=title_size)

    ax.set_xticks([])
    ax.set_xlabel(algorithms_data.printing_name_dict[dataset_name], fontdict={"size": title_size})
    # ax.set_title("Ranks of methods per dataset", fontdict={"size": title_size})


plt.savefig(path + "norm_loss_per_dataset.pdf", dpi=600, bbox_inches="tight")
