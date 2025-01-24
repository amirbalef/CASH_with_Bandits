import os
import pickle
import algorithms_data
import pandas as pd
import matplotlib.pyplot as plt
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

    alg_name = "MaxUCB"
    result_directory = "../results/"
    with open(
        result_directory + dataset_name + "/" + alg_name + "/result.pkl", "rb"
    ) as file:
        alg_res = pickle.load(file)
    with open(
        result_directory + dataset_name + "/" + alg_name + "/pulled_arms.pkl", "rb"
    ) as file:
        alg_pulls = pickle.load(file)
        
    data_list_s = []
    pull_s = []
    for intance_num, intance in enumerate(instances):
        data_list = []
        pull_list = []
        for trail in range(number_of_trails):
            data = raw_dataset[intance_num][:, trail]

            top_value = np.min(raw_dataset[intance_num][:, trail, :])
            baseline_value = np.max(raw_dataset[intance_num][:, trail, :])

            denominator = max(1e-5, baseline_value - top_value)
            data = (data - top_value) / denominator
            data = data[:, :horizon_time]
            data_min = np.min(data, axis=1)
            selected_arms = np.argsort(data_min)

            pulled_arms = np.bincount(
                alg_pulls[intance_num][trail], minlength=number_of_arms
            )

            data_list.append(data[selected_arms, :])
            pull_list.append(pulled_arms[selected_arms])
        data_list_s.append(data_list)
        pull_s.append(pull_list)
    dataset = np.asarray(data_list_s)
    pulls = np.asarray(pull_s)

    arms_data = []
    delta = []
    for i in range(number_of_arms):
        L = []
        for d in range(len(instances)):
            res = []
            data = 1 - dataset[d, :, i, :].flatten()
            high_epsilon = 1 - np.quantile(data, q=0.99)
            low_epsilon = 1 - np.quantile(data, q=0.01) 
            epsilon = np.linspace(high_epsilon, low_epsilon, 10)
            for eps in epsilon:
                if  eps <= 1 - np.quantile(data, q=0.99) or eps >= 1-np.quantile(data, q=0.01):
                    res.append(np.nan)
                else:
                    res.append(np.mean(data >= (1 - eps)) / eps)
            res = np.asarray(res)
            res = res[~np.isnan(res)]

            if i == 0:
                res = np.min(res)  # np.quantile(res, q=0.05)
            else:
                res = np.max(res)  #  np.quantile(res, q=0.95)

            L.append(res)
        arms_data.append(L)
        res = np.mean(
            np.min(dataset[:, :, i, :], axis=2) - np.min(dataset[:, :, 0, :], axis=2),
            axis=1,
        )
        delta.append(res)
    arms_data = np.asarray(arms_data)
    delta = np.asarray(delta)



    ###########################
    path = result_directory + "/plots_for_paper/app/fig_pulls_vs_theory/"
    if not os.path.exists(path):
        os.makedirs(path)


    alpha = 0.5
    T = horizon_time
    first_term = [np.nan]
    second_term = [np.nan]
    for i in range(1, number_of_arms):
        first_term.append(
            (T ** (1 - 2 * alpha * arms_data[0] * delta[i]))
            / (1 - 2 * alpha * arms_data[0] * delta[i])
        )
        second_term.append(2 * alpha * np.sqrt(arms_data[i] * T) * np.log(T))

    dfs = []
    data = {}
    for i in range(0, number_of_arms):
        data[str(i)] = pulls[:, :, i].flatten()
    df = pd.DataFrame(data)
    df["Dataset"] = "Real Experiment"
    dfs.append(df)

    data = {}
    for i in range(0, number_of_arms):
        data[str(i)] = first_term[i]
    df = pd.DataFrame(data)
    df["Dataset"] = "Theory"
    dfs.append(df)

    combined_df = pd.concat(dfs)
    # Melt the combined DataFrame
    melted_df = pd.melt(
        combined_df, id_vars=["Dataset"], var_name="arms", value_name="Value"
    )

    plt.rcParams["text.usetex"] = True
    # Fig size
    if(order==3):
        plt.rcParams["figure.figsize"] = 6, 6
    else:
        plt.rcParams["figure.figsize"] = 8, 6
    plt.rcParams.update({"font.size": 26})

    fig, ax = plt.subplots()
    my_pal = sns.color_palette("tab10")[0:number_of_arms]

    sns.barplot(
        x="arms",
        y="Value",
        hue="Dataset",
        data=melted_df,
        width=0.5,
        palette=my_pal,
    )
    if(order==3):
        plt.legend(
            loc="center left",
            bbox_to_anchor=(1, 0.5),
        )
    else:
        ax.legend().set_visible(False)

    # plt.yscale('log')
    # axins.set_xlim(0.95, 1.05)
    plt.xticks(range(number_of_arms))
    plt.ylabel("Number of pulls")
    plt.xlabel("arms")

    plt.title(algorithms_data.printing_name_dict[dataset_name])
    plt.savefig(path + dataset_name + "_pulls_vs_theory.pdf", dpi=600, bbox_inches="tight")

for i, dataset_name in enumerate(dataset_names):
    print(dataset_name)
    get_plot(i, dataset_name)