import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import algorithms_data
import os
from seaborn.utils import desaturate
from matplotlib.ticker import FormatStrFormatter

### run for each datasets
dataset_name = "TabRepoRaw"
dataset_name = "YaHPOGym"

dataset = pd.read_csv("../datasets/" + dataset_name + ".csv")

instances = sorted(dataset["instance"].unique())
all_arm_index_list = dataset["arm_index"].unique()
valid_arm_index_list = [item for item in all_arm_index_list if item >= 0]
number_of_arms = len(valid_arm_index_list)
number_of_trails = len(dataset["repetition"].unique())
horizon_time = len(dataset["iteration"].unique())
classes = dataset["classifier"].unique()
##
df = dataset[(dataset["arm_index"] >= 0)]
raw_data = df.sort_values(by=["instance", "arm_index", "repetition", "iteration"])[
    "loss"
].values.reshape(len(instances), number_of_arms, number_of_trails, horizon_time)
time = horizon_time

path = "../results/plots_for_paper/fig_assumption/"
if not os.path.exists(path):
    os.makedirs(path)

data_list_s = []
for intance_num in range(len(instances)):
    data_list = []
    scale_factor = []
    for trail in range(number_of_trails):
        data = raw_data[intance_num][:, trail]
        top_value = np.min(raw_data[intance_num][:, :, :])
        baseline_value = np.max(raw_data[intance_num][:, :, :])
        denominator = max(1e-5, baseline_value - top_value)
        data = (data - top_value) / denominator
        data = data[:, :time]
        data_min = np.min(data, axis=1)
        selected_arms = np.argsort(data_min)
        data_list.append(data[selected_arms, :])
    data_list_s.append(data_list)
dataset = np.asarray(data_list_s)



# ################################################################
# Ploting L and U values (in the paper])
# ################################################################

plt.rcParams["figure.figsize"] = 8, 4.5
plt.rcParams.update({"font.size": 20})

high_epsilon = 1.0
low_epsilon = 0.00

epsilon = np.linspace(high_epsilon, low_epsilon, 40)

fig, axs = plt.subplots(1, 3, sharey=True)  # , sharey=True
plt.subplots_adjust(left=0.0, right=1.0, top=0.9, bottom=0.3)
arm_listt = [0, number_of_arms // 2, number_of_arms - 1]
for i in range(3):
    L = []
    for d in range(len(instances)):
        res = []
        for eps in epsilon:
            data = 1 - dataset[d, :, arm_listt[i], :].flatten()
            if eps >= np.quantile(data, q=0.95) or eps <= np.quantile(data, q=0.05):
                res.append(np.nan)
            else:
                res.append(np.mean(data >= (1 - eps)) / eps)
        L.append(res)
    L = np.asarray(L)
    # axs[i].plot(
    #     epsilon,
    #     np.nanmean(L, axis=0),
    #     label="Average $\\frac{G_i(b- \epsilon)}{\epsilon}$",
    #     linewidth=3,
    #     linestyle="--",
    #     color=desaturate("tab:orange", 0.75),
    # )

    for d in range(len(instances)):
        if d == 0:
            label = "$\\frac{G_i(b- \epsilon)}{\epsilon}$ (per dataset)"
            axs[i].plot(
                epsilon,
                L[d],
                linewidth=3,
                color=desaturate("tab:orange", 0.75),
                alpha=0.4,
                label=label,
            )
        else:
            axs[i].plot(
                epsilon,
                L[d],
                linewidth=3,
                color=desaturate("tab:orange", 0.75),
                alpha=0.4,
            )
    # axs[i].fill_between(
    #     epsilon,
    #     np.nanquantile(L, q=0.05, axis=0),
    #     np.nanquantile(L, q=0.95, axis=0),
    #     alpha=0.2,
    #     linewidth=2,
    #     color="tab:orange",
    # )
    axs[i].hlines(
        np.nanmean(np.nanquantile(L, q=0.00, axis=1)),
        xmin=low_epsilon,
        xmax=high_epsilon,
        label="Average $L_i$",
        linestyle="--",
        color=desaturate("tab:red", 0.75),
        linewidth=3,
    )

    axs[i].hlines(
        np.nanmean(np.nanquantile(L, q=1.0, axis=1)),
        xmin=low_epsilon,
        xmax=high_epsilon,
        label="Average $U_i$",
        linestyle="--",
        color=desaturate("tab:blue", 0.75),
        linewidth=3,
    )

    axs[i].yaxis.set_major_formatter(FormatStrFormatter("%.1f"))
    if arm_listt[i] == 0:
        title = "Optimal arm"
    elif arm_listt[i] == number_of_arms - 1:
        title = "Worst arm"
    else:
        Ordinal_numbers = [
        "1st",
        "2nd",
        "3rd",
        "4th",
        "5th",
        "6th",
        "7th",
        "8th",
        "9th",
        "10th",
    ]
        title = Ordinal_numbers[arm_listt[i]] + " arm"

    if dataset_name != "YaHPOGym" :
        axs[i].title.set_text(title)

    if dataset_name == "YaHPOGym" :
        axs[i].set_ylim(top=5)

if dataset_name == "YaHPOGym" :
    handles, labels = axs[0].get_legend_handles_labels()
    fig.legend(
        handles,
        labels,
        bbox_to_anchor=(-0.06, -0.02, 1, 1),
        loc="lower center",
        ncol=3,
        prop={"size": 19},
    )

fig.add_subplot(111, frameon=False)
plt.tick_params(
    labelcolor="none", which="both", top=False, bottom=False, left=False, right=False
)
plt.ylabel(algorithms_data.printing_name_dict[dataset_name], position=(1, 25))
# plt.yscale("log")
plt.xlabel("$\epsilon$", position=(1, 25))
# fig.suptitle(algorithms_data.algoirthm_names_dict[dataset_name])
plt.savefig(path + dataset_name + "_L_U_S.pdf", dpi=600, bbox_inches="tight")