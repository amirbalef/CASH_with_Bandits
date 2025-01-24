import os
import pickle
import plotting_utils
import algorithms_data
import analysis_utils
import pandas as pd
import exp_utils
import matplotlib.pyplot as plt
import pylab
import numpy as np
import seaborn as sns
import matplotlib

dataset_name = "YaHPOGym"
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

#b_steps = np.arange(5, 9, 1)
b_steps = [5,6,7, 8]
for item in b_steps:
    policy_algorithms["MaxUCB_burn-in_" + str(item)] = 1

policy_algorithms["Rising_Bandit"] = 1

policy_algorithms["SMAC"] = 1
policy_algorithms["SMAC_NoInit"] = 1
policy_algorithms["RandomSearch"] = 1
policy_algorithms["Oracle_Arm"] = 1

result_directory = "../results/"
all_result = exp_utils.fetch_results(
    policy_algorithms, result_directory, dataset_name
)

path = result_directory + "/plots_for_paper/app/fig_burn-in/"

if not os.path.exists(path):
    os.mkdir(path)

#shift_metric = "median"
shift_metric = "mean"
selected_arm = 0
number_of_stationary_pieces = 20

shifts = []
changes = []
df_ranks = []

window = horizon_time // number_of_stationary_pieces


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
    if "MaxUCB_burn-in" in key:
        k = "MaxUCB-Burn-in(C="+key.split("_")[-1]+")"

    df[k] = [np.mean(res_per_instance[i][j]) for i, item in enumerate(instances)]


shifts_list = []
changes_list = []
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

    change_per_peice = []
    metric_list_per_peice = []
    for n in range(number_of_stationary_pieces):
        data_stationary_piece = 1 - data_list[:, :, (n) * window : (n + 1) * window]
        if shift_metric == "median":
            metric_list_per_peice.append(
                np.median(data_stationary_piece[:, selected_arm])
            )
        if shift_metric == "mean":
            metric_list_per_peice.append(
                np.mean(np.max(data_stationary_piece[:, selected_arm], axis=1))
            )
        fraction_u = np.mean(data_stationary_piece[:, selected_arm], axis=0)
        data_stationary_piece[:, selected_arm] = 0
        fraction_d = np.max(data_stationary_piece[:, :], axis=1)
        fraction_d = np.mean(fraction_d, axis=0)

        change_per_peice.append(np.mean(fraction_d > fraction_u))
    metric_list_per_peice = np.asarray(metric_list_per_peice)
    change_list_per_peice = np.asarray(change_per_peice)

    shift = np.max(
        (metric_list_per_peice - metric_list_per_peice[0]) / metric_list_per_peice[0]
    )
    change = np.mean(change_list_per_peice)
    shifts_list.append(shift)
    changes_list.append(change)
changes_list = np.asarray(changes_list)
shifts_list = np.asarray(shifts_list)
indexes = np.argsort(shifts_list)
changes.append(changes_list[indexes])
shifts.append(shifts_list[indexes])

df_rank = df.T
df_rank = df_rank.reindex(columns=indexes)
for item in indexes:
    print(instances[int(item)])
df_ranks.append(df_rank)
result = pd.concat(df_ranks, axis=1)

list_lenghth = [len(df_ranks[i].T) for i in range(len(df_ranks))]

figsize = (18, 10)
title_size = 20
plt.rcParams.update({"font.size": 22})

fig, axs = plt.subplots(
    2,
    1,
    figsize=figsize,
    sharey=False,
    gridspec_kw={
        "width_ratios": [100],
        "height_ratios": [10, 50],
    },
)
plt.subplots_adjust(wspace=0.1, hspace=0.1)


cbar_ax = fig.add_axes([0.91, 0.2, 0.03, 0.6])
i = 0
axs[0].plot(shifts[i], "-", color="tab:orange")
# axs[0].plot(changes[i], "-", color="tab:blue")
axs[0].set_xticks([])
axs[0].margins(x=0)
axs[0].set_ylim([0, 1])

axs[0].set_ylabel("Maximum shift\nof mean", fontsize=24, rotation=0, labelpad=100)

ax = axs[1]
cmap = matplotlib.colormaps.get_cmap("RdYlGn_r")
cmap.set_bad("white")

sax = sns.heatmap(
    result.iloc[:, sum(list_lenghth[:i]) : sum(list_lenghth[:i]) + list_lenghth[i]],
    cmap=cmap,
    ax=ax,
    cbar=i == 0,
    cbar_ax=None if i else cbar_ax,
    # vmin=0,
    # vmax=1,
    # norm=LogNorm(),
    # cbar_kws={"ticks": [0, 1, 10, 1e2, 1e3, 1e4, 1e5]},
    # norm=LogNorm(vmin=10e-3, vmax=1, clip=True),
)
for _, spine in sax.spines.items():
    spine.set_visible(True)
if i != 0:
    sax.set_yticks([])
else:
    sax.tick_params(labelsize=title_size)

ax.set_xticks([])
ax.set_xlabel(
    "Tasks-" + algorithms_data.printing_name_dict[dataset_name]+"",
    fontdict={"size": title_size},
)
# ax.set_title("Ranks of methods per dataset", fontdict={"size": title_size})

plt.savefig(path + "norm_loss_per_dataset_shifts.pdf", dpi=600, bbox_inches="tight")


############


##############
import plotting_utils


from cycler import cycler

colors = [plotting_utils.CB_color_cycle[0]]
all_cyclers = cycler(color=colors) * cycler(linestyle=["-"])  # , "--"])  # ,"--"

colors = [
   # plotting_utils.CB_color_cycle[5],
    plotting_utils.CB_color_cycle[1],
    plotting_utils.CB_color_cycle[2],
]
all_cyclers = all_cyclers.concat(cycler(color=colors) * cycler(linestyle=["-", ":"]))  # , "--"])  # ,"--"

all_cyclers = all_cyclers.concat(
    cycler(color=[plotting_utils.CB_color_cycle[3]]) * cycler(linestyle=["-"])
)

if dataset_name != "hebo":
    colorcycler = cycler(color=["black"])
    lines = ["-", ":"]
    if dataset_name == "TabRepo_RS":
        lines = [":"]
    if "SMAC_NoInit" in policy_algorithms:
        lines = ["-", "--", ":"]
    linecycler = cycler(linestyle=lines)
    all_cyclers = all_cyclers.concat(colorcycler * linecycler)
else:
    colorcycler = cycler(color=["black"])
    lines = [":"]
    linecycler = cycler(linestyle=lines)
    all_cyclers = all_cyclers.concat(colorcycler * linecycler)


if "Oracle_Arm" in policy_algorithms:
    colorcycler = cycler(color=["grey"])
    lines = ["-"]
    linecycler = cycler(linestyle=lines)
    all_cyclers = all_cyclers.concat(colorcycler * linecycler)


if "Arm_0" in policy_algorithms:
    colorcycler = cycler(color=plotting_utils.CB_color_cycle)
    lines = [":"]
    linecycler = cycler(linestyle=lines)
    all_cyclers = all_cyclers.concat(colorcycler * linecycler)


data = {}
data["all_result"] = all_result
data["dataset"]  = dataset
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
if data["dataset_name"] == "YaHPOGym":
    data["set_ylim"] = (0.3, 0.05)
    data["legend"] = "seperate"

data["plot_type"] = "Ranking"
data["saving_name"] = "ranking"
data["ylabel"] = "Ranking"
data["plot_confidence_type"] = "mean_std"
plotting_utils.plot_averaged_on_datasets(data)

data["plot_type"] = "Normalized loss"
data["saving_name"] = "norm_loss"
data["plot_confidence_type"] = "mean_std"
data["ylabel"] = "Normalized loss"
plotting_utils.plot_averaged_on_datasets(data)