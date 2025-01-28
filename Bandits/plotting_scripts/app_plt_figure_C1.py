import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import os
import seaborn as sns
import pylab


### run for each datasets
dataset_names = ["TabRepo", "TabRepoRaw", "YaHPOGym", "Reshuffling"]
dataset_name = dataset_names[0]


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

path = "../results/plots_for_paper/app/fig_reward_dist/"
if not os.path.exists(path):
    os.makedirs(path)

data_list_s = []
for intance_num in range(len(instances)):
    data_list = []
    scale_factor = []
    for trail in range(number_of_trails):
        data = raw_data[intance_num][:, trail]
        top_value = np.min(raw_data[intance_num][:, trail, :])
        baseline_value = np.max(raw_data[intance_num][:, trail, :])
        denominator = max(1e-5, baseline_value - top_value)
        data = (data - top_value) / denominator
        data = data[:, :time]
        data_min = np.min(data, axis=1)
        selected_arms = np.argsort(data_min)
        data_list.append(data[selected_arms, :])
    data_list_s.append(data_list)
dataset = np.asarray(data_list_s)


# ################################################################
# Ploting Delta
# ################################################################
Delta_i = []
Delta_i_mean = []
Delta_i_U = []
Delta_i_L = []
df = {}
for i in range(1, number_of_arms):
    res = np.min(dataset[:, :, i, :], axis=2) - np.min(dataset[:, :, 0, :], axis=2)
    Delta_i.append(res)
    Delta_i_mean.append(np.mean(res))
    Delta_i_U.append(np.quantile(res, q=0.95))
    Delta_i_L.append(np.quantile(res, q=0.05))
    df[str(i + 1)] = res.flatten()
df = pd.DataFrame.from_dict(df)

plt.rcParams["text.usetex"] = True
# Fig size
plt.rcParams["figure.figsize"] = 8, 4
plt.rcParams.update({"font.size": 26})

order = list(range(number_of_arms + 1))
order.remove(1)
order[number_of_arms - 1] = 1
order.remove(0)
my_pal = [sns.color_palette("tab10")[i] for i in order]

sns.boxplot(
    data=df,
    width=0.5,
    showmeans=True,
    meanline=True,
    showfliers=False,
    meanprops={"color": "k", "ls": "--", "lw": 2},
    palette=my_pal,
    saturation=0.75,
)
plt.xticks(range(number_of_arms))
plt.ylabel("$\Delta_i$")
plt.xlabel("Sub-optimal arm")
plt.savefig(path + dataset_name + "_Delta_arms.pdf", dpi=600, bbox_inches="tight")

# ################################################################
# Ploting CDF (with distribution shifts)
# ################################################################
from seaborn.utils import desaturate

plt.rcParams["text.usetex"] = True
# Fig size
plt.rcParams["figure.figsize"] = 8, 4
plt.rcParams.update({"font.size": 26})
fig, ax = plt.subplots()

number_of_stationary_pieces = 5

arm = number_of_arms - 1
window = time // number_of_stationary_pieces

order = list(range(number_of_arms + 1))
order.remove(1)
order[number_of_arms - 1] = 1
my_pal = [sns.color_palette("tab10")[i] for i in order]


for arm in range(0, number_of_arms):
    list_of_cdfs_per_seqments = []
    for n in range(number_of_stationary_pieces):
        list_of_cdfs = []
        for i, intance in enumerate(instances):
            list_of_cdfs.append(
                np.sort(
                    1 - dataset[i, :, arm, (n) * window : (n + 1) * window].flatten()
                )
            )
        list_of_cdfs = np.asarray(list_of_cdfs)
        list_of_cdfs_per_seqments.append(list_of_cdfs)

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
    label = Ordinal_numbers[arm] + " arm"
    if arm == number_of_arms - 1:
        label = "Worst arm"
    if arm == 0:
        label = "Optimal arm"

    for n in range(number_of_stationary_pieces):
        list_of_cdfs = list_of_cdfs_per_seqments[n]
        data = np.mean(list_of_cdfs, axis=0).flatten()
        ax.plot(
            np.sort(data),
            1- np.linspace(0, 1, len(data), endpoint=False),
            linewidth=1,
            color=desaturate(my_pal[arm], 0.75),
            alpha=0.5,
        )
    data = np.mean(list_of_cdfs_per_seqments, axis=0)
    data = np.mean(data, axis=0).flatten()

    ax.plot(
        np.sort(data),
        1 - np.linspace(0, 1, len(data), endpoint=False),
        label=label,
        linewidth=3,
        linestyle="--",
        color=desaturate(my_pal[arm], 0.75),
    )

# axins.set_xlim(0.95, 1.05)
plt.ylabel("Survival function")
plt.xlabel("Reward")

# plt.title(algorithms_data.algoirthm_names_dict[dataset_name])
# plt.legend()
# Put a legend below current axis
# show the graph
# if dataset_name == "TabRepo":
#     plt.legend(
#         loc="center left", bbox_to_anchor=(1, 0.5), ncol=(number_of_arms + 1) // 2
#     )

# show the graph
plt.savefig(
    path + dataset_name + "_HPO_eCDF_shifts.pdf", dpi=600, bbox_inches="tight"
)

figlegend = pylab.figure(figsize=(3, 8))
pylab.figlegend(
    *ax.get_legend_handles_labels(), loc="upper left", ncol=(number_of_arms + 1)
)
figlegend.savefig(
    path  + dataset_name + "_legend.pdf",
    dpi=600,
    bbox_inches="tight",
)


# ################################################################
# Ploting eCDF (lines)
# ################################################################

from seaborn.utils import desaturate

plt.rcParams["text.usetex"] = True
# Fig size
plt.rcParams["figure.figsize"] = 8, 4
plt.rcParams.update({"font.size": 26})
fig, ax = plt.subplots()

arm = number_of_arms - 1


for i, intance in enumerate(instances):
    # for i in [1]:
    # dataframe
    df = pd.DataFrame(
        {
            "Score": 1 - dataset[i, :, 0, :].flatten(),
            "Suboptimal": 1 - dataset[i, :, arm, :].flatten(),
        }
    )
    # plot
    ax.plot(
        np.sort(df.Score),
        1 - np.linspace(0, 1, len(df.Score), endpoint=False),
        linewidth=2,
        color=desaturate("tab:blue", 0.75),
        alpha=0.1,
    )
    ax.plot(
        np.sort(df.Suboptimal),
        1 - np.linspace(0, 1, len(df.Suboptimal), endpoint=False),
        linewidth=2,
        color=desaturate("tab:orange", 0.75),
        alpha=0.1,
    )


list_of_cdfs = []
for i, intance in enumerate(instances):
    list_of_cdfs.append(np.sort(1 - dataset[i, :, 0, :].flatten()))
list_of_cdfs = np.asarray(list_of_cdfs)
print(list_of_cdfs.shape)

list_of_cdfs_arm = []
for i, intance in enumerate(instances):
    list_of_cdfs_arm.append(np.sort(1 - dataset[i, :, arm, :].flatten()))

# plot
ax.plot(
    np.sort(np.mean(list_of_cdfs, axis=0)),
    1 - np.linspace(0, 1, len(list_of_cdfs[0]), endpoint=False),
    label="Optimal arm",
    color=desaturate("tab:blue", 0.75),
    linewidth=3,
)
ax.plot(
    np.sort(np.mean(list_of_cdfs_arm, axis=0)),
    1 - np.linspace(0, 1, len(list_of_cdfs[0]), endpoint=False),
    label="Worst arm",
    color=desaturate("tab:orange", 0.75),
    linewidth=3,
)

if 0:  # dataset_name == "TabRepo_gen":
    # inset Axes....
    x1, x2, y1, y2 = 0.9, 1.1, 0.9, 1.1  # subregion of the original image
    axins = ax.inset_axes(
        [0.1, 0.2, 0.4, 0.4],
        xlim=(x1, x2),
        ylim=(y1, y2),
        xticklabels=[],
        yticklabels=[],
    )

    axins.plot(
        np.sort(df.Score),
        np.linspace(0, 1, len(df.Score), endpoint=False),
        label="Optimal arm",
        linewidth=4,
    )
    axins.plot(
        np.sort(df.Suboptimal),
        np.linspace(0, 1, len(df.Suboptimal), endpoint=False),
        label="Worst arm",
        linewidth=4,
    )

    axins.annotate(
        "$\Delta_i$",
        xy=(1.00, 1.00),
        xycoords="data",
        xytext=(0.34, 0.55),
        textcoords="figure fraction",
        color="Black",
    )
    axins.annotate(
        " ",
        xy=(1.0, 1.005),
        xycoords="data",
        xytext=(0.32, 0.505),
        textcoords="figure fraction",
        arrowprops=dict(arrowstyle="<->"),
    )

    axins.set_ylim(0.90, 1.05)
    axins.set_xlim(0.97, 1.02)
    ax.indicate_inset_zoom(axins, edgecolor="black")


# axins.set_xlim(0.95, 1.05)
plt.ylabel("Survival function")
plt.xlabel("Reward")

# plt.title(algorithms_data.algoirthm_names_dict[dataset_name] )
# plt.legend()


# show the graph
plt.savefig(path + dataset_name + "_HPO_eCDF_lines.pdf", dpi=600, bbox_inches="tight")
