import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import algorithms_data
import os
from seaborn.utils import desaturate
from matplotlib.ticker import FormatStrFormatter
import pylab

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
# Ploting L and U values (in the paper])
# ################################################################

values_L = []
values_U = []

W_values_L = []
W_values_U = []


plt.rcParams["figure.figsize"] = 8, 4.5
plt.rcParams.update({"font.size": 24})

high_epsilon = 1.0
low_epsilon = 0.00

epsilon = np.linspace(low_epsilon, high_epsilon, 1000)

fig, axs = plt.subplots(1, 3, sharey=True)  # , sharey=True
plt.subplots_adjust(left=0.0, right=1.0, top=0.9, bottom=0.3)
arm_listt = [0, number_of_arms // 2, number_of_arms - 1]
for i in range(len(arm_listt)):
    L = []
    for d in range(len(instances)):
        data = 1 - dataset[d, :, arm_listt[i], :].flatten()
        # high_epsilon = 1 - np.quantile(data, q=0.99)
        # low_epsilon = 1 - np.quantile(data, q=0.01) 
        # epsilon = np.linspace(high_epsilon, low_epsilon, 100)
        res = []
        for eps in epsilon:
            if   eps <= 1 - np.quantile(data, q=0.99) or eps >= 1 -np.quantile(data, q=0.01):
                res.append(np.nan)
            else:
                res.append(np.mean(data >= (np.quantile(data, q=0.99) - eps)) / eps)
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
                linewidth=1.5,
                color=desaturate("tab:orange", 0.75),
                alpha=0.3,
                label=label,
            )
            label = "$L_i=\\min(\\frac{G_i(b- \epsilon)}{\epsilon})$"
            axs[i].scatter(
                epsilon[np.nanargmin(L[d])],
                np.nanmin(L[d]),
                color=desaturate("tab:red", 0.75),
                alpha=0.3,
                s=3,
                label=label,
                zorder = 99
            )
            label = "$U_i=\\max(\\frac{G_i(b- \epsilon)}{\epsilon})$"
            axs[i].scatter(
                epsilon[np.nanargmax(L[d])],
                np.nanmax(L[d]),
                color=desaturate("tab:blue", 0.75),
                alpha=0.3,
                s=3,
                label=label,
                zorder=99,
            )
        else:
            axs[i].plot(
                epsilon,
                L[d],
                linewidth=1.5,
                color=desaturate("tab:orange", 0.75),
                alpha=0.3,
            )

            axs[i].scatter(
                epsilon[np.nanargmin(L[d])],
                np.nanmin(L[d]),
                color=desaturate("tab:red", 0.75),
                alpha=0.3,
                s=3,
                zorder=99,
            )
            axs[i].scatter(
                epsilon[np.nanargmax(L[d])],
                np.nanmax(L[d]),
                color=desaturate("tab:blue", 0.75),
                alpha=0.3,
                s=3,
                zorder=99,
            )
        if i==0:
            values_L.append(np.nanmin(L[d]))
            values_U.append(np.nanmax(L[d]))

        if i == len(arm_listt) - 1:
            W_values_L.append(np.nanmin(L[d]))
            W_values_U.append(np.nanmax(L[d]))

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

    if dataset_name == "TabRepoRaw":
        axs[i].set_yscale("log")
        axs[i].set_ylim(top=200, bottom=0.4)

    if dataset_name == "YaHPOGym" :
        axs[i].set_yscale("log")
        axs[i].set_ylim(top=500, bottom=0.1)


if dataset_name == "YaHPOGym" :
    handles, labels = axs[0].get_legend_handles_labels()
    order = [0, 3, 1, 4, 2]
    labels, handles =[labels[idx] for idx in order], [handles[idx] for idx in order]
    leg = fig.legend(
        handles,
        labels,
        bbox_to_anchor=(-0.06, -0.06, 1, 1),
        loc="lower center",
        ncol=3,
        prop={"size": 15},
    )
    for handle in leg.legend_handles:
        handle._sizes = [30]
        handle.set_alpha(1)

fig.add_subplot(111, frameon=False)
plt.tick_params(
    labelcolor="none", which="both", top=False, bottom=False, left=False, right=False
)
plt.ylabel(algorithms_data.printing_name_dict[dataset_name], position=(1, 25))
# plt.yscale("log")
plt.xlabel("$\epsilon$", position=(1, 25))
# fig.suptitle(algorithms_data.algoirthm_names_dict[dataset_name])
plt.savefig(path + dataset_name + "_L_U_S.pdf", dpi=600, bbox_inches="tight")


print(W_values_U)

# Create a figure with two subplots
fig, axes = plt.subplots(1, 2, figsize=(8, 4), sharey=True)
plt.subplots_adjust(wspace=0)

bins = np.logspace(-1.1, 1.5, num=20)  # Custom bins for list1
# Plot histogram for list1
axes[0].hist(values_L, bins=bins, color="tab:blue", alpha=0.3, edgecolor="tab:blue", label = "Optimal arm")
axes[0].hist(
    W_values_L,
    bins=bins,
    color="tab:orange",
    alpha=0.3,
    edgecolor="tab:orange",
    label="Worst arm",
)
#axes[0].set_title("Histogram of L")
axes[0].set_xlabel("Values for L")
axes[0].set_ylabel("Frequency")
# axes[0].set_xticks(bins)  # Set ticks to bin edges
# axes[0].set_xticklabels(bins)  # Use bin edges as tick labels
axes[0].set_xscale("log")
#axes[0].legend(loc="upper left", fontsize=16)


bins = np.logspace(-1.0, 3, num=20)  # Custom bins for list1

# Plot histogram for list2
axes[1].hist(
    values_U,
    bins=bins,
    color="tab:blue",
    alpha=0.3,
    edgecolor="tab:blue",
    label="Optimal arm",
)
axes[1].hist(W_values_U, bins=bins, color="tab:orange", alpha=0.3, edgecolor="tab:orange", label="Worst arm",
)
#axes[1].set_title("Histogram of U")
axes[1].set_xlabel("Values for U")
# axes[1].set_xticks(bins)  # Set ticks to bin edges
# axes[1].set_xticklabels(bins)  # Use bin edges as tick labels
axes[1].set_xscale("log")
#axes[1].legend(loc="upper left", fontsize=16)

plt.savefig(path + dataset_name + "_L_U_Hist.pdf", dpi=600, bbox_inches="tight")

figlegend = pylab.figure(figsize=(8, 2))
pylab.figlegend(*axes[0].get_legend_handles_labels(), loc="upper left", ncol=2)
figlegend.savefig(
    path + "legend.pdf",
    dpi=600,
    bbox_inches="tight",
)

