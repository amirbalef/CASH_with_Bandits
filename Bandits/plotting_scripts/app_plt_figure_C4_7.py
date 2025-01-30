import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import algorithms_data
import os
from seaborn.utils import desaturate
from matplotlib.ticker import FormatStrFormatter

### run for each datasets
dataset_names = ["TabRepo", "TabRepoRaw", "YaHPOGym", "Reshuffling"]
dataset_name = dataset_names[1]


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

path = "../results/plots_for_paper/app/fig_assumption/"
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
# Ploting L and U values
# ################################################################

plt.rcParams["figure.figsize"] = 24, 18
plt.rcParams.update({"font.size": 26})

high_epsilon = 1.0
low_epsilon = 0.00

epsilon = np.linspace(low_epsilon, high_epsilon, 1000)

fig, axs = plt.subplots(3, number_of_arms)  # , sharey=True
print(axs.shape)
plt.subplots_adjust(left=0.0, right=1.0, top=0.9, bottom=0.1)
plt.subplots_adjust(wspace=0.1, hspace= 0.3)

for i in range(number_of_arms):
    L = []
    selected_instances =[]
    for d in range(len(instances)):
        all_stds = np.asarray([np.std(dataset[d, :, i, :].flatten())  for i in range(number_of_arms)])
        if np.any(all_stds <= 0.001):
            continue
        res = []
        data = 1 - dataset[d, :, i, :].flatten()
        for eps in epsilon:
            if   eps <= 1 - np.quantile(data, q=0.99) or eps >= 1 -np.quantile(data, q=0.01):
                res.append(np.nan)
            else:
                res.append(np.mean(data > (1 - eps)) / eps)
        L.append(res)
        selected_instances.append(instances[d])
    L = np.asarray(L)
    print("number of selected instances", len(selected_instances))
    axs[0,i].plot(
        epsilon,
        np.nanmean(L, axis=0),
        label="Average $\\frac{G_i(b- \epsilon)}{\epsilon}$",
        linewidth=5,
        linestyle="--",
        color=desaturate("tab:orange", 0.50),
        zorder = 1000,
    )
    for d in range(len(selected_instances)):
        

        scatter_size = 5 + (500 // len(selected_instances))
        line_width = 1 + (30 / len(selected_instances))
        if d == 0:
            label = "$\\frac{G_i(b- \epsilon)}{\epsilon}$ (per dataset)"
            axs[0, i].plot(
                epsilon,
                L[d],
                linewidth=line_width,
                color=desaturate("tab:orange", 0.75),
                alpha=0.5,
                label=label,
            )

            label = "$L_i=\\min(\\frac{G_i(b- \epsilon)}{\epsilon})$"
            axs[0, i].scatter(
                epsilon[np.nanargmin(L[d])],
                np.nanmin(L[d]),
                color=desaturate("#377eb8", 0.75),
                alpha=0.8,
                s=scatter_size,
                label=label,
                zorder=999,
            )
            label = "$U_i=\\max(\\frac{G_i(b- \epsilon)}{\epsilon})$"
            axs[0, i].scatter(
                epsilon[np.nanargmax(L[d])],
                np.nanmax(L[d]),
                color=desaturate("#e41a1c", 0.75),
                alpha=0.8,
                s=scatter_size,
                label=label,
                zorder=999,
            )
        else:
                axs[0, i].plot(
                    epsilon,
                    L[d],
                    linewidth=line_width,
                    color=desaturate("tab:orange", 0.75),
                    alpha=0.5,
                )

                axs[0, i].scatter(
                    epsilon[np.nanargmin(L[d])],
                    np.nanmin(L[d]),
                    color=desaturate("#377eb8", 0.75),
                    alpha=0.8,
                    s=scatter_size,
                    zorder=999,
                )
                axs[0, i].scatter(
                    epsilon[np.nanargmax(L[d])],
                    np.nanmax(L[d]),
                    color=desaturate("#e41a1c", 0.75),
                    alpha=0.8,
                    s=scatter_size,
                    zorder=999,
                )

    axs[0, i].yaxis.set_major_formatter(FormatStrFormatter("%.1f"))
    if i == 0:
        title = "Optimal arm"
        axs[0, i].set_ylabel("$\\frac{G(b - \\epsilon)}{\\epsilon}$")

    elif i == number_of_arms - 1:
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
        title = Ordinal_numbers[i] + " arm"
    axs[0, i].title.set_text(title)

    if dataset_name == "YaHPOGym" :
        axs[0, i].set_yscale("log")
        axs[0, i].set_ylim(top=500, bottom=0.1)
    if dataset_name == "TabRepo":
        axs[0, i].set_yscale("log")
        axs[0, i].set_ylim(top=200, bottom=0.4)
    if dataset_name == "TabRepoRaw":
        axs[0, i].set_yscale("log")
        axs[0, i].set_ylim(top=200, bottom=0.4)
    if dataset_name == "Reshuffling":
        axs[0, i].set_yscale("log")
        axs[0, i].set_ylim(top=250, bottom=0.1)
    axs[0, i].set_xlabel("$\\epsilon$")
    

    bins = np.logspace(-2, 2, num=20)  # Custom bins for list1
    # Plot histogram for list1
    axs[1, i].hist(
        np.nanmin(L, axis=1),
        bins=bins,
        color=desaturate("#377eb8", 0.75),
        alpha=0.8,
        edgecolor="#377eb8",
    )
    # axes[0].set_title("Histogram of L")
    axs[1, i].set_xlabel("Values for L")
    axs[1, i].set_xscale("log")
    axs[1, i].set_xticks([0.1, 1, 10])  # Set ticks to bin edges
    axs[1, i].set_xticklabels([0.1, 1, 10])  # Use bin edges as tick labels

    bins = np.logspace(-2, 3, num=20)  # Custom bins for list1

    # Plot histogram for list2
    axs[2, i].hist(
        np.nanmax(L, axis=1),
        bins=bins,
        color=desaturate("#e41a1c", 0.65),
        alpha=0.8,
        edgecolor=desaturate("#e41a1c", 0.45),
    )

    # axes[1].set_title("Histogram of U")
    axs[2, i].set_xlabel("Values for U")
    axs[2, i].set_xscale("log")
    axs[2, i].set_xticks([0.01, 1, 100])  # Set ticks to bin edges
    axs[2, i].set_xticklabels([0.01, 1, 100])  # Use bin edges as tick labels

    if(i==0):
        axs[1, i].set_ylabel("Frequency")
        axs[2, i].set_ylabel("Frequency")
    else:
        axs[0, i].set_yticks([])
        axs[1, i].set_yticks([])
        axs[2, i].set_yticks([])

handles, labels = axs[0,0].get_legend_handles_labels()
leg = fig.legend(
    handles, labels, bbox_to_anchor=(0, -0.05, 1, 1), loc="lower center", ncol=6
)
for handle in leg.legend_handles:
    handle._sizes = [30]
    handle.set_alpha(1)

fig.add_subplot(111, frameon=False)
plt.tick_params(
    labelcolor="none", which="both", top=False, bottom=False, left=False, right=False
)
#plt.ylabel(algorithms_data.printing_name_dict[dataset_name], position=(1, 25))
#plt.xlabel("$\epsilon$", position=(1, 25))
plt.savefig(path + dataset_name + "_L_U.pdf", dpi=600, bbox_inches="tight")
