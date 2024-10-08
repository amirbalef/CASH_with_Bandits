import sys
sys.path.append("../")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from cycler import cycler
import plotting_utils
import exp_utils
import algorithms_data
from scipy.stats import gaussian_kde

dataset_name = "TabRepoRaw"
dataset = pd.read_csv("../../datasets/" + dataset_name + ".csv")

instances = sorted(dataset["instance"].unique())
all_arm_index_list = dataset["arm_index"].unique()
valid_arm_index_list = [item for item in all_arm_index_list if item >= 0]
number_of_arms = len(valid_arm_index_list)
number_of_trails = len(dataset["repetition"].unique())
horizon_time = len(dataset["iteration"].unique())
classes = dataset["classifier"].unique()
##
instance_num = instances.index("ilpd")

policy_algorithms = {}
policy_algorithms["SMAC"] = 1
policy_algorithms["MaxUCB"] = 1
result_directory = "../../results/"
all_result = exp_utils.fetch_results(policy_algorithms, result_directory, dataset_name)


def improve_legend(ax=None):
    if ax is None:
        ax = plt.gca()
    adjustment = [0] * len(policy_algorithms)
    adjustment.extend([ -8, -5, 0, 7, 0, 0, 0])
    for i, line in enumerate(ax.lines):
        print(i, line.get_label())
        data_x, data_y = line.get_data()
        right_most_x = data_x[-1]
        right_most_y = data_y[-1]
        ax.annotate(
            line.get_label(),
            xy=(right_most_x, right_most_y),
            xytext=(5, adjustment[i]),
            textcoords="offset points",
            va="center",
            color=line.get_color(),
        )
    ax.legend().set_visible(False)


plt.rcParams.update({"font.size": 30})

colors = ["black"]
if("MaxUCB" in policy_algorithms):
    colors = ["black", "blue"]
all_cyclers = cycler(color=colors) * cycler(
    linestyle=["-"]
)
colors = plotting_utils.CB_color_cycle
myorder = [0, 1, 6, 3, 4, 5, 2]
colors = [colors[i] for i in myorder]
colorcycler = cycler(color=colors)
lines = ["-"]
linecycler = cycler(linestyle=lines)
all_cyclers = all_cyclers.concat(colorcycler * linecycler)


fig, ax = plt.subplots(figsize=(7, 8))
ax.set_prop_cycle(all_cyclers)
for j, item in enumerate(all_result):
    label = item 
    if(item=="SMAC"):
        label = "Combined Search"
    ax.plot(
        np.mean(
            np.maximum.accumulate(
                1 - np.asarray(all_result[item][instance_num])[:, 1:], axis=1
            ),
            axis=0,
        ),
        label=label,  # item,
        linewidth=4.0,
    )
for arm in range(number_of_arms):
    df = dataset[
        (dataset["instance"] == instances[instance_num])
        & (dataset["arm_index"] == int(arm))
    ]
    data = df.sort_values(by=["repetition", "iteration"])["loss"].values.reshape( number_of_trails, horizon_time)
    ax.plot(
        np.mean(np.maximum.accumulate(1 - data, axis=1), axis=0),
        label=algorithms_data.printing_name_dict[classes[arm]],
        linewidth=4.0,
    ) 

ax.set(xlabel="HPO iteration", ylabel="Accuracy")

improve_legend(ax)
plt.gca().spines["left"].set_position(("data", -5))
plt.gca().spines["right"].set_position(("data", 316))
ax.spines["top"].set_bounds(-5, 316)
ax.spines["bottom"].set_bounds(-5, 316)
fig.savefig(
    result_directory + "extra_plots/performance.pdf", dpi=600, bbox_inches="tight"
)
plt.close()


#################################################################
#################################################################
#################################################################

plt.rcParams.update({"font.size": 18})
arm1 = 2
arm2 = 0

df = dataset[(dataset["instance"] == instances[instance_num])]

df1 = df[df["arm_index"] == arm1]
df2 = df[df["arm_index"] == arm2]

data1 = 1 - np.mean(
    df1.sort_values(by=["repetition", "iteration"])["loss"].values.reshape(
        number_of_trails, horizon_time
    ),
    axis=0,
).reshape(-1)
data2 = 1 - np.mean(
    df2.sort_values(by=["repetition", "iteration"])["loss"].values.reshape(
        number_of_trails, horizon_time
    ),
    axis=0,
).reshape(-1)

density1 = gaussian_kde(data1, bw_method=0.15)
density2 = gaussian_kde(data2, bw_method=0.15)
xs = np.linspace(0, 1, 200)

x = np.arange(0.74, 0.775, 0.0001)
y1 = density1(x)
y1[x < np.min(data1)] = 0
y1[np.max(data1) < x] = 0

y2 = density2(x)
y2[x < np.min(data2)] = 0
y2[np.max(data2) < x] = 0

# define multiple normal distributions
plt.plot(x, y1, color=colors[arm1])  # , label="$A^1$:"+classes[arm1],


x1 = np.min(data1)
plt.fill_between(x, y1, where=x >= x1, color=colors[arm1], alpha=0.3)
plt.axvline(
    data1.mean(),
    color=colors[arm1],
    ls="dashed",
    lw=1,
    label="mean(" + classes[arm1] + ")",
)
# plt.axvline(x1, color=colors[arm1], ls='-.', lw=2, label="$q_{0.95}$("+classes[arm1]+")")
plt.axvline(
    np.max(data1),
    color=colors[arm1],
    ls="--",
    lw=2,
    label="$\max$(" + classes[arm1] + ")",
)


plt.plot(x, y2, color=colors[arm2])  # , label="$A^2$:" + classes[arm2]
x2 = np.min(data2)
plt.fill_between(x, y2, where=x >= x2, color=colors[arm2], alpha=0.3)
plt.axvline(
    data2.mean(), color=colors[arm2], ls="dashed", label="$\mu$(" + classes[arm2] + ")"
)
plt.axvline(
    np.max(data2),
    color=colors[arm2],
    ls="--",
    lw=2,
    label="$\max$(" + classes[arm2] + ")",
)

plt.ylabel("Probability Density Function")
plt.xlabel("Accuracy")

# add legend to plot
plt.legend(fontsize=14, loc="upper left")
# plt.savefig("quantile-comparison.png", dpi = 600, bbox_inches = 'tight')
plt.savefig(
    result_directory + "extra_plots/distribution-comparison.pdf",
    dpi=600,
    bbox_inches="tight",
)