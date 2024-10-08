import sys
sys.path.append("../")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from cycler import cycler
import plotting_utils
import exp_utils
import algorithms_data
from matplotlib.animation import FuncAnimation, PillowWriter
import matplotlib as mpl

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
#policy_algorithms["MaxUCB"] = 1
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
time_data = np.arange(horizon_time)
fig, ax = plt.subplots(figsize=(9, 8))



def animate(i):
    ax.clear()
    ax.set_ylim(bottom=0.70, top = 0.79)

    colors = ["black"]
    if("MaxUCB" in policy_algorithms):
        colors = ["black", "blue"]
    all_cyclers = cycler(color=colors) * cycler(
        linestyle=["-"]
    )

    ax.set_prop_cycle(all_cyclers)
    for j, item in enumerate(all_result):
        label = item
        if item == "SMAC":
            label = "Combined Search"
        ax.plot(
            np.mean(
                np.maximum.accumulate(
                    1 - np.asarray(all_result[item][instance_num])[:, 1:], axis=1
                ),
                axis=0,
            ),
            label=label,  # item,
            linewidth=4.0, alpha = 0.1
        )

    

    improve_legend(ax)
    plt.gca().spines["left"].set_position(("data", -5))
    plt.gca().spines["right"].set_position(("data", 366))
    ax.spines["top"].set_bounds(-5, 366)
    ax.spines["bottom"].set_bounds(-5, 366)

    ax.set(xlabel="HPO iteration", ylabel="Observed accuracy")

    colors = ["black"]
    if "MaxUCB" in policy_algorithms:
        colors = ["black", "blue"]
    all_cyclers = cycler(color=colors) * cycler(linestyle=["-"])
    ax.set_prop_cycle(all_cyclers)
    lines_points = []

    for j, item in enumerate(all_result):
        label = item
        if(item=="SMAC"):
            label = "Combined Search"
        ax_data_r = (1 - np.asarray(all_result[item][instance_num])[:, 1:])
        ax_data = np.mean(
            np.maximum.accumulate(
                1 - np.asarray(all_result[item][instance_num])[:, 1:], axis=1
            ),
            axis=0,
        )

        line = ax.scatter(
            time_data[0:i],
            ax_data_r[0][0:i],
            label=label,
            linewidth=1.0,
            alpha=0.2,
        )
        lines_points.append(line)
        line = ax.scatter(
            time_data[0:i],
            ax_data[0:i],
            label=label,
            linewidth=1.0,
            alpha=0.2,
        )
        lines_points.append(line)

        line,  = ax.plot(
            time_data[0:i],
            ax_data[0:i],
            label=label,
            linewidth=1.0,
            alpha=0.8,
        )
        lines_points.append(line)

        (point,) = ax.plot(time_data[i], ax_data[i], marker=".")

        lines_points.append(point)

    
    remaining_budget = (horizon_time - i)//10
    print("remaining_budget", remaining_budget)
    time_text = ax.annotate(
        "Remaining Budget:\n " + ("\$" * remaining_budget) ,
        xy=(120, 0.715),
        xytext=(120, 0.715),
        textcoords="offset points",
        va="center",
        color="black",
        fontsize=20,
    )

    lines_points.append(time_text)

    return lines_points


fig.subplots_adjust(left=0.15, bottom=0.12, right=0.6, top=0.95, wspace=None, hspace=None)
ani = FuncAnimation(
    fig, animate, interval=100, blit=True, repeat=False, repeat_delay=10000,  frames=(horizon_time)
)

ani.save("HPO_combined_search.gif", dpi=300, writer='imagemagick')
# Show the plot
plt.close()
