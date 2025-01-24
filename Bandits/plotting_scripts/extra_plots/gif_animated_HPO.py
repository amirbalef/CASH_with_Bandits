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
    if i > horizon_time // number_of_arms:
        i = horizon_time // number_of_arms

    ax.clear()

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
    lines = ["--"]
    linecycler = cycler(linestyle=lines)
    all_cyclers = all_cyclers.concat(colorcycler * linecycler)

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
    for arm in range(number_of_arms):
        df = dataset[
            (dataset["instance"] == instances[instance_num])
            & (dataset["arm_index"] == int(arm))
        ]
        data = df.sort_values(by=["repetition", "iteration"])["loss"].values.reshape(
            number_of_trails, horizon_time
        )
        ax.plot(
            np.mean(np.maximum.accumulate(1 - data, axis=1), axis=0),
            label=algorithms_data.printing_name_dict[classes[arm]],
            linewidth=4.0, alpha = 0.3
        )


    improve_legend(ax)
    plt.gca().spines["left"].set_position(("data", -5))
    plt.gca().spines["right"].set_position(("data", 366))
    ax.spines["top"].set_bounds(-5, 366)
    ax.spines["bottom"].set_bounds(-5, 366)

    ax.set(xlabel="HPO iteration", ylabel="Maximum observed accuracy")


    colors = plotting_utils.CB_color_cycle
    myorder = [0, 1, 6, 3, 4, 5, 2]
    colors = [colors[i] for i in myorder]
    colorcycler = cycler(color=colors)
    lines = ["-", "-"]
    linecycler = cycler(linestyle=lines)
    all_cyclers = colorcycler * linecycler
    ax.set_prop_cycle(all_cyclers)

    # for j, item in enumerate(all_result):
    #     label = item
    #     if(item=="SMAC"):
    #         label = "Combined Search"
    #     axs[j].plot(
    #         np.mean(
    #             np.maximum.accumulate(
    #                 1 - np.asarray(all_result[item][instance_num])[:, 1:], axis=1
    #             ),
    #             axis=0,
    #         ),
    #         label=label,  # item,
    #         linewidth=4.0,
    #     )
    lines_points = []
    for arm in range(number_of_arms):
        df = dataset[
            (dataset["instance"] == instances[instance_num])
            & (dataset["arm_index"] == int(arm))
        ]
        data = df.sort_values(by=["repetition", "iteration"])["loss"].values.reshape( number_of_trails, horizon_time)
        ax_data = np.mean(np.maximum.accumulate(1 - data, axis=1), axis=0)
        (line,) = ax.plot(time_data[0:i], ax_data[0:i],
            label=algorithms_data.printing_name_dict[classes[arm]],
            linewidth=4.0,
        ) 
        lines_points.append(line)
        (point,) = ax.plot(time_data[i], ax_data[i], marker=".")
        lines_points.append(point)
    
    remaining_budget = (horizon_time - i * number_of_arms)//10
    print(remaining_budget)
    time_text = ax.annotate(
        "Remaining Budget:\n " + ("\$" * remaining_budget) ,
        xy=(120, 0.735),
        xytext=(120, 0.735),
        textcoords="offset points",
        va="center",
        color="black",
        fontsize=20,
    )

    lines_points.append(time_text)

    return lines_points


fig.subplots_adjust(left=0.15, bottom=0.12, right=0.6, top=0.95, wspace=None, hspace=None)
delay = 30
ani = FuncAnimation(
    fig,
    animate,
    interval=700,
    blit=True,
    repeat=False,
    repeat_delay=10000,
    frames=(horizon_time // number_of_arms + delay),
)

ani.save("HPO.gif", dpi=300, writer='imagemagick')
# Show the plot
plt.close()

# run command  "gifsicle -O3 --colors 256 --lossy=30 -o HPO_c.gif HPO.gif"


# fig, ax = plt.subplots(figsize=(7, 8))
# ax.set_prop_cycle(all_cyclers)
# for j, item in enumerate(all_result):
#     label = item 
#     if(item=="SMAC"):
#         label = "Combined Search"
#     ax.plot(
#         np.mean(
#             np.maximum.accumulate(
#                 1 - np.asarray(all_result[item][instance_num])[:, 1:], axis=1
#             ),
#             axis=0,
#         ),
#         label=label,  # item,
#         linewidth=4.0,
#     )
# for arm in range(number_of_arms):
#     df = dataset[
#         (dataset["instance"] == instances[instance_num])
#         & (dataset["arm_index"] == int(arm))
#     ]
#     data = df.sort_values(by=["repetition", "iteration"])["loss"].values.reshape( number_of_trails, horizon_time)
#     ax.plot(
#         np.mean(np.maximum.accumulate(1 - data, axis=1), axis=0),
#         label=algorithms_data.printing_name_dict[classes[arm]],
#         linewidth=4.0,
#     ) 

# ax.set(xlabel="HPO iteration", ylabel="Accuracy")

# improve_legend(ax)
# plt.gca().spines["left"].set_position(("data", -5))
# plt.gca().spines["right"].set_position(("data", 316))
# ax.spines["top"].set_bounds(-5, 316)
# ax.spines["bottom"].set_bounds(-5, 316)
# fig.savefig(
#     result_directory + "extra_plots/performance.pdf", dpi=600, bbox_inches="tight"
# )
# plt.close()

#fig, axs = plt.subplots(2, 4)
# for j, item in enumerate(all_result):
#     label = item 
#     if(item=="SMAC"):
#         label = "Combined Search"
#     axs[j].plot(
#         np.mean(
#             np.maximum.accumulate(
#                 1 - np.asarray(all_result[item][instance_num])[:, 1:], axis=1
#             ),
#             axis=0,
#         ),
#         label=label,  # item,
#         linewidth=4.0,
#     )
#for arm in range(number_of_arms):
    

# ax.set(xlabel="HPO iteration", ylabel="Accuracy")


# fig.savefig(
#     result_directory + "extra_plots/performance.pdf", dpi=600, bbox_inches="tight"
# )

