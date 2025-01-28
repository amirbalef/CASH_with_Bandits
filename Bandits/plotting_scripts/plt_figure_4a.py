import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import os
from seaborn.utils import desaturate

plt.rcParams["text.usetex"] = True
# Fig size
plt.rcParams["figure.figsize"] = 8, 5
plt.rcParams.update({"font.size": 26})


### run for each datasets
dataset_name = "TabRepoRaw"
#dataset_name = "YaHPOGym"

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
print(dataset.shape)


from scipy.stats import pareto

from scipy.stats import lognorm


def truncated_log_normal_dist(a_trunc, b_trunc, seq_len, a, loc, scale):
    quantile1 = lognorm.cdf(a_trunc, a, loc=loc, scale=scale)
    quantile2 = lognorm.cdf(b_trunc, a, loc=loc, scale=scale)

    return lognorm.ppf(
        np.random.uniform(quantile1, quantile2, size=seq_len),
        a,
        loc=loc,
        scale=scale,
    )


def truncated_pareto_dist(a_trunc, b_trunc, seq_len, a=None, loc=None, scale=None):
    quantile1 = pareto.cdf(a_trunc, a, loc=loc, scale=scale)
    quantile2 = pareto.cdf(b_trunc, a, loc=loc, scale=scale)

    return pareto.ppf(
        np.random.uniform(quantile1, quantile2, size=seq_len),
        a,
        loc=loc,
        scale=scale,
    )

X = np.linspace(0, 1, 100)

list_of_cdfs_arm_1 = []
for i, intance in enumerate(instances):
    list_of_cdfs_arm_1.append(np.sort(1 - dataset[i, :, 0, :].flatten()))
list_of_cdfs_arm_2 = []
for i, intance in enumerate(instances):
    list_of_cdfs_arm_2.append(np.sort(1 - dataset[i, :, -1, :].flatten()))

if dataset_name == "YaHPOGym":

    loc = 0
    scale = 0.2
    skewness = 1.0
    r2 = truncated_log_normal_dist(0.0, 1.0, 100000, a=skewness, loc=loc, scale=scale)

    plt.plot(
        np.sort(np.mean(list_of_cdfs_arm_1, axis=0)),
        1 - np.linspace(0, 1, len(list_of_cdfs_arm_1[0]), endpoint=False),
        label="Optimal arm",
        color=desaturate("tab:blue", 0.75),
        linewidth=3,
    )

    xrange = np.linspace(0.9935, 1.0, 100)
    yrange = (1.0 - (xrange)) * 100
    plt.plot(xrange, yrange, color="tab:blue", linewidth=3, linestyle="dotted")
    plt.annotate(
        "$U>100$",
        xy=(0.97, 0.7),
        xycoords="data",
        fontsize=16,
        color="tab:blue",
        rotation=-(180 / np.pi) * np.arctan(100),
    )

    # Define x data range for tangent line
    xrange = np.linspace(0.6, 1.0, 100)
    yrange = (1.0 - (xrange)) * 1.4
    plt.plot(xrange, yrange, color="tab:blue", linewidth=3, linestyle="--")
    plt.annotate(
        "$L>1$",
        xy=(0.48, 0.6),
        xycoords="data",
        fontsize=16,
        color="tab:blue",
        rotation=-(180 / np.pi) * np.arctan(1.4)+20 ,
    )


    plt.plot(
        np.sort(np.mean(list_of_cdfs_arm_2, axis=0)),
        1 - np.linspace(0, 1, len(list_of_cdfs_arm_2[0]), endpoint=False),
        label="Worst arm",
        color=desaturate("tab:orange", 0.75),
        linewidth=3,
    )


    # Define x data range for tangent line
    xrange = np.linspace(0.685, 0.79, 100)
    yrange = (0.79 - (xrange)) * 7
    plt.plot(xrange, yrange, color="tab:orange", linewidth=3, linestyle="dotted")
    plt.annotate(
        "$U>7$",
        xy=(0.63, 0.78),
        xycoords="data",
        fontsize=16,
        color="tab:orange",
        rotation=-(180 / np.pi) * np.arctan(7)+5,
    )


    # Define x data range for tangent line
    xrange = np.linspace(0.5, 0.79, 100)
    yrange = (0.79 - (xrange)) * 1.4
    plt.plot(xrange, yrange, color="tab:orange", linewidth=3, linestyle="--")
    plt.annotate(
        "$L>1$",
        xy=(0.4, 0.4),
        xycoords="data",
        fontsize=16,
        color="tab:orange",
        rotation=-(180 / np.pi) * np.arctan(1.4) + 20,
    )


    plt.plot(
        np.sort(r2),
        1 - np.linspace(0, 1, len(r2), endpoint=False),
        label="Log-normal",
        color=desaturate("tab:red", 0.75),
        linewidth=1,
    )


    xrange = np.linspace(0.22, 1.0, 100)
    yrange = (1.0 - (xrange)) * 1.0
    plt.plot(xrange, yrange, color="tab:red", linewidth=1, linestyle="dotted")
    plt.annotate(
        "$U\\approx1$",
        xy=(0.1, 0.78),
        xycoords="data",
        fontsize=16,
        color="tab:red",
        rotation=-(180 / np.pi) * np.arctan(1.0) + 20,
    )

    # Define x data range for tangent line
    xrange = np.linspace(0.53, 1.0, 100)
    yrange = (1.0 - (xrange)) * 0.1
    plt.plot(xrange, yrange, color="tab:red", linewidth=1, linestyle="--")
    plt.annotate(
        "$L<0.1$",
        xy=(0.4, 0.04),
        xycoords="data",
        fontsize=16,
        color="tab:red",
        rotation=(180 / np.pi) * np.arctan(0.1) -10,
    )


if dataset_name == "TabRepoRaw":
    loc = -0.1
    scale = 0.1
    skewness = 1.0

    r2 = truncated_pareto_dist(0.0, 1.0, 100000, a=skewness, loc=loc, scale=scale)
    # plot
    plt.plot(
        np.sort(np.mean(list_of_cdfs_arm_1, axis=0)),
        1 - np.linspace(0, 1, len(list_of_cdfs_arm_1[0]), endpoint=False),
        label="Optimal arm",
        color=desaturate("tab:blue", 0.75),
        linewidth=3,
    )


    # Define x data range for tangent line
    xrange = np.linspace(0.972, 1.0, 100)
    yrange = (1.0 - (xrange)) * 25
    plt.plot(xrange, yrange, color="tab:blue", linewidth=3, linestyle="dotted")
    plt.annotate(
        "$U>25$",
        xy=(0.94, 0.75),
        xycoords="data",
        fontsize=16,
        color="tab:blue",
        rotation=-(180 / np.pi) * np.arctan(25),
    )

    # Define x data range for tangent line
    xrange = np.linspace(0.6, 1.0, 100)
    yrange = (1.0 - (xrange)) * 1.4
    plt.plot(xrange, yrange, color="tab:blue", linewidth=3, linestyle="--")
    plt.annotate(
        "$L>1$",
        xy=(0.48, 0.6),
        xycoords="data",
        fontsize=16,
        color="tab:blue",
        rotation=-(180 / np.pi) * np.arctan(1.4) + 20,
    )


    plt.plot(
        np.sort(np.mean(list_of_cdfs_arm_2, axis=0)),
        1 - np.linspace(0, 1, len(list_of_cdfs_arm_2[0]), endpoint=False),
        label="Worst arm",
        color=desaturate("tab:orange", 0.75),
        linewidth=3,
    )


    # Define x data range for tangent line
    xrange = np.linspace(0.69, 0.87, 100)
    yrange = (0.87 - (xrange)) * 4
    plt.plot(xrange, yrange, color="tab:orange", linewidth=3, linestyle="dotted")
    plt.annotate(
        "$U>4$",
        xy=(0.63, 0.76),
        xycoords="data",
        fontsize=16,
        color="tab:orange",
        rotation=-(180 / np.pi) * np.arctan(4) + 10,
    )


    # Define x data range for tangent line
    xrange = np.linspace(0.45, 0.87, 100)
    yrange = (0.87 - (xrange)) * 1.4
    plt.plot(xrange, yrange, color="tab:orange", linewidth=3, linestyle="--")
    plt.annotate(
        "$L>1$",
        xy=(0.35, 0.6),
        xycoords="data",
        fontsize=16,
        color="tab:orange",
        rotation=-(180 / np.pi) * np.arctan(1.4) + 20,
    )


    plt.plot(
        np.sort(r2),
        1 - np.linspace(0, 1, len(r2), endpoint=False),
        label="Pareto",
        color=desaturate("tab:red", 0.75),
        linewidth=1,
    )


    xrange = np.linspace(0.22, 1.0, 100)
    yrange = (1.0 - (xrange)) * 1.0
    plt.plot(xrange, yrange, color="tab:red", linewidth=1, linestyle="dotted")
    plt.annotate(
        "$U\\approx1$",
        xy=(0.1, 0.78),
        xycoords="data",
        fontsize=16,
        color="tab:red",
        rotation=-(180 / np.pi) * np.arctan(1.0) + 20,
    )

    # Define x data range for tangent line
    xrange = np.linspace(0.53, 1.0, 100)
    yrange = (1.0 - (xrange)) * 0.05
    plt.plot(xrange, yrange, color="tab:red", linewidth=1, linestyle="--")
    plt.annotate(
        "$L<0.05$",
        xy=(0.38, 0.02),
        xycoords="data",
        fontsize=16,
        color="tab:red",
        rotation=(180 / np.pi) * np.arctan(0.05) - 5,
    )


# Add labels, legend, and grid
plt.xlabel("reward")
plt.ylabel("Survival function")
# plt.title("Comparison of $G_1(x)$ and $G_2(x)$", fontsize=18)
plt.legend(loc="lower left", fontsize=18)
# plt.grid(True)
plt.xlim(0, 1.02)  # Ensure x is between 0 and 1
# plt.ylim(0, 1.1)  # Set y-axis limits slightly above 1 for clarity

# Show the plot
plt.tight_layout()
plt.savefig(path+ dataset_name +"_L_U_demo.pdf")