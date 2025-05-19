import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import os
from seaborn.utils import desaturate

plt.rcParams["text.usetex"] = True
# Fig size
plt.rcParams["figure.figsize"] = 5, 5
plt.rcParams.update({"font.size": 26})


### run for each datasets
dataset_name = "TabRepoRaw"
#dataset_name = "YaHPOGym"
#dataset_name = "Synth"

path = "../results/plots_for_paper/fig_assumption/"
if not os.path.exists(path):
    os.makedirs(path)


def lighten_color(color, amount=0.5):
    """
    Lightens the given color by multiplying (1-luminosity) by the given amount.
    Input can be matplotlib color string, hex string, or RGB tuple.

    Examples:
    >> lighten_color('g', 0.3)
    >> lighten_color('#F034A3', 0.6)
    >> lighten_color((.3,.55,.1), 0.5)
    """
    import matplotlib.colors as mc
    import colorsys

    try:
        c = mc.cnames[color]
    except:
        c = color
    c = colorsys.rgb_to_hls(*mc.to_rgb(c))
    return colorsys.hls_to_rgb(c[0], 1 - amount * (1 - c[1]), c[2])


if(dataset_name!="Synth"):
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

    list_of_cdfs_arm_1 = []
    for i, intance in enumerate(instances):
        list_of_cdfs_arm_1.append(np.sort(1 - dataset[i, :, 0, :].flatten()))
    list_of_cdfs_arm_2 = []
    for i, intance in enumerate(instances):
        list_of_cdfs_arm_2.append(np.sort(1 - dataset[i, :, -1, :].flatten()))

if dataset_name == "YaHPOGym":


    plt.plot(
        np.sort(np.mean(list_of_cdfs_arm_1, axis=0)),
        1 - np.linspace(0, 1, len(list_of_cdfs_arm_1[0]), endpoint=False),
        label="Optimal arm",
        color=desaturate("tab:blue", 0.75),
        linewidth=3,
    )

    xrange = np.linspace(0.99, 1.0, 100)
    yrange = (1.0 - (xrange)) * 100
    plt.plot(
        xrange,
        yrange,
        color=lighten_color(desaturate("tab:blue", 0.75), 1.25),
        linewidth=3,
        linestyle="dotted",
    )
    angle = -np.rad2deg(np.arctan(100))

    t = plt.annotate(
        "$U>100$",
        xy=(xrange[0]-0.05, yrange[40]),
        xycoords="data",
        fontsize=16,
        color=lighten_color(desaturate("tab:blue", 0.75), 1.25),
        rotation=angle,
    )
    t.set_bbox(dict(facecolor="white", alpha=0.9, edgecolor="none", pad=0.0))


    # Define x data range for tangent line
    xrange = np.linspace(0.0, 1.0, 100)
    yrange = (1.0 - (xrange)) * 1.05
    plt.plot(
        xrange,
        yrange,
        color=lighten_color(desaturate("tab:blue", 0.75), 1.25),
        linewidth=2,
        linestyle="dotted",
    )

    angle = -np.rad2deg(np.arctan(1.05))

    t = plt.annotate(
        "$L>1$",
        xy=(xrange[50], yrange[65]),
        xycoords="data",
        fontsize=16,
        color=lighten_color(desaturate("tab:blue", 0.75), 1.25),
        rotation=angle,
    )
    t.set_bbox(dict(facecolor="white", alpha=0.9, edgecolor="none", pad=0.0))



    plt.plot(
        np.sort(np.mean(list_of_cdfs_arm_2, axis=0)),
        1 - np.linspace(0, 1, len(list_of_cdfs_arm_2[0]), endpoint=False),
        label="Worst arm",
        color=desaturate("tab:orange", 0.75),
        linewidth=3,
    )

    xrange = np.linspace(0.6, 0.79, 100)
    yrange = (0.79 - (xrange)) * 8
    plt.plot(
        xrange,
        yrange,
        color=lighten_color(desaturate("tab:orange", 0.75), 1.25),
        linewidth=2,
        linestyle="dotted",
    )

    angle = -np.rad2deg(np.arctan(8))

    t = plt.annotate(
        "$U>8$",
        xy=(xrange[25], yrange[57]),
        xycoords="data",
        fontsize=16,
        color=lighten_color(desaturate("tab:orange", 0.75), 1.25),
        rotation=angle,
    )
    t.set_bbox(dict(facecolor="white", alpha=0.9, edgecolor="none", pad=0.0))


    # Define x data range for tangent line
    xrange = np.linspace(0.0, 0.79, 100)
    yrange = (0.79 - (xrange)) * 1.38
    plt.plot(
        xrange,
        yrange,
        color=lighten_color(desaturate("tab:orange", 0.75), 1.25),
        linewidth=2,
        linestyle="--",
    )
    angle = -np.rad2deg(np.arctan(1.38))
    
    t = plt.annotate(
        "$L>1$",
        xy=(xrange[33], yrange[50]),
        xycoords="data",
        fontsize=16,
        color=lighten_color(desaturate("tab:orange", 0.75), 1.25),
        rotation=angle,
    )
    t.set_bbox(dict(facecolor="white", alpha=0.9, edgecolor="none", pad=0.0))


    plt.legend(handlelength=0.9, handletextpad=0.3, loc="lower left", fontsize=18)


if dataset_name == "TabRepoRaw":
    # plot
    plt.plot(
        np.sort(np.mean(list_of_cdfs_arm_1, axis=0)),
        1 - np.linspace(0, 1, len(list_of_cdfs_arm_1[0]), endpoint=False),
        label="Optimal arm",
        color=desaturate("tab:blue", 0.75),
        linewidth=3,
    )


    # Define x data range for tangent line
    xrange = np.linspace(0.93, 1.0, 100)
    yrange = (1.0 - (xrange)) * 25
    plt.plot(xrange, yrange, color=lighten_color(desaturate("tab:blue", 0.75), 1.25), linewidth=2, linestyle="dotted")
    angle = -np.rad2deg(np.arctan(25))

    t = plt.annotate(
        "$U>25$",
        xy=(xrange[0], yrange[60]),
        xycoords="data",
        fontsize=16,
        color=lighten_color(desaturate("tab:blue", 0.75), 1.25),
        rotation=angle,
    )
    t.set_bbox(dict(facecolor="white", alpha=0.9, edgecolor="none", pad=0.0))

    # Define x data range for tangent line
    xrange = np.linspace(0.1, 1.0, 100)
    yrange = (1.0 - (xrange)) * 1.1
    plt.plot(
        xrange,
        yrange,
        color=lighten_color(desaturate("tab:blue", 0.75), 1.25),
        linewidth=2,
        linestyle="--",
    )
    angle = -np.rad2deg(np.arctan(1.1))

    t = plt.annotate(
        "$L>1$",
        xy=(xrange[45], yrange[60]),
        xycoords="data",
        fontsize=16,
        color=lighten_color(desaturate("tab:blue", 0.75), 1.25),
        rotation=angle,
    )
    t.set_bbox(dict(facecolor="white", alpha=0.9, edgecolor="none", pad=0.0))




    plt.plot(
        np.sort(np.mean(list_of_cdfs_arm_2, axis=0)),
        1 - np.linspace(0, 1, len(list_of_cdfs_arm_2[0]), endpoint=False),
        label="Worst arm",
        color=desaturate("tab:orange", 0.75),
        linewidth=3,
    )


    # Define x data range for tangent line
    xrange = np.linspace(0.50, 0.87, 100)
    yrange = (0.87 - (xrange)) * 4
    plt.plot(
        xrange,
        yrange,
        color=lighten_color(desaturate("tab:orange", 0.75), 1.25),
        linewidth=2,
        linestyle="dotted",
        alpha=0.7,
    )
    angle = -np.rad2deg(np.arctan(4))

    t = plt.annotate(
        "$U>4$",
        xy=(xrange[37], yrange[50]),
        xycoords="data",
        fontsize=16,
        color=lighten_color(desaturate("tab:orange", 0.75), 1.25),
        rotation=angle,
    )
    t.set_bbox(dict(facecolor="white", alpha=0.9, edgecolor="none", pad=0.0))

    # Define x data range for tangent line
    xrange = np.linspace(0.0, 0.87, 100)
    yrange = (0.87 - (xrange)) * 1.25
    plt.plot(
        xrange,
        yrange,
        color=lighten_color(desaturate("tab:orange", 0.75), 1.25),
        linewidth=2,
        linestyle="--",
    )
    angle = -np.rad2deg(np.arctan(1.25))
    
    t = plt.annotate(
        "$L>1$",
        xy=(xrange[33], yrange[50]),
        xycoords="data",
        fontsize=16,
        color=lighten_color(desaturate("tab:orange", 0.75), 1.25),
        rotation=angle,
    )
    t.set_bbox(dict(facecolor="white", alpha=0.9, edgecolor="none", pad=0.0))

    plt.legend(handlelength=0.9, handletextpad=0.3, loc="lower left", fontsize=18)


if dataset_name == "Synth":

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

    loc = 0
    scale = 0.2
    skewness = 1.0
    r2 = truncated_log_normal_dist(0.0, 1.0, 100000, a=skewness, loc=loc, scale=scale)


    plt.plot(
        np.sort(r2),
        1 - np.linspace(0, 1, len(r2), endpoint=False),
        label="Log-normal",
        color=desaturate("#1b9e77", 0.75),
        linewidth=3,
    )
    xrange = np.linspace(0.0, 1.0, 100)
    yrange = (1.0 - (xrange)) * 1.02
    plt.plot(
        xrange,
        yrange,
        color=lighten_color(desaturate("#1b9e77", 0.75), 1.25),
        linewidth=2,
        linestyle="dotted",
    )

    angle = -np.rad2deg(np.arctan(1.02))
    t = plt.annotate(
        "$U\\approx1$",
        xy=(xrange[45], yrange[55]),
        xycoords="data",
        fontsize=16,
        color=lighten_color(desaturate("#1b9e77", 0.75), 1.25),
        rotation=angle,
    )
    t.set_bbox(dict(facecolor="white", alpha=0.9, edgecolor="none", pad=0.0))


    # Define x data range for tangent line
    xrange = np.linspace(0.0, 1.0, 100)
    yrange = (1.0 - (xrange)) * 0.1
    plt.plot(
        xrange,
        yrange,
        color=lighten_color(desaturate("#1b9e77", 0.75), 1.25),
        linewidth=2,
        linestyle="dotted",
    )
    angle = -np.rad2deg(np.arctan(0.1))
    t = plt.annotate(
        "$L<0.1$",
        xy=(xrange[3], yrange[30]),
        xycoords="data",
        fontsize=16,
        color=lighten_color(desaturate("#1b9e77", 0.75), 1.25),
        rotation=angle,
    )
    t.set_bbox(dict(facecolor="white", alpha=0.9, edgecolor="none", pad=0.0))


    loc = -0.1
    scale = 0.1
    skewness = 1.0

    r2 = truncated_pareto_dist(0.0, 1.0, 100000, a=skewness, loc=loc, scale=scale)

    plt.plot(
        np.sort(r2),
        1 - np.linspace(0, 1, len(r2), endpoint=False),
        label="Pareto",
        color=desaturate("#7570b3", 0.75),
        linewidth=3,
    )

    xrange = np.linspace(0.0, 1.0, 100)
    yrange = (1.0 - (xrange)) * 1.0
    plt.plot(
        xrange,
        yrange,
        color=lighten_color(desaturate("#7570b3", 0.75), 1.25),
        linewidth=2,
        linestyle="dotted",
    )

    angle = -np.rad2deg(np.arctan(1.02))
    t = plt.annotate(
        "$U\\approx1$",
        xy=(xrange[40], yrange[63]),
        xycoords="data",
        fontsize=16,
        color=lighten_color(desaturate("#7570b3", 0.75), 1.25),
        rotation=angle,
    )
    t.set_bbox(dict(facecolor="white", alpha=0.9, edgecolor="none", pad=0.0))


    # Define x data range for tangent line
    xrange = np.linspace(0.0, 1.0, 100)
    yrange = (1.0 - (xrange)) * 0.05
    plt.plot(
        xrange,
        yrange,
        color=lighten_color(desaturate("#7570b3", 0.75), 1.25),
        linewidth=1,
        linestyle="--",
    )

    angle = -np.rad2deg(np.arctan(0.05))
    t = plt.annotate(
        "$L<0.05$",
        xy=(xrange[2], yrange[70]),
        xycoords="data",
        fontsize=16,
        color=lighten_color(desaturate("#7570b3", 0.75), 1.25),
        rotation=angle,
    )
    t.set_bbox(dict(facecolor="white", alpha=0.9, edgecolor="none", pad=0.0))


    plt.legend(loc="upper right", fontsize=18, handlelength=0.9, handletextpad=0.3)


# Add labels, legend, and grid
#plt.xlabel("Reward", labelpad=0)
plt.ylabel(r"$P(\mathrm{reward } \geq x)$")
plt.xlabel("x", labelpad=0)

# plt.title("Comparison of $G_1(x)$ and $G_2(x)$", fontsize=18)

# plt.grid(True)
plt.xlim(0, 1.02)  # Ensure x is between 0 and 1
plt.ylim(0, 1.01)  # Set y-axis limits slightly above 1 for clarity

# Show the plot
plt.tight_layout()
plt.savefig(path + dataset_name + "_L_U_demo.pdf", dpi=600, bbox_inches="tight")