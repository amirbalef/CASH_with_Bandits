import os
import plotting_utils
import pandas as pd
import exp_utils
import matplotlib.pyplot as plt
import numpy as np

dataset_names = [
    "TabRepo",
    "TabRepoRaw",
    "YaHPOGym",
    "Reshuffling" , 
]


def get_plot(order,  dataset_name):

    dataset = pd.read_csv("../datasets/" + dataset_name + ".csv")

    instances = sorted(dataset["instance"].unique())
    all_arm_index_list = dataset["arm_index"].unique()
    valid_arm_index_list = [item for item in all_arm_index_list if item >= 0]
    number_of_arms = len(valid_arm_index_list)
    number_of_trails = len(dataset["repetition"].unique())
    horizon_time = len(dataset["iteration"].unique())
    classes = dataset["classifier"].unique()
    combined_search_algorithms = dataset[dataset["arm_index"] < 0]["optimizer"].unique()

    policy_algorithms = {}
    alphas = np.arange(0.0, 3.0, 0.1)
    alphas = np.round(alphas, 2)
    for item in alphas:
        policy_algorithms["MaxUCB_" + str(item)] = 1

    result_directory = "../results/"
    all_result = exp_utils.fetch_results(policy_algorithms, result_directory, dataset_name)

    from cycler import cycler
    all_cyclers = cycler(color=[plotting_utils.CB_color_cycle[0]]) * cycler(linestyle=["-"])  # ,"--"

    colors = [plt.cm.Reds(i) for i in range(50, 250, 2*200//len(alphas)) ]
    # ['red', 'orange', 'blue', 'green', 'cyan', 'brown', 'olive'] #, 'purple ']
    all_cyclers =  all_cyclers.concat(cycler(color=colors[:len(alphas)//4 ]) * cycler(linestyle=["-", "--"]) ) # ,"--"
    all_cyclers =  all_cyclers.concat(cycler(color=[colors[len(alphas)//4]]) * cycler(linestyle=["-"]) )
    all_cyclers =  all_cyclers.concat(cycler(color=colors[len(alphas)//4+1:]) * cycler(linestyle=["-", "--"]) ) 


    path = result_directory + "plots_for_paper/app/fig_ablation/" 
    if not os.path.exists(path):
            os.makedirs(path) 

    data= {}
    data["all_result"] = all_result
    data["dataset"]  = dataset
    data["number_of_arms"] = number_of_arms
    data["instances"] = instances
    data["number_of_trails"] = number_of_trails
    data["saving_path"] = path
    data["dataset_name"] = dataset_name
    data["test_data"] = -1
    data["third_axis"] = alphas
    data["horizon_time"] = horizon_time
    data["clip_upper"] = [0.4, 0.5, 0.25, 0.1][order]

    import matplotlib.colors as colors
    # colourmap from green to red, biased towards the blue end.
    # Try out different gammas > 1.0
    data["cmap"] = colors.LinearSegmentedColormap.from_list(
        "nameofcolormap", ["g", "y", "r", "black"], gamma=1.0
    )


    data["plot_type"] = "Normalized loss"
    data["saving_name"] = "norm_loss"
    data["plot_confidence_type"] = "mean_std"
    plotting_utils.plot_heatmap(data)

    data["plot_type"] = "Ranking"
    data["saving_name"] = "ranking"
    data["plot_confidence_type"] = "mean_std"
    plotting_utils.plot_heatmap(data)


for i, dataset_name in enumerate(dataset_names):
    get_plot(i, dataset_name)
plt.show()
