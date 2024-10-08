import pandas as pd
import exp_utils
import analysis_utils
import algorithms_data
import scipy.stats
import numpy as np

dataset_names = [
    "TabRepo",
    "TabRepoRaw",
    "YaHPOGym",]  # , "hebo"]
baselines_names = [
    "RandomSearch",
    "SMAC",
    "SMAC"]  # ,     "UCB"]

competitors = [
    "MaxUCB",
    "Q_BayesUCB",
    "ER_UCB_S",
    "Rising_Bandit",
    "QoMax_SDA",
    "Max_Median",
    "UCB",
]  # ,, "QoMax_SDA",  "Max_Median", "UCB", "Oracle_Arm"]


def compute_wins(baseline, competitor, results):
    baseline_wins_against_competitor = 0
    competitor_wins_against_baseline = 0
    baseline_and_competitor_tie = 0

    for concat in results:
        baseline_arr = np.mean(concat[baseline])
        competitor_arr = np.mean(concat[competitor])
        baseline_wins_against_competitor += np.sum(
            (baseline_arr < competitor_arr) * (1-np.isclose(baseline_arr, competitor_arr))
        )
        competitor_wins_against_baseline += np.sum((competitor_arr < baseline_arr)* (1-np.isclose(baseline_arr, competitor_arr)))
        baseline_and_competitor_tie += np.sum(np.isclose(baseline_arr,competitor_arr))

    return (
        competitor_wins_against_baseline,
        baseline_and_competitor_tie,
        baseline_wins_against_competitor,
    )

def compute_sign_test(
    competitor_wins_against_baseline,
    baseline_and_competitor_tie,
    baseline_wins_against_competitor,
):
    """Compute the sign test according to Demsar, 2006.

    We use the sign test because different benchmarks measure different metrics,
    making them incommensurable."""
    remainder = int(baseline_and_competitor_tie / 2)
    p = scipy.stats.binom_test(
        (
            baseline_wins_against_competitor + remainder,
            competitor_wins_against_baseline + remainder,
        ),
        alternative="less",
    )
    return p


def extract_results(dataset_name, baseline_name, competitors):
    policy_algorithms = {}
    for item in competitors:
        policy_algorithms[item] = 1
    policy_algorithms[baseline_name] = 1

    dataset = pd.read_csv("../datasets/" + dataset_name + ".csv")

    instances = sorted(dataset["instance"].unique())
    all_arm_index_list = dataset["arm_index"].unique()
    valid_arm_index_list = [item for item in all_arm_index_list if item >= 0]
    number_of_arms = len(valid_arm_index_list)
    number_of_trails = len(dataset["repetition"].unique())
    horizon_time = len(dataset["iteration"].unique())

    time = horizon_time

    result_directory = "../results/"
    all_result = exp_utils.fetch_results(
        policy_algorithms, result_directory, dataset_name
    )

    res = analysis_utils.get_error_per_instance_time(
        all_result, number_of_arms, instances, number_of_trails, time
    )

    results = {}
    for competitor in competitors:
        (
            competitor_wins_against_baseline,
            baseline_and_competitor_tie,
            baseline_wins_against_competitor,
        ) = compute_wins(baseline_name, competitor, res)
        p = compute_sign_test(
            competitor_wins_against_baseline,
            baseline_and_competitor_tie,
            baseline_wins_against_competitor,
        )
        # wtl = (competitor_wins_against_baseline/(len(instances)), baseline_and_competitor_tie/(len(instances)), baseline_wins_against_competitor/(len(instances)))
        wtl = (
            competitor_wins_against_baseline,
            baseline_and_competitor_tie,
            baseline_wins_against_competitor,
        )
        results[competitor] = wtl, p
    return results


init_table = """\\begin{table}[htbp]
\\centering
\\scriptsize
\\begin{tabular}{ll"""
init_table += "c" * len(competitors)
init_table += "}\n"
init_table += "Benchmark & "
for item in competitors:
    if item == "Q_BayesUCB":
        init_table += " &  \\makecell{Quantile\\\\Bayes UCB}"
    elif item == "Rising Bandit":
          init_table += " &  \\makecell{Rising\\\\Bandit}"
    else:
        init_table += " & " + algorithms_data.printing_name_dict[item]

init_table += "\\\\"
rows_string = ""
for dataset_name, baseline_name in zip(dataset_names, baselines_names):
    results = extract_results(dataset_name, baseline_name, competitors)
    rows_string += "\\midrule \n"

    rows_string += (
        "\\multicolumn{1}{l}{\\multirow{2}{*}{"
        + algorithms_data.printing_name_dict[dataset_name]
        + "}} & "
    )
    rows_string += "p-value "
    for key, value in results.items():
        p = value[-1]
        if p < (0.05 / len(competitors)):
            rows_string += " & " + "$\mathbf{\\underline{%.5f}}$" % p
        elif p < 0.05:
            rows_string += " & " + "$\mathbf{%.5f}$" % p
        else:
            rows_string += " & " + "$%.5f$" % p
    rows_string += " \\\\\n"

    rows_string += " &  w/t/l "
    for key, value in results.items():
        rows_string += " & " + "${:.0f}$/${:.0f}$/${:.0f}$".format(
            value[0][0], value[0][1], value[0][2]
        )

    rows_string += "\\\\\n"
end_table = """\\bottomrule
\\end{tabular}
\\end{table}"""
print(init_table)
print(rows_string)
print(end_table)