import pandas as pd
import exp_utils
import analysis_utils
import algorithms_data
import scipy.stats
import numpy as np

dataset_names = [
    "TabRepo",
    "TabRepoRaw",
    "YaHPOGym",
    "Reshuffling"]


times_dataset_names = {"TabRepo": [100, 200],  "TabRepoRaw": [100, 200], "YaHPOGym": [100, 200], "Reshuffling":[125, 250]}



baselines_names = ["MaxUCB", "MaxUCB", "MaxUCB", "MaxUCB"]  # ,     "UCB"]

competitors = [
    #"MaxUCB_adaptive_alpha",
    "Successive_Halving",
    "R_SR",
    "R_UCBE"
]  # ,,


# competitors = []
# alphas = np.arange(0.0, 1.0, 0.1)
# alphas = np.round(alphas, 2)
# for item in alphas:
#     competitors.append("MaxUCB_" + str(item))


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


def extract_results(dataset_name, baseline_name, competitors, time):
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

    result_directory = "../results/"
    all_result = exp_utils.fetch_results(
        policy_algorithms, result_directory, dataset_name
    )
    
    seed_res = analysis_utils.get_error_per_seed(
        all_result, dataset, number_of_arms, instances, number_of_trails
    )
    df = pd.concat(seed_res, axis=0).sort_index().ffill()
    m_norm_err_res = df.groupby(level=0,axis=1).mean().T.to_dict()[time]

    std_norm_err_res = df.groupby(level=0,axis=1).std().T.to_dict()[time]

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
        results[competitor] = wtl, p, (m_norm_err_res[competitor],std_norm_err_res[competitor])
    return results



init_table = """\\begin{table}[htbp]
\\centering
\\scriptsize
\\begin{tabular}{lll"""
init_table += "l" * len(competitors)
init_table += "}\n"
init_table += "& Time & "
for item in competitors:
    if item == "Q_BayesUCB":
        init_table += " &  \\makecell{Quantile\\\\Bayes UCB}"
    if item == "Rising Bandit":
        init_table += " &  \\makecell{Rising\\\\Bandit}"
    else:
        init_table += " & " + algorithms_data.printing_name_dict[item]

init_table += "\\\\"
rows_string = ""
rows_string += "\\midrule \n"

for dataset_name, baseline_name in zip(dataset_names, baselines_names):
    rows_string += (
        "\\multirow{"
        + str(3 * len(times_dataset_names[dataset_name]))
        + "}{*}{\\rotatebox[origin=c]{90}{"
        + dataset_name
        + "}}\\\\\n"
    )
    for time in times_dataset_names[dataset_name]:
        results = extract_results(dataset_name, baseline_name, competitors, time)
        rows_string += "\\cmidrule{2-" + str(len(competitors)+3) + "} \n"

        rows_string += " & \\multicolumn{1}{l}{\\multirow{3}{*}{" + str(time) + "}}"
        # rows_string += " & p-value "
        # for key, value in results.items():
        #     p = value[1]
        #     if p < (0.05 / len(competitors)):
        #         rows_string += " & " + "$\mathbf{\\underline{%.5f}}$" % p
        #     elif p < 0.05:
        #         rows_string += " & " + "$\mathbf{%.5f}$" % p
        #     else:
        #         rows_string += " & " + "$%.5f$" % p
        # rows_string += " \\\\\n"

        rows_string += "& &  w/t/l "
        for key, value in results.items():
            rows_string += " & " + "${:.0f}$/${:.0f}$/${:.0f}$".format(
                value[0][0], value[0][1], value[0][2]
            )
        rows_string += " \\\\\n"

        # rows_string += "& &  loss"
        # for key, value in results.items():
        #     rows_string += " & " + "${:.4f}$".format(
        #         value[2][0]
        #     )  # + "\\\\$\\pm{:.0f}$".format(value[2][1]) +"}"
        # rows_string += "\\\\\n"

    rows_string += "\\bottomrule\n"
    # wtl = '%f/%f/%f' % (competitor_wins_against_baseline/(len(instances)*number_of_trails), baseline_and_competitor_tie/(len(instances)*number_of_trails), baseline_wins_against_competitor/(len(instances)*number_of_trails))
    # print("wins/ties/losses against RS",wtl)
    # print("p-value against RS =", p )

end_table = """\\bottomrule
\\end{tabular}
\\end{table}"""
print(init_table)
print(rows_string)
print(end_table)