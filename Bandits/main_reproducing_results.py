import os
import numpy as np 
import pandas as pd
import pickle 
from functools import partial

from policies.QUCB import QuantileUCB
from policies.Max_Median import Max_Median
from policies.QoMax_ETC import QoMax_ETC
from policies.QoMax_SDA import QoMax_SDA
from policies.UCB import UCB 
from policies.ER_UCB import ER_UCB_S, ER_UCB_N
from policies.Q_BayesUCB import Q_BayesUCB 
from policies.Exp3 import Exp3
from policies.MaxUCB import MaxUCB
from policies.Rising_Bandit import Rising_Bandit
from policies.Successive_Halving import Successive_Halving
from policies.ThompsonSampling import ThompsonSampling
from policies.Threshold_Ascent import Threshold_Ascent
from policies.R_UCBE import R_UCBE
from policies.R_SR import R_SR
from policies.MaxUCB_adaptive_alpha import MaxUCB_adaptive_alpha

from policies.MaxSearch import MaxSearch_Gaussian, MaxSearch_SubGaussian
from policies.Random import Random
from plotting_scripts import exp_utils


multiprocess = "joblib"
#multiprocess = " "
if(multiprocess == "joblib"):
     import joblib

dataset_name = "TabRepo"
#dataset_name = "TabRepoRaw"
#dataset_name = "YaHPOGym"
#dataset_name = "Reshuffling"
#dataset_name = "SubSupernet"

dataset = pd.read_csv("./datasets/" + dataset_name + ".csv")

instances = sorted(dataset["instance"].unique())

print(instances)
all_arm_index_list = dataset["arm_index"].unique()
valid_arm_index_list = [item for item in all_arm_index_list if item >= 0]
number_of_arms = len(valid_arm_index_list)
number_of_trails = len(dataset["repetition"].unique())
horizon_time = len(dataset["iteration"].unique())
classes = dataset["classifier"].unique()
combined_search_algorithms = dataset[dataset["arm_index"] < 0]["optimizer"].unique()


result_directory = "./results/" + dataset_name + "/"
###########################
policy_algorithms = {}
policy_algorithms["Random"] = Random
policy_algorithms["QuantileUCB"] = QuantileUCB
policy_algorithms["Q_BayesUCB"] = Q_BayesUCB
policy_algorithms["Successive_Halving"] = Successive_Halving
policy_algorithms["R_UCBE"] = R_UCBE
policy_algorithms["R_SR"] = R_SR
policy_algorithms["UCB"] = UCB
policy_algorithms["MaxUCB"] = MaxUCB
policy_algorithms["MaxUCB_adaptive_alpha"] = MaxUCB_adaptive_alpha

policy_algorithms["Rising_Bandit"] = Rising_Bandit
policy_algorithms["QoMax_ETC"] = QoMax_ETC
policy_algorithms["QoMax_SDA"] = QoMax_SDA
policy_algorithms["ER_UCB_S"] = ER_UCB_S
policy_algorithms["ER_UCB_N"] = ER_UCB_N
policy_algorithms["Max_Median"] = Max_Median
policy_algorithms["Exp3"] = Exp3
policy_algorithms["Threshold_Ascent"] = Threshold_Ascent
policy_algorithms["MaxSearch_Gaussian"] = MaxSearch_Gaussian
policy_algorithms["MaxSearch_SubGaussian"] = MaxSearch_SubGaussian
policy_algorithms["ThompsonSampling"] = ThompsonSampling

# alphas = np.arange(0.0, 3, 0.1)
# alphas = np.round(alphas, 2)
# for item in alphas:
#     policy_algorithms["MaxUCB_" + str(item)] = partial(MaxUCB, alpha=item)

# bruning_steps = np.arange(1, 9, 1)
# for item in bruning_steps:
#     policy_algorithms["MaxUCB_burn-in_" + str(item)] = partial(
#         MaxUCB, burn_in_steps=item
#     )

policy_algorithms["Oracle_Arm"] =  None


df = dataset[(dataset["arm_index"]>=0)]
df = df[["instance", "arm_index", "repetition", "iteration", "loss"]]
data = df.sort_values(by=["instance", "arm_index", "repetition", "iteration"])[
    "loss"
].values.reshape(len(instances), number_of_arms, number_of_trails, horizon_time)


for alg_name,alg in policy_algorithms.items():
    if(alg_name=="Oracle_Arm"):
        run_expriment = exp_utils.run_fake_expriment
    else:
        run_expriment = exp_utils.run_expriment
    if not os.path.exists(result_directory + alg_name):
        if(multiprocess=='joblib'):
            print(alg_name)
            result, result_pulled_arms = zip(
                *joblib.Parallel(n_jobs=-1, backend="multiprocessing")(
                    joblib.delayed(partial(run_expriment, alg))(
                        data[instance_num]
                    )
                    for instance_num in range(len(instances))
                )
            )
        else:
            result = []
            result_pulled_arms = []
            for instance_num in range(len(instances)):
                print(alg_name, instances[instance_num])
                res, res_t, res_pulled = run_expriment(alg, data[instance_num])
                result.append(res)
                result_pulled_arms.append(res_pulled)

        os.makedirs(result_directory + alg_name)
        with open(result_directory + alg_name + "/result.pkl", "wb") as file:
            pickle.dump(result,file )
        with open(result_directory + alg_name + "/pulled_arms.pkl", "wb") as file:
            pickle.dump(result_pulled_arms,file )
    else:
        print(alg_name +" does exist")

for alg_name in combined_search_algorithms:
    df = dataset[(dataset["arm_index"] < 0) & (dataset["optimizer"] == alg_name)]
    df = df[["instance", "arm_index", "repetition", "iteration", "loss"]]
    data = df.sort_values(by=["instance", "arm_index", "repetition", "iteration"])[
        "loss"
    ].values.reshape(len(instances), 1, number_of_trails, horizon_time)

    if not os.path.exists(result_directory + alg_name):
        if(multiprocess=='joblib'):
            print(alg_name)
            result, _ = zip(
                *joblib.Parallel(n_jobs=-1, backend="multiprocessing")(
                    joblib.delayed(partial(exp_utils.run_fake_expriment, alg_name))(
                        data[instance_num]
                    )
                    for instance_num in range(len(instances))
                )
            )
        else:
            result = []
            for instance_num in range(len(instances)):
                print(alg_name, instances[instance_num])
                res, _ = exp_utils.run_fake_expriment(alg_name, data[instance_num])
                result.append(res)

        os.makedirs(result_directory + alg_name)
        with open(result_directory + alg_name + "/result.pkl", "wb") as file:
            pickle.dump(result,file )
    else:
        print(alg_name +" does exist")