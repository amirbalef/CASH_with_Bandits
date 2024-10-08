import os
import numpy as np
import pandas as pd
import pickle

# dataset_name = "TabRepo"
#dataset_name = "TabRepoRaw"
dataset_name = "YaHPOGym"

res_dir = "./results/"
root_dir = res_dir + dataset_name + "/"

if dataset_name == "TabRepo":
    number_of_arms = 7
    optimizer_per_arm = "RandomSearch"
    optimizers = ["RandomSearch"]
    number_of_trails = 32
    classifier_key = "config:methods:__choice__"

if dataset_name == "YaHPOGym":
    number_of_arms = 6
    optimizer_per_arm = "SMAC"
    optimizers = ["RandomSearch", "SMAC", "SMAC_NoInit"]
    number_of_trails = 32
    classifier_key = "config:Model:learner_id"

if dataset_name == "TabRepoRaw":
    number_of_arms = 7
    optimizer_per_arm = "SMAC"
    optimizers = ["RandomSearch", "SMAC"]
    number_of_trails = 32
    classifier_key = "config:methods:__choice__"


instances = [instance
for instance in os.listdir(root_dir)
    if (os.path.isdir(root_dir + instance) and instance[0] != "_")
]

results_list=[]
errors = []
for instace in instances:
    print(instace)
    try:
        instace_dir = root_dir + instace + "/"
        arm_index_method_list = [
            (
                int(arm_index),
                optimizer_per_arm, optimizer_per_arm + "_Arm_" + str(arm_index),
            )
            for arm_index in range(number_of_arms)
        ]
        arm_index_method_list.extend([ -1, optimizers[i], optimizers[i]] for i in range(len(optimizers)) )

        for arm_index, optimizer, optimizer_method in arm_index_method_list:
            for trial in range(number_of_trails):
                result = pd.read_pickle(
                    instace_dir + optimizer_method + "/" + str(trial) + "/result.pkl"
                )
                try:
                    if(instace == "cylinder-bands"):
                        pass
                        #print(result.columns)
                        #print(result["summary:model_error"].to_numpy())
                    losses = result["summary:model_error"].to_numpy()
                    classifiers = result[classifier_key].to_numpy()
                except Exception as e:
                    print("error", optimizer_method, instace, trial, arm_index)
                    print(result.columns)
                    print(result["summary:model_error"].to_numpy())
                    print(e)
                
                for iteration, (loss, classifier) in enumerate(zip(losses, classifiers)):
                    dict1 = {
                        "instance": instace,
                        "repetition": trial,
                        "arm_index": arm_index,
                        "iteration": iteration,
                        "loss": loss,
                        "optimizer": optimizer,
                        "classifier": classifier,
                    }
                    results_list.append(dict1)

    except Exception as e:
        print(e)

        errors.append(instace)
        pass

df = pd.DataFrame(results_list)

df.to_csv(dataset_name + ".csv", index=False)