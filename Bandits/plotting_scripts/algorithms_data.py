
printing_name_dict = {}
printing_name_dict["YaHPOGym"] = "YaHPOGym[SMAC]" #  \\citep{pfisterer-automl22a}"
printing_name_dict["Reshuffling"] = "Reshuffling[HEBO]"
printing_name_dict["TabRepo"] = "TabRepo[RS]"
printing_name_dict["TabRepoRaw"] = "TabRepoRaw[SMAC]"

printing_name_dict["SMAC_NoInit"] = "SMAC-no-init"
printing_name_dict["SMAC"]= "SMAC" #  \\citep{lindauer-jmlr22a}"
printing_name_dict["RandomSearch"] = "Random Search"

printing_name_dict["UCB"]= "UCB"
printing_name_dict["QuantileUCB"]= "Quantile UCB"
printing_name_dict["Exp3_TB"]= "Exp3"
printing_name_dict["Exp3_OG"]= "Exp3"

printing_name_dict["ER_UCB_N"]= "ER-UCB-N" #   \\citep{hu2021cascaded}"
printing_name_dict["ER_UCB_S"]= "ER-UCB-S" #  \\citep{hu2021cascaded}"

printing_name_dict["QoMax_SDA"]= "QoMax-SDA" #   \\citep{baudry2022efficient}"
printing_name_dict["QoMax_ETC"]= "QoMax-ETC" #  \\citep{baudry2022efficient}"
printing_name_dict["Max_Median"]= "Max-Median" #   \\citep{bhatt2022extreme}"
printing_name_dict["Rising_Bandit"]= "Rising Bandit" #   \\citep{li2020efficient}"
printing_name_dict["Random"]= "Random Policy"

printing_name_dict["Q_BayesUCB"]= "Quantile Bayes UCB"
printing_name_dict["Successive_Halving"]= "Successive Halving"

printing_name_dict["MaxUCB"] = "\\textbf{MaxUCB }"

printing_name_dict["Oracle"] = "Oracle"
printing_name_dict["Oracle_Arm"] = "Oracle Arm"



alphas = [round(x, 2) for x in [i * 0.1 for i in range(30)]]
for item in alphas:
    printing_name_dict["MaxUCB_" + str(item)] = "MaxtUCB ( $\\alpha$=" + str(item) + ")"
    printing_name_dict["MaxUCB_" + str(item)] = "$\\alpha$=" + str(item) + ""

intial_steps = [1,2,3,4,5,6,7,8,9,10]
for item in intial_steps:
    printing_name_dict["MaxUCB_init_steps_" + str(item)] = (
        "MaxUCB ( init steps=" + str(item) + ")"
    )
bruning_steps = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
for item in bruning_steps:
    printing_name_dict["MaxUCB_burn-in_" + str(item)] = (
        "\\textbf{MaxUCB-Burn-in(C=" + str(item) + ")}"
    )

for i in range(7):
    printing_name_dict["Arm_" + str(i)] =  "Arm " + str(i)


printing_name_dict["Threshold_Ascent"] = "Threshold Ascent"
printing_name_dict["MaxSearch_Gaussian"] = "MaxSearch Gaussian"
printing_name_dict["MaxSearch_SubGaussian"] = "MaxSearch SubGaussian"


printing_name_dict["CatBoost"] = "CatBoost"
printing_name_dict["ExtraTrees"] = "ExtraTrees"
printing_name_dict["LightGBM"] = "LightGBM"
printing_name_dict["NeuralNetFastAI"] = "NN(FastAI)"
printing_name_dict["NeuralNetTorch"] = "NN(PyTorch)"
printing_name_dict["RandomForest"] = "Random Forest"
printing_name_dict["XGBoost"] = "XGBoost"