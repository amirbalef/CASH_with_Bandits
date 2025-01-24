# **HPO_Runs**  
This directory is dedicated to extracting HPO trajectories for benchmarking and analysis.  

---

## **Dependencies**  
Using a Conda environment is recommended for managing dependencies.  

### Required Packages  
You may need to install and set up the following repositories:  
- **[TabRepo](https://github.com/autogluon/tabrepo)**: A collection of tabular datasets and utilities for benchmarking.  
- **[YAHPO Gym](https://github.com/slds-lmu/yahpo_gym)**: A benchmarking suite for hyperparameter optimization tasks.  

### Python Version  
Ensure that you are using **Python 3.9–3.11**, as other Python versions may not be supported.  Only Linux support has been tested.

### Installation  
To install the required dependencies, run:  
```bash
pip install -r requirements.txt
```

## Running Experiments
To run experiments, execute the following command:

```bash
python main_run_hpo.py 
```

## Extracting HPO trajectoriess.
After running the experiments, extract the generated HPO trajectories with:

```bash
python extract_HPO_runs.py 
```