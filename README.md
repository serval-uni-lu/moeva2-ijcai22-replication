# Replication package

## Introduction

This package contains all the necessary data and scripts to replicate the experiments of our research paper "A Unified Framework for Adversarial Attack and Defense in Constrained Feature Space".

## Create the environment

We recommend using [miniconda](https://docs.conda.io/en/latest/miniconda.html) and Python 3.8.8 on a Linux system.
```
conda create -n moeva2 python=3.8.8
conda activate moeva2
pip install -r requirements.txt
```
Download the additional data from https://figshare.com/s/84ae808ce6999fafd192 and place the content of the downloaded data folder in the data folder of this folder.

Alternatively, run the script './prepare.sh' to create the conda enviornment and place the data automatically.

## File structure

- config: configurations files used to prepare and run the experiments
- data: the source data used to craft the attacks
- models: pre-trained models
- out: results of the experiments
- src: scripts and modules to run the attacks
  - attacks: module containing the implementations of the attacks
  - examples: usecase-specific scripts
  - botnet and lcld: scripts to prepare data and models for the experiments for CTU-13 and LCLD respectively
  - united: scripts to run a single attack

## Usage

Run the following command to launch all the experiments at once.
```
conda activate moeva2
./run_all.sh
```

The results are found in ./out/attacks/[project]/rq[X]/metrics_[attack_name]_[hash].json

Feel free to comment out experiments in `run_all.sh` and/or reduce the number of original examples (n_initial_state parameter) in the config file 
ending with ```static.yaml``` to speed up the experiments.

The data used to train the models are also provided in the archive ```data-1``` and ```data-2```.
Simply merge the folders if you intend to use them.

Use the following commands to retrain the models:
```
python -m src.experiments.lcld.01_train_robust -c ./config/rq0.lcld.yaml
python -m src.experiments.botnet.01_train_robust -c ./config/rq0.botnet.yaml
```

## Citation 

If you have used our framework for research purposes, you can cite our publication by: TBD.