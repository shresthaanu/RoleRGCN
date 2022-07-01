# Joint Credibility Estimation of News, User, and Publisher via Role-Relational Graph Convolutional Networks

This repository contains the code for the paper 
"Joint Credibility Estimation of News, User, and Publisher via Role-Relational Graph Convolutional Networks"

# Data Sources
FakeNewsNet-PolitiFact dataset can be downloaded using the code provided at https://github.com/KaiDMML/FakeNewsNet

PolitiFact-2021 dataset can be downloaded using the instructions provided at ```Data/README.md```

# Content

* Script to generate train-test split : ```Code/Utils/five_fold_train_test_split.ipynb```
* Script to run baseline models for FakeNewsNet-PolitiFact dataset: ```Code/Experiments/FakeNewsNet-PolitiFact/Baseline_exp.ipynb```
* Script to run Role-RGCN model for FakeNewsNet-PolitiFact dataset: ```Code/Experiments/FakeNewsNet-PolitiFact/Role_RGCN_exp.ipynb```
* * Script to run baseline models for PolitiFact-2021 dataset: ```Code/Experiments/PolitiFact-2021/Baseline_exp.ipynb```
* Script to run Role-RGCN model for PolitiFact-2021 dataset: ```Code/Experiments/PolitiFact-2021/Role_RGCN_exp.ipynb```
* Helper files: ```Utils/features.py```, ```Models/models.py```
