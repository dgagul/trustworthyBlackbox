
# Trustworthiness of Black Box models

This repository contains code and additional resources including execution examples of the work done during the period of the Master's Thesis about "Computing the Trustworthiness Level of Black Box Machine and Deep Learning Models". The code extends the capabilities of the [`TRUSTED.AI`](https://github.com/JoelLeupp/Trusted-AI/tree/main) application developed by Leupp et al. and improves the trustworthiness assessment of Black Box Machine Learning models with limited information available.


## Repository Structure

The structure consists of four main folders:
- Resources: contains various document types used during the thesis to generate illustrative Figures
- Score: contains Python code responsible for calculating and plotting various metrics and trust scores 
- Solutions: subfolders contain all important information for the evaluation scenarios presented in the thesis
- synDS: contains Python code capable of creating synthetic datasets, offering two approaches (MUST and MAY)
to balance privacy and accuracy concerns.

Execution examples can be found in the `compute_true_score.ipynb` and `compute_advanced_score.ipynb` Jupyter Notebooks. 
The cells are documented and intuitive. Before executing all cells, make sure to specify the variables indicatded by a  `Todo:`. 
Plots are automatically stored in the respective folder of the evaluation scenario.


```bash
├── Resources # documents used to generate illustrations for report
│   ├── *.psd 
│   ├── *.png
│   ├── *.ppt
│   └── *.xlsx
├── Score # compute trust scores
│   ├── algorithms
│   │   ├── explainability.py
│   │   ├── fairness.py
│   │   ├── methodology.py
│   │   ├── robustness.py
│   │   └── trustworthiness.py
│   ├── configs
│   │   ├── mappings
│   │   ├── metrics
│   │   └── weights
│   ├── helpers.py
│   └── plot.py   
├── Solutions # hold information of evaluation scenarios
│   ├── S1DiabetesPredictionRF
│   │   ├── AdvancedScoreMAY
│   │   ├── AdvancedScoreMUST
│   │   ├── LimitedScore
│   │   ├── TrueScore
│   │   ├── *.png
│   │   └── train_RF.ipynb
│   ├── S2HomeCreditDefaultRiskDNN
│   │   ├── *Score*
│   │   ├── *.png
│   │   ├── DNN.py
│   │   ├── model.keras
│   │   └── train_DNN.ipynb
│   └── compare_datasets.ipynb
├── synDS # generate synthetic dataset
│   ├── generator.py
│   ├── propertiesScanner.py
│   ├── time-complexity.ipynb
│   └── timecomplexity.png
├── .gitignore
├── compute_advanced_score.ipynb # may and must
├── compute_true_score.ipynb # and limited
├── config.py 
├── ReadMe.md
└── requirements.txt
```


## Installation

The following steps describe the installation process:


1. Clone the GitHub Repository
```
git clone https://github.com/dgagul/trustworthyBlackbox
cd trustworthyBlackbox
```

2. Create a virtual environment and install the dependencies
```
python -m venv BBenv
BBenv\Scripts\activate
pip install -r requirements.txt
```


3. Launch the Jupyter Notebook and execute desired scenario
```
jupyter notebook
```


