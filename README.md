# Titanic Survival Prediction - Data Preparation Pipeline

## Project Overview

This project prepares the Titanic dataset for machine learning through a three-stage workflow:

1. Data cleaning
2. Feature engineering
3. Feature selection

The repository includes scripts, a notebook, and generated CSV outputs for each stage of the pipeline.

## Objective

The main goal is to transform the raw Titanic dataset into model-ready and reduced-feature datasets by:

- handling missing values
- treating outliers
- creating informative features
- encoding and scaling predictors
- selecting the most useful features for downstream modeling

## Project Structure

```text
COMP 334 AI/
|-- data/
|   |-- train.csv
|   |-- test.csv
|   |-- gender_submission.csv
|   |-- train_cleaned.csv
|   |-- test_cleaned.csv
|   |-- train_engineered.csv
|   |-- test_engineered.csv
|   |-- train_model_ready.csv
|   |-- test_model_ready.csv
|   |-- train_selected.csv
|   |-- test_selected.csv
|   |-- feature_importance.csv
|   `-- feature_correlation_review.csv
|-- notebooks/
|   |-- Titanic_Feature_Engineering.ipynb
|   `-- outlier_treatment.png
|-- scripts/
|   |-- data_cleaning.py
|   |-- feature_engineering.py
|   `-- feature_selection.py
|-- README.md
`-- Requirements.txt
```

## Pipeline Summary

### 1. Data Cleaning

Implemented in `scripts/data_cleaning.py`.

This stage:

- loads `data/train.csv` and `data/test.csv`
- analyzes missing values in both datasets
- imputes missing `Age` values using the training median
- fills missing `Embarked` values using the mode
- fills missing test-set `Fare` values using the training median
- extracts `Deck` from `Cabin`
- adds `Age_Missing` and `Cabin_Missing` indicator columns
- drops the original `Cabin` column
- standardizes `Sex` values
- removes duplicate training rows if any exist
- detects and caps outliers in `Age` and `Fare` using the IQR rule
- saves an outlier visualization to `notebooks/outlier_treatment.png`

Outputs:

- `data/train_cleaned.csv`
- `data/test_cleaned.csv`

### 2. Feature Engineering

Implemented in `scripts/feature_engineering.py`.

This stage:

- loads cleaned train and test datasets
- creates `FamilySize = SibSp + Parch + 1`
- creates `IsAlone` as an indicator for solo travelers
- creates `FarePerPerson`
- extracts passenger `Title` from `Name`
- normalizes titles such as `Mlle`, `Ms`, and `Mme`
- groups uncommon titles into `Rare`
- creates `AgeGroup` with these bands:
  `Child`, `Teen`, `Adult`, `Senior`
- creates `FareLog` using `log1p`
- preserves or reconstructs `Deck` when needed
- one-hot encodes categorical variables:
  `Sex`, `Embarked`, `Title`, `Deck`, and `AgeGroup`
- scales numeric predictors using `StandardScaler`
- keeps `PassengerId` and `Survived` in the final training output

Outputs:

- `data/train_engineered.csv`
- `data/test_engineered.csv`
- `data/train_model_ready.csv`
- `data/test_model_ready.csv`

### 3. Feature Selection

Implemented in `scripts/feature_selection.py`.

This stage:

- loads the model-ready datasets
- separates identifiers and target labels
- removes highly correlated features using a correlation threshold of `0.85`
- keeps the feature with stronger correlation to `Survived` when redundant pairs are found
- ranks retained features using `RandomForestClassifier`
- selects the top `12` features by importance
- exports both reduced datasets and analysis reports

Outputs:

- `data/train_selected.csv`
- `data/test_selected.csv`
- `data/feature_importance.csv`
- `data/feature_correlation_review.csv`

## How to Run

### 1. Install dependencies

```bash
pip install -r Requirements.txt
```

### 2. Run the pipeline scripts

Run the scripts from the `scripts` folder so their relative paths resolve correctly:

```bash
cd scripts
python data_cleaning.py
python feature_engineering.py
python feature_selection.py
```

You can also explore the workflow interactively in:

```bash
jupyter notebook notebooks/Titanic_Feature_Engineering.ipynb
```

## Requirements

The project uses:

- Python 3.x
- pandas 1.5.3
- numpy 1.24.3
- matplotlib 3.7.1
- seaborn 0.12.2
- scikit-learn 1.2.2
- jupyter 1.0.0

## Key Deliverables

By the end of the workflow, the project produces:

- cleaned datasets ready for further transformation
- engineered datasets with domain-based features
- model-ready datasets with encoded and scaled predictors
- reduced datasets containing the selected top features
- feature importance and correlation review reports

## Author

**Moses Onyango**  
BSc Computer Science - Egerton University
