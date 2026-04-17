# Titanic Survival Prediction – Data Cleaning & Feature Engineering

## 📌 Project Overview

This project focuses on analyzing the Titanic dataset to build a strong foundation for predictive modeling. The goal is to improve data quality and extract meaningful features that enhance the prediction of passenger survival.

The workflow follows three key stages:

* Data Cleaning
* Feature Engineering
* Feature Selection

---

## 🎯 Objective

To preprocess the Titanic dataset by:

* Handling missing values and inconsistencies
* Engineering informative features
* Selecting the most relevant variables for modeling

---

## 📂 Project Structure

```
titanic_assignment/
│
├── data/
│   ├── train.csv
│   ├── test.csv
│   └── train_cleaned.csv
│
├── notebooks/
│   └── Titanic_Feature_Engineering.ipynb
│
├── scripts/
│   ├── data_cleaning.py
│   ├── feature_engineering.py
│   └── feature_selection.py
│
├── README.md
└── requirements.txt
```

---

## Data Cleaning

### Missing Value Handling

* **Age**: Imputed using median due to skewness and presence of outliers
* **Embarked**: Filled using mode since only a few values were missing
* **Cabin**: Transformed into a new feature (`Deck`) before dropping due to excessive missing values

### Outlier Handling

* Extreme values in **Fare** were capped at the 99th percentile to reduce their influence

### Data Consistency

* Standardized categorical values (e.g., `Sex`)
* Removed duplicate records

---

## Feature Engineering

Feature engineering was applied to improve the dataset’s predictive power:

### 🔹 Derived Features

* **FamilySize** = SibSp + Parch + 1
* **IsAlone** = Indicator for passengers traveling alone
* **FarePerPerson** = Fare divided by family size

### 🔹 Title Extraction

* Extracted titles (Mr, Mrs, Miss, etc.) from passenger names
* Grouped rare titles into a single category

### 🔹 Deck Feature

* Extracted from Cabin to represent passenger location

### 🔹 Age Groups

* Categorized into Child, Teen, Adult, and Senior

---

## Feature Transformation

* **Log Transformation** applied to Fare to reduce skewness
* **Scaling** performed on numerical features for model compatibility

---

## 🔢 Categorical Encoding

* One-hot encoding applied to:

  * Sex
  * Embarked
  * Title
  * Deck
  * AgeGroup

---

## Feature Selection

### Correlation Analysis

* Identified relationships between features
* Removed redundant variables

### Feature Importance

* Used Random Forest to rank features
* Most important predictors:

  * Sex
  * Pclass
  * Fare
  * Title

---

## Key Insights

* Female passengers had significantly higher survival rates
* Higher passenger class (Pclass) increased survival likelihood
* Smaller families or individuals traveling alone had lower survival chances
* Socioeconomic factors (Fare, Title) strongly influenced survival

---

## ⚙️ How to Run the Project

### 1. Clone the repository

```bash
git clone <your-repo-link>
cd titanic_assignment
```

### 2. Install dependencies

```bash
pip install -r requirements.txt
```

### 3. Run the notebook

```bash
jupyter notebook notebooks/Titanic_Feature_Engineering.ipynb
```

---

## Requirements

* Python 3.x
* pandas
* numpy
* matplotlib
* seaborn
* scikit-learn

---

## Conclusion

This project demonstrates the importance of:

* Proper data preprocessing
* Thoughtful feature engineering
* Data-driven feature selection

These steps significantly improve the performance and reliability of machine learning models.

---

## Author

**Moses Onyango**
BSc Computer Science – Egerton University

---
