"""
Titanic Dataset - Feature Engineering Module
This script creates derived features, encodes categoricals, and scales numeric data
for downstream modeling.
"""

import re

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler


TITLE_PATTERN = re.compile(r",\s*([^\.]+)\.")
RARE_TITLES = {
    "Lady", "Countess", "Capt", "Col", "Don", "Dr", "Major", "Rev",
    "Sir", "Jonkheer", "Dona", "the Countess"
}
TITLE_NORMALIZATION = {
    "Mlle": "Miss",
    "Ms": "Miss",
    "Mme": "Mrs",
}
AGE_GROUP_BINS = [0, 12, 19, 60, np.inf]
AGE_GROUP_LABELS = ["Child", "Teen", "Adult", "Senior"]


def load_cleaned_data(train_path, test_path):
    """Load cleaned train and test data."""
    train = pd.read_csv(train_path)
    test = pd.read_csv(test_path)
    print(f"Cleaned training data loaded: {train.shape}")
    print(f"Cleaned test data loaded: {test.shape}")
    return train, test


def extract_title(name):
    """Extract and normalize title from passenger name."""
    match = TITLE_PATTERN.search(str(name))
    title = match.group(1).strip() if match else "Unknown"
    title = TITLE_NORMALIZATION.get(title, title)
    if title in RARE_TITLES:
        return "Rare"
    return title


def create_engineered_features(df):
    """Create domain features from the cleaned Titanic dataset."""
    engineered = df.copy()

    engineered["FamilySize"] = engineered["SibSp"] + engineered["Parch"] + 1
    engineered["IsAlone"] = (engineered["FamilySize"] == 1).astype(int)
    engineered["FarePerPerson"] = engineered["Fare"] / engineered["FamilySize"].replace(0, 1)
    engineered["Title"] = engineered["Name"].apply(extract_title)
    engineered["AgeGroup"] = pd.cut(
        engineered["Age"],
        bins=AGE_GROUP_BINS,
        labels=AGE_GROUP_LABELS,
        include_lowest=True
    )
    engineered["FareLog"] = np.log1p(engineered["Fare"].clip(lower=0))

    # Keep feature engineering resilient even if older cleaned files do not contain Deck.
    if "Deck" not in engineered.columns:
        engineered["Deck"] = np.where(
            engineered.get("Cabin_Missing", 1).astype(int) == 1,
            "Unknown",
            "Known"
        )
    else:
        engineered["Deck"] = engineered["Deck"].fillna("Unknown").replace("", "Unknown")

    return engineered


def encode_and_scale(train_df, test_df):
    """Encode categoricals, align columns, and scale numeric features."""
    target = train_df["Survived"].copy() if "Survived" in train_df.columns else None
    train_ids = train_df["PassengerId"].copy() if "PassengerId" in train_df.columns else None
    test_ids = test_df["PassengerId"].copy() if "PassengerId" in test_df.columns else None

    drop_columns = ["Name", "Ticket"]
    train_features = train_df.drop(columns=drop_columns, errors="ignore")
    test_features = test_df.drop(columns=drop_columns, errors="ignore")

    if "Survived" in train_features.columns:
        train_features = train_features.drop(columns=["Survived"])

    categorical_columns = ["Sex", "Embarked", "Title", "Deck", "AgeGroup"]
    available_categoricals = [col for col in categorical_columns if col in train_features.columns or col in test_features.columns]

    combined = pd.concat([train_features, test_features], axis=0, sort=False)
    combined = pd.get_dummies(combined, columns=available_categoricals, drop_first=False, dtype=int)

    train_rows = len(train_features)
    train_encoded = combined.iloc[:train_rows].copy()
    test_encoded = combined.iloc[train_rows:].copy()

    numeric_columns = [
        col for col in train_encoded.select_dtypes(include=[np.number]).columns
        if col not in {"PassengerId"}
    ]

    scaler = StandardScaler()
    train_encoded[numeric_columns] = scaler.fit_transform(train_encoded[numeric_columns])
    test_encoded[numeric_columns] = scaler.transform(test_encoded[numeric_columns])

    if train_ids is not None:
        train_encoded["PassengerId"] = train_ids.values
    if test_ids is not None:
        test_encoded["PassengerId"] = test_ids.values
    if target is not None:
        train_encoded["Survived"] = target.values

    train_priority = [col for col in ["PassengerId", "Survived"] if col in train_encoded.columns]
    test_priority = [col for col in ["PassengerId"] if col in test_encoded.columns]

    train_encoded = train_encoded[train_priority + [col for col in train_encoded.columns if col not in train_priority]]
    test_encoded = test_encoded[test_priority + [col for col in test_encoded.columns if col not in test_priority]]

    return train_encoded, test_encoded


def save_outputs(train_engineered, test_engineered, train_model_ready, test_model_ready):
    """Persist engineered datasets."""
    train_engineered.to_csv("../data/train_engineered.csv", index=False)
    test_engineered.to_csv("../data/test_engineered.csv", index=False)
    train_model_ready.to_csv("../data/train_model_ready.csv", index=False)
    test_model_ready.to_csv("../data/test_model_ready.csv", index=False)

    print("\nFiles saved:")
    print("  - data/train_engineered.csv")
    print("  - data/test_engineered.csv")
    print("  - data/train_model_ready.csv")
    print("  - data/test_model_ready.csv")


def main():
    """Run the feature engineering pipeline."""
    print("=" * 60)
    print("TITANIC FEATURE ENGINEERING PIPELINE")
    print("=" * 60)

    train_clean, test_clean = load_cleaned_data("../data/train_cleaned.csv", "../data/test_cleaned.csv")

    train_engineered = create_engineered_features(train_clean)
    test_engineered = create_engineered_features(test_clean)
    train_model_ready, test_model_ready = encode_and_scale(train_engineered, test_engineered)

    save_outputs(train_engineered, test_engineered, train_model_ready, test_model_ready)

    print("\n" + "=" * 60)
    print("FEATURE ENGINEERING COMPLETED SUCCESSFULLY!")
    print("=" * 60)
    print(f"Engineered train shape: {train_engineered.shape}")
    print(f"Engineered test shape: {test_engineered.shape}")
    print(f"Model-ready train shape: {train_model_ready.shape}")
    print(f"Model-ready test shape: {test_model_ready.shape}")


if __name__ == "__main__":
    main()
