"""
Titanic Dataset - Feature Selection Module
This script removes redundant features and selects the most informative
variables for downstream modeling.
"""

import pandas as pd
from sklearn.ensemble import RandomForestClassifier


CORRELATION_THRESHOLD = 0.85
TOP_N_FEATURES = 12
RANDOM_STATE = 42


def load_model_ready_data(train_path, test_path):
    """Load model-ready train and test datasets."""
    train = pd.read_csv(train_path)
    test = pd.read_csv(test_path)
    print(f"Model-ready training data loaded: {train.shape}")
    print(f"Model-ready test data loaded: {test.shape}")
    return train, test


def split_features_and_target(train_df, test_df):
    """Separate target/ID columns from predictor columns."""
    identifier_columns = ["PassengerId"]
    target_column = "Survived"

    y_train = train_df[target_column].copy()
    X_train = train_df.drop(columns=identifier_columns + [target_column], errors="ignore")
    X_test = test_df.drop(columns=identifier_columns, errors="ignore")

    return X_train, y_train, X_test


def remove_redundant_features(X_train, y_train, threshold=CORRELATION_THRESHOLD):
    """Drop highly correlated features, keeping the one more related to survival."""
    feature_target_corr = X_train.apply(lambda col: col.corr(y_train)).abs().fillna(0)
    feature_corr = X_train.corr().abs()

    columns_to_drop = set()
    columns = feature_corr.columns.tolist()

    for i in range(len(columns)):
        left = columns[i]
        if left in columns_to_drop:
            continue

        for j in range(i + 1, len(columns)):
            right = columns[j]
            if right in columns_to_drop:
                continue

            if feature_corr.loc[left, right] > threshold:
                keep = left if feature_target_corr[left] >= feature_target_corr[right] else right
                drop = right if keep == left else left
                columns_to_drop.add(drop)

    retained_columns = [col for col in X_train.columns if col not in columns_to_drop]
    reduction_report = pd.DataFrame({
        "feature": columns,
        "target_correlation": feature_target_corr.reindex(columns).values,
        "dropped_for_redundancy": [col in columns_to_drop for col in columns]
    })

    return retained_columns, reduction_report


def rank_feature_importance(X_train, y_train):
    """Rank features with a Random Forest classifier."""
    model = RandomForestClassifier(
        n_estimators=300,
        random_state=RANDOM_STATE,
        n_jobs=1
    )
    model.fit(X_train, y_train)

    importance_df = pd.DataFrame({
        "feature": X_train.columns,
        "importance": model.feature_importances_
    }).sort_values("importance", ascending=False, ignore_index=True)

    return importance_df


def select_top_features(importance_df, top_n=TOP_N_FEATURES):
    """Select a practical set of top-ranked features."""
    selected = importance_df.head(min(top_n, len(importance_df)))["feature"].tolist()
    return selected


def build_selected_datasets(train_df, test_df, selected_features):
    """Create selected train/test datasets while preserving key columns."""
    train_selected = train_df[["PassengerId", "Survived"] + selected_features].copy()
    test_selected = test_df[["PassengerId"] + selected_features].copy()
    return train_selected, test_selected


def save_outputs(train_selected, test_selected, importance_df, reduction_report):
    """Persist selected datasets and reports."""
    train_selected.to_csv("../data/train_selected.csv", index=False)
    test_selected.to_csv("../data/test_selected.csv", index=False)
    importance_df.to_csv("../data/feature_importance.csv", index=False)
    reduction_report.to_csv("../data/feature_correlation_review.csv", index=False)

    print("\nFiles saved:")
    print("  - data/train_selected.csv")
    print("  - data/test_selected.csv")
    print("  - data/feature_importance.csv")
    print("  - data/feature_correlation_review.csv")


def main():
    """Run the feature selection pipeline."""
    print("=" * 60)
    print("TITANIC FEATURE SELECTION PIPELINE")
    print("=" * 60)

    train_df, test_df = load_model_ready_data("../data/train_model_ready.csv", "../data/test_model_ready.csv")
    X_train, y_train, _ = split_features_and_target(train_df, test_df)

    retained_columns, reduction_report = remove_redundant_features(X_train, y_train)
    print(f"\nFeatures before redundancy filtering: {X_train.shape[1]}")
    print(f"Features after redundancy filtering: {len(retained_columns)}")

    X_train_reduced = X_train[retained_columns]
    importance_df = rank_feature_importance(X_train_reduced, y_train)
    selected_features = select_top_features(importance_df)

    print(f"\nSelected features ({len(selected_features)}):")
    for feature in selected_features:
        print(f"  - {feature}")

    train_selected, test_selected = build_selected_datasets(train_df, test_df, selected_features)
    save_outputs(train_selected, test_selected, importance_df, reduction_report)

    print("\nTop feature importances:")
    print(importance_df.head(10).to_string(index=False))

    print("\n" + "=" * 60)
    print("FEATURE SELECTION COMPLETED SUCCESSFULLY!")
    print("=" * 60)
    print(f"Selected train shape: {train_selected.shape}")
    print(f"Selected test shape: {test_selected.shape}")


if __name__ == "__main__":
    main()
