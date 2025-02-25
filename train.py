import numpy as np
import pandas as pd

from model import Model


def loss(real_rates, predicted_rates):
    return np.average(abs(predicted_rates / real_rates - 1.0)) * 100.0


def remove_outliers(df, column, percentile=99.98):
    """
    Removes rows where values exceed the specified percentile threshold.
    """
    threshold = np.percentile(df[column], percentile)
    return df[df[column] <= threshold], threshold


def add_custom_features(df):
    """
    Adds custom features like is_kma_equal and is_rate_outlier.
    """
    # is_kma_equal: True if origin and destination are the same
    df["is_kma_equal"] = df["destination_kma"] == df["origin_kma"]

    # df["is_rate_outlier"] = detect_outliers_percentile(df, column="rate", percentile=99.7)

    return df


def add_interaction_features(df):
    """
    Adds interaction features to the dataframe.
    """
    df["miles_weight_interaction"] = df["valid_miles"] * df["weight"]
    df["kma_interaction"] = df["origin_kma"] + "_" + df["destination_kma"]
    return df


def train_and_validate():
    df = pd.read_csv("dataset/train.csv")
    print(f"Initial shape: {df.shape}")
    df = df.dropna().drop_duplicates()
    print(f"Without NaN and duplicates: {df.shape}")

    len_before = df.shape[0]
    df, rate_threshold = remove_outliers(df, "rate", percentile=99.98)
    len_after = df.shape[0]
    print(f"Rate threshold: {rate_threshold}")
    print(f"Without outliers: {df.shape}")
    print(f"Outliers removed: {len_before - len_after}")

    df["rate"] = np.log1p(df["rate"])

    df = add_interaction_features(df)
    df = add_custom_features(df)

    # Define features
    numerical_features = [
        "valid_miles",
        "weight",
        # "miles_weight_interaction",
    ]

    temporal_features = [
        # "month_sin",
        # "month_cos",
        # "day_of_week_sin",
        # "day_of_week_cos",
        # "hour_sin",
        # "hour_cos",
        "month",
        "day_of_week",
        "hour",
        "season_num",
    ]
    high_cardinality_categorical_features = ["origin_kma", "destination_kma", "kma_interaction"]

    low_cardinality_categorical_features = [
        "transport_type",
        "is_kma_equal",
    ]

    print(f"Numerical Features: {numerical_features}")
    print(f"Temporal Features: {temporal_features}")
    print(f"High Cardinality Categorical Features: {high_cardinality_categorical_features}")
    print(f"Low Cardinality Categorical Features: {low_cardinality_categorical_features}")

    model = Model()
    model.build_pipeline(
        numerical_features=numerical_features,
        temporal_features=temporal_features,
        high_cardinality_categorical_features=high_cardinality_categorical_features,
        low_cardinality_categorical_features=low_cardinality_categorical_features,
    )

    # Best Params : {'regressor__learning_rate': 0.12, 'regressor__max_depth': 12, 'regressor__n_estimators': 250, 'regressor__subsample': 1.0}
    model.fit(df, df["rate"], n_trials=5)

    df = pd.read_csv("dataset/validation.csv")
    df["rate"] = np.log1p(df["rate"])
    df = add_interaction_features(df)
    df = add_custom_features(df)
    predicted_log_rates = model.predict(df)
    predicted_rates = np.expm1(predicted_log_rates)
    real_rates = np.expm1(df["rate"])

    mape = loss(real_rates, predicted_rates)
    mape = np.round(mape, 2)
    print(f"Accuracy of validation is {mape}%")
    return mape


def generate_final_solution():
    # combine train and validation to improve final predictions
    df = pd.read_csv("dataset/train.csv")
    df_val = pd.read_csv("dataset/validation.csv")
    df = df.append(df_val).reset_index(drop=True)
    df["rate"] = np.log1p(df["rate"])
    df = add_interaction_features(df)
    df = add_custom_features(df)

    model = Model()
    model.load_model()
    # fit again!

    # generate and save test predictions
    df_test = pd.read_csv("dataset/test.csv")
    df_test = add_interaction_features(df_test)
    df_test = add_custom_features(df_test)
    predicted_log_rates = model.predict(df_test)
    df_test["predicted_rate"] = np.expm1(predicted_log_rates)
    df_test.to_csv("dataset/predicted.csv", index=False)


if __name__ == "__main__":
    mape = train_and_validate()

    print(f"Accuracy of validation is {mape}%")

    if mape < 9:  # try to reach 9% or less for validation
        generate_final_solution()
        print("'predicted.csv' is generated.")
    else:
        print("No model met the MAPE < 9% requirement.")
