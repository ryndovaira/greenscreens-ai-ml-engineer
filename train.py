import pandas as pd
import numpy as np
from model import Model
from tqdm import tqdm


def loss(real_rates, predicted_rates):
    return np.average(abs(predicted_rates / real_rates - 1.0)) * 100.0


def add_interaction_features(df):
    df = df.copy()
    df["kma_interaction"] = df["origin_kma_mean_rate"] * df["destination_kma_mean_rate"]
    df["miles_weight_interaction"] = df["valid_miles"] * df["weight"]
    return df


def train_and_validate():
    df = pd.read_csv("dataset/train.csv")

    # Define features
    numerical_features = ["valid_miles", "weight", "miles_weight_interaction"]
    categorical_features = ["origin_kma", "destination_kma", "kma_interaction"]

    # Create interaction features
    df["kma_interaction"] = df["origin_kma"] + "_" + df["destination_kma"]
    df["miles_weight_interaction"] = df["valid_miles"] * df["weight"]

    model = Model()
    model.build_pipeline(numerical_features, categorical_features)

    # Hyperparameter grid
    param_grid = {
            "regressor__n_estimators": [100, 200, 300],
            "regressor__max_depth": [ 6, 10, 12, 15],
            "regressor__learning_rate": [0.8, 0.1, 0.12, 0.18],
            "regressor__subsample": [0.8, 1.0],
        }

    # Train the model
    model.fit(df, df["rate"], param_grid=param_grid)

    # Validate on validation set
    df_val = pd.read_csv("dataset/validation.csv")
    df_val["kma_interaction"] = df_val["origin_kma"] + "_" + df_val["destination_kma"]
    df_val["miles_weight_interaction"] = df_val["valid_miles"] * df_val["weight"]

    predicted_rates = model.predict(df_val)
    mape = loss(df_val["rate"], predicted_rates)
    print(f"Validation MAPE: {mape}%")
    return mape


def generate_final_solution(best_model_type):
    # combine train and validation to improve final predictions
    df = pd.read_csv("dataset/train.csv")
    df_val = pd.read_csv("dataset/validation.csv")
    df = pd.concat([df, df_val]).reset_index(drop=True)
    df = add_interaction_features(df)

    features = [
        "valid_miles",
        "weight",
        "origin_kma_mean_rate",
        "destination_kma_mean_rate",
        "kma_interaction",
        "miles_weight_interaction",
    ]

    model = Model(model_type=best_model_type)
    model.build_pipeline(features, ["origin_kma", "destination_kma", "kma_interaction"])
    model.fit(df[features], df["rate"])

    # generate and save test predictions
    df_test = pd.read_csv("dataset/test.csv")
    df_test = add_interaction_features(df_test)
    df_test["predicted_rate"] = model.predict(df_test[features])
    df_test.to_csv("dataset/predicted.csv", index=False)
    print(f"'predicted.csv' generated using {best_model_type}")


if __name__ == "__main__":
    mape = train_and_validate()

    # Select best model
    print(f"Best MAPE: {mape}%")

    if results[best_model] < 9:  # try to reach 9% or less for validation
        generate_final_solution(best_model)
        print("Final prediction generated.")
    else:
        print("No model met the MAPE < 9% requirement.")
