import h2o
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from h2o.automl import H2OAutoML
import os
import json


def loss(real_rates, predicted_rates):
    """
    Calculate Mean Absolute Percentage Error (MAPE).
    """
    return np.mean(np.abs((real_rates - predicted_rates) / real_rates)) * 100


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
    df["is_kma_equal"] = df["destination_kma"] == df["origin_kma"]
    return df


def log_skewed_columns(df, columns):
    """
    Log transform skewed columns.
    """
    for col in columns:
        df[f"log_{col}"] = np.log1p(df[col])
    return df


def add_interaction_features(df):
    """
    Adds interaction features to the dataframe.
    """
    df["miles_weight_interaction"] = df["valid_miles"] * df["weight"]
    df["kma_interaction"] = df["origin_kma"] + "_" + df["destination_kma"]
    return df


def extract_temporal_features(df, datetime_col="pickup_date"):
    """
    Extracts temporal and cyclic features from the pickup_date.
    """
    df[datetime_col] = pd.to_datetime(df[datetime_col])
    df["month"] = df[datetime_col].dt.month
    df["day_of_week"] = df[datetime_col].dt.dayofweek
    df["hour"] = df[datetime_col].dt.hour

    # Cyclic encoding
    df["month_sin"] = np.sin(2 * np.pi * df["month"] / 12)
    df["month_cos"] = np.cos(2 * np.pi * df["month"] / 12)
    df["day_of_week_sin"] = np.sin(2 * np.pi * df["day_of_week"] / 7)
    df["day_of_week_cos"] = np.cos(2 * np.pi * df["day_of_week"] / 7)
    df["hour_sin"] = np.sin(2 * np.pi * df["hour"] / 24)
    df["hour_cos"] = np.cos(2 * np.pi * df["hour"] / 24)

    # Season feature
    df["season"] = df["month"].map(
        {
            12: "winter",
            1: "winter",
            2: "winter",
            3: "spring",
            4: "spring",
            5: "spring",
            6: "summer",
            7: "summer",
            8: "summer",
            9: "autumn",
            10: "autumn",
            11: "autumn",
        }
    )
    season_mapping = {"winter": 0, "spring": 1, "summer": 2, "autumn": 3}
    df["season_num"] = df["season"].map(season_mapping)

    return df


class Model:
    def __init__(self, experiment_name="experiment"):
        h2o_port = int(os.getenv("H2O_PORT", 54321))
        log_dir = os.getenv("H2O_LOG_DIR", "./logs")

        h2o.init(
            log_dir=log_dir, ip="localhost", port=h2o_port, nthreads=-1, bind_to_localhost=False
        )

        self.aml = None
        self.leader = None
        self.experiment_name = experiment_name
        os.makedirs(self.experiment_name, exist_ok=True)

    def fit(self, df, target_column, validation_df):
        """
        Train H2O AutoML on the provided DataFrame.
        """
        # Log-transform the target to handle outliers better
        df["log_" + target_column] = np.log1p(df[target_column])
        validation_df["log_" + target_column] = np.log1p(validation_df[target_column])

        # Convert DataFrame to H2OFrame
        h2o_df = h2o.H2OFrame(df)
        h2o_valid_df = h2o.H2OFrame(validation_df)

        # Feature selection (exclude original target and date)
        features = [
            col
            for col in df.columns
            if col not in [target_column, "log_" + target_column, "pickup_date"]
        ]
        target = "log_" + target_column

        # Run H2O AutoML
        self.aml = H2OAutoML(
            project_name=self.experiment_name,
            max_models=1,
            max_runtime_secs=60 * 5,
            seed=42,
            sort_metric="MAE",
            stopping_metric="MAE",
            stopping_rounds=1,
            exclude_algos=["DeepLearning", "StackedEnsemble"],  # Exclude heavy models
        )

        self.aml.train(x=features, y=target, training_frame=h2o_df, leaderboard_frame=h2o_valid_df)

        # Select the best model
        self.leader = self.aml.leader

        # Save model and experiment details
        self.save_model()
        self.save_feature_importance()
        self.save_model_params()

    def predict(self, df):
        """
        Make predictions using the trained H2O model.
        """
        h2o_df = h2o.H2OFrame(df)
        preds = self.leader.predict(h2o_df).as_data_frame()
        return np.expm1(preds["predict"])  # Convert back from log scale

    def save_model(self):
        """
        Save the best model.
        """
        model_path = h2o.save_model(model=self.leader, path=self.experiment_name, force=True)
        print(f"Model saved at: {model_path}")

    def save_model_params(self):
        """
        Save the hyperparameters of the best model.
        """
        params = self.leader.params
        params_path = os.path.join(self.experiment_name, "best_model_params.json")
        with open(params_path, "w") as f:
            json.dump(params, f, indent=4)
        print(f"Hyperparameters saved at: {params_path}")

    def save_feature_importance(self):
        """
        Save feature importance and plot the result.
        """
        varimp = self.leader.varimp(use_pandas=True)
        varimp_path = os.path.join(self.experiment_name, "feature_importance.csv")
        varimp.to_csv(varimp_path, index=False)
        print(f"Feature importance saved at: {varimp_path}")

        # Plot feature importance
        plt.figure(figsize=(12, 6))
        plt.barh(varimp["variable"], varimp["relative_importance"])
        plt.title("Feature Importance")
        plt.xlabel("Relative Importance")
        plt.savefig(os.path.join(self.experiment_name, "feature_importance.png"))
        plt.close()
        print(
            f"Feature importance plot saved at: {os.path.join(self.experiment_name, 'feature_importance.png')}"
        )

    def plot_shap_summary(self, df):
        """
        Plot SHAP summary plot if supported by the model.
        """
        try:
            h2o_df = h2o.H2OFrame(df)
            self.leader.shap_summary_plot(h2o_df)
        except Exception as e:
            print("SHAP summary plot not supported for this model:", e)


def prepare_df(df: pd.DataFrame) -> pd.DataFrame:
    df = extract_temporal_features(df)
    df = add_interaction_features(df)
    df = add_custom_features(df)
    df = log_skewed_columns(df, columns=["valid_miles", "weight"])
    return df


def prepare_train_df(df: pd.DataFrame) -> pd.DataFrame:
    """
    Prepare the dataframe for training.
    """
    df = df.dropna().drop_duplicates()

    len_before = df.shape[0]
    df, rate_threshold = remove_outliers(df, "rate", percentile=99.98)
    len_after = df.shape[0]
    print(f"Rate threshold: {rate_threshold}, Outliers removed: {len_before - len_after}")

    df = prepare_df(df)

    return df


def train_and_validate():
    """
    Train the model on training data and validate it on validation data.
    """
    df = pd.read_csv("dataset/train.csv")
    df_valid = pd.read_csv("dataset/validation.csv")

    df = prepare_train_df(df)

    df_valid = prepare_df(df_valid)

    model = Model(experiment_name="experiment_train_validate")
    model.fit(df, "rate", df_valid)
    predicted_rates = model.predict(df_valid)
    mape = loss(df_valid["rate"], predicted_rates)
    mape = np.round(mape, 2)

    # Save MAPE score
    log_path = os.path.join(model.experiment_name, "validation_results.json")
    with open(log_path, "w") as f:
        json.dump({"MAPE": mape}, f, indent=4)
    print(f"Validation MAPE: {mape}% (saved in {log_path})")

    return mape


def generate_final_solution():
    """
    Train the model on combined train and validation data and generate predictions for the test set.
    """
    df_train = pd.read_csv("dataset/train.csv")

    df_train = prepare_train_df(df_train)

    df_valid = pd.read_csv("dataset/validation.csv")
    df_full = pd.concat([df_train, df_valid]).reset_index(drop=True)

    df_full = prepare_df(df_full)
    model = Model(experiment_name="experiment_final")
    model.fit(df_full, "rate", df_valid)

    df_test = pd.read_csv("dataset/test.csv")
    df_test = prepare_df(df_test)
    df_test["predicted_rate"] = model.predict(df_test)
    output_path = os.path.join(model.experiment_name, "predicted.csv")
    df_test.to_csv(output_path, index=False)
    print(f"Predictions saved at: {output_path}")

    # Plot SHAP if possible
    model.plot_shap_summary(df_full)


if __name__ == "__main__":
    mape = train_and_validate()
    print(f"Accuracy of validation is {mape}%")

    if mape < 9:
        generate_final_solution()
        print("'predicted.csv' is generated, please send it to us")
    else:
        print("MAPE >= 9%. Consider tuning the model further.")
