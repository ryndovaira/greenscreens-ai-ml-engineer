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
            max_models=1,  # Reduce number of models (was 20)
            # max_runtime_secs=60 * 5,  # 2 minutes max runtime
            seed=42,
            sort_metric="mape",  # Keep MAPE as main evaluation
            stopping_metric="MAE",  # Simple metric for stopping
            stopping_rounds=1,  # Faster early stopping
            # exclude_algos=["DeepLearning", "StackedEnsemble"],  # Exclude heavy models
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


def train_and_validate():
    """
    Train the model on training data and validate it on validation data.
    """
    df = pd.read_csv("dataset/train.csv")
    df_valid = pd.read_csv("dataset/validation.csv")

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
    df_valid = pd.read_csv("dataset/validation.csv")
    df_full = pd.concat([df_train, df_valid]).reset_index(drop=True)

    model = Model(experiment_name="experiment_final")
    model.fit(df_full, "rate")

    df_test = pd.read_csv("dataset/test.csv")
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
