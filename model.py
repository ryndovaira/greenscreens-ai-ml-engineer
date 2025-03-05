import json
import os
from pathlib import Path

import numpy as np
import pandas as pd

import h2o
from h2o.automl import H2OAutoML
from h2o.estimators import H2OGradientBoostingEstimator
from matplotlib import pyplot as plt


class Model:
    def __init__(self, experiment_name: str = "experiment"):
        self.leader_model_path = None
        h2o_port = int(os.getenv("H2O_PORT", 54321))

        self.experiment_name = experiment_name
        self.experiment_dir = Path("results") / experiment_name
        self.experiment_dir.mkdir(parents=True, exist_ok=True)
        log_dir = Path(self.experiment_dir / os.getenv("H2O_LOG_DIR", "logs"))

        h2o.init(
            log_dir=str(log_dir),
            ip="localhost",
            port=h2o_port,
            nthreads=-1,
            bind_to_localhost=False,
        )

        self.aml = None
        self.leader = None

    @staticmethod
    def extract_temporal_features(
        df: pd.DataFrame, datetime_col: str = "pickup_date"
    ) -> pd.DataFrame:
        """
        Extracts temporal and cyclic features from the pickup_date.
        """
        df[datetime_col] = pd.to_datetime(df[datetime_col])
        df["month"] = df[datetime_col].dt.month
        df["day_of_week"] = df[datetime_col].dt.dayofweek
        df["hour"] = df[datetime_col].dt.hour
        df["year"] = df[datetime_col].dt.year

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

        return df

    @staticmethod
    def remove_outliers(df: pd.DataFrame, column: str, percentile=99.98) -> pd.DataFrame:
        """
        Removes rows where values exceed the specified percentile threshold.
        """

        len_before = df.shape[0]
        threshold = np.percentile(df[column], percentile)
        df = df[df[column] <= threshold]
        len_after = df.shape[0]
        print(f"Rate threshold: {threshold}, Outliers removed: {len_before - len_after}")

        return df

    @staticmethod
    def log_skewed_columns(df: pd.DataFrame, columns: tuple) -> pd.DataFrame:
        """
        Log transform skewed columns.
        """
        for col in columns:
            df[f"log_{col}"] = np.log1p(df[col])
        return df

    @classmethod
    def prepare_df(
        cls, df: pd.DataFrame, skewed_columns: tuple = ("valid_miles", "weight", "rate")
    ) -> pd.DataFrame:
        df = cls.extract_temporal_features(df)
        df = cls.log_skewed_columns(df, skewed_columns)

        return df

    @classmethod
    def prepare_train_df(
        cls, df: pd.DataFrame, skewed_columns, outliers_percentile: float = 99.98
    ) -> pd.DataFrame:
        df = df.dropna().drop_duplicates()
        df = cls.remove_outliers(df=df, column="rate", percentile=outliers_percentile)
        df = cls.prepare_df(df=df, skewed_columns=skewed_columns)
        return df

    def fit_gbm(
        self,
        x: list[str],
        y: str,
        train_df: pd.DataFrame,
        params,
    ):
        self.save_json(x, "used_features.json")

        training_frame = h2o.H2OFrame(train_df[x + [y]])
        self.leader = H2OGradientBoostingEstimator(**params)
        self.leader.train(x=x, y=y, training_frame=training_frame)

        self.print_model_performance()
        self.save_artifacts()

    def fit(
        self,
        x: list[str],
        y: str,
        train_df: pd.DataFrame,
        leaderboard_df: pd.DataFrame,
        h2_o_auto_ml_params: dict[str, any],
    ):
        """
        Train H2O AutoML on the provided DataFrame.
        """
        self.save_json(x, "used_features.json")

        training_frame = h2o.H2OFrame(train_df[x + [y]])
        leaderboard_frame = h2o.H2OFrame(leaderboard_df[x + [y]])

        self.aml = H2OAutoML(project_name=self.experiment_name, **h2_o_auto_ml_params)
        self.aml.train(x=x, y=y, training_frame=training_frame, leaderboard_frame=leaderboard_frame)
        self.leader = self.aml.leader

        self.print_model_performance()
        self.save_artifacts()

    def save_artifacts(self):
        self.save_h2o_model()
        self.save_mojo_model()
        self.save_model_params()
        self.save_feature_importance()

    def print_model_performance(self):
        perf = self.leader.model_performance()
        print(perf)

    def predict(self, x):
        h2o_x = h2o.H2OFrame(x)
        preds = self.leader.predict(h2o_x).as_data_frame()

        if "log" in self.leader.params["response_column"]["actual"]["column_name"]:
            return np.expm1(preds["predict"])
        else:
            return preds["predict"]

    def save_h2o_model(self):
        if self.leader is None:
            raise ValueError("Model is not trained yet. Please train the model first.")
        self.leader_model_path = h2o.save_model(
            model=self.leader, path=str(self.experiment_dir), force=True
        )
        print(f"Model saved at: {self.leader_model_path}")
        return self.leader_model_path

    def load_h2o_model(self, path: str):
        model_path = self.leader_model_path if self.leader_model_path else path
        if model_path is None:
            raise ValueError("Model is not trained yet. Please train the model first.")
        self.leader = h2o.load_model(model_path)
        print(f"Model loaded from: {model_path}")

    def save_mojo_model(self):
        if self.leader is None:
            raise ValueError("Model is not trained yet. Please train the model first.")
        self.mojo_path = self.leader.save_mojo(str(self.experiment_dir), force=True)
        print(f"MOJO model saved at: {self.mojo_path}")
        return self.mojo_path

    def load_mojo_model(self):
        if self.mojo_path is None:
            raise ValueError("Model is not trained yet. Please train the model first.")
        self.leader = h2o.import_mojo(self.mojo_path)
        print(f"MOJO model loaded from: {self.mojo_path}")

    def save_model_params(self):
        """
        Save the hyperparameters of the best model.
        """
        params = self.leader.params
        params_path = os.path.join(self.experiment_dir, "best_model_params.json")
        with open(params_path, "w") as f:
            json.dump(params, f, indent=4)
        print(f"Hyperparameters saved at: {params_path}")

    def save_json(self, data, filename):
        """
        Utility to save data as JSON.
        """
        path = self.experiment_dir / filename
        with open(path, "w") as f:
            json.dump(data, f, indent=4)
        print(f"Saved {filename} at {path}")

    def save_feature_importance(self):
        """
        Save feature importance and plot the result.
        """
        try:
            varimp = self.leader.varimp(use_pandas=True)
            varimp_path = self.experiment_dir / "feature_importance.csv"
            varimp.to_csv(varimp_path, index=False)
            print(f"Feature importance saved at: {varimp_path}")

            plt.figure(figsize=(16, 16))
            plt.barh(varimp["variable"], varimp["relative_importance"])
            plt.title("Feature Importance")
            plt.xlabel("Relative Importance")
            fig_path = os.path.join(self.experiment_dir, "feature_importance.png")
            plt.savefig(fig_path)
            plt.close()
            print(f"Feature importance plot saved at: {fig_path}")
        except Exception as e:
            print("Feature importance not supported:", e)
