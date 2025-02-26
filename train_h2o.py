import json
import os
import pickle
from pathlib import Path

import h2o
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from category_encoders import CatBoostEncoder, OneHotEncoder
from h2o.automl import H2OAutoML
from sklearn.preprocessing import KBinsDiscretizer


def loss(real_rates, predicted_rates):
    """
    Calculate Mean Absolute Percentage Error (MAPE).
    """
    return np.mean(np.abs((real_rates - predicted_rates) / real_rates)) * 100


def remove_outliers(df, column, percentile=99.98):
    """
    Removes rows where values exceed the specified percentile threshold.
    """

    len_before = df.shape[0]
    threshold = np.percentile(df[column], percentile)
    df = df[df[column] <= threshold]
    len_after = df.shape[0]
    print(f"Rate threshold: {threshold}, Outliers removed: {len_before - len_after}")

    return df


def add_custom_features(df):
    """
    Adds custom features like is_kma_equal and is_rate_outlier.
    """
    df["is_kma_equal"] = df["destination_kma"] == df["origin_kma"]

    # Рассчитываем агрегированные метрики по парам (origin_kma, destination_kma)
    agg_features = (
        df.groupby(["origin_kma", "destination_kma"])
        .agg(
            valid_miles_min=("valid_miles", "min"),
            valid_miles_mean=("valid_miles", "mean"),
            valid_miles_median=("valid_miles", "median"),
            valid_miles_max=("valid_miles", "max"),
            rate_min=("rate", "min"),
            # rate_mean=("rate", "mean"),
            # rate_median=("rate", "median"),
            # rate_max=("rate", "max"),
            # log_rate_min=("log_rate", "min"),
            # log_rate_mean=("log_rate", "mean"),
            # log_rate_median=("log_rate", "median"),
            # log_rate_max=("log_rate", "max"),
        )
        .reset_index()
    )

    # Добавляем эти агрегированные признаки обратно в исходный датафрейм
    df = df.merge(agg_features, on=["origin_kma", "destination_kma"], how="left")

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
    df["year"] = df[datetime_col].dt.year

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


def bin_features(df, columns, n_bins=8, kbins=None):
    bin_columns = [f"bin_{col}" for col in columns]

    if kbins is None:
        kbins = KBinsDiscretizer(
            n_bins=n_bins, encode="ordinal", strategy="quantile", random_state=42
        )
        transformed = kbins.fit_transform(df[columns])
    else:
        transformed = kbins.transform(df[columns])

    # Преобразуем результат обратно в DataFrame
    df[bin_columns] = pd.DataFrame(transformed, columns=bin_columns, index=df.index)

    return df, kbins


def prepare_categorical_features(
    df, target_feature: str, high_cardinality, low_cardinality, train_encoders=None
):
    if train_encoders is None:
        train_encoders = {
            "high": CatBoostEncoder(),
            "low": OneHotEncoder(handle_unknown="ignore"),
        }
        df_high_cardinality = (
            train_encoders["high"]
            .fit_transform(df[high_cardinality], df[target_feature])
            .add_prefix("encoded_")
        )
        df_low_cardinality = (
            train_encoders["low"].fit_transform(df[low_cardinality]).add_prefix("encoded_")
        )
    else:
        df_high_cardinality = (
            train_encoders["high"]
            .transform(df[high_cardinality])
            .fillna(train_encoders["high"]._mean)
            .add_prefix("encoded_")
        )
        df_low_cardinality = (
            train_encoders["low"].transform(df[low_cardinality]).add_prefix("encoded_")
        )
    # add high and low cardinality features to the dataframe without dropping the original columns
    df = df.join(df_high_cardinality).join(df_low_cardinality)

    return df, train_encoders


def save_encoders(encoders, kbins, path="encoders.pkl"):
    with open(path, "wb") as f:
        pickle.dump({"encoders": encoders, "kbins": kbins}, f)
    print(f"Encoders saved at: {path}")


def load_encoders(path="encoders.pkl"):
    with open(path, "rb") as f:
        data = pickle.load(f)
    print(f"Encoders loaded from: {path}")
    return data["encoders"], data["kbins"]


class Model:
    def __init__(self, experiment_name="experiment"):
        h2o_port = int(os.getenv("H2O_PORT", 54321))

        self.experiment_name = experiment_name
        self.experiment_dir = Path("experiments") / experiment_name
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

    def fit(
        self,
        features: list[str],
        target: str,
        df: pd.DataFrame,
        validation_df: pd.DataFrame,
    ):
        """
        Train H2O AutoML on the provided DataFrame.
        """
        # Convert DataFrame to H2OFrame
        h2o_df = h2o.H2OFrame(df[features + [target]])
        h2o_valid_df = h2o.H2OFrame(validation_df[features + [target]])

        self.save_json(features, "used_features.json")

        # Run H2O AutoML
        self.aml = H2OAutoML(
            project_name=self.experiment_name,
            max_models=1,
            seed=42,
            sort_metric="MAE",
            stopping_metric="MAE",
            exclude_algos=[
                "DeepLearning",
                # "StackedEnsemble"
            ],
        )

        self.aml.train(x=features, y=target, training_frame=h2o_df, leaderboard_frame=h2o_valid_df)

        # Select the best model
        self.leader = self.aml.leader

        # Save model and experiment details
        self.save_model()
        self.plot_shap_summary(df[features], "train")
        self.plot_shap_summary(validation_df, "valid")
        self.save_feature_importance()
        self.save_model_params()
        self.explain_model(h2o_df, "train")
        self.explain_model(h2o_valid_df, "valid")

        try:
            learning_curve_plot = self.leader.learning_curve_plot()
            learning_curve_plot_path = os.path.join(self.experiment_dir, "learning_curve_plot.png")
            learning_curve_plot.savefig(learning_curve_plot_path)
            print(f"Learning curve plot saved at: {learning_curve_plot_path}")
        except Exception as e:
            print("Learning curve plot not supported:", e)

        try:
            pf = self.aml.pareto_front()
            pf_path = os.path.join(self.experiment_dir, "pareto_front.png")
            pf.savefig(pf_path)
            print(f"Pareto front saved at: {pf_path}")
        except Exception as e:
            print("Pareto front not supported:", e)

    def explain_model(self, h2o_valid_df, name):
        """
        Uses H2O explainability methods to visualize feature importance and SHAP values.
        """
        try:
            explanation = self.leader.explain(h2o_valid_df)  # 🔧 Вызов explain() вместо SHAP
            explanation_path = os.path.join(self.experiment_dir, f"model_explanation_{name}.json")
            with open(explanation_path, "w") as f:
                json.dump(str(explanation), f)
            print(f"Model explanation saved at: {explanation_path}")
        except Exception as e:
            print("Model explainability not supported:", e)

    def analyze_feature_importance(self, df):
        """Computes SHAP and feature importance."""
        try:
            h2o_df = h2o.H2OFrame(df)
            # explainer = shap.Explainer(self.leader.predict, h2o_df)
            explainer = self.leader.varimp(use_pandas=True)
            shap_values = explainer(h2o_df)

            importance_df = pd.DataFrame(
                {"feature": df.columns, "shap_importance": np.abs(shap_values.values).mean(axis=0)}
            ).sort_values("shap_importance", ascending=False)

            importance_df.to_csv(self.experiment_dir / "shap_feature_importance.csv", index=False)
            return importance_df["feature"].tolist()
        except Exception as e:
            print("SHAP not supported:", e)
            return self.leader.varimp(use_pandas=True)["variable"].tolist()

    def predict(self, df):
        """
        Make predictions using the trained H2O model.
        """
        h2o_df = h2o.H2OFrame(df)
        preds = self.leader.predict(h2o_df).as_data_frame()

        if "log" in self.leader.params["response_column"]["actual"]["column_name"]:
            return np.expm1(preds["predict"])
        else:
            return preds["predict"]

    def save_model(self):
        """
        Save the best model.
        """
        model_path = h2o.save_model(model=self.leader, path=str(self.experiment_dir), force=True)
        print(f"Model saved at: {model_path}")

    def save_model_params(self):
        """
        Save the hyperparameters of the best model.
        """
        params = self.leader.params
        params_path = os.path.join(self.experiment_dir, "best_model_params.json")
        with open(params_path, "w") as f:
            json.dump(params, f, indent=4)
        print(f"Hyperparameters saved at: {params_path}")

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

        try:
            varimp_heatmap = self.leader.varimp_heatmap()
            varimp_heatmap_path = os.path.join(
                self.experiment_dir, "feature_importance_heatmap.png"
            )
            varimp_heatmap.savefig(varimp_heatmap_path)
            print(f"Feature importance heatmap saved at: {varimp_heatmap_path}")
        except Exception as e:
            print("Feature importance heatmap not supported:", e)

        try:
            model_correlation_heatmap = self.leader.model_correlation_heatmap()
            model_correlation_heatmap_path = os.path.join(
                self.experiment_dir, "model_correlation_heatmap.png"
            )
            model_correlation_heatmap.savefig(model_correlation_heatmap_path)
            print(f"Model correlation heatmap saved at: {model_correlation_heatmap_path}")
        except Exception as e:
            print("Model correlation heatmap not supported:", e)

    def plot_shap_summary(self, df, name: str):
        """
        Plot SHAP summary plot if supported by the model.
        """
        try:
            h2o_df = h2o.H2OFrame(df)
            shap_plot = self.leader.shap_summary_plot(h2o_df)
            shap_plot_path = os.path.join(self.experiment_dir, f"shap_summary_plot_{name}.png")
            shap_plot.savefig(shap_plot_path)
            print(f"SHAP summary plot saved at: {shap_plot_path}")
        except Exception as e:
            print("SHAP summary plot not supported for this model:", e)
        try:
            shap_explain_row_plot = self.leader.shap_explain_row_plot(h2o_df)
            shap_explain_row_plot_path = os.path.join(
                self.experiment_dir, "shap_explain_row_plot.png"
            )
            shap_explain_row_plot.savefig(shap_explain_row_plot_path)
            print(f"SHAP explain row plot saved at: {shap_explain_row_plot_path}")

        except Exception as e:
            print("SHAP explain row plot not supported for this model:", e)

    def save_json(self, data, filename):
        """
        Utility to save data as JSON.
        """
        path = self.experiment_dir / filename
        with open(path, "w") as f:
            json.dump(data, f, indent=4)
        print(f"Saved {filename} at {path}")


def prepare_df(df: pd.DataFrame, target_feature: str, train_encoders=None, kbins=None):
    df = extract_temporal_features(df)
    df = add_interaction_features(df)
    df = log_skewed_columns(df, ["valid_miles", "weight", "rate"])
    df = add_custom_features(df)
    df, kbins = bin_features(df, ["valid_miles", "weight"], kbins=kbins)
    df, encoders = prepare_categorical_features(
        df,
        target_feature,
        ["origin_kma", "destination_kma"],
        ["transport_type", "season"],
        train_encoders,
    )
    return df, encoders, kbins


def prepare_train_df(df: pd.DataFrame, target_feature: str) -> pd.DataFrame:
    """
    Prepare the dataframe for training.
    """
    df = df.dropna().drop_duplicates()
    df = remove_outliers(df, "rate", percentile=99.98)
    df, encoders, kbins = prepare_df(df, target_feature)

    save_encoders(encoders, kbins, "encoders.pkl")

    return df


"""
Full set of features:
rate
valid_miles
transport_type
weight
pickup_date
origin_kma
destination_kma
month
day_of_week
hour
year
month_sin
month_cos
day_of_week_sin
day_of_week_cos
hour_sin
hour_cos
season
season_num
miles_weight_interaction
kma_interaction
log_valid_miles
log_weight
log_rate
is_kma_equal
valid_miles_min
valid_miles_mean
valid_miles_median
valid_miles_max
bin_valid_miles
bin_weight
encoded_origin_kma
encoded_destination_kma
encoded_transport_type_1
encoded_transport_type_2
encoded_transport_type_3
encoded_season_1
encoded_season_2
encoded_season_3
encoded_season_4

"""


def train_and_validate():
    """
    Train the model on training data and validate it on validation data.
    """
    target_feature = "log_rate"
    df = pd.read_csv("dataset/train.csv")
    df_valid = pd.read_csv("dataset/validation.csv")

    df = prepare_train_df(df, target_feature)
    print(f"Columns:\n{'\n'.join(df.columns)}\n\n")

    encoders, kbins = load_encoders("encoders.pkl")
    df_valid, _, _ = prepare_df(df_valid, target_feature, encoders, kbins)
    print(f"Validation columns:\n{'\n'.join(df_valid.columns)}\n\n")

    print(f"Columns diff:\n{set(df.columns) - set(df_valid.columns)}\n\n")

    feature_sets = {
        "all": df.drop(["rate", "log_rate"], axis=1).columns.tolist(),
        # "basic_log": {"log_valid_miles", "log_weight"},
        # "basic_log_transport": {"log_valid_miles", "log_weight", "transport_type"},
        # "basic_log_equal_kma": {"log_valid_miles", "log_weight", "is_kma_equal"},
        # "basic_log_transport_equal_kma": {
        #     "log_valid_miles",
        #     "log_weight",
        #     "transport_type",
        #     "is_kma_equal",
        # },
        # "basic_log_transport_equal_kma_temporal": {
        #     "log_valid_miles",
        #     "log_weight",
        #     "transport_type",
        #     "is_kma_equal",
        #     "season",
        #     "month",
        #     "day_of_week",
        #     "hour",
        # },
        # "basic_log_transport_kma": {
        #     "log_valid_miles",
        #     "log_weight",
        #     "transport_type",
        #     "origin_kma",
        #     "destination_kma",
        # },
        "favourite": {
            "log_valid_miles",
            "log_weight",
            "bin_valid_miles",
            "bin_weight",
            "hour",
            "month",
            "day_of_week",
            "encoded_season_1",
            "encoded_season_2",
            "encoded_season_3",
            "encoded_season_4",
            "valid_miles_min",
            "valid_miles_mean",
            "valid_miles_median",
            "valid_miles_max",
            "encoded_transport_type_1",
            "encoded_transport_type_2",
            "encoded_transport_type_3",
        },
        "basic_log_transport_kma_temporal": {
            "log_valid_miles",
            "log_weight",
            "transport_type",
            "origin_kma",
            "destination_kma",
            "season_num",
            "month",
            "day_of_week",
            "hour",
        },
        "everything_log": {
            "log_valid_miles",
            "log_weight",
            "transport_type",
            "month",
            "day_of_week",
            "hour",
            "origin_kma",
            "destination_kma",
            "is_kma_equal",
            "season_num",
            "day_of_week",
            "hour",
        },
    }

    leader_board = {}

    for idx, (name, features) in enumerate(feature_sets.items()):
        experiment_name = f"experiment_train_validate_{idx + 1}_{name}_{target_feature}"
        print(
            f"\nRunning experiment: {experiment_name} with target {target_feature} and features: {features}"
        )

        model = Model(experiment_name=experiment_name)
        model.fit(features=features, target=target_feature, df=df, validation_df=df_valid)

        important_features = model.analyze_feature_importance(df)
        print(f"Important features: {important_features}")

        predicted_rates = model.predict(df_valid)
        mape = loss(df_valid["rate"], predicted_rates)
        mape = np.round(mape, 2)

        leader_board[experiment_name] = mape

        model.save_json({"MAPE": mape}, "validation_results.json")
        print(f"Validation MAPE: {mape}%")

    print("\nLeaderboard:")
    for name, mape in leader_board.items():
        print(f"{name}: {mape}%")

    print(
        f"\nBest model: {min(leader_board, key=leader_board.get)} with MAPE: {min(leader_board.values())}%"
    )

    return min(leader_board.values())


def generate_final_solution():
    """
    Train the model on combined train and validation data and generate predictions for the test set.
    """
    target_feature = "log_rate"
    df_train = pd.read_csv("dataset/train.csv")
    df_train = prepare_train_df(df_train, target_feature)

    encoders, kbins = load_encoders("encoders.pkl")
    df_valid = pd.read_csv("dataset/validation.csv")
    df_valid, _, _ = prepare_df(df_valid, target_feature, encoders, kbins)

    df_full = pd.concat([df_train, df_valid]).reset_index(drop=True)
    df_full, _, _ = prepare_df(df_full, target_feature, encoders, kbins)

    model = Model(experiment_name="experiment_final")
    model.fit(df_full, "rate", df_valid)

    df_test = pd.read_csv("dataset/test.csv")
    df_test, _, _ = prepare_df(df_test, target_feature, encoders, kbins)
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
