import joblib
import numpy as np
import optuna
import pandas as pd
from category_encoders import CatBoostEncoder
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.compose import ColumnTransformer
from sklearn.metrics import mean_absolute_percentage_error
from sklearn.model_selection import KFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer, OneHotEncoder, KBinsDiscretizer
from xgboost import XGBRegressor


class TemporalFeaturesExtractor(BaseEstimator, TransformerMixin):
    """
    Custom transformer to extract temporal features from 'pickup_date'.
    """

    def __init__(self, datetime_col="pickup_date"):
        self.datetime_col = datetime_col

    def fit(self, X, y=None):
        return self

    def transform(self, X_original):
        X = X_original.copy()
        X[self.datetime_col] = pd.to_datetime(X[self.datetime_col])

        # Extract temporal features
        X["month"] = X[self.datetime_col].dt.month
        X["day_of_week"] = X[self.datetime_col].dt.dayofweek
        X["hour"] = X[self.datetime_col].dt.hour

        # Apply cyclic encoding for temporal features
        X["month_sin"] = np.sin(2 * np.pi * X["month"] / 12)
        X["month_cos"] = np.cos(2 * np.pi * X["month"] / 12)

        X["day_of_week_sin"] = np.sin(2 * np.pi * X["day_of_week"] / 7)
        X["day_of_week_cos"] = np.cos(2 * np.pi * X["day_of_week"] / 7)

        X["hour_sin"] = np.sin(2 * np.pi * X["hour"] / 24)
        X["hour_cos"] = np.cos(2 * np.pi * X["hour"] / 24)

        X["season"] = X["month"].map(
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
        X["season_num"] = X["season"].map(season_mapping)

        return X.drop(columns=[self.datetime_col])


class Model:
    def __init__(self):
        self.pipeline = None
        self.best_model = None
        self.best_params = None

    def build_pipeline(
        self,
        numerical_features: list,
        temporal_features: list,
        high_cardinality_categorical_features: list,
        low_cardinality_categorical_features: list,
    ):
        """
        Build a pipeline with preprocessing, target encoding, and the model.
        """
        # Numerical preprocessing pipeline: Imputation, Scaling, Log Transform
        numerical_transformer = Pipeline(
            steps=[
                ("log_transform", FunctionTransformer(np.log1p, validate=True)),
                ("binning", KBinsDiscretizer(n_bins=8, encode="ordinal", strategy="quantile")),
            ]
        )

        # Categorical preprocessing pipeline
        catboost_transformer = Pipeline(steps=[("catboost_encoder", CatBoostEncoder())])

        one_hot_transformer = Pipeline(
            steps=[("one_hot_encoder", OneHotEncoder(handle_unknown="ignore"))]
        )

        # Combine numerical and categorical transformers
        preprocessor = ColumnTransformer(
            transformers=[
                ("numerical", numerical_transformer, numerical_features),
                ("temporal", "passthrough", temporal_features),
                ("catboost", catboost_transformer, high_cardinality_categorical_features),
                ("onehot", one_hot_transformer, low_cardinality_categorical_features),
            ]
        )

        model = XGBRegressor(
            random_state=42,
            eval_metric="mape",
            n_jobs=-1,
            tree_method="hist",
        )

        self.pipeline = Pipeline(
            steps=[
                ("temporal_features", TemporalFeaturesExtractor(datetime_col="pickup_date")),
                ("preprocessor", preprocessor),
                ("regressor", model),
            ]
        )

    def optuna_objective(self, trial, X, y):
        """
        Optuna objective function using raw MAPE as the optimization target.
        """
        params = {
            "n_estimators": trial.suggest_int("n_estimators", 100, 500),
            "max_depth": trial.suggest_int("max_depth", 4, 15),
            "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
            "subsample": trial.suggest_float("subsample", 0.5, 1.0),
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.5, 1.0),
            "gamma": trial.suggest_float("gamma", 0, 10),
            "reg_alpha": trial.suggest_float("reg_alpha", 0, 5),
            "reg_lambda": trial.suggest_float("reg_lambda", 0, 5),
            "min_child_weight": trial.suggest_int("min_child_weight", 1, 20),
        }

        prefixed_params = {f"regressor__{key}": value for key, value in params.items()}
        self.pipeline.set_params(**prefixed_params)

        # rate_buckets = pd.qcut(y, q=6, labels=False, duplicates='drop')
        # stratified_kf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
        kf = KFold(n_splits=5, shuffle=True, random_state=42)

        # Cross-validation loop with MAPE
        mape_scores = []

        for train_index, valid_index in kf.split(X):
            # for train_index, valid_index in stratified_kf.split(X, rate_buckets):
            X_train, X_valid = X.iloc[train_index], X.iloc[valid_index]
            y_train, y_valid = y.iloc[train_index], y.iloc[valid_index]

            self.pipeline.fit(
                X_train,
                y_train,
                # regressor__eval_set=[(X_valid, y_valid)],
                # regressor__early_stopping_rounds=50,
                # regressor__verbose=False,
            )
            preds = self.pipeline.predict(X_valid)
            mape = mean_absolute_percentage_error(y_valid, preds) * 100  # MAPE in percentage
            mape_scores.append(mape)

        avg_mape = np.mean(mape_scores)
        return avg_mape  # Optuna will minimize this value

    def fit(self, X, y, n_trials=50):
        """
        Train the model with optional hyperparameter tuning and tqdm progress.
        """
        study = optuna.create_study(direction="minimize")
        study.optimize(lambda trial: self.optuna_objective(trial, X, y), n_trials=n_trials)

        self.best_params = study.best_params
        print(f"Best Params : {self.best_params}")

        prefixed_params = {f"regressor__{key}": value for key, value in self.best_params.items()}
        self.pipeline.set_params(**prefixed_params)
        self.pipeline.fit(X, y)
        self.best_model = self.pipeline

        self.save_model()

    def predict(self, X):
        return self.best_model.predict(X)

    def save_model(self, model_path="best_model.joblib", params_path="best_params.joblib"):
        """
        Save the trained model and best parameters.
        """
        joblib.dump(self.best_model, model_path)
        joblib.dump(self.best_params, params_path)
        print("Model and parameters saved.")

    def load_model(self, model_path="best_model.joblib", params_path="best_params.joblib"):
        """
        Load the trained model and parameters.
        """
        self.best_model = joblib.load(model_path)
        self.best_params = joblib.load(params_path)
        print("Model and parameters loaded.")
