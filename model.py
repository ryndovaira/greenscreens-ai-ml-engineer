import joblib
import numpy as np
import optuna
from category_encoders import TargetEncoder
from sklearn.compose import ColumnTransformer
from sklearn.metrics import mean_absolute_percentage_error
from sklearn.model_selection import KFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer
from xgboost import XGBRegressor


class Model:
    def __init__(self):
        self.pipeline = None
        self.best_model = None
        self.best_params = None

    def build_pipeline(self, numerical_features, categorical_features):
        """
        Build a pipeline with preprocessing, target encoding, and the model.
        """
        # Numerical preprocessing pipeline: Imputation, Scaling, Log Transform
        numerical_transformer = Pipeline(
            steps=[
                ("log_transform", FunctionTransformer(np.log1p, validate=True)),
            ]
        )

        # Categorical preprocessing pipeline: Target Encoding
        categorical_transformer = Pipeline(steps=[("target_encoder", TargetEncoder())])

        # Combine numerical and categorical transformers
        preprocessor = ColumnTransformer(
            transformers=[
                ("num", numerical_transformer, numerical_features),
                ("cat", categorical_transformer, categorical_features),
            ]
        )

        model = XGBRegressor(
                    random_state=42,
                    eval_metric="mape",
                    n_jobs=-1,
                    tree_method="hist",
                )

        self.pipeline = Pipeline(steps=[("preprocessor", preprocessor), ("regressor", model)])

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

        # Cross-validation loop with MAPE
        mape_scores = []
        kf = KFold(n_splits=5, shuffle=True, random_state=42)

        for train_index, valid_index in kf.split(X):
            X_train, X_valid = X.iloc[train_index], X.iloc[valid_index]
            y_train, y_valid = y.iloc[train_index], y.iloc[valid_index]

            self.pipeline.fit(X_train, y_train)
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