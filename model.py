import numpy as np
from lightgbm import LGBMRegressor
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, FunctionTransformer
from xgboost import XGBRegressor
from category_encoders import TargetEncoder
from tqdm import tqdm


class Model:
    def __init__(self, model_type="random_forest"):
        self.model_type = model_type
        self.pipeline = None
        self.best_model = None

    def build_pipeline(self, numerical_features, categorical_features):
        """
        Build a pipeline with preprocessing, target encoding, and the model.
        """

        # Numerical preprocessing pipeline: Imputation, Scaling, Log Transform
        numerical_transformer = Pipeline(
            steps=[
                ("imputer", SimpleImputer(strategy="median")),
                ("log_transform", FunctionTransformer(np.log1p, validate=True)),
                ("scaler", StandardScaler()),
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

        # Select model
        if self.model_type == "random_forest":
            model = RandomForestRegressor(random_state=42)
        elif self.model_type == "xgboost":
            model = XGBRegressor(random_state=42, eval_metric="mae", n_jobs=-1)
        elif self.model_type == "lightgbm":
            model = LGBMRegressor(random_state=42, n_jobs=-1)
        else:
            raise ValueError(
                "Unsupported model_type. Choose 'random_forest', 'xgboost', or 'lightgbm'."
            )

        print(f"Building pipeline for {self.model_type} model...")
        # Build pipeline
        self.pipeline = Pipeline(steps=[("preprocessor", preprocessor), ("regressor", model)])

    def fit(self, X, y, param_grid=None):
        """
        Train the model with optional hyperparameter tuning and tqdm progress.
        """
        if param_grid:
            total_combinations = 1
            for param in param_grid.values():
                total_combinations *= len(param)

            with tqdm(total=total_combinations, desc=f"Grid Search ({self.model_type})") as pbar:

                class TQDMGridSearchCV(GridSearchCV):
                    def _run_search(self_inner, evaluate_candidates):
                        def wrapped(candidate_params):
                            results = evaluate_candidates(candidate_params)
                            pbar.update(len(candidate_params))
                            return results

                        super(TQDMGridSearchCV, self_inner)._run_search(wrapped)

                grid_search = TQDMGridSearchCV(
                    self.pipeline,
                    param_grid,
                    cv=5,
                    scoring="neg_mean_absolute_percentage_error",
                    n_jobs=-1,
                )
                grid_search.fit(X, y)
                self.best_model = grid_search.best_estimator_
                print(f"Best Params for {self.model_type}: {grid_search.best_params_}")
        else:
            self.pipeline.fit(X, y)
            self.best_model = self.pipeline

    def predict(self, X):
        return self.best_model.predict(X)
