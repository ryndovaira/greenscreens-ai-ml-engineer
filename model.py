import numpy as np
from category_encoders import TargetEncoder
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, FunctionTransformer
from xgboost import XGBRegressor


class Model:
    def __init__(self):
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

        model = XGBRegressor(
                    random_state=42,
                    eval_metric="mae",
                    n_jobs=-1,
                    tree_method="hist",
                )

        self.pipeline = Pipeline(steps=[("preprocessor", preprocessor), ("regressor", model)])

    def fit(self, X, y, param_grid=None):
        """
        Train the model with optional hyperparameter tuning and tqdm progress.
        """
        if param_grid:
            grid_search = GridSearchCV(
                self.pipeline,
                param_grid,
                cv=5,
                scoring="neg_mean_absolute_percentage_error",
                n_jobs=-1,
                verbose=3,
            )
            grid_search.fit(X, y)
            self.best_model = grid_search.best_estimator_
            print(f"Best Params : {grid_search.best_params_}")
        else:
            self.pipeline.fit(X, y)
            self.best_model = self.pipeline

    def predict(self, X):
        return self.best_model.predict(X)
