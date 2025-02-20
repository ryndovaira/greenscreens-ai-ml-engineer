import joblib
import numpy as np
from category_encoders import TargetEncoder
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import GridSearchCV
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

    def fit(self, X, y, param_grid):
        """
        Train the model with optional hyperparameter tuning and tqdm progress.
        """
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
        self.best_params = grid_search.best_params_
        print(f"Best Params : {grid_search.best_params_}")

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