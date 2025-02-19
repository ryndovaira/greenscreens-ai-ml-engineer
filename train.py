import pandas as pd
import numpy as np
from model import Model


def loss(real_rates, predicted_rates):
    return np.average(abs(predicted_rates / real_rates - 1.0)) * 100.0


def add_interaction_features(df):
    df = df.copy()
    df['kma_interaction'] = df['origin_kma_mean_rate'] * df['destination_kma_mean_rate']
    df['miles_weight_interaction'] = df['valid_miles'] * df['weight']
    return df


def train_and_validate(model_type):
    df = pd.read_csv('dataset/train.csv')
    df = add_interaction_features(df)

    model = Model(model_type=model_type)

    # Features for the pipeline
    features = ['valid_miles', 'weight', 'origin_kma_mean_rate', 'destination_kma_mean_rate',
                'kma_interaction', 'miles_weight_interaction']

    # Build pipeline
    model.build_pipeline(features)

    # Hyperparameter grid
    param_grid = {}
    if model_type == 'random_forest':
        param_grid = {
            'regressor__n_estimators': [100, 200],
            'regressor__max_depth': [10, 20, None],
            'regressor__min_samples_split': [2, 5]
        }
    elif model_type == 'xgboost':
        param_grid = {
            'regressor__n_estimators': [100, 200],
            'regressor__max_depth': [3, 6, 10],
            'regressor__learning_rate': [0.01, 0.1],
            'regressor__subsample': [0.8, 1.0]
        }
    elif model_type == 'lightgbm':
        param_grid = {
            'regressor__n_estimators': [100, 200],
            'regressor__max_depth': [3, 6, 10],
            'regressor__learning_rate': [0.01, 0.1],
            'regressor__subsample': [0.8, 1.0]
        }

    model.fit(df[features], df['rate'], param_grid=param_grid)

    df_val = pd.read_csv('dataset/validation.csv')
    df_val = add_interaction_features(df_val)
    predicted_rates = model.predict(df_val[features])
    mape = loss(df_val['rate'], predicted_rates)
    print(f"Validation MAPE for {model_type}: {mape}%")
    return mape


def generate_final_solution(best_model_type):
    # combine train and validation to improve final predictions
    df = pd.read_csv('dataset/train.csv')
    df_val = pd.read_csv('dataset/validation.csv')
    df = pd.concat([df, df_val]).reset_index(drop=True)
    df = add_interaction_features(df)

    features = ['valid_miles', 'weight', 'origin_kma_mean_rate', 'destination_kma_mean_rate',
                'kma_interaction', 'miles_weight_interaction']

    model = Model(model_type=best_model_type)
    model.build_pipeline(features)
    model.fit(df[features], df['rate'])

    # generate and save test predictions
    df_test = pd.read_csv('dataset/test.csv')
    df_test = add_interaction_features(df_test)
    df_test['predicted_rate'] = model.predict(df_test[features])
    df_test.to_csv('dataset/predicted.csv', index=False)
    print(f"'predicted.csv' generated using {best_model_type}")


if __name__ == "__main__":
    models = ['random_forest', 'xgboost', 'lightgbm']
    results = {}

    for model_type in models:
        mape = train_and_validate(model_type)
        results[model_type] = mape

    # Select best model
    best_model = min(results, key=results.get)
    print(f"Best Model: {best_model} with MAPE: {results[best_model]}%")

    if results[best_model] < 9:  # try to reach 9% or less for validation
        generate_final_solution(best_model)
        print("Final prediction generated.")
    else:
        print("No model met the MAPE < 9% requirement.")
