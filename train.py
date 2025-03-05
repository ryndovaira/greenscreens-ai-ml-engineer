import pandas as pd
import numpy as np

from model import Model


def loss(real_rates, predicted_rates):
    return np.average(abs(predicted_rates / real_rates - 1.0)) * 100.0


def train_and_validate():
    df_train = pd.read_csv("dataset/train.csv")
    df_val = pd.read_csv("dataset/validation.csv")

    skewed_columns = ("valid_miles", "weight", "rate")

    df_train = Model.prepare_train_df(df=df_train, skewed_columns=skewed_columns)
    df_val = Model.prepare_df(df=df_val, skewed_columns=skewed_columns)

    experiment_features = {
        "main": {
            "day_of_week",
            "destination_kma",
            "hour",
            "log_valid_miles",
            "log_weight",
            "month",
            "origin_kma",
            "pickup_date",
            "transport_type",
            "year",
        },
    }
    leader_board = {}
    models = {}

    h2_o_auto_ml_params = dict(
        max_models=10,
        seed=42,
        sort_metric="MAE",
        stopping_metric="MAE",
        include_algos=["GBM"],
        # exclude_algos=[
        #     "DeepLearning",
        #     "StackedEnsemble",
        #     "DRF",
        # ],
    )
    y = "log_rate"
    for idx, (name, x) in enumerate(experiment_features.items()):
        x = sorted(list(x))
        experiment_name = f"train_validate_{idx + 1}_{name}_{y}"
        print(f"\nRunning experiment: {experiment_name} with target {y} and features: {x}")

        model = Model(experiment_name=experiment_name)

        model.fit(
            x=x,
            y=y,
            train_df=df_train,
            leaderboard_df=df_val,
            h2_o_auto_ml_params=h2_o_auto_ml_params,
        )

        predicted_rates = model.predict(df_val)
        mape = loss(df_val["rate"], predicted_rates)
        mape = np.round(mape, 2)

        leader_board[experiment_name] = mape
        models[experiment_name] = (model, x, y)

        model.save_json({"MAPE": mape}, "validation_results.json")
        print(f"Validation MAPE: {mape}%")

    print("\nLeaderboard:")
    for name, mape in leader_board.items():
        print(f"{name}: {mape}%")

    best_model_name = min(leader_board, key=leader_board.get)
    best_mape = leader_board[best_model_name]
    best_model_x_y = models[best_model_name]
    print(f"\nBest model: {best_model_name} with MAPE: {best_mape}%")

    return best_mape, best_model_x_y


def generate_final_solution(best_model, x, y):
    df_train = pd.read_csv("dataset/train.csv")
    df_val = pd.read_csv("dataset/validation.csv")
    df_test = pd.read_csv("dataset/test.csv")

    skewed_columns = ("valid_miles", "weight", "rate")

    df_train = Model.prepare_train_df(df=df_train, skewed_columns=skewed_columns)
    df_val = Model.prepare_df(df=df_val, skewed_columns=skewed_columns)
    df_test = Model.prepare_df(df=df_test, skewed_columns=("valid_miles", "weight"))

    # combine train and validation to improve final predictions
    df = pd.concat([df_train, df_val], ignore_index=True).reset_index(drop=True)

    best_model_type = best_model.leader.algo  # e.g., 'gbm', 'xgboost'
    best_model_params = {
        k: v
        for k, v in best_model.leader.actual_params.items()
        if k
        not in [
            "model_id",
            "training_frame",
            "response_column",
            "validation_frame",
            "ignored_columns",
        ]
    }

    if best_model_type == "gbm":
        # Check if the model has the same MAPE as before
        # model_train = Model(experiment_name="final_train_val")
        # model_train.fit_gbm(x=x, y=y, train_df=df_train, params=best_model_params)
        # predicted_rates = model_train.predict(df_val)
        # mape_train = np.round(loss(df_val["rate"], predicted_rates), 2)
        # model_train.save_json({"MAPE": mape_train}, "validation_results.json")
        # print(f"MAPE (Train -> Val): {mape_train}%")

        model = Model(experiment_name="final_train_val_test")
        model.fit_gbm(x=x, y=y, train_df=df, params=best_model_params)

        # Check if the model works on the validation set as expected
        # predicted_rates = model.predict(df_val)
        # mape_val = np.round(loss(df_val["rate"], predicted_rates), 2)
        # model.save_json({"MAPE": mape_val}, "validation_results.json")
        # print(f"MAPE (Train + Val -> Val): {mape_val}%")

        # Generate and save test predictions
        df_test["predicted_rate"] = model.predict(df_test)
        df_test.to_csv("dataset/predicted.csv", index=False)

    else:
        raise ValueError(f"Unknown model type: {best_model_type}")


if __name__ == "__main__":
    mape, (best_model, x, y) = train_and_validate()
    print(f"Accuracy of validation is {mape}%")

    if mape < 9:  # try to reach 9% or less for validation
        generate_final_solution(best_model, x, y)
        print("'predicted.csv' is generated, please send it to us")
