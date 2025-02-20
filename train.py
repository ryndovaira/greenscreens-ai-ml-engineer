import numpy as np
import pandas as pd

from model import Model


def loss(real_rates, predicted_rates):
    return np.average(abs(predicted_rates / real_rates - 1.0)) * 100.0




def train_and_validate():
    df = pd.read_csv("dataset/train.csv")
    df = df.dropna().drop_duplicates()

    # Define features
    numerical_features = ["valid_miles",
                          "weight",
                          # "miles_weight_interaction"
                          ]
    categorical_features = ["origin_kma",
                            "destination_kma",
                            # "kma_interaction"
                            ]

    model = Model()
    model.build_pipeline(numerical_features, categorical_features)

    param_grid = {
        "regressor__n_estimators": [250, 300, 350],
        "regressor__max_depth": [12,],
        "regressor__learning_rate": [0.12,],
        "regressor__subsample": [1.0,],
    }
    model.fit(df, df["rate"], param_grid=param_grid)

    df = pd.read_csv('dataset/validation.csv')
    predicted_rates = model.predict(df)
    mape = loss(df.rate, predicted_rates)
    mape = np.round(mape, 2)
    return mape


def generate_final_solution():
    # combine train and validation to improve final predictions
    df = pd.read_csv("dataset/train.csv")
    df_val = pd.read_csv("dataset/validation.csv")
    df = df.append(df_val).reset_index(drop=True)

    model = Model()
    model.load_model()

    # generate and save test predictions
    df_test = pd.read_csv("dataset/test.csv")
    # df_test = add_interaction_features(df_test)
    df_test["predicted_rate"] = model.predict(df_test)
    df_test.to_csv("dataset/predicted.csv", index=False)


if __name__ == "__main__":
    mape = train_and_validate()

    print(f'Accuracy of validation is {mape}%')

    if mape < 9:  # try to reach 9% or less for validation
        generate_final_solution()
        print("'predicted.csv' is generated.")
    else:
        print("No model met the MAPE < 9% requirement.")
