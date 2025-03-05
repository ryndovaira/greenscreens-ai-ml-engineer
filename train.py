import pandas as pd
import numpy as np

from model import Model


def loss(real_rates, predicted_rates):
    return np.average(abs(predicted_rates / real_rates - 1.0)) * 100.0


def train_and_validate():
    df = pd.read_csv("dataset/train.csv")
    model = Model()
    model.fit(df, df.rate)

    df = pd.read_csv("dataset/validation.csv")
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
    model.fit(df, df.rate)

    # generate and save test predictions
    df_test = pd.read_csv("dataset/test.csv")
    df_test["predicted_rate"] = model.predict(df_test)
    df_test.to_csv("dataset/predicted.csv", index=False)


if __name__ == "__main__":
    mape = train_and_validate()
    print(f"Accuracy of validation is {mape}%")

    if mape < 9:  # try to reach 9% or less for validation
        generate_final_solution()
        print("'predicted.csv' is generated, please send it to us")
