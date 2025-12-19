# Script to train machine learning model.

import pickle

import pandas as pd
from sklearn.model_selection import train_test_split

from starter.ml import process_data
from starter.ml.model import compute_model_metrics_overall, train_model

cat_features = [
    "workclass",
    "education",
    "marital-status",
    "occupation",
    "relationship",
    "race",
    "sex",
    "native-country",
]

if __name__ == "__main__":
    data = pd.read_csv("starter/data/census.csv")
    # print(data.columns)  # check csv has been cleaned

    train, _ = train_test_split(data, test_size=0.20)

    X_train, y_train, encoder, lb = process_data(
        train, categorical_features=cat_features, label="salary", training=True
    )

    model = train_model(X_train, y_train)

    with open("model/model.pkl", "wb") as f:
        pickle.dump(model, f)

    with open("model/encoder.pkl", "wb") as f:
        pickle.dump(encoder, f)

    with open("model/lb.pkl", "wb") as f:
        pickle.dump(lb, f)

    # overall performance
    compute_model_metrics_overall(
        df=data,
        cat_features=cat_features,
        model=model,
        encoder=encoder,
        lb=lb,
        label="salary",
    )

    # slice based performance
    # compute_model_metrics_on_slices(
    #     df=data,
    #     cat_features=cat_features,
    #     model=model,
    #     encoder=encoder,
    #     lb=lb,
    #     label="salary"
    # )
