# Script to train machine learning model.

from sklearn.model_selection import train_test_split

import pandas as pd
import pickle
from sklearn.model_selection import train_test_split

from ml.data import process_data
from ml.model import train_model

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
        train,
        categorical_features=cat_features,
        label="salary",
        training=True
    )

    model = train_model(X_train, y_train)

    with open("model/model.pkl", "wb") as f:
        pickle.dump(model, f)

    with open("model/encoder.pkl", "wb") as f:
        pickle.dump(encoder, f)

    with open("model/lb.pkl", "wb") as f:
        pickle.dump(lb, f)
