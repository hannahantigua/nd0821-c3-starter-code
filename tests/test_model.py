import numpy as np
import pandas as pd

from ml.data import process_data
from ml.model import inference, train_model

# Sample data
SAMPLE_DATA = pd.DataFrame(
    {
        "age": [25, 38],
        "workclass": ["Private", "Self-emp-not-inc"],
        "fnlgt": [226802, 89814],
        "education": ["11th", "HS-grad"],
        "education-num": [7, 9],
        "marital-status": ["Never-married", "Married-civ-spouse"],
        "occupation": ["Machine-op-inspct", "Farming-fishing"],
        "relationship": ["Own-child", "Husband"],
        "race": ["Black", "White"],
        "sex": ["Male", "Male"],
        "capital-gain": [0, 0],
        "capital-loss": [0, 0],
        "hours-per-week": [40, 50],
        "native-country": ["United-States", "United-States"],
        "salary": ["<=50K", ">50K"],
    }
)

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


def test_process_data():
    X, y, encoder, lb = process_data(
        SAMPLE_DATA,
        categorical_features=cat_features,
        label="salary",
        training=True,
    )
    assert X.shape[0] == 2
    assert len(y) == 2
    assert encoder is not None
    assert lb is not None


def test_train_model():
    X, y, encoder, lb = process_data(
        SAMPLE_DATA,
        categorical_features=cat_features,
        label="salary",
        training=True,
    )
    model = train_model(X, y)
    assert model is not None
    # Check model has a predict method
    assert hasattr(model, "predict")


def test_inference():
    X, y, encoder, lb = process_data(
        SAMPLE_DATA,
        categorical_features=cat_features,
        label="salary",
        training=True,
    )
    model = train_model(X, y)
    preds = inference(model, X)
    assert preds.shape[0] == X.shape[0]
    # Check predictions are integers (0 or 1 for LogisticRegression)
    assert set(np.unique(preds)).issubset({0, 1})
