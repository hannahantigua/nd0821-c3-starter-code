# api/app.py

import pickle
from typing import Dict

import pandas as pd
from fastapi import FastAPI
from pydantic import BaseModel, Field

from ml.data import process_data
from ml.model import inference

# Load trained artifacts
with open("model/model.pkl", "rb") as f:
    model = pickle.load(f)
with open("model/encoder.pkl", "rb") as f:
    encoder = pickle.load(f)
with open("model/lb.pkl", "rb") as f:
    lb = pickle.load(f)

# List of categorical features
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


# Pydantic model
class CensusData(BaseModel):
    age: int
    workclass: str
    fnlgt: int
    education: str
    education_num: int = Field(..., alias="education-num")
    marital_status: str = Field(..., alias="marital-status")
    occupation: str
    relationship: str
    race: str
    sex: str
    capital_gain: int = Field(..., alias="capital-gain")
    capital_loss: int = Field(..., alias="capital-loss")
    hours_per_week: int = Field(..., alias="hours-per-week")
    native_country: str = Field(..., alias="native-country")

    class Config:
        allow_population_by_field_name = True
        schema_extra = {
            "example": {
                "age": 37,
                "workclass": "Private",
                "fnlgt": 34146,
                "education": "HS-grad",
                "education-num": 9,
                "marital-status": "Married-civ-spouse",
                "occupation": "Handlers-cleaners",
                "relationship": "Husband",
                "race": "White",
                "sex": "Male",
                "capital-gain": 0,
                "capital-loss": 0,
                "hours-per-week": 40,
                "native-country": "United-States",
            }
        }


# FastAPI app
app = FastAPI(title="Census Income Classifier API")


@app.get("/", response_model=Dict[str, str])
def read_root() -> Dict[str, str]:
    """
    Root endpoint. Returns a welcome message.
    """
    return {"message": "Welcome to the Census Income Classifier API!"}


@app.post("/predict", response_model=Dict[str, str])
def predict_income(data: CensusData) -> Dict[str, str]:
    """
    Predict if the salary is >50K based on Census data.
    """
    # Convert Pydantic model to DataFrame using aliases (hyphen columns)
    df = pd.DataFrame([data.dict(by_alias=True)])
    df["salary"] = "<=50K"

    # Process the data
    X, _, _, _ = process_data(
        df,
        categorical_features=cat_features,
        label="salary",  # label not used but required by function
        training=False,
        encoder=encoder,
        lb=lb,
    )

    # Make prediction
    pred = inference(model, X)

    # Convert back to original label
    pred_label = lb.inverse_transform(pred)[0]

    return {"prediction": pred_label}
