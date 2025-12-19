import requests

API_URL = "https://nd0821-c3-starter-code-gdwb.onrender.com/predict"

payload = {
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

response = requests.post(API_URL, json=payload)

print("Status code:", response.status_code)
print("Response JSON:", response.json())
