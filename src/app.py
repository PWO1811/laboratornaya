from fastapi import FastAPI
from pydantic import BaseModel
import joblib
import pandas as pd

app = FastAPI()

model = joblib.load('models/model.joblib')


class LaptopFeatures(BaseModel):
    Brand: str
    Processor_Speed: float
    RAM_Size: int
    Storage_Capacity: int
    Screen_Size: float
    Weight: float


@app.post("/predict")
def predict(features: LaptopFeatures):
    input_data = pd.DataFrame([features.dict()])

    prediction = model.predict(input_data)[0]

    return {"predicted_price": prediction}


@app.get("/")
def read_root():
    return {"message": "Laptop Price Prediction API"}