from fastapi import FastAPI
from pydantic import BaseModel
import joblib
import pandas as pd


# ==========================================
# CREATE FASTAPI APPLICATION
# ==========================================

app = FastAPI(
    title="Shimla Crop Predictor API",
    description="API for weather prediction and crop recommendation",
    version="1.0.0"
)


# ==========================================
# LOAD TRAINED MODELS
# ==========================================

temperature_model = joblib.load(
    "models/temperature_model.pkl"
)

rainfall_model = joblib.load(
    "models/rainfall_model.pkl"
)

crop_model = joblib.load(
    "models/crop_recommendation_model.pkl"
)


# ==========================================
# WEATHER INPUT
# ==========================================

class WeatherInput(BaseModel):
    temperature: float
    rainfall: float
    humidity: float
    wind_speed: float
    month: int
    frost_risk: int


# ==========================================
# CROP INPUT
# ==========================================

class CropInput(BaseModel):
    N: float
    P: float
    K: float
    temperature: float
    humidity: float
    ph: float
    rainfall: float


# ==========================================
# HOME ENDPOINT
# ==========================================

@app.get("/")
def home():
    return {
        "message": "Shimla Crop Predictor API is running"
    }


# ==========================================
# WEATHER PREDICTION
# ==========================================

@app.post("/predict/weather")
def predict_weather(data: WeatherInput):

    input_data = pd.DataFrame([{
        "temperature_lag_1": data.temperature,
        "rainfall_lag_1": data.rainfall,
        "humidity_lag_1": data.humidity,
        "wind_speed_lag_1": data.wind_speed,
        "month": data.month,
        "frost_risk": data.frost_risk
    }])

    predicted_temperature = temperature_model.predict(
        input_data
    )[0]

    predicted_rainfall = rainfall_model.predict(
        input_data
    )[0]

    # Rainfall cannot be negative
    predicted_rainfall = max(
        0,
        predicted_rainfall
    )

    return {
        "predicted_temperature": round(
            float(predicted_temperature), 2
        ),
        "predicted_rainfall": round(
            float(predicted_rainfall), 2
        )
    }


# ==========================================
# CROP RECOMMENDATION
# ==========================================

@app.post("/predict/crop")
def predict_crop(data: CropInput):

    input_data = pd.DataFrame([{
        "N": data.N,
        "P": data.P,
        "K": data.K,
        "temperature": data.temperature,
        "humidity": data.humidity,
        "ph": data.ph,
        "rainfall": data.rainfall
    }])

    prediction = crop_model.predict(
        input_data
    )[0]

    return {
        "recommended_crop": str(prediction)
    }