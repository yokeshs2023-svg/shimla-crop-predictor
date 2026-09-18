import os
import pandas as pd
import joblib
import requests


# ==========================================
# TEST 1 — Crop dataset loads
# ==========================================

def test_crop_dataset_loads():
    file_path = "data/raw/crop_recommendation.csv"

    df = pd.read_csv(file_path)

    assert not df.empty
    assert len(df.columns) >= 7


# ==========================================
# TEST 2 — Weather dataset loads
# ==========================================

def test_weather_dataset_loads():
    file_path = "data/raw/shimla_weather.csv"

    df = pd.read_csv(file_path)

    assert not df.empty

    required_columns = [
        "date",
        "temperature",
        "rainfall",
        "humidity",
        "wind_speed"
    ]

    for column in required_columns:
        assert column in df.columns


# ==========================================
# TEST 3 — Clean weather dataset
# ==========================================

def test_clean_weather_dataset():
    file_path = "data/processed/shimla_weather_clean.csv"

    df = pd.read_csv(file_path)

    assert not df.empty
    assert df.isnull().sum().sum() == 0
    assert df.duplicated().sum() == 0


# ==========================================
# TEST 4 — Crop model exists and loads
# ==========================================

def test_crop_model_loads():
    model_path = "models/crop_recommendation_model.pkl"

    assert os.path.exists(model_path)

    model = joblib.load(model_path)

    assert model is not None


# ==========================================
# TEST 5 — Weather models exist
# ==========================================

def test_weather_models_load():
    temperature_model = "models/temperature_model.pkl"
    rainfall_model = "models/rainfall_model.pkl"

    assert os.path.exists(temperature_model)
    assert os.path.exists(rainfall_model)

    temp_model = joblib.load(temperature_model)
    rain_model = joblib.load(rainfall_model)

    assert temp_model is not None
    assert rain_model is not None