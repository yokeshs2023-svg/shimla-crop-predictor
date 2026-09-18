import os
import sys
from pathlib import Path

import joblib
import pandas as pd

# Make the project root available so "src" can be imported reliably by pytest.
PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))


# ============================================================
# 1. CROP DATASET
# ============================================================

def test_crop_dataset_loads():
    file_path = PROJECT_ROOT / "data" / "raw" / "crop_recommendation.csv"
    df = pd.read_csv(file_path)

    assert not df.empty
    assert len(df.columns) >= 7


def test_crop_dataset_required_columns():
    file_path = PROJECT_ROOT / "data" / "raw" / "crop_recommendation.csv"
    df = pd.read_csv(file_path)

    required_columns = [
        "N", "P", "K", "temperature",
        "humidity", "ph", "rainfall", "label"
    ]

    for column in required_columns:
        assert column in df.columns


# ============================================================
# 2. WEATHER DATASET
# ============================================================

def test_weather_dataset_loads():
    file_path = PROJECT_ROOT / "data" / "raw" / "shimla_weather.csv"
    df = pd.read_csv(file_path)

    assert not df.empty


def test_weather_dataset_required_columns():
    file_path = PROJECT_ROOT / "data" / "raw" / "shimla_weather.csv"
    df = pd.read_csv(file_path)

    required_columns = [
        "date",
        "temperature",
        "rainfall",
        "humidity",
        "wind_speed"
    ]

    for column in required_columns:
        assert column in df.columns


# ============================================================
# 3. CLEANED WEATHER DATASET
# ============================================================

def test_clean_weather_dataset():
    file_path = (
        PROJECT_ROOT
        / "data"
        / "processed"
        / "shimla_weather_clean.csv"
    )
    df = pd.read_csv(file_path)

    assert not df.empty
    assert df.isnull().sum().sum() == 0
    assert df.duplicated().sum() == 0


def test_clean_weather_date_column():
    file_path = (
        PROJECT_ROOT
        / "data"
        / "processed"
        / "shimla_weather_clean.csv"
    )
    df = pd.read_csv(file_path)

    dates = pd.to_datetime(df["date"], errors="coerce")

    assert dates.notna().all()


# ============================================================
# 4. MODELS
# ============================================================

def test_crop_model_loads():
    model_path = (
        PROJECT_ROOT
        / "models"
        / "crop_recommendation_model.pkl"
    )

    assert model_path.exists()

    model = joblib.load(model_path)

    assert model is not None


def test_weather_models_load():
    temperature_model = (
        PROJECT_ROOT
        / "models"
        / "temperature_model.pkl"
    )
    rainfall_model = (
        PROJECT_ROOT
        / "models"
        / "rainfall_model.pkl"
    )

    assert temperature_model.exists()
    assert rainfall_model.exists()

    assert joblib.load(temperature_model) is not None
    assert joblib.load(rainfall_model) is not None


# ============================================================
# 5. CROP RECOMMENDATION
# ============================================================

def test_crop_recommendation_function():
    from src.crop_recommendation import recommend_crop

    crop = recommend_crop(
        90,
        40,
        40,
        20,
        75,
        6.5,
        150
    )

    assert isinstance(crop, str)
    assert len(crop) > 0


# ============================================================
# 6. FROST RISK
# ============================================================

def test_frost_risk_high():
    from src.frost_risk import calculate_frost_risk

    assert calculate_frost_risk(-2) == "HIGH"


def test_frost_risk_low():
    from src.frost_risk import calculate_frost_risk

    assert calculate_frost_risk(5) == "LOW"


def test_frost_advisory_high():
    from src.frost_risk import get_frost_advisory

    advisory = get_frost_advisory("HIGH")

    assert "Frost risk is high" in advisory


def test_frost_advisory_low():
    from src.frost_risk import get_frost_advisory

    advisory = get_frost_advisory("LOW")

    assert "Frost risk is low" in advisory


# ============================================================
# 7. SEASON AND FROST FEATURES
# ============================================================

def test_season_feature():
    file_path = (
        PROJECT_ROOT
        / "data"
        / "processed"
        / "shimla_weather_clean.csv"
    )
    df = pd.read_csv(file_path)

    assert "season" in df.columns

    valid_seasons = {
        "Winter",
        "Spring",
        "Monsoon",
        "Autumn"
    }

    assert set(df["season"].dropna()).issubset(valid_seasons)


def test_prepare_frost_function():
    file_path = (
        PROJECT_ROOT
        / "data"
        / "processed"
        / "shimla_weather_clean.csv"
    )
    df = pd.read_csv(file_path)

    assert "frost_risk" in df.columns
    assert set(df["frost_risk"].dropna().unique()).issubset({0, 1})

    cold_rows = df[df["temperature"] <= 0]

    if not cold_rows.empty:
        assert (cold_rows["frost_risk"] == 1).all()


# ============================================================
# 8. FASTAPI
# ============================================================

def test_api_home():
    from fastapi.testclient import TestClient
    from src.api import app

    client = TestClient(app)

    response = client.get("/")

    assert response.status_code == 200
    assert "message" in response.json()


def test_api_crop_prediction():
    from fastapi.testclient import TestClient
    from src.api import app

    client = TestClient(app)

    payload = {
        "N": 90,
        "P": 40,
        "K": 40,
        "temperature": 20,
        "humidity": 75,
        "ph": 6.5,
        "rainfall": 150
    }

    response = client.post("/predict/crop", json=payload)

    assert response.status_code == 200
    assert "recommended_crop" in response.json()
    assert isinstance(response.json()["recommended_crop"], str)


def test_api_weather_prediction():
    from fastapi.testclient import TestClient
    from src.api import app

    client = TestClient(app)

    payload = {
        "temperature": 5,
        "rainfall": 2,
        "humidity": 75,
        "wind_speed": 5,
        "month": 1,
        "frost_risk": 0
    }

    response = client.post("/predict/weather", json=payload)

    assert response.status_code == 200

    result = response.json()

    assert "predicted_temperature" in result
    assert "predicted_rainfall" in result
    assert result["predicted_rainfall"] >= 0


# ============================================================
# 9. WEATHER COLLECTION CONFIGURATION
# ============================================================

def test_weather_collection():
    file_path = PROJECT_ROOT / "src" / "collect_weather.py"

    source = file_path.read_text(encoding="utf-8")

    assert "archive-api.open-meteo.com" in source
    assert "shimla_weather.csv" in source
    assert "temperature_2m" in source
    assert "precipitation" in source
