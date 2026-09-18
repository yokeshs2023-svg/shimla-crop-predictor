import pandas as pd
import numpy as np
import joblib

from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    mean_absolute_error,
    mean_squared_error,
    r2_score
)


# ==================================================
# CROP MODEL EVALUATION
# ==================================================

print("==========================================")
print("       CROP MODEL EVALUATION")
print("==========================================")

crop_df = pd.read_csv(
    "data/processed/crop_recommendation_clean.csv"
)

crop_features = [
    "N",
    "P",
    "K",
    "temperature",
    "humidity",
    "ph",
    "rainfall"
]

X_crop = crop_df[crop_features]
y_crop = crop_df["label"]


# Use the same split as the training program
X_train, X_test, y_train, y_test = train_test_split(
    X_crop,
    y_crop,
    test_size=0.20,
    random_state=42,
    stratify=y_crop
)


# Load trained crop model
crop_model = joblib.load(
    "models/crop_recommendation_model.pkl"
)

y_crop_pred = crop_model.predict(X_test)


# Classification metrics
crop_accuracy = accuracy_score(
    y_test,
    y_crop_pred
)

crop_precision = precision_score(
    y_test,
    y_crop_pred,
    average="weighted",
    zero_division=0
)

crop_recall = recall_score(
    y_test,
    y_crop_pred,
    average="weighted",
    zero_division=0
)

crop_f1 = f1_score(
    y_test,
    y_crop_pred,
    average="weighted",
    zero_division=0
)

crop_cm = confusion_matrix(
    y_test,
    y_crop_pred
)


print("\nCrop Classification Metrics")

print("Accuracy :", crop_accuracy)
print("Precision:", crop_precision)
print("Recall   :", crop_recall)
print("F1-score :", crop_f1)

print("\nConfusion Matrix:")
print(crop_cm)


# ==================================================
# WEATHER MODEL EVALUATION
# ==================================================

print("\n==========================================")
print("       WEATHER MODEL EVALUATION")
print("==========================================")


weather_df = pd.read_csv(
    "data/processed/shimla_weather_clean.csv"
)

weather_df["date"] = pd.to_datetime(
    weather_df["date"]
)

weather_df = weather_df.sort_values(
    "date"
).reset_index(drop=True)


# Create lag features
weather_df["temperature_lag_1"] = (
    weather_df["temperature"].shift(1)
)

weather_df["rainfall_lag_1"] = (
    weather_df["rainfall"].shift(1)
)

weather_df["humidity_lag_1"] = (
    weather_df["humidity"].shift(1)
)

weather_df["wind_speed_lag_1"] = (
    weather_df["wind_speed"].shift(1)
)


# Create next-day targets
weather_df["temperature_target"] = (
    weather_df["temperature"].shift(-1)
)

weather_df["rainfall_target"] = (
    weather_df["rainfall"].shift(-1)
)


# Remove missing rows
weather_df = weather_df.dropna().reset_index(drop=True)


weather_features = [
    "temperature_lag_1",
    "rainfall_lag_1",
    "humidity_lag_1",
    "wind_speed_lag_1",
    "month",
    "frost_risk"
]

X_weather = weather_df[weather_features]

y_temperature = weather_df[
    "temperature_target"
]

y_rainfall = weather_df[
    "rainfall_target"
]


# Same chronological 80/20 split
split_index = int(len(weather_df) * 0.8)

X_train = X_weather.iloc[:split_index]
X_test = X_weather.iloc[split_index:]

y_temp_test = y_temperature.iloc[split_index:]
y_rain_test = y_rainfall.iloc[split_index:]


# Load trained models
temperature_model = joblib.load(
    "models/temperature_model.pkl"
)

rainfall_model = joblib.load(
    "models/rainfall_model.pkl"
)


# Predictions
temperature_pred = temperature_model.predict(
    X_test
)

rainfall_pred = rainfall_model.predict(
    X_test
)


# ==================================================
# TEMPERATURE METRICS
# ==================================================

temperature_mae = mean_absolute_error(
    y_temp_test,
    temperature_pred
)

temperature_rmse = np.sqrt(
    mean_squared_error(
        y_temp_test,
        temperature_pred
    )
)

temperature_r2 = r2_score(
    y_temp_test,
    temperature_pred
)


print("\nTemperature Regression Metrics")

print("MAE :", temperature_mae)
print("RMSE:", temperature_rmse)
print("R²  :", temperature_r2)


# ==================================================
# RAINFALL METRICS
# ==================================================

rainfall_mae = mean_absolute_error(
    y_rain_test,
    rainfall_pred
)

rainfall_rmse = np.sqrt(
    mean_squared_error(
        y_rain_test,
        rainfall_pred
    )
)

rainfall_r2 = r2_score(
    y_rain_test,
    rainfall_pred
)


print("\nRainfall Regression Metrics")

print("MAE :", rainfall_mae)
print("RMSE:", rainfall_rmse)
print("R²  :", rainfall_r2)


# ==================================================
# COMPLETE
# ==================================================

print("\n==========================================")
print("       MODEL EVALUATION COMPLETE")
print("==========================================")