import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import numpy as np
import joblib


# ==========================================
# 1. LOAD CLEAN WEATHER DATA
# ==========================================

file_path = "data/processed/shimla_weather_clean.csv"

df = pd.read_csv(file_path)

# Convert date to datetime
df["date"] = pd.to_datetime(df["date"])

# Sort data by date
df = df.sort_values("date").reset_index(drop=True)


# ==========================================
# 2. CREATE LAG FEATURES
# ==========================================

df["temperature_lag_1"] = df["temperature"].shift(1)
df["rainfall_lag_1"] = df["rainfall"].shift(1)
df["humidity_lag_1"] = df["humidity"].shift(1)
df["wind_speed_lag_1"] = df["wind_speed"].shift(1)


# ==========================================
# 3. CREATE NEXT-DAY TARGETS
# ==========================================

df["temperature_target"] = df["temperature"].shift(-1)
df["rainfall_target"] = df["rainfall"].shift(-1)


# Remove rows containing missing values
df = df.dropna().reset_index(drop=True)


# ==========================================
# 4. SELECT FEATURES
# ==========================================

features = [
    "temperature_lag_1",
    "rainfall_lag_1",
    "humidity_lag_1",
    "wind_speed_lag_1",
    "month",
    "frost_risk"
]

X = df[features]

y_temperature = df["temperature_target"]
y_rainfall = df["rainfall_target"]


# ==========================================
# 5. TRAIN-TEST SPLIT
# ==========================================

# Use chronological split because weather is time-series data

split_index = int(len(df) * 0.8)

X_train = X.iloc[:split_index]
X_test = X.iloc[split_index:]

y_temp_train = y_temperature.iloc[:split_index]
y_temp_test = y_temperature.iloc[split_index:]

y_rain_train = y_rainfall.iloc[:split_index]
y_rain_test = y_rainfall.iloc[split_index:]


# ==========================================
# 6. TRAIN TEMPERATURE MODEL
# ==========================================

temperature_model = LinearRegression()

temperature_model.fit(X_train, y_temp_train)

temperature_predictions = temperature_model.predict(X_test)


# ==========================================
# 7. TRAIN RAINFALL MODEL
# ==========================================

rainfall_model = LinearRegression()

rainfall_model.fit(X_train, y_rain_train)

rainfall_predictions = rainfall_model.predict(X_test)


# ==========================================
# 8. EVALUATE TEMPERATURE MODEL
# ==========================================

temperature_mae = mean_absolute_error(
    y_temp_test,
    temperature_predictions
)

temperature_rmse = np.sqrt(
    mean_squared_error(
        y_temp_test,
        temperature_predictions
    )
)

temperature_r2 = r2_score(
    y_temp_test,
    temperature_predictions
)


# ==========================================
# 9. EVALUATE RAINFALL MODEL
# ==========================================

rainfall_mae = mean_absolute_error(
    y_rain_test,
    rainfall_predictions
)

rainfall_rmse = np.sqrt(
    mean_squared_error(
        y_rain_test,
        rainfall_predictions
    )
)

rainfall_r2 = r2_score(
    y_rain_test,
    rainfall_predictions
)


# ==========================================
# 10. DISPLAY RESULTS
# ==========================================

print("========== WEATHER MODEL RESULTS ==========")

print("\nTemperature Model")
print("MAE:", temperature_mae)
print("RMSE:", temperature_rmse)
print("R2 Score:", temperature_r2)

print("\nRainfall Model")
print("MAE:", rainfall_mae)
print("RMSE:", rainfall_rmse)
print("R2 Score:", rainfall_r2)


# ==========================================
# 11. SAVE MODELS
# ==========================================

joblib.dump(
    temperature_model,
    "models/temperature_model.pkl"
)

joblib.dump(
    rainfall_model,
    "models/rainfall_model.pkl"
)

print("\nModels saved successfully!")
print("Temperature model: models/temperature_model.pkl")
print("Rainfall model: models/rainfall_model.pkl")