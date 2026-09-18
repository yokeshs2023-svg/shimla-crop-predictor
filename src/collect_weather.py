import requests
import pandas as pd

# Shimla coordinates
LATITUDE = 31.1048
LONGITUDE = 77.1734

# Historical date range
START_DATE = "2020-01-01"
END_DATE = "2024-12-31"

# Open-Meteo Historical Weather API
URL = "https://archive-api.open-meteo.com/v1/archive"

params = {
    "latitude": LATITUDE,
    "longitude": LONGITUDE,
    "start_date": START_DATE,
    "end_date": END_DATE,
    "hourly": "temperature_2m,relative_humidity_2m,precipitation,wind_speed_10m",
    "timezone": "Asia/Kolkata",
    "temperature_unit": "celsius",
    "precipitation_unit": "mm",
    "wind_speed_unit": "kmh"
}

# Get weather data
response = requests.get(URL, params=params, timeout=60)

# Stop if the API gives an error
response.raise_for_status()

# Convert API response to Python dictionary
data = response.json()

# Convert hourly data into a DataFrame
df = pd.DataFrame(data["hourly"])

# Convert time into datetime
df["time"] = pd.to_datetime(df["time"])

# Extract date
df["date"] = df["time"].dt.date

# Convert hourly data into daily data
daily_df = df.groupby("date").agg(
    temperature=("temperature_2m", "mean"),
    rainfall=("precipitation", "sum"),
    humidity=("relative_humidity_2m", "mean"),
    wind_speed=("wind_speed_10m", "mean")
).reset_index()

# Round the values
daily_df["temperature"] = daily_df["temperature"].round(2)
daily_df["rainfall"] = daily_df["rainfall"].round(2)
daily_df["humidity"] = daily_df["humidity"].round(2)
daily_df["wind_speed"] = daily_df["wind_speed"].round(2)

# Save the final CSV
OUTPUT_FILE = "data/raw/shimla_weather.csv"

daily_df.to_csv(OUTPUT_FILE, index=False)

print("Shimla weather dataset created successfully!")
print("File:", OUTPUT_FILE)
print("Number of records:", len(daily_df))
print()
print(daily_df.head())