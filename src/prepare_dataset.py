import pandas as pd


# ==========================================
# 1. LOAD RAW DATASETS
# ==========================================

crop_file = "data/raw/crop_recommendation.csv"
weather_file = "data/raw/shimla_weather.csv"

crop_df = pd.read_csv(crop_file)
weather_df = pd.read_csv(weather_file)


# ==========================================
# 2. CLEAN CROP DATASET
# ==========================================

print("========== CROP DATASET ==========")

print("Original rows:", len(crop_df))

# Check missing values
print("Missing values:")
print(crop_df.isnull().sum())

# Remove duplicate rows
crop_df = crop_df.drop_duplicates()

print("Rows after removing duplicates:", len(crop_df))

# Save cleaned crop dataset
crop_df.to_csv(
    "data/processed/crop_recommendation_clean.csv",
    index=False
)

print("Clean crop dataset saved successfully!")


# ==========================================
# 3. CLEAN WEATHER DATASET
# ==========================================

print("\n========== SHIMLA WEATHER DATASET ==========")

print("Original rows:", len(weather_df))

# Check missing values
print("Missing values:")
print(weather_df.isnull().sum())

# Remove duplicate rows
weather_df = weather_df.drop_duplicates()

# Convert date column
weather_df["date"] = pd.to_datetime(weather_df["date"])


# ==========================================
# 4. CREATE DATE FEATURES
# ==========================================

weather_df["year"] = weather_df["date"].dt.year
weather_df["month"] = weather_df["date"].dt.month


# ==========================================
# 5. CREATE SEASON
# ==========================================

def get_season(month):
    if month in [12, 1, 2]:
        return "Winter"
    elif month in [3, 4, 5]:
        return "Spring"
    elif month in [6, 7, 8, 9]:
        return "Monsoon"
    else:
        return "Autumn"


weather_df["season"] = weather_df["month"].apply(get_season)


# ==========================================
# 6. CREATE FROST RISK
# ==========================================

def get_frost_risk(temperature):
    if temperature <= 0:
        return 1
    return 0


weather_df["frost_risk"] = weather_df["temperature"].apply(
    get_frost_risk
)


# ==========================================
# 7. SAVE CLEAN WEATHER DATASET
# ==========================================

weather_df.to_csv(
    "data/processed/shimla_weather_clean.csv",
    index=False
)

print("Rows after removing duplicates:", len(weather_df))

print("Weather features created:")
print("year")
print("month")
print("season")
print("frost_risk")

print("\nClean weather dataset saved successfully!")


# ==========================================
# 8. FINAL SUMMARY
# ==========================================

print("\n========== PREPROCESSING COMPLETE ==========")

print(
    "Crop dataset:",
    "data/processed/crop_recommendation_clean.csv"
)

print(
    "Weather dataset:",
    "data/processed/shimla_weather_clean.csv"
)