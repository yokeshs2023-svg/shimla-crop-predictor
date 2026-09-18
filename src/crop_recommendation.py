import joblib
import pandas as pd


# Load trained crop recommendation model
MODEL_PATH = "models/crop_recommendation_model.pkl"

crop_model = joblib.load(MODEL_PATH)


def recommend_crop(
    nitrogen,
    phosphorus,
    potassium,
    temperature,
    humidity,
    ph,
    rainfall
):
    """Predict a crop using soil and weather information."""

    input_data = pd.DataFrame([{
        "N": nitrogen,
        "P": phosphorus,
        "K": potassium,
        "temperature": temperature,
        "humidity": humidity,
        "ph": ph,
        "rainfall": rainfall
    }])

    prediction = crop_model.predict(input_data)

    return prediction[0]


if __name__ == "__main__":

    print("========== CROP RECOMMENDATION ==========")

    print("\nEnter soil information:")

    nitrogen = float(input("Nitrogen (N): "))
    phosphorus = float(input("Phosphorus (P): "))
    potassium = float(input("Potassium (K): "))
    ph = float(input("Soil pH: "))

    print("\nEnter weather information:")

    temperature = float(input("Temperature (°C): "))
    humidity = float(input("Humidity (%): "))
    rainfall = float(input("Rainfall (mm): "))

    crop = recommend_crop(
        nitrogen,
        phosphorus,
        potassium,
        temperature,
        humidity,
        ph,
        rainfall
    )

    print("\nRecommended Crop:", crop)