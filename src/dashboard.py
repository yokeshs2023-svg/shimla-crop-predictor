import streamlit as st
import requests
import pandas as pd


# ==========================================
# PAGE CONFIGURATION
# ==========================================

st.set_page_config(
    page_title="Shimla Crop Advisory",
    page_icon="🌱",
    layout="wide"
)


# ==========================================
# TITLE
# ==========================================

st.title("🌱 Shimla Crop Advisory")
st.write(
    "Weather information, frost risk and crop recommendation"
)


# ==========================================
# LOAD SHIMLA WEATHER DATA
# ==========================================

weather_file = "data/raw/shimla_weather.csv"

weather_df = pd.read_csv(weather_file)

weather_df["date"] = pd.to_datetime(weather_df["date"])

latest_weather = weather_df.iloc[-1]


temperature = float(latest_weather["temperature"])
rainfall = float(latest_weather["rainfall"])
humidity = float(latest_weather["humidity"])
wind_speed = float(latest_weather["wind_speed"])


# ==========================================
# FROST RISK
# ==========================================

if temperature <= 0:
    frost_risk = "HIGH"
else:
    frost_risk = "LOW"


# ==========================================
# WEATHER DISPLAY
# ==========================================

st.header("🌦️ Shimla Weather")

col1, col2, col3, col4 = st.columns(4)

with col1:
    st.metric(
        "Temperature",
        f"{temperature:.2f} °C"
    )

with col2:
    st.metric(
        "Rainfall",
        f"{rainfall:.2f} mm"
    )

with col3:
    st.metric(
        "Humidity",
        f"{humidity:.2f} %"
    )

with col4:
    st.metric(
        "Wind Speed",
        f"{wind_speed:.2f} km/h"
    )


# ==========================================
# FROST RISK DISPLAY
# ==========================================

st.header("❄️ Frost Risk")

if frost_risk == "HIGH":
    st.error(
        "❄️ HIGH FROST RISK — Protect frost-sensitive crops."
    )
else:
    st.success(
        "Frost Risk: LOW"
    )


# ==========================================
# CROP RECOMMENDATION
# ==========================================

st.header("🌾 Crop Recommendation")

st.write(
    "Enter soil information to generate a crop recommendation."
)

col1, col2, col3, col4 = st.columns(4)

with col1:
    nitrogen = st.number_input(
        "Nitrogen (N)",
        min_value=0.0,
        value=90.0
    )

with col2:
    phosphorus = st.number_input(
        "Phosphorus (P)",
        min_value=0.0,
        value=40.0
    )

with col3:
    potassium = st.number_input(
        "Potassium (K)",
        min_value=0.0,
        value=40.0
    )

with col4:
    ph = st.number_input(
        "Soil pH",
        min_value=0.0,
        max_value=14.0,
        value=6.5
    )


# ==========================================
# RECOMMEND BUTTON
# ==========================================

if st.button("🌱 Recommend Crop"):

    request_data = {
        "N": nitrogen,
        "P": phosphorus,
        "K": potassium,
        "temperature": temperature,
        "humidity": humidity,
        "ph": ph,
        "rainfall": rainfall
    }

    try:

        response = requests.post(
            "http://host.docker.internal:8000/predict/crop",
            json=request_data,
            timeout=10
        )

        if response.status_code == 200:

            result = response.json()

            crop = result["recommended_crop"]

            st.success(
                f"🌾 Recommended Crop: **{crop.upper()}**"
            )

        else:

            st.error(
                f"API Error: {response.status_code}"
            )

    except requests.exceptions.RequestException:

        st.error(
            "Could not connect to FastAPI. "
            "Make sure the FastAPI server is running."
        )


# ==========================================
# HISTORICAL WEATHER
# ==========================================

st.header("📊 Historical Weather")

st.subheader("Temperature")

st.line_chart(
    weather_df.set_index("date")["temperature"]
)

st.subheader("Rainfall")

st.line_chart(
    weather_df.set_index("date")["rainfall"]
)


# ==========================================
# INFORMATION
# ==========================================

st.info(
    "Note: Crop recommendation requires soil parameters "
    "such as N, P, K and pH. Weather data alone does not "
    "provide a complete soil-based recommendation."
)