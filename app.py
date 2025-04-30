import streamlit as st
import pandas as pd
import numpy as np
import pickle
import random
import matplotlib.pyplot as plt
import plotly.graph_objects as go
import os

# --- UI Setup ---
st.set_page_config(page_title="AQI Predictor", layout="centered")
st.title("🌍 AQI Prediction App")
st.markdown("Enter pollutant values to predict Air Quality Index (AQI)")

# --- Helper: compute nominal confidence from quantile levels ---
def compute_confidence(alpha_low: float = 0.05, alpha_high: float = 0.95) -> float:
    """
    Returns the confidence percentage corresponding to the
    interval between alpha_low and alpha_high quantiles.
    """
    return (alpha_high - alpha_low) * 100

# --- Load Quantile Models ---
@st.cache_resource
def load_models():
    # Find the directory this script lives in
    base_dir = os.path.dirname(__file__)
    model_dir = os.path.join(base_dir, "models")

    paths = {
        "lo":  os.path.join(model_dir, "lightgbm_quantile_05.pkl"),
        "med": os.path.join(model_dir, "lightgbm_quantile_50.pkl"),
        "hi":  os.path.join(model_dir, "lightgbm_quantile_95.pkl"),
    }

    models = {}
    for key, path in paths.items():
        if not os.path.exists(path):
            raise FileNotFoundError(f"Model file not found: {path}")
        with open(path, "rb") as f:
            models[key] = pickle.load(f)
    return models

models = load_models()
model_lo, model_med, model_hi = models["lo"], models["med"], models["hi"]

# --- Feature Mapping ---
pollutant_mapping = {
    "Carbon Monoxide (CO)": 0,
    "Nitrogen Dioxide (NO2)": 1,
    "Ozone": 2,
    "Ozone (8hr)": 3,
    "PM10": 4,
    "PM2.5": 5,
    "Sulfur Dioxide (SO2)": 6,
}

# --- Random Example Generator ---
def generate_random_example():
    return {
        'so2': round(random.uniform(0.5, 20), 2),
        'co': round(random.uniform(0.1, 1.5), 2),
        'o3': round(random.uniform(10, 100), 2),
        'o3_8hr': round(random.uniform(10, 100), 2),
        'pm10': round(random.uniform(10, 150), 2),
        'pm25': round(random.uniform(5, 100), 2),
        'no2': round(random.uniform(5, 80), 2),
        'nox': round(random.uniform(5, 100), 2),
        'no': round(random.uniform(1, 40), 2),
        'windspeed': round(random.uniform(0.5, 6), 2),
        'winddirec': round(random.uniform(0, 360), 2),
        'co_8hr': round(random.uniform(0.1, 1.2), 2),
        'pm25_avg': round(random.uniform(5, 80), 2),
        'pm10_avg': round(random.uniform(10, 140), 2),
        'so2_avg': round(random.uniform(0.5, 15), 2),
        'pollutant': random.choice(list(pollutant_mapping.keys()))
    }

# --- Example Button ---
if st.button("🎲 Use Random Example"):
    st.session_state.example = generate_random_example()
else:
    st.session_state.setdefault("example", generate_random_example())
ex = st.session_state.example

# --- Input Form ---
col1, col2 = st.columns(2)
with col1:
    so2       = st.number_input("SO₂ (μg/m³)", value=ex['so2'])
    co        = st.number_input("CO (mg/m³)", value=ex['co'])
    o3        = st.number_input("O₃ (μg/m³)", value=ex['o3'])
    o3_8hr    = st.number_input("O₃ (8hr, μg/m³)", value=ex['o3_8hr'])
    pm10      = st.number_input("PM10 (μg/m³)", value=ex['pm10'])
    pm25      = st.number_input("PM2.5 (μg/m³)", value=ex['pm25'])
    pollutant = st.selectbox(
        "Main Pollutant",
        list(pollutant_mapping.keys()),
        index=list(pollutant_mapping.keys()).index(ex['pollutant'])
    )
with col2:
    no2      = st.number_input("NO₂ (μg/m³)", value=ex['no2'])
    nox      = st.number_input("NOx (μg/m³)", value=ex['nox'])
    no       = st.number_input("NO (μg/m³)", value=ex['no'])
    windspeed= st.number_input("Wind Speed (m/s)", value=ex['windspeed'])
    winddirec= st.number_input("Wind Direction (°)", value=ex['winddirec'])
    co_8hr   = st.number_input("CO (8hr, mg/m³)", value=ex['co_8hr'])
    pm25_avg = st.number_input("PM2.5 Avg (μg/m³)", value=ex['pm25_avg'])
    pm10_avg = st.number_input("PM10 Avg (μg/m³)", value=ex['pm10_avg'])
    so2_avg  = st.number_input("SO₂ Avg (μg/m³)", value=ex['so2_avg'])

input_data = pd.DataFrame([{
    'pollutant': pollutant_mapping[pollutant],
    'so2': so2, 'co': co, 'o3': o3, 'o3_8hr': o3_8hr,
    'pm10': pm10, 'pm2.5': pm25, 'no2': no2, 'nox': nox, 'no': no,
    'windspeed': windspeed, 'winddirec': winddirec, 'co_8hr': co_8hr,
    'pm2.5_avg': pm25_avg, 'pm10_avg': pm10_avg, 'so2_avg': so2_avg
}])

sensitive_group = st.selectbox(
    "👤 Are you part of a sensitive group?",
    [
      "No, I'm generally healthy",
      "Yes, I have respiratory issues",
      "Yes, I am elderly (65+)",
      "Yes, I am a child (below 12)"
    ]
)

# --- Predict & Display ---
if st.button("📈 Predict AQI"):
    lo = model_lo.booster.predict(input_data.values)[0]
    med= model_med.booster.predict(input_data.values)[0]
    hi = model_hi.booster.predict(input_datav)[0]
    margin = (hi - lo) / 2
    confidence = compute_confidence(0.05, 0.95)

    st.subheader(
        f"Predicted AQI: {med:.1f} ± {margin:.1f}  "
        f"({confidence:.0f}% CI: [{lo:.1f}, {hi:.1f}])"
    )

    st.markdown(
        f"""
        **What this means:**  
        • We estimate today’s AQI at **{med:.1f}** (the median).  
        • “±{margin:.1f}” is half of our {confidence:.0f}% interval.  
        • So we’re about **{confidence:.0f}% confident** the true AQI lies between **{lo:.1f}** and **{hi:.1f}**.  
        • Because the upper bound ({hi:.1f}) crosses “Unhealthy for Sensitive Groups,”
          please take extra care if you’re sensitive.
        """
    )

    # --- AQI Category & Gauge ---
    def get_category(aqi):
        if aqi <= 50:  return "Good", "🟢😊"
        elif aqi <=100: return "Moderate", "🟡😐"
        elif aqi <=150: return "Unhealthy for Sensitive Groups", "🟠😷"
        elif aqi <=200: return "Unhealthy", "🔴😷"
        elif aqi <=300: return "Very Unhealthy", "🟣😨"
        else:          return "Hazardous", "⚫☠️"

    category, emoji = get_category(med)
    st.markdown(f"### AQI Category: {emoji} {category}")

    # build gauge with CI band
    fig = go.Figure()
    # add CI band as a grey rectangle
    fig.update_layout(
        shapes=[{
            "type": "rect",
            "xref": "paper", "yref": "paper",
            "x0": lo/500, "x1": hi/500,
            "y0": 0,     "y1": 0.1,
            "fillcolor": "lightgrey",
            "opacity": 0.5,
            "layer": "below"
        }]
    )
    # add the gauge needle
    fig.add_trace(go.Indicator(
        mode="gauge+number",
        value=med,
        gauge={
            "axis": {"range": [0,500]},
            "bar": {"color": "black"},
            "steps": [
                {"range":[0,50],   "color":"green"},
                {"range":[50,100], "color":"yellow"},
                {"range":[100,150],"color":"orange"},
                {"range":[150,200],"color":"red"},
                {"range":[200,300],"color":"purple"},
                {"range":[300,500],"color":"maroon"},
            ],
        },
        title={"text":"Air Quality Index"}
    ))
    st.plotly_chart(fig)

    fig.update_layout(
        shapes=[
            dict(
                type="rect",
                xref="paper", yref="paper",
                x0=lo/500, x1=hi/500,   # normalize [0,500] → [0.0,1.0]
                y0=0,   y1=0.1,         # height of the band (10% of plot)
                fillcolor="lightgrey",
                opacity=0.5,
                layer="below"
            )
        ]
    )

        # --- AQI Health Info ---
    st.markdown("### 📊 AQI Levels and Health Implications")

    # --- Personalized Health Advice ---
    st.markdown("### 🩺 Personalized Health Advisory")

    if sensitive_group == "No, I'm generally healthy":
        if med <= 100:
            st.success("Air quality is acceptable. You can continue regular outdoor activities.")
        elif med <= 150:
            st.warning("Moderate health risk. Limit prolonged outdoor exertion if possible.")
        else:
            st.error("Air quality is poor. Consider limiting outdoor exposure and using air purifiers indoors.")
    else:
        if med <= 50:
            st.success("Good air quality. Safe for sensitive groups.")
        elif med <= 100:
            st.warning("Moderate risk for sensitive individuals. Limit strenuous outdoor activities.")
        elif med <= 150:
            st.warning("Increased risk for sensitive individuals. Reduce outdoor activities, consider wearing masks.")
        elif med <= 200:
            st.error("Unhealthy air for sensitive groups. Stay indoors and use air purification if available.")
        else:
            st.error("⚠️ Hazardous conditions! Avoid outdoor activities and stay in a controlled indoor environment.")


    # --- AQI Table ---
    aqi_html_table = """
    <style>
        table.aqi-table {
            width: 100%;
            border-collapse: collapse;
        }
        .aqi-table th, .aqi-table td {
            border: 1px solid #ddd;
            padding: 8px;
            text-align: left;
            font-size: 14px;
        }
        .aqi-table th {
            background-color: #f2f2f2;
            font-weight: bold;
        }
    </style>

    <table class="aqi-table">
        <thead>
            <tr>
                <th>AQI</th>
                <th>Air Pollution Level</th>
                <th>Health Implications</th>
                <th>Cautionary Statement (for PM2.5)</th>
            </tr>
        </thead>
        <tbody>
            <tr style="background-color: #00e400;">
                <td>0 - 50</td>
                <td>Good</td>
                <td>Air quality is considered satisfactory, and air pollution poses little or no risk.</td>
                <td>None</td>
            </tr>
            <tr style="background-color: #ffff00;">
                <td>51 - 100</td>
                <td>Moderate</td>
                <td>Air quality is acceptable; however, there may be a moderate health concern for a small number of sensitive people.</td>
                <td>People with respiratory disease should limit prolonged outdoor exertion.</td>
            </tr>
            <tr style="background-color: #ff7e00;">
                <td>101 - 150</td>
                <td>Unhealthy for Sensitive Groups</td>
                <td>Members of sensitive groups may experience health effects. The general public is not likely to be affected.</td>
                <td>Limit prolonged outdoor exertion.</td>
            </tr>
            <tr style="background-color: #ff0000;">
                <td>151 - 200</td>
                <td>Unhealthy</td>
                <td>Everyone may begin to experience health effects; sensitive groups may experience more serious effects.</td>
                <td>Avoid prolonged or heavy exertion. Sensitive groups should avoid outdoor activity.</td>
            </tr>
            <tr style="background-color: #8f3f97; color: white;">
                <td>201 - 300</td>
                <td>Very Unhealthy</td>
                <td>Health warnings of emergency conditions. The entire population is more likely to be affected.</td>
                <td>Avoid all outdoor exertion. Stay indoors.</td>
            </tr>
            <tr style="background-color: #7e0023; color: white;">
                <td>301+</td>
                <td>Hazardous</td>
                <td>Health alert: everyone may experience more serious health effects.</td>
                <td>Everyone should avoid all outdoor exertion.</td>
            </tr>
        </tbody>
    </table>
    """
    st.markdown(aqi_html_table, unsafe_allow_html=True)


    

