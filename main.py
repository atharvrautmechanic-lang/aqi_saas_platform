import os
import joblib
import numpy as np
import requests
from fastapi import FastAPI, HTTPException
from database import save_new_telemetry_to_cloud

app = FastAPI(title="Multi-Pollutant Satellite Inference Engine API")

# Initialize and lock the model into memory layout on cloud boot-up
MODEL_BIN = "multi_pollutant_forest.pkl"
if os.path.exists(MODEL_BIN):
    model = joblib.load(MODEL_BIN)
else:
    raise RuntimeError(f"Missing core model weight file: {MODEL_BIN}")

FEATURES = [
    'Latitude', 'Longitude', 'Elevation', 'Distance_To_Coast',
    'Temp', 'Wind_U', 'Wind_V', 'Wind_Speed', 'AOD_to_Elevation_Ratio',
    'AOD', 'Fire_Count',
    'NO2_column_number_density', 'SO2_column_number_density',
    'CO_column_number_density', 'O3_column_number_density'
]


@app.get("/api/v1/predict")
def predict_air_quality(lat: float, lon: float):
    """
    Accepts coordinates, polls real-time weather parameters, runs AI inference,
    logs metadata to the cloud database, and returns data payload to frontend.
    """
    try:
        # 1. Dynamic Live Weather/Elevation Proxy Extraction Layer
        elev_res = requests.get(f"https://api.open-meteo.com/v1/elevation?latitude={lat}&longitude={lon}",
                                timeout=10).json()
        elevation = elev_res['elevation'][0]

        weather_res = requests.get(
            f"https://api.open-meteo.com/v1/forecast?latitude={lat}&longitude={lon}&current=temperature_2m,wind_speed_10m,wind_direction_10m",
            timeout=3).json()
        w_data = weather_res['current']

        aq_res = requests.get(
            f"https://air-quality-api.open-meteo.com/v1/air-quality?latitude={lat}&longitude={lon}&current=pm2_5,pm10,nitrogen_dioxide,sulphur_dioxide,carbon_monoxide,ozone",
            timeout=10).json()
        aq_data = aq_res['current']
    except Exception as e:
        raise HTTPException(status_code=503, detail=f"External Meteorological API Failure: {str(e)}")

    # 2. Physics & Kinematic Vector Calculations
    temp_k = w_data['temperature_2m'] + 273.15
    w_speed = w_data['wind_speed_10m'] / 3.6
    w_dir = np.radians(w_data['wind_direction_10m'])
    wind_u = -w_speed * np.sin(w_dir)
    wind_v = -w_speed * np.cos(w_dir)
    live_aod = aq_data['pm2_5'] / 50.0

    profile = {
        'Latitude': lat, 'Longitude': lon, 'Elevation': elevation, 'Distance_To_Coast': 6.0,
        'Temp': temp_k, 'Wind_U': wind_u, 'Wind_V': wind_v, 'Wind_Speed': w_speed,
        'AOD_to_Elevation_Ratio': live_aod / (elevation + 1), 'AOD': live_aod, 'Fire_Count': 0.0,
        'NO2_column_number_density': aq_data['nitrogen_dioxide'] * 1e-6,
        'SO2_column_number_density': aq_data['sulphur_dioxide'] * 1e-6,
        'CO_column_number_density': aq_data['carbon_monoxide'] * 1e-4,
        'O3_column_number_density': aq_data['ozone'] * 1e-6
    }

    # 3. Shape features array and run Random Forest Inference Matrix
    input_vector = np.array([[profile[f] for f in FEATURES]])
    preds = model.predict(input_vector)[0]

    # Add predicted values to profile for continuous learning cloud logging
    targets = ['PM2.5_SubIndex', 'PM10', 'NO2', 'SO2', 'CO', 'HCHO']
    for idx, t_name in enumerate(targets):
        profile[t_name] = preds[idx]

    # Save log asynchronously to Supabase/Neon
    save_new_telemetry_to_cloud(profile)

    return {
        "status": "success",
        "coordinates": {"lat": lat, "lon": lon},
        "predictions": {
            "pm25_aqi": round(preds[0], 2),
            "pm10": round(preds[1], 2),
            "no2": round(preds[2], 2),
            "so2": round(preds[3], 2),
            "co": round(preds[4], 2),
            "hcho": round(preds[5], 6)
        }
    }
