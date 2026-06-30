import streamlit as st
import requests

st.set_page_config(page_title="EcoInference AI", page_icon="🛰️", layout="centered")

st.title("🛰️ Satellite Air Quality Inference Platform")
st.write("Real-time proxy-based machine learning estimations across geographic sectors.")

# 1. User Interface Inputs
city_name = st.text_input("Enter target city to inspect:", placeholder="e.g., Mumbai")

# Link to your live Hugging Face Space URL (change this after deploying your backend)
BACKEND_API_URL = "https://your-username-your-space-name.hf.space/api/v1/predict"

if st.button("Initiate Satellite Scan"):
    if not city_name.strip():
        st.warning("Please enter a valid location name.")
    else:
        with st.spinner("Resolving coordinates and running model inference calculations..."):
            try:
                # Resolve city text string to precise floats using Geocoding
                geo_url = f"https://geocoding-api.open-meteo.com/v1/search?name={city_name}&count=1&language=en&format=json"
                geo_res = requests.get(geo_url).json()

                if 'results' not in geo_res or len(geo_res['results']) == 0:
                    st.error("Could not resolve location coordinates. Try a major city.")
                else:
                    lat = geo_res['results'][0]['latitude']
                    lon = geo_res['results'][0]['longitude']

                    # Request prediction output fields from your hosted FastAPI backend
                    api_call_url = f"{BACKEND_API_URL}?lat={lat}&lon={lon}"
                    response = requests.get(api_call_url).json()

                    if response.get("status") == "success":
                        data = response["predictions"]
                        st.success(f"Scan complete for {city_name.upper()} ({lat}°N, {lon}°E)")

                        # Render metric UI cards
                        col1, col2, col3 = st.columns(3)
                        col1.metric("PM2.5 Sub-Index", f"{data['pm25_aqi']} AQI")
                        col2.metric("PM10 Particle Level", f"{data['pm10']} µg/m³")
                        col3.metric("Nitrogen Dioxide (NO2)", f"{data['no2']} µg/m³")

                        col4, col5, col6 = st.columns(3)
                        col4.metric("Sulphur Dioxide (SO2)", f"{data['so2']} µg/m³")
                        col5.metric("Carbon Monoxide (CO)", f"{data['co']} mg/m³")
                        col6.metric("Formaldehyde (HCHO)", f"{data['hcho']}")
                    else:
                        st.error("Error running inference calculation pipeline.")
            except Exception as e:
                st.error(f"Network error linking to cloud microservices: {e}")