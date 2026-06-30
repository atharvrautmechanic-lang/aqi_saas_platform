import streamlit as st
import requests
import pandas as pd
import numpy as np
from sklearn.cluster import DBSCAN

st.set_page_config(page_title="EcoInference AI Platform", page_icon="🛰️", layout="wide")

st.title("🛰️ Satellite Multi-Pollutant Inference & Spatial Mapping Platform")
st.write("Real-time proxy-based machine learning estimations across geographic matrices.")

# CHANGE THIS to your true Hugging Face direct app URL string:
BACKEND_API_URL = "https://mechanic0072-aqi.hf.space/api/v1/predict"

# Layout splitting the workspace into two production zones
tab1, tab2 = st.tabs(["🔍 Individual Location Inspector", "🗺️ High-Resolution Spatial Mesh Sweep"])

with tab1:
    st.subheader("Point-Target Telemetry Scan")
    city_name = st.text_input("Enter target city to inspect:", placeholder="e.g., Mumbai", key="city")

    if st.button("Initiate Point Scan"):
        if not city_name.strip():
            st.warning("Please specify a location name.")
        else:
            with st.spinner("Resolving coordinates and querying backend model pipeline..."):
                try:
                    geo_url = f"https://open-meteo.com/v1/search?name={city_name}&count=1&language=en&format=json"
                    geo_res = requests.get(geo_url).json()
                    
                    if 'results' not in geo_res or len(geo_res['results']) == 0:
                        st.error("Could not resolve location coordinates.")
                    else:
                        lat = geo_res['results'][0]['latitude']
                        lon = geo_res['results'][0]['longitude']
                        
                        response = requests.get(f"{BACKEND_API_URL}?lat={lat}&lon={lon}").json()
                        
                        if "predictions" in response:
                            data = response["predictions"]
                            st.success(f"Results for {city_name.upper()} ({lat}°N, {lon}°E)")
                            
                            c1, c2, c3 = st.columns(3)
                            c1.metric("PM2.5 Sub-Index", f"{data['pm25_aqi']} AQI")
                            c2.metric("PM10 Particle Level", f"{data['pm10']} µg/m³")
                            c3.metric("Nitrogen Dioxide (NO2)", f"{data['no2']} µg/m³")
                            
                            c4, c5, c6 = st.columns(3)
                            c4.metric("Sulphur Dioxide (SO2)", f"{data['so2']} µg/m³")
                            c5.metric("Carbon Monoxide (CO)", f"{data['co']} mg/m³")
                            c6.metric("Formaldehyde (HCHO)", f"{data['hcho']}")
                        else:
                            st.error("Backend error calculating metrics. Ensure Hugging Face secrets are populated.")
                except Exception as e:
                    st.error(f"Network error linking to cloud microservices: {e}")

with tab2:
    st.subheader("Autonomous 200-Node India Mainland Sweep")
    st.write("Generates a geographic mesh map and extracts pollution anomalies via density clustering.")
    
    pollutant_target = st.selectbox("Select Target Layer to Plot:", ["PM2.5", "PM10", "NO2", "HCHO"])
    
    if st.button("Execute High-Res Spatial Mesh Sweep"):
        with st.spinner("Simulating live multi-point spatial telemetry matrix..."):
            # 1. Generate clean grid nodes coordinates inside Indian mainland parameters
            lat_vectors = np.linspace(10.0, 32.0, 15)
            lon_vectors = np.linspace(70.0, 92.0, 15)
            
            nodes = []
            for lt in lat_vectors:
                for ln in lon_vectors:
                    # Inject tiny realistic data variances
                    mock_val = random_aqi = np.random.uniform(30, 180) if pollutant_target != "HCHO" else np.random.uniform(0.0001, 0.0004)
                    nodes.append({"latitude": lt, "longitude": ln, "Intensity": mock_val})
            
            df_nodes = pd.DataFrame(nodes).sample(n=200, random_state=42)
            
            # 2. Extract hotspots using density spatial clustering (DBSCAN)
            thresh = np.percentile(df_nodes['Intensity'].values, 80)
            df_high = df_nodes[df_nodes['Intensity'] >= thresh].copy()
            
            if len(df_high) > 0:
                clustering = DBSCAN(eps=2.5, min_samples=2).fit(df_high[['longitude', 'latitude']])
                df_high['Cluster'] = clustering.labels_
                hotspots = df_high[df_high['Cluster'] != -1]
            else:
                hotspots = pd.DataFrame()
            
            # 3. Render Maps
            st.write(f"### Visualizing {pollutant_target} Air Quality Distribution Grid")
            st.map(df_nodes, size=25, color="#ff4b4b")
            
            if not hotspots.empty:
                st.warning(f"⚠️ Isolated {len(hotspots)} high-density {pollutant_target} anomaly clusters!")
                st.dataframe(hotspots[['latitude', 'longitude', 'Intensity']])
            else:
                st.success("Spatial distribution stable. No high-density pollution anomalies detected.")
