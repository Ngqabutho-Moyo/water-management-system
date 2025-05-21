from datetime import datetime
import streamlit as st
import pandas as pd
import joblib
import numpy as np
import plotly.graph_objects as go
import folium
from streamlit_folium import st_folium
from pathlib import Path

# Configuration
ROOT = Path(r'C:\Users\Ngqabutho Moyo\Documents\Extra Curriculars\Leak Detection (Anesu)\Streamlit dashboard\app')

@st.cache_resource
def load_models():
    """Load trained models with caching"""
    return (
        joblib.load(ROOT/'ml_models/leak_model.sav'),
        joblib.load(ROOT/'ml_models/burst_model.sav'),
        joblib.load(ROOT/'ml_models/spatial_model.sav'),
        joblib.load(ROOT/'ml_models/scaler.sav')
    )

@st.cache_data
def load_data():
    """Load and cache water data"""
    return pd.read_csv(ROOT/'datasets/water_data.csv')

# Load models and data
try:
    leak_model, burst_model, spatial_model, scaler = load_models()
    df = load_data()
    spatial_features = ['pressure', 'flow_rate', 'temperature',
                      'latitude', 'longitude', 'pipe_age', 'pipe_material']
except Exception as e:
    st.error(f"Failed to load models/data: {str(e)}")
    st.stop()

# Initialize session state
if 'last_sample' not in st.session_state:
    st.session_state.last_sample = df.sample(1).iloc[0]
    
if 'map_sample' not in st.session_state:
    st.session_state.map_sample = df.sample(500)
    st.session_state.map_sample['leak_risk'] = spatial_model.predict_proba(
        st.session_state.map_sample[spatial_features])[:,1]
    st.session_state.map_sample['burst_risk'] = burst_model.predict_proba(
        st.session_state.map_sample[['pressure', 'flow_rate', 'temperature']])[:,1]

# Title and tabs
st.title('Smart Water Management System')
tab1, tab2, tab3, tab4 = st.tabs([
    "Real-time Monitoring", 
    "Leak Predictions", 
    "Burst Predictions",
    "Risk Location Map"
])

# Threshold configurations
THRESHOLDS = {
    'pressure': {'safe': 4, 'danger': 4.5},
    'flow_rate': {'safe': 180, 'danger': 190},
    'temperature': {'safe': 25, 'danger': 28},
    'leak': 0.4,
    'burst': 0.4
}

RANGES = {
    'pressure': {'min': 0, 'max': 5, 'unit': 'bar'},
    'flow_rate': {'min': 0, 'max': 200, 'unit': 'L/s'},
    'temperature': {'min': 10, 'max': 30, 'unit': '°C'}
}

def get_gauge_color(value, param):
    """Determine gauge color based on thresholds"""
    if value >= THRESHOLDS[param]['danger']:
        return "#FF0000"  # Red for danger
    elif value >= THRESHOLDS[param]['safe']:
        return "#FFA500"  # Orange for warning
    return "#2ca02c"  # Green for safe

# Tab 1: Real-time Monitoring
with tab1:
    st.header("Current Sensor Readings")
    
    if st.button("Refresh Data"):
        st.session_state.last_sample = df.sample(1).iloc[0]
        st.rerun()
    
    sample = st.session_state.last_sample
    
    # Create gauge columns
    cols = st.columns(3)
    for i, param in enumerate(['pressure', 'flow_rate', 'temperature']):
        with cols[i]:
            fig = go.Figure(go.Indicator(
                mode="gauge+number",
                value=sample[param],
                domain={'x': [0, 1], 'y': [0, 1]},
                title={'text': f"{param.title()} ({RANGES[param]['unit']})"},
                gauge={
                    'axis': {'range': [None, RANGES[param]['max']]},
                    'bar': {'color': get_gauge_color(sample[param], param)},
                    'steps': [
                        {'range': [0, THRESHOLDS[param]['safe']], 'color': "white"},
                        {'range': [THRESHOLDS[param]['safe'], THRESHOLDS[param]['danger']], 'color': "lightyellow"}
                    ],
                    'threshold': {
                        'line': {'color': "red", 'width': 4},
                        'thickness': 0.8,
                        'value': THRESHOLDS[param]['safe']}
                }
            ))
            st.plotly_chart(fig, use_container_width=True)
    
    st.caption(f"Last updated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

# Tab 2: Leak Predictions
with tab2:
    st.header("Leak Detection Analysis")
    sample = df.sample(1)
    features = scaler.transform(sample[['pressure', 'flow_rate', 'temperature']])
    proba = leak_model.predict_proba(features)[0][1]
    
    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=proba*100,
        domain={'x': [0, 1], 'y': [0, 1]},
        title={'text': "Leak Probability (%)"},
        gauge={
            'axis': {'range': [0, 100]},
            'bar': {'color': "#2ca02c"},
            'steps': [
                {'range': [0, 40], 'color': "white"},
                {'range': [40, 70], 'color': "lightyellow"},
                {'range': [70, 100], 'color': "red"}
            ],
            'threshold': {'line': {'color': "red", 'width': 4}, 'value': THRESHOLDS['leak']*100}
        }
    ))
    st.plotly_chart(fig, use_container_width=True)
    
    if proba > THRESHOLDS['leak']:
        st.error(f"ALERT: {proba:.1%} leak risk detected", icon="⚠️")
    else:
        st.success("No leak detected", icon="✅")

# Tab 3: Burst Predictions
with tab3:
    st.header("Pipe Burst Prediction")
    sample = df.sample(1)
    features = scaler.transform(sample[['pressure', 'flow_rate', 'temperature']])
    proba = burst_model.predict_proba(features)[0][1]
    
    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=proba*100,
        domain={'x': [0, 1], 'y': [0, 1]},
        title={'text': "Burst Probability (%)"},
        gauge={
            'axis': {'range': [0, 100]},
            'bar': {'color': "#2ca02c"},
            'steps': [
                {'range': [0, 40], 'color': "white"},
                {'range': [40, 70], 'color': "lightyellow"},
                {'range': [70, 100], 'color': "red"}
            ],
            'threshold': {'line': {'color': "red", 'width': 4}, 'value': THRESHOLDS['burst']*100}
        }
    ))
    st.plotly_chart(fig, use_container_width=True)
    
    if proba > THRESHOLDS['burst']:
        st.error(f"💥 CRITICAL: {proba:.1%} burst risk detected", icon="🚨")
    else:
        st.success("No burst detected", icon="✅")

# Tab 4: Risk Location Map
with tab4:
    st.header("Risk Location Map")
    
    # Sample size selector
    # col1, col2 = st.columns([2, 1])
    # with col1:
    #     if st.button("Resample Data"):
    #         st.session_state.map_sample = df.sample(500)
    #         st.session_state.map_sample['leak_risk'] = spatial_model.predict_proba(
    #             st.session_state.map_sample[spatial_features])[:,1]
    #         st.session_state.map_sample['burst_risk'] = burst_model.predict_proba(
    #             st.session_state.map_sample[['pressure', 'flow_rate', 'temperature']])[:,1]
    #         st.rerun()
    
    # Create map with cached sample
    map_df = st.session_state.map_sample
    m = folium.Map(
        location=[map_df['latitude'].mean(), map_df['longitude'].mean()],
        zoom_start=14,
        tiles="CartoDB positron"
    )
    
    # Add markers
    for _, row in map_df.iterrows():
        risk_color = 'red' if row['leak_risk'] > 0.7 else 'orange' if row['leak_risk'] > 0.4 else 'green'
        
        folium.CircleMarker(
            location=[row['latitude'], row['longitude']],
            radius=8,
            color=risk_color,
            fill=True,
            popup=f"""
            <strong>Pipe ID:</strong> {row['pipe_id']}<br>
            <strong>Leak Risk:</strong> {row['leak_risk']:.1%}<br>
            <strong>Burst Risk:</strong> {row['burst_risk']:.1%}<br>
            <strong>Age:</strong> {row['pipe_age']} years
            """
        ).add_to(m)
    
    # Display map
    st_folium(m, width=700, height=500, key='map')
    
    # Legend
    st.markdown("""
    **Risk Legend:**
    - <span style='color:red'>🔴 High Risk</span> (>70%)
    - <span style='color:orange'>🟠 Medium Risk</span> (40-70%)
    - <span style='color:green'>🟢 Low Risk</span> (<40%)
    """, unsafe_allow_html=True)
    
    if st.button("Resample Data"):
            st.session_state.map_sample = df.sample(1000)
            st.session_state.map_sample['leak_risk'] = spatial_model.predict_proba(
                st.session_state.map_sample[spatial_features])[:,1]
            st.session_state.map_sample['burst_risk'] = burst_model.predict_proba(
                st.session_state.map_sample[['pressure', 'flow_rate', 'temperature']])[:,1]
            st.rerun()