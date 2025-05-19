from datetime import datetime, time
from pathlib import Path
import streamlit as st
import plotly.express as px
import pandas as pd
import joblib
import numpy as np
from time import sleep
import plotly.graph_objects as go
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import numpy as np
import joblib

def load_models():    
    return (
        joblib.load('ml_models/leak_model.sav'),
        joblib.load('ml_models/burst_model.sav'),
        joblib.load('ml_models/scaler.sav')
    )

leak_model, burst_model, scaler = load_models()
df = pd.read_csv('datasets/water_data.csv')

# Title
st.title('Smart Water Management System')

# Tabs
tab1, tab2, tab3 = st.tabs([
    "Real-time monitoring", 
    "Leak Predictions", 
    "Burst Predictions"
])

if 'last_update' not in st.session_state:
    st.session_state.last_update = datetime.now()
    st.session_state.current_sample = df.sample(1).iloc[0]

# Tab 1: Real-time monitoring


with tab1:
    st.header("Current Sensor Readings")
    
    # Define safe thresholds (adjust these values as needed)
    thresholds = {
        'pressure': {'safe': 4, 'danger': 4.5},
        'flow_rate': {'safe': 180, 'danger': 190},
        'temperature': {'safe': 25, 'danger': 28}
    }
    
    if st.button("Refresh Data"):
        st.session_state.last_sample = df.sample(1).iloc[0]
        st.rerun()
    
    sample = st.session_state.last_sample if 'last_sample' in st.session_state else df.sample(1).iloc[0]
    
    # Define ranges
    ranges = {
        'pressure': {'min': 0, 'max': 5, 'unit': 'bar'},
        'flow_rate': {'min': 0, 'max': 200, 'unit': 'L/s'},
        'temperature': {'min': 10, 'max': 30, 'unit': '°C'}
    }
    
    # Create 3 columns for gauges
    col1, col2, col3 = st.columns(3)
    
    def get_gauge_color(value, param):
        if value >= thresholds[param]['danger']:
            return "#FF0000"  # Bright red for danger
        elif value >= thresholds[param]['safe']:
            return "#FFA500"  # Orange for warning
        else:
            return "#2ca02c"  # Green for safe
    
    # Pressure Gauge
    with col1:
        fig = go.Figure(go.Indicator(
            mode = "gauge+number+delta",
            value = sample['pressure'],
            domain = {'x': [0, 1], 'y': [0, 1]},
            title = {'text': f"Pressure ({ranges['pressure']['unit']})"},
            gauge = {
                'axis': {'range': [None, ranges['pressure']['max']], 'tickwidth': 1},
                'bar': {'color': get_gauge_color(sample['pressure'], 'pressure')},
                'steps': [
                    {'range': [0, thresholds['pressure']['safe']], 'color': "white"},
                    {'range': [thresholds['pressure']['safe'], thresholds['pressure']['danger']], 'color': "lightyellow"}
                ],
                'threshold': {
                    'line': {'color': "red", 'width': 4},
                    'thickness': 0.75,
                    'value': thresholds['pressure']['safe']}  # Warning threshold
            }
        ))
        fig.update_layout(margin=dict(t=50, b=10))
        st.plotly_chart(fig, use_container_width=True)
    
    # Flow Rate Gauge
    with col2:
        fig = go.Figure(go.Indicator(
            mode = "gauge+number+delta",
            value = sample['flow_rate'],
            domain = {'x': [0, 1], 'y': [0, 1]},
            title = {'text': f"Flow Rate ({ranges['flow_rate']['unit']})"},
            gauge = {
                'axis': {'range': [None, ranges['flow_rate']['max']], 'tickwidth': 1},
                'bar': {'color': get_gauge_color(sample['flow_rate'], 'flow_rate')},
                'steps': [
                    {'range': [0, thresholds['flow_rate']['safe']], 'color': "white"},
                    {'range': [thresholds['flow_rate']['safe'], thresholds['flow_rate']['danger']], 'color': "lightyellow"}
                ],
                'threshold': {
                    'line': {'color': "red", 'width': 4},
                    'thickness': 0.75,
                    'value': 180}
            }
        ))
        fig.update_layout(margin=dict(t=50, b=10))
        st.plotly_chart(fig, use_container_width=True)
    
    # Temperature Gauge
    with col3:
        fig = go.Figure(go.Indicator(
            mode = "gauge+number+delta",
            value = sample['temperature'],
            domain = {'x': [0, 1], 'y': [0, 1]},
            title = {'text': f"Temperature ({ranges['temperature']['unit']})"},
            gauge = {
                'axis': {'range': [None, ranges['temperature']['max']], 'tickwidth': 1},
                'bar': {'color': get_gauge_color(sample['temperature'], 'temperature')},
                'steps': [
                    {'range': [0, thresholds['temperature']['safe']], 'color': "white"},
                    {'range': [thresholds['temperature']['safe'], thresholds['temperature']['danger']], 'color': "lightyellow"}
                ],
                'threshold': {
                    'line': {'color': "red", 'width': 4},
                    'thickness': 0.75,
                    'value': thresholds['temperature']['safe']}
            }
        ))
        fig.update_layout(margin=dict(t=50, b=10))
        st.plotly_chart(fig, use_container_width=True)
        
    st.caption(f"Last updated: {datetime.now().strftime('%H:%M:%S')}")

# Tab 2: Leak predictions
with tab2:
    st.header("Leak Detection Analysis")
    sample = df.sample(1)
    features = scaler.transform(sample[['pressure', 'flow_rate', 'temperature']])
    proba = leak_model.predict_proba(features)[0][1]
    
    fig = go.Figure(go.Indicator(
        mode = "gauge+number",
        value = proba*100,
        domain = {'x': [0, 1], 'y': [0, 1]},
        title = {'text': "Leak Probability (%)"},
        gauge = {
            'axis': {'range': [0, 100], 'tickwidth': 1},
            'bar': {'color': "#2ca02c"},
            'steps': [
                {'range': [0, 40], 'color': "white"},
                {'range': [40, 70], 'color': "lightyellow"},
                {'range': [70, 100], 'color': "red"}
            ],
        }
    ))
    st.plotly_chart(fig, use_container_width=True)
    
    if proba > 0.4:
        st.error(f'ALERT: {proba:.2%} leak risk', icon="⚠️")
        # st.progress(proba)
    else:
        st.success('No leak detected', icon="✅")

# Tab 3: Burst predictions
with tab3:
    st.header("Pipe Burst Prediction")
    sample = df.sample(1)
    
    # Ensure correct column names match your model training
    features = scaler.transform(sample[['pressure', 'flow_rate', 'temperature']])
    proba = burst_model.predict_proba(features)[0][1]
    
    fig = go.Figure(go.Indicator(
        mode = "gauge+number",
        value = proba*100,
        domain = {'x': [0, 1], 'y': [0, 1]},
        title = {'text': "Burst Probability (%)"},
        gauge = {
            'axis': {'range': [0, 100], 'tickwidth': 1},
            'bar': {'color': "#2ca02c"},
            'steps': [
                {'range': [0, 40], 'color': "white"},
                {'range': [40, 70], 'color': "lightyellow"},
                {'range': [70, 100], 'color': "red"}
            ],
        }
    ))
    st.plotly_chart(fig, use_container_width=True)
    
    if proba > 0.4:
        st.error(f'ALERT: {proba:.2%} burst risk', icon="⚠️")
        # st.progress(proba)
    else:
        st.success('No burst detected', icon="✅")