from pathlib import Path  # Changed from matplotlib.path
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from faker import Faker
from datetime import datetime, timedelta
import joblib
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.pipeline import make_pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import cross_val_score
from pathlib import Path

fake = Faker()
model_path = str(Path(__file__).parent)
print(model_path[-13:])

# Harare bounding coordinates
HARARE_BOUNDS = {
    'min_lat': -17.93,
    'max_lat': -17.70,
    'min_lon': 30.95,
    'max_lon': 31.15
}

# Updated to use relative path
ROOT = model_path[-13:]

def generate_water_data(num_samples=1000):
    """Generate synthetic water data with spatial features"""
    data = []
    base_pressure = 3.2
    base_flow = 125.0
    base_temp = 17.4
    
    pipe_materials = ['PVC', 'Cast Iron', 'HDPE', 'Galvanized Iron', 'Asbestos Cement']
    material_probs = [0.5, 0.2, 0.2, 0.05, 0.05]
    
    for _ in range(num_samples):
        lat = np.random.uniform(HARARE_BOUNDS['min_lat'], HARARE_BOUNDS['max_lat'])
        lon = np.random.uniform(HARARE_BOUNDS['min_lon'], HARARE_BOUNDS['max_lon'])
        
        pressure = np.clip(np.random.normal(base_pressure, 0.5), 0.9, 4.0)
        flow_rate = np.clip(np.random.normal(base_flow, 44.0), 50.0, 330.0)
        temp = np.clip(np.random.normal(base_temp, 4.3), 10.0, 25.0)
        
        risk_factor = 1 + (lat + 17.8) * 2
        leak = int(np.random.random() < (0.02 * risk_factor))
        burst = int(np.random.random() < (0.01 * risk_factor))
        
        if leak:
            pressure *= 0.8
            flow_rate *= 1.2
        if burst:
            pressure *= 0.5
            flow_rate *= 1.5
            
        data.append({
            "timestamp": datetime.now() - timedelta(minutes=np.random.randint(0, 1440)),
            "pressure": pressure,
            "flow_rate": flow_rate,
            "temperature": temp,
            "latitude": lat,
            "longitude": lon,
            "pipe_age": np.random.randint(0, 30),
            "pipe_material": np.random.choice(pipe_materials, p=material_probs),
            "leak_status": leak,
            "burst_status": burst,
            "pipe_id": f"ZW-{np.random.randint(1000, 9999)}"
        })
    
    return pd.DataFrame(data)

def load_and_preprocess_data(df):
    """Preprocess data and handle class imbalance"""
    df = df.dropna()
    
    # Basic models features
    X = df[['pressure', 'flow_rate', 'temperature']]
    y_leak = df['leak_status']
    y_burst = df['burst_status']
    
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    def random_oversample(X, y):
        unique, counts = np.unique(y, return_counts=True)
        max_count = max(counts)
        
        resampled_X, resampled_y = [], []
        
        for class_val in unique:
            class_indices = np.where(y == class_val)[0]
            current_count = len(class_indices)
            
            if current_count < max_count:
                indices_to_add = np.random.choice(class_indices, 
                                                 size=max_count-current_count, 
                                                 replace=True)
                resampled_X.append(X[indices_to_add])
                resampled_y.append(y[indices_to_add])
            
            resampled_X.append(X[class_indices])
            resampled_y.append(y[class_indices])
        
        return np.vstack(resampled_X), np.concatenate(resampled_y)
    
    # Apply oversampling
    X_res_leak, y_leak_res = random_oversample(X_scaled, y_leak)
    X_res_burst, y_burst_res = random_oversample(X_scaled, y_burst)
    
    # Split data
    X_train_leak, X_test_leak, y_leak_train, y_leak_test = train_test_split(
        X_res_leak, y_leak_res, test_size=0.2, random_state=42
    )
    
    X_train_burst, X_test_burst, y_burst_train, y_burst_test = train_test_split(
        X_res_burst, y_burst_res, test_size=0.2, random_state=42
    )
    
    return {
        'X_train_leak': X_train_leak, 'X_test_leak': X_test_leak,
        'y_leak_train': y_leak_train, 'y_leak_test': y_leak_test,
        'X_train_burst': X_train_burst, 'X_test_burst': X_test_burst,
        'y_burst_train': y_burst_train, 'y_burst_test': y_burst_test,
        'scaler': scaler,
        'df': df  # Include full dataframe for spatial model
    }

def train_and_evaluate_models(data):
    """Train and evaluate all models"""
    model_params = {
        'n_estimators': 50,
        'max_depth': 5,
        'min_samples_split': 10,
        'random_state': 42,
        'class_weight': 'balanced'
    }
    
    # Basic models
    leak_model = RandomForestClassifier(**model_params)
    burst_model = RandomForestClassifier(**model_params)
    
    # Spatial model
    spatial_features = ['pressure', 'flow_rate', 'temperature',
                      'latitude', 'longitude', 'pipe_age', 'pipe_material']
    
    preprocessor = ColumnTransformer([
        ('num', StandardScaler(), ['pressure', 'flow_rate', 'temperature', 'pipe_age']),
        ('cat', OneHotEncoder(), ['pipe_material'])
    ])
    
    spatial_model = make_pipeline(
        preprocessor,
        RandomForestClassifier(**model_params)
    )

    # Train and validate
    print("Training models...")
    for model, X, y, name in [
        (leak_model, data['X_train_leak'], data['y_leak_train'], "Leak"),
        (burst_model, data['X_train_burst'], data['y_burst_train'], "Burst"),
        (spatial_model, data['df'][spatial_features], data['df']['leak_status'], "Spatial")
    ]:
        scores = cross_val_score(model, X, y, cv=5, scoring='f1')
        print(f'Model scores: {scores}')
        model.fit(X, y)
        print(f"{name} Model - Mean CV F1: {np.mean(scores):.2f}")

    # Save models
    model_dir = ROOT / "ml_models"
    model_dir.mkdir(exist_ok=True)
    
    joblib.dump(leak_model, model_dir/'leak_model.sav')
    joblib.dump(burst_model, model_dir/'burst_model.sav')
    joblib.dump(spatial_model, model_dir/'spatial_model.sav')
    joblib.dump(data['scaler'], model_dir/'scaler.sav')
    
    return leak_model, burst_model, spatial_model

# Main execution
if __name__ == "__main__":
    print('Generating dataset...')
    df = generate_water_data(10000)  # Reduced from 100k for quicker testing
    
    print('Saving dataset...')
    (ROOT / "datasets").mkdir(exist_ok=True)
    df.to_csv(ROOT/"datasets"/"water_data.csv", index=False)
    
    print('Preprocessing data...')
    training_data = load_and_preprocess_data(df)
    
    print('Training models...')
    models = train_and_evaluate_models(training_data)