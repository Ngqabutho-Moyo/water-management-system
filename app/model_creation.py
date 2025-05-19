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

fake = Faker()

def generate_water_data(num_samples=1000):
    data = []
    base_pressure = 3.2  # Mean pressure (bar)
    base_flow = 125.0    # Mean flow rate (L/s)
    base_temp = 17.4     # Mean temperature (°C)
    
    for _ in range(num_samples):
        # Simulate normal fluctuations
        pressure = np.random.normal(base_pressure, 0.5)
        flow_rate = np.random.normal(base_flow, 44.0)
        temp = np.random.normal(base_temp, 4.3)
        
        # Introduce leaks/bursts (rare events)
        leak = 1 if np.random.random() < 0.02 else 0
        burst = 1 if np.random.random() < 0.01 else 0
        
        # Simulate sensor noise
        if leak:
            pressure *= 0.8  # Pressure drops during leaks
            flow_rate *= 1.2  # Flow increases due to leakage
        
        if burst:
            pressure *= 0.5  # Severe drop
            flow_rate *= 1.5  # Big spike
            
        data.append({
            "timestamp": datetime.now() - timedelta(minutes=np.random.randint(0, 1440)),
            "pressure": max(0.9, min(4.0, pressure)),  # Clamp to realistic range
            "flow_rate": max(50.0, min(330.0, flow_rate)),
            "temperature": max(10.0, min(25.0, temp)),
            "leak_status": leak,
            "burst_status": burst
        })
    
    return pd.DataFrame(data)

df = generate_water_data(10000)
df.to_csv('datasets/water_data.csv', index=False)

def load_and_preprocess_data(df):    
    # Handle missing values if any
    df = df.dropna()
    
    # Separate features and targets
    X = df[['pressure', 'flow_rate', 'temperature']]
    y_leak = df['leak_status']
    y_burst = df['burst_status']
    
    # Scale features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # Custom oversampling function
    def random_oversample(X, y):
        # Count number of samples in each class
        unique, counts = np.unique(y, return_counts=True)
        max_count = max(counts)
        
        # For each class that needs oversampling
        resampled_X = []
        resampled_y = []
        
        for class_val in unique:
            # Get indices of current class
            class_indices = np.where(y == class_val)[0]
            current_count = len(class_indices)
            
            # If minority class
            if current_count < max_count:
                # Calculate how many samples to add
                num_to_add = max_count - current_count
                
                # Randomly select samples with replacement
                indices_to_add = np.random.choice(class_indices, size=num_to_add, replace=True)
                
                # Add to resampled data
                resampled_X.append(X[indices_to_add])
                resampled_y.append(y[indices_to_add])
            
            # Always add original samples
            resampled_X.append(X[class_indices])
            resampled_y.append(y[class_indices])
        
        # Combine all samples
        return np.vstack(resampled_X), np.concatenate(resampled_y)
    
    # Apply oversampling to both targets
    X_res_leak, y_leak_res = random_oversample(X_scaled, y_leak)
    X_res_burst, y_burst_res = random_oversample(X_scaled, y_burst)
    
    # Split leak data
    X_train_leak, X_test_leak, y_leak_train, y_leak_test = train_test_split(
        X_res_leak, y_leak_res, test_size=0.2, random_state=42
    )
    
    # Split burst data
    X_train_burst, X_test_burst, y_burst_train, y_burst_test = train_test_split(
        X_res_burst, y_burst_res, test_size=0.2, random_state=42
    )
    
    return {
        'X_train_leak': X_train_leak, 'X_test_leak': X_test_leak,
        'y_leak_train': y_leak_train, 'y_leak_test': y_leak_test,
        'X_train_burst': X_train_burst, 'X_test_burst': X_test_burst,
        'y_burst_train': y_burst_train, 'y_burst_test': y_burst_test,
        'scaler': scaler
    }
    
training_data=load_and_preprocess_data(df)

def train_and_evaluate_models(data):
    # 1. Initialize models with stricter parameters to prevent overfitting
    leak_model = RandomForestClassifier(
        n_estimators=50,  # Reduced from 100
        max_depth=5,      # Added depth limit
        min_samples_split=10,  # Increased from default 2
        random_state=42
    )
    
    burst_model = RandomForestClassifier(
        n_estimators=50,
        max_depth=5,
        min_samples_split=10,
        random_state=42
    )

    # 2. Add cross-validation
    from sklearn.model_selection import cross_val_score
    leak_cv_scores = cross_val_score(leak_model, data['X_train_leak'], data['y_leak_train'], 
                                   cv=5, scoring='f1')
    print(f"Leak Model CV F1 Scores: {leak_cv_scores}")
    print(f"Mean CV F1: {np.mean(leak_cv_scores):.2f}")

    # 3. Train and evaluate with additional diagnostics
    leak_model.fit(data['X_train_leak'], data['y_leak_train'])
    leak_pred = leak_model.predict(data['X_test_leak'])
    
    burst_model.fit(data['X_train_burst'], data['y_burst_train'])
    burst_pred = burst_model.predict(data['X_test_burst'])

    # 4. Enhanced evaluation
    def extended_evaluation(y_true, y_pred, model_name):
        print(f"\n{model_name} Performance:")
        print(classification_report(y_true, y_pred))
        
        # Confusion matrix plot
        '''
        cm = confusion_matrix(y_true, y_pred)
        plt.figure()
        plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
        plt.title(f'{model_name} Confusion Matrix')
        plt.colorbar()
        plt.xticks([0,1], ['No', 'Yes'])
        plt.yticks([0,1], ['No', 'Yes'])
        plt.xlabel('Predicted')
        plt.ylabel('True')
        
        # Add feature importance plot
        plt.figure()
        if hasattr(leak_model, 'feature_importances_'):
            importances = leak_model.feature_importances_
            plt.barh(range(len(importances)), importances, tick_label=['Pressure', 'Flow Rate', 'Temperature'])
            plt.title(f'{model_name} Feature Importance')
        plt.show()
        '''
    extended_evaluation(data['y_leak_test'], leak_pred, "Leak Detection")
    extended_evaluation(data['y_burst_test'], burst_pred, "Burst Prediction")

    # 5. Save models only if not overfit
    if np.mean(leak_cv_scores) < 0.95:  # Reasonable threshold
        joblib.dump(leak_model, 'ml_models/leak_model.sav')
        joblib.dump(burst_model, 'ml_models/burst_model.sav')
        joblib.dump(data['scaler'], 'ml_models/scaler.sav')
    else:
        print("\nWARNING: Potential overfitting detected - models not saved!")
        print("Recommended actions:")
        print("- Check for data leakage")
        print("- Add more regularization")
        print("- Collect more diverse training data")

    return leak_model, burst_model

train_and_evaluate_models(training_data)