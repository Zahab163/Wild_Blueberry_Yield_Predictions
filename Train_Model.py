# train_model.py
import pandas as pd
import numpy as np
import joblib
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import xgboost as xgb
import os

def create_model_directory():
    """Create models directory if it doesn't exist"""
    os.makedirs('models', exist_ok=True)

def train_and_save_model():
    """Train the model and save using joblib"""
    print(" Starting model training pipeline...")
    
    # Load data
    try:
        df = pd.read_csv('data/WildBlueberryPollinationSimulationData.csv')
        print(" Data loaded successfully!")
    except FileNotFoundError:
        print(" Dataset not found. Please download from Kaggle.")
        return
    
    # Prepare features and target
    X = df.drop('yield', axis=1)
    y = df['yield']
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    
    # Feature scaling
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Train model (using XGBoost as example)
    print(" Training XGBoost model...")
    model = xgb.XGBRegressor(
        n_estimators=200,
        max_depth=6,
        learning_rate=0.1,
        subsample=0.9,
        random_state=42
    )
    
    model.fit(X_train_scaled, y_train)
    
    # Evaluate model
    y_pred = model.predict(X_test_scaled)
    r2 = r2_score(y_test, y_pred)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    
    print(f" Model trained successfully!")
    print(f" Model Performance:")
    print(f"   R² Score: {r2:.4f}")
    print(f"   RMSE: {rmse:.4f}")
    
    # Save artifacts using joblib
    create_model_directory()
    
    # Save model
    joblib.dump(model, 'models/blueberry_yield_predictor.pkl')
    print("Model saved as 'models/blueberry_yield_predictor.pkl'")
    
    # Save scaler
    joblib.dump(scaler, 'models/scaler.pkl')
    print("Scaler saved as 'models/scaler.pkl'")
    
    # Save feature names
    joblib.dump(list(X.columns), 'models/feature_names.pkl')
    print("Feature names saved as 'models/feature_names.pkl'")
    
    # Save training metrics
    metrics = {
        'r2_score': r2,
        'rmse': rmse,
        'mae': mean_absolute_error(y_test, y_pred),
        'feature_names': list(X.columns),
        'model_type': 'XGBoost'
    }
    joblib.dump(metrics, 'models/training_metrics.pkl')
    print("Training metrics saved!")
    
    return model, scaler, metrics

def load_saved_model():
    """Load the saved model and artifacts"""
    try:
        model = joblib.load('models/blueberry_yield_predictor.pkl')
        scaler = joblib.load('models/scaler.pkl')
        feature_names = joblib.load('models/feature_names.pkl')
        metrics = joblib.load('models/training_metrics.pkl')
        
        print(" All model artifacts loaded successfully!")
        return model, scaler, feature_names, metrics
    except FileNotFoundError as e:
        print(f" Error loading model: {e}")
        return None

if __name__ == "__main__":
    # Train and save model
    train_and_save_model()
    
    # Test loading
    print("\n Testing model loading...")
    artifacts = load_saved_model()
    
    if artifacts:
        model, scaler, feature_names, metrics = artifacts
        print(f" Loaded Model Info:")
        print(f"   Model Type: {metrics['model_type']}")
        print(f"   R² Score: {metrics['r2_score']:.4f}")
        print(f"   Features: {len(feature_names)}")
        print(f"   Feature Names: {feature_names}")