# utils.py
import joblib
import pandas as pd
import numpy as np
from datetime import datetime

def save_model_artifacts(model, scaler, feature_names, metrics, model_dir='models'):
    """
    Save all model artifacts using joblib
    
    Args:
        model: Trained model object
        scaler: Fitted scaler object
        feature_names: List of feature names
        metrics: Dictionary of training metrics
        model_dir: Directory to save artifacts
    """
    import os
    os.makedirs(model_dir, exist_ok=True)
    
    # Save with timestamps
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    artifacts = {
        'model': model,
        'scaler': scaler,
        'feature_names': feature_names,
        'metrics': metrics,
        'timestamp': timestamp
    }
    
    # Save individual artifacts
    joblib.dump(model, f'{model_dir}/model_{timestamp}.pkl')
    joblib.dump(scaler, f'{model_dir}/scaler_{timestamp}.pkl')
    joblib.dump(feature_names, f'{model_dir}/features_{timestamp}.pkl')
    joblib.dump(metrics, f'{model_dir}/metrics_{timestamp}.pkl')
    
    # Save complete bundle
    joblib.dump(artifacts, f'{model_dir}/model_bundle_{timestamp}.pkl')
    
    # Update latest references
    joblib.dump(model, f'{model_dir}/blueberry_yield_predictor.pkl')
    joblib.dump(scaler, f'{model_dir}/scaler.pkl')
    joblib.dump(feature_names, f'{model_dir}/feature_names.pkl')
    
    print(f" Model artifacts saved with timestamp: {timestamp}")

def load_model_bundle(model_path):
    """
    Load complete model bundle
    
    Args:
        model_path: Path to model bundle file
    
    Returns:
        dict: Model artifacts
    """
    try:
        artifacts = joblib.load(model_path)
        print(f" Model bundle loaded (trained on: {artifacts['timestamp']})")
        return artifacts
    except Exception as e:
        print(f" Error loading model bundle: {e}")
        return None

def get_model_info():
    """Get information about saved models"""
    try:
        metrics = joblib.load('models/training_metrics.pkl')
        feature_names = joblib.load('models/feature_names.pkl')
        
        return {
            'performance': metrics,
            'features': feature_names,
            'num_features': len(feature_names)
        }
    except FileNotFoundError:
        return {"error": "Model not found. Please train the model first."}

def batch_predict(csv_file_path):
    """
    Make batch predictions from CSV file
    
    Args:
        csv_file_path: Path to CSV file with features
    
    Returns:
        pandas.DataFrame: Predictions
    """
    try:
        # Load data
        data = pd.read_csv(csv_file_path)
        
        # Load model
        predictor = BlueberryYieldPredictor()
        if predictor.model is None:
            return {"error": "Model not loaded"}
        
        # Make predictions
        predictions = []
        for _, row in data.iterrows():
            result = predictor.predict_yield(row.to_dict())
            if "error" not in result:
                predictions.append(result['predicted_yield'])
            else:
                predictions.append(None)
        
        # Add predictions to data
        data['predicted_yield'] = predictions
        
        return data
        
    except Exception as e:
        return {"error": f"Batch prediction failed: {str(e)}"}