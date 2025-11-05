# predict.py
import joblib
import pandas as pd
import numpy as np

class BlueberryYieldPredictor:
    def __init__(self):
        self.model = None
        self.scaler = None
        self.feature_names = None
        self.metrics = None
        self.load_model()
    
    def load_model(self):
        """Load the trained model and artifacts"""
        try:
            self.model = joblib.load('models/blueberry_yield_predictor.pkl')
            self.scaler = joblib.load('models/scaler.pkl')
            self.feature_names = joblib.load('models/feature_names.pkl')
            self.metrics = joblib.load('models/training_metrics.pkl')
            print(" Model loaded successfully!")
        except FileNotFoundError as e:
            print(f" Error loading model: {e}")
            print(" Please run train_model.py first to train and save the model.")
    
    def predict_yield(self, input_data):
        """
        Predict blueberry yield for given input data
        
        Args:
            input_data (dict or pandas.DataFrame): Input features
        
        Returns:
            dict: Prediction results
        """
        if self.model is None:
            return {"error": "Model not loaded. Please train the model first."}
        
        try:
            # Convert input to DataFrame
            if isinstance(input_data, dict):
                input_df = pd.DataFrame([input_data])
            else:
                input_df = input_data
            
            # Ensure all required features are present
            missing_features = set(self.feature_names) - set(input_df.columns)
            if missing_features:
                return {"error": f"Missing features: {missing_features}"}
            
            # Reorder columns to match training data
            input_df = input_df[self.feature_names]
            
            # Scale features
            input_scaled = self.scaler.transform(input_df)
            
            # Make prediction
            prediction = self.model.predict(input_scaled)[0]
            
            # Interpret prediction
            interpretation = self.interpret_prediction(prediction)
            
            return {
                "predicted_yield": round(prediction, 2),
                "units": "kg/hectare",
                "interpretation": interpretation,
                "model_performance": {
                    "r2_score": round(self.metrics['r2_score'], 4),
                    "rmse": round(self.metrics['rmse'], 2)
                }
            }
            
        except Exception as e:
            return {"error": f"Prediction failed: {str(e)}"}
    
    def interpret_prediction(self, yield_value):
        """Interpret the yield prediction"""
        if yield_value > 50:
            return "Excellent yield potential! Conditions are optimal."
        elif yield_value > 35:
            return "Good yield potential. Consider minor optimizations."
        elif yield_value > 25:
            return "Average yield potential. Review farming practices."
        else:
            return "Low yield potential. Significant improvements needed."
    
    def get_model_info(self):
        """Get information about the trained model"""
        if self.metrics:
            return self.metrics
        return {"error": "Model information not available."}

# Example usage
if __name__ == "__main__":
    # Initialize predictor
    predictor = BlueberryYieldPredictor()
    
    # Example prediction
    sample_input = {
        'clonesize': 25.0,
        'honeybee': 0.6,
        'bumbles': 0.4,
        'andrena': 0.2,
        'osmia': 0.15,
        'MaxOfUpperTRange': 30.0,
        'MinOfUpperTRange': 20.0,
        'AverageOfUpperTRange': 25.0,
        'MaxOfLowerTRange': 18.0,
        'MinOfLowerTRange': 12.0,
        'AverageOfLowerTRange': 15.0,
        'RainingDays': 5.0,
        'AverageRainingDays': 4.0,
        'fruitset': 0.7,
        'fruitmass': 0.6,
        'seeds': 35.0
    }
    
    print("\n Making sample prediction...")
    result = predictor.predict_yield(sample_input)
    
    if "error" not in result:
        print(f" Prediction Results:")
        print(f"   Predicted Yield: {result['predicted_yield']} {result['units']}")
        print(f"   Interpretation: {result['interpretation']}")
        print(f"   Model R² Score: {result['model_performance']['r2_score']}")
    else:
        print(f" Error: {result['error']}")