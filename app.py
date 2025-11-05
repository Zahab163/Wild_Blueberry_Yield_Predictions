# app.py
import streamlit as st
import pandas as pd
import numpy as np
import joblib
from predict import BlueberryYieldPredictor

# Page configuration
st.set_page_config(
    page_title="Wild Blueberry Yield Predictor",
    page_icon="🌿",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        color: #2E8B57;
        text-align: center;
        margin-bottom: 2rem;
    }
    .prediction-card {
        background-color: #f0f8f0;
        padding: 2rem;
        border-radius: 10px;
        border-left: 5px solid #2E8B57;
    }
    .feature-importance {
        background-color: #fffaf0;
        padding: 1rem;
        border-radius: 10px;
        margin: 1rem 0;
    }
</style>
""", unsafe_allow_html=True)

def main():
    # Header
    st.markdown('<h1 class="main-header"> Wild Blueberry Yield Predictor</h1>', 
                unsafe_allow_html=True)
    
    st.write("""
    Predict blueberry yield based on environmental and agricultural factors to optimize your farming practices.
    Adjust the parameters in the sidebar and see how they affect the predicted yield.
    """)
    
    # Initialize predictor
    @st.cache_resource
    def load_predictor():
        return BlueberryYieldPredictor()
    
    predictor = load_predictor()
    
    # Sidebar for input parameters
    st.sidebar.header(" Agricultural Parameters")
    
    # Group related parameters
    st.sidebar.subheader(" Pollinator Density (bees/m²/min)")
    honeybee = st.sidebar.slider("Honeybee", 0.1, 1.0, 0.6, 0.05)
    bumbles = st.sidebar.slider("Bumblebee", 0.05, 0.8, 0.4, 0.05)
    andrena = st.sidebar.slider("Andrena Bee", 0.01, 0.5, 0.2, 0.05)
    osmia = st.sidebar.slider("Osmia Bee", 0.01, 0.4, 0.15, 0.05)
    
    st.sidebar.subheader(" Temperature Ranges (°C)")
    max_upper_temp = st.sidebar.slider("Max Upper Temp", 20.0, 40.0, 30.0, 1.0)
    min_upper_temp = st.sidebar.slider("Min Upper Temp", 10.0, 30.0, 20.0, 1.0)
    avg_upper_temp = st.sidebar.slider("Avg Upper Temp", 15.0, 35.0, 25.0, 1.0)
    
    st.sidebar.subheader("Rainfall")
    raining_days = st.sidebar.slider("Raining Days", 0.0, 15.0, 5.0, 1.0)
    avg_raining_days = st.sidebar.slider("Avg Raining Days", 0.0, 10.0, 4.0, 1.0)
    
    st.sidebar.subheader(" Fruit Characteristics")
    clonesize = st.sidebar.slider("Clone Size (m²)", 10.0, 40.0, 25.0, 1.0)
    fruitset = st.sidebar.slider("Fruit Set Ratio", 0.1, 1.0, 0.7, 0.05)
    fruitmass = st.sidebar.slider("Fruit Mass (g)", 0.1, 1.0, 0.6, 0.05)
    seeds = st.sidebar.slider("Number of Seeds", 10.0, 50.0, 35.0, 1.0)
    
    # Create input dictionary
    input_data = {
        'clonesize': clonesize,
        'honeybee': honeybee,
        'bumbles': bumbles,
        'andrena': andrena,
        'osmia': osmia,
        'MaxOfUpperTRange': max_upper_temp,
        'MinOfUpperTRange': min_upper_temp,
        'AverageOfUpperTRange': avg_upper_temp,
        'MaxOfLowerTRange': 15.0,  # Default values
        'MinOfLowerTRange': 8.0,
        'AverageOfLowerTRange': 11.5,
        'RainingDays': raining_days,
        'AverageRainingDays': avg_raining_days,
        'fruitset': fruitset,
        'fruitmass': fruitmass,
        'seeds': seeds
    }
    
    # Prediction button
    if st.sidebar.button(" Predict Yield", type="primary"):
        with st.spinner('Calculating yield prediction...'):
            result = predictor.predict_yield(input_data)
        
        if "error" not in result:
            # Display results in a nice card
            st.markdown('<div class="prediction-card">', unsafe_allow_html=True)
            
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.metric(
                    label="Predicted Yield",
                    value=f"{result['predicted_yield']} {result['units']}",
                    delta="Optimal" if result['predicted_yield'] > 40 else "Good" if result['predicted_yield'] > 30 else "Needs Improvement"
                )
            
            with col2:
                st.metric(
                    label="Model Confidence",
                    value=f"{result['model_performance']['r2_score'] * 100:.1f}%",
                    delta="R² Score"
                )
            
            with col3:
                st.metric(
                    label="Prediction Accuracy",
                    value=f"±{result['model_performance']['rmse']} {result['units']}",
                    delta="RMSE"
                )
            
            # Interpretation
            st.info(f" **Interpretation**: {result['interpretation']}")
            
            st.markdown('</div>', unsafe_allow_html=True)
            
            # Recommendations based on prediction
            st.subheader(" Recommendations")
            if result['predicted_yield'] > 50:
                st.success("""
                **Excellent Conditions!** Maintain your current practices:
                - Continue optimal pollinator management
                - Monitor temperature ranges regularly
                - Maintain current irrigation schedule
                """)
            elif result['predicted_yield'] > 35:
                st.warning("""
                **Good Performance - Room for Improvement:**
                - Consider increasing pollinator habitats
                - Optimize temperature control in greenhouses
                - Review fruit set optimization techniques
                """)
            else:
                st.error("""
                **Needs Significant Improvement:**
                - Implement pollinator attraction strategies
                - Review temperature management systems
                - Optimize clone size and spacing
                - Consult agricultural expert
                """)
        else:
            st.error(f"Prediction error: {result['error']}")
    
    # Model information section
    st.sidebar.markdown("---")
    st.sidebar.subheader(" Model Information")
    
    if predictor.metrics:
        st.sidebar.write(f"**Model Type**: {predictor.metrics.get('model_type', 'XGBoost')}")
        st.sidebar.write(f"**R² Score**: {predictor.metrics.get('r2_score', 0):.3f}")
        st.sidebar.write(f"**Features Used**: {len(predictor.feature_names)}")
    
    # Main area additional information
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader(" Feature Importance")
        st.write("""
        Top factors affecting blueberry yield:
        1. **Honeybee Density** - Critical for pollination
        2. **Fruit Set Ratio** - Direct yield indicator
        3. **Clone Size** - Plant capacity
        4. **Temperature Ranges** - Growth optimization
        5. **Bumblebee Density** - Secondary pollinators
        """)
    
    with col2:
        st.subheader(" Farming Tips")
        st.write("""
        - Maintain diverse pollinator populations
        - Monitor temperature fluctuations
        - Optimize plant spacing and clone size
        - Implement proper irrigation based on rainfall
        - Regular soil testing and nutrient management
        """)

if __name__ == "__main__":
    main()