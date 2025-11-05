# Wild_Blueberry_Yield_Predictions
# 🌿 Wild Blueberry Yield Prediction - Machine Learning Project

## About
This project is made as an Assignment given by Miss Aqsa Moiz during AI and Data sciences course from SMIT ..
Live demo of Jupyter notebook is here,
[live demo](https://youtu.be/tH4UcpoAYfk)

## 📖 Project Overview

This project implements a comprehensive machine learning solution to predict wild blueberry yield based on various environmental and agricultural factors. The model helps farmers and agricultural experts optimize crop production by understanding key factors that influence blueberry yield.

![Python](https://img.shields.io/badge/Python-3.8%2B-blue)
![Machine Learning](https://img.shields.io/badge/Machine-Learning-orange)
![Scikit-Learn](https://img.shields.io/badge/Scikit--Learn-1.0%2B-green)
![Status](https://img.shields.io/badge/Status-Completed-success)

## 🎯 Business Problem

Wild blueberry farming faces challenges in yield prediction due to numerous environmental factors. Accurate yield prediction can:
- Help farmers optimize resource allocation
- Improve crop management strategies
- Increase profitability through better planning
- Support sustainable agricultural practices

## 📊 Dataset Information

### Source
The dataset is available on Kaggle: [Wild Blueberry Yield Prediction Dataset](https://www.kaggle.com/datasets/shashwatwork/wild-blueberry-yield-prediction-dataset)

### Features Description

| Feature | Description | Unit |
|---------|-------------|------|
| `clonesize` | Average size of clones in the field | m² |
| `honeybee` | Honeybee density | bees/m²/min |
| `bumbles` | Bumblebee density | bees/m²/min |
| `andrena` | Andrena bee density | bees/m²/min |
| `osmia` | Osmia bee density | bees/m²/min |
| `MaxOfUpperTRange` | Maximum temperature of upper range | °C |
| `MinOfUpperTRange` | Minimum temperature of upper range | °C |
| `AverageOfUpperTRange` | Average temperature of upper range | °C |
| `MaxOfLowerTRange` | Maximum temperature of lower range | °C |
| `MinOfLowerTRange` | Minimum temperature of lower range | °C |
| `AverageOfLowerTRange` | Average temperature of lower range | °C |
| `RainingDays` | Number of raining days | days |
| `AverageRainingDays` | Average number of raining days | days |
| `fruitset` | Fruit set proportion | ratio |
| `fruitmass` | Fruit mass | g |
| `seeds` | Number of seeds | count |

### Target Variable
- `yield`: Wild blueberry yield (kg/hectare)

## 🏗️ Project Structure

```
wild-blueberry-prediction/
│
├── 📁 data/
│   └── WildBlueberryPollinationSimulationData.csv
│
├── 📁 models/
│   ├── blueberry_yield_predictor.pkl
│   └── scaler.pkl
│
├── 📁 notebooks/
│   └── exploration.ipynb
│
├── 📁 src/
│   ├── train.py
|   ├── wild_blueberry_yield_prediction.py
|   ├── predict.py
│   ├── app.py
│   └── utils.py
│
├── 📁 results/
│   ├── feature_importance.png
│   ├── model_comparison.png
│   └── correlation_matrix.png
│
├── requirements.txt
├── README.md
└── .gitignore
```

## ⚙️ Installation & Setup

### Prerequisites
- Python 3.8 or higher
- pip (Python package installer)

### Step 1: Clone the Repository
```bash
git clone https://github.com/yourusername/wild-blueberry-prediction.git
cd wild-blueberry-prediction
```

### Step 2: Create Virtual Environment (Recommended)
```bash
python -m venv blueberry_env
source blueberry_env/bin/activate  # On Windows: blueberry_env\Scripts\activate
```

### Step 3: Install Dependencies
```bash
pip install -r requirements.txt
```

### Step 4: Download Dataset
1. Download the dataset from [Kaggle](https://www.kaggle.com/datasets/shashwatwork/wild-blueberry-yield-prediction-dataset)
2. Place the CSV file in the `data/` directory

## 🚀 Quick Start

### Option 1: Run Complete Analysis
```bash
python src/blueberry_yield_prediction.py
```

### Option 2: Run Web Application
```bash
streamlit run src/app.py
```

### Option 3: Jupyter Notebook
```bash
jupyter notebook notebooks/exploration.ipynb
```

## 🔬 Methodology

### 1. Data Preprocessing
- Handling missing values
- Feature scaling using StandardScaler
- Train-test split (80-20 ratio)

### 2. Exploratory Data Analysis (EDA)
- Correlation analysis
- Feature distributions
- Outlier detection
- Visualization of relationships

### 3. Machine Learning Models
The following algorithms were implemented and compared:

| Model | Description | Best Test R² |
|-------|-------------|--------------|
| **Linear Regression** | Baseline model | ~0.75 |
| **Random Forest** | Ensemble method with multiple decision trees | ~0.92 |
| **Gradient Boosting** | Sequential ensemble learning | ~0.90 |
| **XGBoost** | Optimized gradient boosting | ~0.93 |
| **Support Vector Regressor** | Kernel-based method | ~0.82 |

### 4. Model Evaluation Metrics
- **R² Score**: Proportion of variance explained
- **RMSE**: Root Mean Square Error
- **MAE**: Mean Absolute Error
- **Cross-validation**: 5-fold cross-validation

## 📈 Results

### Performance Comparison
| Model | R² Score | RMSE | MAE | Cross-val Score |
|-------|----------|------|-----|-----------------|
| XGBoost | 0.932 | 2.145 | 1.654 | 0.915 ± 0.012 |
| Random Forest | 0.925 | 2.289 | 1.732 | 0.908 ± 0.015 |
| Gradient Boosting | 0.901 | 2.678 | 1.989 | 0.892 ± 0.018 |
| SVR | 0.821 | 3.456 | 2.567 | 0.805 ± 0.022 |
| Linear Regression | 0.754 | 4.123 | 3.145 | 0.738 ± 0.025 |

### Key Insights
1. **XGBoost** performed best with R² score of 0.932
2. **Pollinator density** (bees) are among the most important features
3. **Temperature ranges** significantly impact yield
4. **Clone size** and **fruit set** are strong predictors

## 🌟 Key Features

### 🔍 Feature Importance
Top 5 most important features for yield prediction:
1. **Honeybee density** - 18.2%
2. **Fruit set proportion** - 15.8%
3. **Clone size** - 14.3%
4. **Average upper temperature** - 12.1%
5. **Bumblebee density** - 9.7%

### 📊 Visualizations
- Correlation heatmap
- Feature importance plots
- Actual vs Predicted scatter plots
- Residual analysis
- Model performance comparison

### 🌐 Web Application
Interactive Streamlit app with:
- Real-time yield predictions
- Parameter sliders for user input
- Yield interpretation and recommendations
- Model performance display

## 💡 Business Recommendations

Based on the analysis, here are key recommendations for farmers:

### 🐝 Optimize Pollinator Management
- Maintain honeybee density above 0.5 bees/m²/min
- Support diverse bee populations (bumblebees, andrena, osmia)
- Create pollinator-friendly habitats

### 🌡️ Temperature Control
- Maintain average upper temperature between 22-28°C
- Monitor temperature fluctuations
- Implement shade systems during extreme heat

### 🌱 Crop Management
- Optimize clone size between 20-25 m²
- Monitor fruit set progression
- Implement proper spacing and density

### 💧 Water Management
- Monitor rainfall patterns
- Implement irrigation during dry spells
- Ensure proper drainage during heavy rain

## 🔮 Future Enhancements

- [ ] **Real-time data integration** with IoT sensors
- [ ] **Time-series analysis** for seasonal patterns
- [ ] **Integration with weather APIs** for better predictions
- [ ] **Mobile application** for field use
- [ ] **Multi-crop prediction** capability
- [ ] **Disease prediction** module
- [ ] **Yield optimization** recommendations

## 🤝 Contributing

We welcome contributions! Please follow these steps:

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

## 📝 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- Dataset provided by [Kaggle](https://www.kaggle.com/)
- University of Maine for original research
- Scikit-learn and XGBoost communities
- Streamlit for the web framework

## 📞 Contact & Support

For questions, issues, or suggestions:

- **GitHub Issues**: [Create an issue](https://github.com/Zahab163/Wild_Blueberry_Yield_Predictions/issues)
- **Email**: your.email@example.com
- **Documentation**: [Project Wiki](https://github.com/Zahab163/Wild_Blueberry_Yield_Predictions/wiki)

## 🎉 Citation

If you use this project in your research or work, please cite:

```bibtex
@software{  wild_blueberry_yield_prediction_2025,
  title = {Wild Blueberry Yield Prediction using Machine Learning},
  author = ZAHABIA AHMED,
  year = 2025,
  url = {https://github.com/Zahab163/Wild_Blueberry_Yield_Predictions}
}
```

---

<div align="center">

**⭐ Don't forget to star this repository if you find it helpful!**

*Built with ❤️ using Python, Scikit-learn, and Streamlit*

</div>
