import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

# Set style for better visualizations
plt.style.use('seaborn-v0_8')
sns.set_palette("husl")

import kagglehub

# Download latest version
path = kagglehub.dataset_download("shashwatwork/wild-blueberry-yield-prediction-dataset")

print("Path to dataset files:", path)

print("=" * 50)
print("WILD BLUEBERRY YIELD PREDICTION ANALYSIS")
print("=" * 50)

def load_data():
    """
    Load the wild blueberry yield prediction dataset
    You can download it from:
    https://www.kaggle.com/datasets/shashwatwork/wild-blueberry-yield-prediction-dataset
    """
    try:
        df = pd.read_csv('WildBlueberryPollinationSimulationData.csv')
        print(" Dataset loaded successfully!")
        return df
    except FileNotFoundError:
        print(" Dataset file not found. Creating sample data for demonstration...")
        return create_sample_data()

def create_sample_data():
    """Create sample data if actual dataset is not available"""
    np.random.seed(42)
    n_samples = 1000

    data = {
        'clonesize': np.random.uniform(10, 30, n_samples),
        'honeybee': np.random.uniform(0.2, 0.8, n_samples),
        'bumbles': np.random.uniform(0.1, 0.6, n_samples),
        'andrena': np.random.uniform(0.05, 0.4, n_samples),
        'osmia': np.random.uniform(0.02, 0.3, n_samples),
        'MaxOfUpperTRange': np.random.uniform(25, 35, n_samples),
        'MinOfUpperTRange': np.random.uniform(15, 25, n_samples),
        'AverageOfUpperTRange': np.random.uniform(20, 30, n_samples),
        'MaxOfLowerTRange': np.random.uniform(10, 20, n_samples),
        'MinOfLowerTRange': np.random.uniform(5, 15, n_samples),
        'AverageOfLowerTRange': np.random.uniform(7, 17, n_samples),
        'RainingDays': np.random.uniform(0, 10, n_samples),
        'AverageRainingDays': np.random.uniform(2, 8, n_samples),
        'fruitset': np.random.uniform(0.3, 0.9, n_samples),
        'fruitmass': np.random.uniform(0.3, 0.8, n_samples),
        'seeds': np.random.uniform(20, 40, n_samples)
    }

    # Create yield based on some relationships (for demonstration)
    data['yield'] = (
        0.3 * data['clonesize'] +
        0.2 * data['honeybee'] * 50 +
        0.15 * data['bumbles'] * 50 +
        0.1 * data['andrena'] * 50 +
        0.08 * data['AverageOfUpperTRange'] +
        0.05 * data['fruitset'] * 100 +
        np.random.normal(0, 5, n_samples)
    )

    return pd.DataFrame(data)
# Load data
df = load_data()

# Display basic information
print(f"\n Dataset Shape: {df.shape}")
print(f"\n First 5 rows:")
df.head()

print(f"\n Dataset Info:")
df.info()

print(f"\n Statistical Summary:")
df.describe()

def perform_eda(df):
    """Perform comprehensive exploratory data analysis"""
    print("\n" + "="*50)
    print("EXPLORATORY DATA ANALYSIS")
    print("="*50)

    # Check for missing values
    print(f"\n Missing Values:")
    missing_data = df.isnull().sum()
    print(missing_data[missing_data > 0])

    # Correlation analysis
    plt.figure(figsize=(10, 8))
    correlation_matrix = df.corr()
    sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', center=0,
                fmt='.2f', linewidths=0.5)
    plt.title('Feature Correlation Matrix', fontsize=16, fontweight='bold')
    plt.tight_layout()
    plt.show()

    # Feature distributions
    numerical_features = df.select_dtypes(include=[np.number]).columns
    df[numerical_features].hist(bins=20, figsize=(20, 15))
    plt.suptitle('Distribution of Numerical Features', fontsize=16, fontweight='bold')
    plt.tight_layout()
    plt.show()

    perform_eda(df)

    # Distribution of target variable and relationships with top features
plt.figure(figsize=(15, 5))

plt.subplot(1, 3, 1)
sns.histplot(df['yield'], kde=True)
plt.title('Distribution of Blueberry Yield')
plt.xlabel('Yield')

plt.subplot(1, 3, 2)
sns.boxplot(y=df['yield'])
plt.title('Box Plot of Yield')

plt.subplot(1, 3, 3)
# Top 5 features correlated with yield (excluding yield itself)
correlation_matrix = df.corr()
top_features = correlation_matrix['yield'].abs().sort_values(ascending=False).index[1:6]
for feature in top_features:
    plt.scatter(df[feature], df['yield'], alpha=0.5, label=feature)
plt.legend()
plt.title('Top Features vs Yield')
plt.xlabel('Feature Values')
plt.ylabel('Yield')

plt.tight_layout()
plt.show()

from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV

print("\n" + "="*50)
print("DATA PREPROCESSING")
print("="*50)

# Separate features and target
X = df.drop('yield', axis=1)
y = df['yield']

# Handle categorical variables if any
categorical_cols = X.select_dtypes(include=['object']).columns
if len(categorical_cols) > 0:
    X = pd.get_dummies(X, columns=categorical_cols, drop_first=True)

# Split the data
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

print(f"Training set size: {X_train.shape}")
print(f"Testing set size: {X_test.shape}")

from sklearn.preprocessing import StandardScaler

# Feature scaling
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.svm import SVR
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.inspection import permutation_importance
import xgboost as xgb
import warnings
warnings.filterwarnings('ignore')

print("\n" + "="*50)
print("MODEL TRAINING AND EVALUATION")
print("="*50)

# Initialize models
models = {
    'Linear Regression': LinearRegression(),
    'Random Forest': RandomForestRegressor(n_estimators=100, random_state=42),
    'Gradient Boosting': GradientBoostingRegressor(n_estimators=100, random_state=42),
    'XGBoost': xgb.XGBRegressor(n_estimators=100, random_state=42),
    'SVR': SVR(kernel='rbf')
}

results = {}

def evaluate_model(model, X_train, X_test, y_train, y_test, model_name):
    """Evaluate a model and return metrics"""
    # Train model
    model.fit(X_train, y_train)

    # Predictions
    y_train_pred = model.predict(X_train)
    y_test_pred = model.predict(X_test)

    # Calculate metrics
    train_mae = mean_absolute_error(y_train, y_train_pred)
    test_mae = mean_absolute_error(y_test, y_test_pred)
    train_rmse = np.sqrt(mean_squared_error(y_train, y_train_pred))
    test_rmse = np.sqrt(mean_squared_error(y_test, y_test_pred))
    train_r2 = r2_score(y_train, y_train_pred)
    test_r2 = r2_score(y_test, y_test_pred)

    # Cross-validation
    cv_scores = cross_val_score(model, X_train, y_train, cv=5, scoring='r2')

    return {
        'model': model,
        'train_mae': train_mae,
        'test_mae': test_mae,
        'train_rmse': train_rmse,
        'test_rmse': test_rmse,
        'train_r2': train_r2,
        'test_r2': test_r2,
        'cv_mean': cv_scores.mean(),
        'cv_std': cv_scores.std(),
        'predictions': y_test_pred
    }

# Train and evaluate all models
for name, model in models.items():
    print(f"\n Training {name}...")
    results[name] = evaluate_model(model, X_train_scaled, X_test_scaled, y_train, y_test, name)

# Display results
results_df = pd.DataFrame({
    'Model': list(results.keys()),
    'Train R²': [results[name]['train_r2'] for name in results.keys()],
    'Test R²': [results[name]['test_r2'] for name in results.keys()],
    'Test RMSE': [results[name]['test_rmse'] for name in results.keys()],
    'Test MAE': [results[name]['test_mae'] for name in results.keys()],
    'CV R² Mean': [results[name]['cv_mean'] for name in results.keys()],
    'CV R² Std': [results[name]['cv_std'] for name in results.keys()]
})

print("\n MODEL PERFORMANCE COMPARISON:")
print(results_df.round(4))

plt.figure(figsize=(15, 10))

# Model performance comparison
plt.subplot(2, 2, 1)
models_list = list(results.keys())
test_r2_scores = [results[name]['test_r2'] for name in models_list]
bars = plt.bar(models_list, test_r2_scores, color=['skyblue', 'lightcoral', 'lightgreen', 'gold', 'violet'])
plt.title('Model Performance (Test R² Score)', fontweight='bold')
plt.ylabel('R² Score')
plt.xticks(rotation=45)
# Add value labels on bars
for bar, value in zip(bars, test_r2_scores):
    plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
             f'{value:.3f}', ha='center', va='bottom')

# Actual vs Predicted for best model
best_model_name = max(results, key=lambda x: results[x]['test_r2'])
best_model = results[best_model_name]['model']
best_predictions = results[best_model_name]['predictions']

plt.subplot(2, 2, 2)
plt.scatter(y_test, best_predictions, alpha=0.6)
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', lw=2)
plt.xlabel('Actual Yield')
plt.ylabel('Predicted Yield')
plt.title(f'Actual vs Predicted - {best_model_name}\n(R² = {results[best_model_name]["test_r2"]:.3f})')
plt.tight_layout()
plt.show()

# Residual plot
plt.subplot(2, 2, 3)
residuals = y_test - best_predictions
plt.scatter(best_predictions, residuals, alpha=0.6)
plt.axhline(y=0, color='r', linestyle='--')
plt.xlabel('Predicted Values')
plt.ylabel('Residuals')
plt.title('Residual Plot')
plt.tight_layout()
plt.show()

# Error distribution
plt.subplot(2, 2, 4)
sns.histplot(residuals, kde=True)
plt.title('Distribution of Residuals')
plt.xlabel('Residuals')

plt.tight_layout()
plt.show()

print("\n" + "="*50)
print("FEATURE IMPORTANCE ANALYSIS")
print("="*50)

# Get feature importance from the best model
if hasattr(best_model, 'feature_importances_'):
    feature_importance = pd.DataFrame({
        'feature': X.columns,
        'importance': best_model.feature_importances_
    }).sort_values('importance', ascending=False)

    print("\n TOP 10 MOST IMPORTANT FEATURES:")
    print(feature_importance.head(10))

    # Plot feature importance
    plt.figure(figsize=(12, 8))
    top_features = feature_importance.head(10)
    sns.barplot(data=top_features, x='importance', y='feature')
    plt.title(f'Top 10 Feature Importance - {best_model_name}', fontweight='bold')
    plt.xlabel('Importance Score')
    plt.tight_layout()
    plt.show()

# Permutation importance (model-agnostic)
print("\n Calculating permutation importance...")
perm_importance = permutation_importance(
    best_model, X_test_scaled, y_test, n_repeats=10, random_state=42
)

perm_df = pd.DataFrame({
    'feature': X.columns,
    'importance': perm_importance.importances_mean,
    'std': perm_importance.importances_std
}).sort_values('importance', ascending=False)

print("\n TOP 10 FEATURES BY PERMUTATION IMPORTANCE:")
print(perm_df.head(10))


print("\n" + "="*50)
print("HYPERPARAMETER TUNING")
print("="*50)

# Tune the best model
if best_model_name == 'Random Forest':
    param_grid = {
        'n_estimators': [50, 100, 200],
        'max_depth': [10, 20, None],
        'min_samples_split': [2, 5, 10],
        'min_samples_leaf': [1, 2, 4]
    }
elif best_model_name == 'XGBoost':
    param_grid = {
        'n_estimators': [50, 100, 200],
        'max_depth': [3, 6, 9],
        'learning_rate': [0.01, 0.1, 0.2],
        'subsample': [0.8, 0.9, 1.0]
    }
else:
    param_grid = {}  # Skip tuning for other models for demonstration

if param_grid:
    print(f" Tuning {best_model_name}...")
    grid_search = GridSearchCV(
        best_model, param_grid, cv=5, scoring='r2', n_jobs=-1
    )
    grid_search.fit(X_train_scaled, y_train)

    print(f"Best parameters: {grid_search.best_params_}")
    print(f"Best cross-validation score: {grid_search.best_score_:.4f}")

    # Update best model with tuned parameters
    best_model = grid_search.best_estimator_

    # Evaluate tuned model
    y_pred_tuned = best_model.predict(X_test_scaled)
    tuned_r2 = r2_score(y_test, y_pred_tuned)
    print(f"Tuned model test R²: {tuned_r2:.4f}")

    print("\n" + "="*50)
print("PREDICTION ON SAMPLE DATA")
print("="*50)

# Create sample new data for prediction
sample_data = X_test_scaled[:5]  # Take first 5 samples from test set
predictions = best_model.predict(sample_data)

print("\n SAMPLE PREDICTIONS:")
print("Actual vs Predicted Values:")
for i, (actual, predicted) in enumerate(zip(y_test.iloc[:5], predictions)):
    print(f"Sample {i+1}: Actual = {actual:.2f}, Predicted = {predicted:.2f}, "
          f"Error = {abs(actual - predicted):.2f}")

print("\n" + "="*50)
print("BUSINESS INSIGHTS & RECOMMENDATIONS")
print("="*50)

# Calculate key metrics
avg_yield = df['yield'].mean()
best_yield = df['yield'].max()
worst_yield = df['yield'].min()

print(f"\n Yield Statistics:")
print(f"Average Yield: {avg_yield:.2f} kg/hectare")
print(f"Best Yield: {best_yield:.2f} kg/hectare")
print(f"Worst Yield: {worst_yield:.2f} kg/hectare")

# Feature impact analysis
if 'feature_importance' in locals():
    top_3_features = feature_importance.head(3)['feature'].tolist()
    print(f"\n Top 3 Factors Affecting Yield:")
    for i, feature in enumerate(top_3_features, 1):
        print(f"  {i}. {feature}")

print(f"\n RECOMMENDATIONS FOR FARMERS:")
print("1. Optimize pollinator habitats to increase bee populations")
print("2. Monitor and maintain optimal temperature ranges during growing season")
print("3. Implement proper clone size management techniques")
print("4. Use rainfall data to optimize irrigation schedules")
print("5. Regular soil testing and nutrient management")

print("\n" + "="*50)
print("MODEL DEPLOYMENT SUMMARY")
print("="*50)

print(f"\n FINAL MODEL: {best_model_name}")
print(f" TEST R² SCORE: {results[best_model_name]['test_r2']:.4f}")
print(f" TEST RMSE: {results[best_model_name]['test_rmse']:.4f}")
print(f" TEST MAE: {results[best_model_name]['test_mae']:.4f}")

print("\n" + "="*50)
print("PROJECT COMPLETED SUCCESSFULLY! ")
print("="*50)


