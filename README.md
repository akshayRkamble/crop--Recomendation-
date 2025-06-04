# Crop Recommendation System

## Overview
This project implements a machine learning-based crop recommendation system that helps farmers and agricultural experts determine the most suitable crops to grow based on various environmental and soil parameters.

## Project Structure
```
Crop-Recomendation/
├── model/
│   └── crop_recommendation_model.py    # Core ML model implementation
├── utils/
│   ├── pdf_generator.py               # PDF report generation utilities
│   ├── visualization.py              # Data visualization functions
│   └── advanced_features.py          # Additional feature implementations
├── styles/
│   └── custom.css                    # Custom styling for the web interface
├── attached_assets/
│   └── Crop_recommendation (1).csv   # Dataset for training
├── .streamlit/                       # Streamlit configuration
├── .devcontainer/                    # Development container configuration
├── main.py                           # Main application entry point
└── pyproject.toml                    # Project dependencies and metadata
```

## Technologies Used

### Core Technologies
- **Python**: Primary programming language used for development
- **Machine Learning**: Multiple models implemented for crop prediction
- **Data Science**: Data processing and analysis tools
- **Streamlit**: Web application framework for the user interface

### Key Libraries and Frameworks
1. **Machine Learning Libraries**:
   - `scikit-learn`: Used for implementing Random Forest and SVM models
   - `XGBoost`: Advanced gradient boosting framework for improved predictions
   - `numpy`: Numerical computing and array operations
   - `pandas`: Data manipulation and analysis

2. **Data Processing**:
   - `StandardScaler`: For feature normalization
   - `train_test_split`: For dataset splitting
   - `cross_val_score`: For model validation

3. **Model Evaluation**:
   - `accuracy_score`: For measuring model performance
   - `classification_report`: For detailed model evaluation metrics

4. **Web Interface**:
   - `Streamlit`: For creating interactive web applications
   - `Custom CSS`: For enhanced UI/UX

5. **Additional Features**:
   - PDF Report Generation
   - Data Visualization
   - Advanced Analytics

### Models Implemented
1. **Random Forest Classifier**
   ```python
   from sklearn.ensemble import RandomForestClassifier
   
   rf_model = RandomForestClassifier(
       n_estimators=100,
       random_state=42
   )
   ```

2. **XGBoost Classifier**
   ```python
   from xgboost import XGBClassifier
   
   xgb_model = XGBClassifier(
       n_estimators=100,
       random_state=42
   )
   ```

3. **Support Vector Machine (SVM)**
   ```python
   from sklearn.svm import SVC
   
   svm_model = SVC(
       kernel='rbf',
       probability=True,
       random_state=42
   )
   ```

## Features
- Multiple model implementation for robust predictions
- Automatic model selection based on performance
- Feature importance analysis
- Cross-validation for model reliability
- Probability estimates for predictions
- Interactive web interface
- PDF report generation
- Data visualization capabilities
- Support for various environmental parameters:
  - Nitrogen (N)
  - Phosphorus (P)
  - Potassium (K)
  - Temperature
  - Humidity
  - pH level
  - Rainfall

## Setup and Installation
1. Ensure Python 3.x is installed
2. Install required dependencies:
   ```bash
   pip install numpy pandas scikit-learn xgboost joblib streamlit
   ```
3. Clone the repository:
   ```bash
   git clone [repository-url]
   cd Crop-Recomendation
   ```
4. Run the application:
   ```bash
   streamlit run main.py
   ```

## Usage

### 1. Web Interface Usage
```python
# Example of Streamlit interface code
import streamlit as st

def main():
    st.title("Crop Recommendation System")
    
    # Input fields
    N = st.number_input("Nitrogen (N)", min_value=0, max_value=140)
    P = st.number_input("Phosphorus (P)", min_value=5, max_value=145)
    K = st.number_input("Potassium (K)", min_value=5, max_value=205)
    temperature = st.number_input("Temperature (°C)", min_value=8.0, max_value=44.0)
    humidity = st.number_input("Humidity (%)", min_value=14.0, max_value=100.0)
    ph = st.number_input("pH", min_value=3.5, max_value=10.0)
    rainfall = st.number_input("Rainfall (mm)", min_value=20.0, max_value=300.0)
    
    if st.button("Get Recommendation"):
        # Get prediction
        prediction, probabilities = recommender.predict([N, P, K, temperature, humidity, ph, rainfall])
        st.success(f"Recommended Crop: {prediction}")
```

### 2. Programmatic Usage
```python
from model.crop_recommendation_model import CropRecommender

# Initialize the recommender
recommender = CropRecommender()

# Example input data
features = [
    90,  # N
    42,  # P
    43,  # K
    20.8,  # temperature
    82.0,  # humidity
    6.5,   # ph
    202.9  # rainfall
]

# Get prediction
prediction, probabilities = recommender.predict(features)
print(f"Recommended Crop: {prediction}")
print("Probability Scores:", probabilities)
```

### 3. Data Visualization Example
```python
import matplotlib.pyplot as plt
import seaborn as sns

def plot_feature_importance(model, feature_names):
    importance = model.get_feature_importance()
    plt.figure(figsize=(10, 6))
    sns.barplot(x=importance, y=feature_names)
    plt.title("Feature Importance")
    plt.show()
```

### 4. PDF Report Generation
```python
from utils.pdf_generator import generate_report

def create_crop_report(prediction, probabilities, features):
    report_data = {
        "crop": prediction,
        "probabilities": probabilities,
        "soil_parameters": features,
        "recommendations": get_crop_recommendations(prediction)
    }
    generate_report(report_data, "crop_recommendation.pdf")
```

## Model Performance
The system automatically selects the best performing model based on:
- Accuracy scores
- Cross-validation results
- Model stability

Example of model evaluation:
```python
# Cross-validation example
from sklearn.model_selection import cross_val_score

cv_scores = cross_val_score(model, X_train, y_train, cv=5)
print(f"Cross-validation scores: {cv_scores}")
print(f"Average CV score: {cv_scores.mean():.3f} (+/- {cv_scores.std() * 2:.3f})")
```

## Future Enhancements
- Integration with weather APIs for real-time data
- Enhanced mobile responsiveness
- Additional model implementations
- Real-time monitoring capabilities
- Advanced data analytics dashboard
- Multi-language support
- API endpoints for integration

## License
[Add your license information here]

## Contributing
[Add contribution guidelines here]
