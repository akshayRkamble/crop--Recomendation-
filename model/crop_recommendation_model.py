import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from xgboost import XGBClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import accuracy_score, classification_report
import joblib
import os

class CropRecommender:
    def __init__(self):
        # Initialize multiple models
        self.models = {
            'random_forest': RandomForestClassifier(
                n_estimators=100,
                random_state=42
            ),
            'xgboost': XGBClassifier(
                n_estimators=100,
                random_state=42
            ),
            'svm': SVC(
                kernel='rbf',
                probability=True,
                random_state=42
            )
        }

        self.best_model = None
        self.best_model_name = None
        self.scaler = StandardScaler()

        try:
            # Load and prepare data
            data_path = "attached_assets/Crop_recommendation (1).csv"
            if not os.path.exists(data_path):
                raise FileNotFoundError(f"Dataset not found at {data_path}")

            df = pd.read_csv(data_path)
            if df.empty:
                raise ValueError("Dataset is empty")

            # Prepare features and target
            X = df[['N', 'P', 'K', 'temperature', 'humidity', 'ph', 'rainfall']].values
            y = df['label'].values

            # Split data
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=0.2, random_state=42
            )

            # Scale features
            self.scaler.fit(X_train)
            X_train_scaled = self.scaler.transform(X_train)
            X_test_scaled = self.scaler.transform(X_test)

            # Train and evaluate models
            best_accuracy = 0
            model_scores = {}

            for name, model in self.models.items():
                try:
                    # Train model
                    model.fit(X_train_scaled, y_train)

                    # Make predictions
                    y_pred = model.predict(X_test_scaled)
                    accuracy = accuracy_score(y_test, y_pred)

                    # Cross validation
                    cv_scores = cross_val_score(
                        model, X_train_scaled, y_train, cv=5
                    )

                    model_scores[name] = {
                        'accuracy': accuracy,
                        'cv_mean': cv_scores.mean(),
                        'cv_std': cv_scores.std()
                    }

                    # Update best model
                    if accuracy > best_accuracy:
                        best_accuracy = accuracy
                        self.best_model = model
                        self.best_model_name = name
                except Exception as model_error:
                    print(f"Error training {name} model: {str(model_error)}")
                    continue

            if not self.best_model:
                raise RuntimeError("No models were successfully trained")

            self.model_scores = model_scores
            # Store unique crop labels
            self.crop_labels = sorted(df['label'].unique())

        except Exception as e:
            error_msg = f"Failed to initialize models: {str(e)}"
            print(error_msg)  # For debugging
            raise RuntimeError(error_msg)

    def predict(self, features):
        """
        Predict crop using the best performing model
        features: [N, P, K, temperature, humidity, ph, rainfall]
        """
        if not self.best_model:
            raise RuntimeError("Models not properly initialized")

        features_scaled = self.scaler.transform(features.reshape(1, -1))
        prediction = self.best_model.predict(features_scaled)
        probabilities = self.best_model.predict_proba(features_scaled)
        return prediction[0], probabilities[0]

    def get_model_scores(self):
        """Return evaluation metrics for all models"""
        if not hasattr(self, 'model_scores'):
            return {}
        return self.model_scores

    def get_feature_importance(self):
        """Return feature importance scores from Random Forest model"""
        if self.best_model_name == 'random_forest':
            return self.best_model.feature_importances_
        else:
            return self.models['random_forest'].feature_importances_

    def get_crop_labels(self):
        """Return list of possible crops"""
        if not hasattr(self, 'crop_labels'):
            return []
        return self.crop_labels