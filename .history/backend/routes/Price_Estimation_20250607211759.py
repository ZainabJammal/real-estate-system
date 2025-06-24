import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor
from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import train_test_split
import joblib

class EnsemblePropertyPredictor:
    def __init__(self):
        self.rf_model = None  # Random Forest for location
        self.xgb_model = None  # XGBoost for other features
        self.district_encoder = OneHotEncoder(handle_unknown='ignore')  # Encode districts
        self.is_trained = False

    def _preprocess_data(self, df):
        """Clean and split features for RF/XGBoost."""
        # Convert numeric fields (handle missing values)
        numeric_cols = ['bedrooms', 'bathrooms', 'size_m2', 'price_$', 'latitude', 'longitude']
        df[numeric_cols] = df[numeric_cols].apply(pd.to_numeric, errors='coerce')
        df = df.dropna(subset=numeric_cols)
        
        # Encode districts (fit during training, transform during prediction)
        if hasattr(self, 'district_encoder'):
            district_encoded = self.district_encoder.transform(df[['district']]).toarray()
            district_cols = [f"district_{d}" for d in self.district_encoder.categories_[0]]
            df_district = pd.DataFrame(district_encoded, columns=district_cols)
        else:
            raise ValueError("District encoder not fitted. Call train() first.")
        
        # Split features
        X_location = pd.concat([df_district, df[['latitude', 'longitude']], axis=1])
        X_other = df[['bedrooms', 'bathrooms', 'size_m2']]
        y = df['price_$']
        
        return X_location, X_other, y

    def train(self, df, test_size=0.2):
        """Train RF and XGBoost on preprocessed data."""
        # Preprocess and split
        X_location, X_other, y = self._preprocess_data(df)
        X_loc_train, X_loc_test, X_other_train, X_other_test, y_train, y_test = \
            train_test_split(X_location, X_other, y, test_size=test_size, random_state=42)
        
        # Train Random Forest (location features)
        self.rf_model = RandomForestRegressor(n_estimators=100, random_state=42)
        self.rf_model.fit(X_loc_train, y_train)
        
        # Train XGBoost (other features)
        self.xgb_model = XGBRegressor(
            n_estimators=200, 
            learning_rate=0.1, 
            max_depth=5, 
            random_state=42
        )
        self.xgb_model.fit(X_other_train, y_train)
        
        self.is_trained = True
        print("Training complete. Models ready for prediction.")

    def predict(self, df_input):
        """Predict price for new property data."""
        if not self.is_trained:
            raise ValueError("Models not trained. Call train() first.")
        
        # Preprocess input
        X_location, X_other, _ = self._preprocess_data(df_input)
        
        # Get predictions
        pred_rf = self.rf_model.predict(X_location)
        pred_xgb = self.xgb_model.predict(X_other)
        
        # Ensemble average (adjust weights as needed)
        return (pred_rf * 0.5 + pred_xgb * 0.5).tolist()