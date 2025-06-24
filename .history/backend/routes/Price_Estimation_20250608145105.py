
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor
from sklearn.preprocessing import OneHotEncoder
import joblib

class EnsemblePropertyPredictor:
    def __init__(self):
        self.rf_model = RandomForestRegressor(n_estimators=100, random_state=42)
        self.xgb_model = XGBRegressor(n_estimators=200, learning_rate=0.1, max_depth=5, random_state=42)
        self.district_encoder = OneHotEncoder(handle_unknown='ignore', sparse_output=False)
        self.is_trained = False
        self.feature_names_loc = None
        self.feature_names_other = None

    def _clean_data(self, df):
        """Shared data cleaning logic."""
        numeric_cols = ['bedrooms', 'bathrooms', 'size_m2', 'price_$', 'latitude', 'longitude']
        cols_to_clean = [col for col in numeric_cols if col in df.columns]
        df[cols_to_clean] = df[cols_to_clean].apply(pd.to_numeric, errors='coerce')
        
        # Determine which columns to check for NaNs based on context (training vs. prediction)
        cols_to_check_nans = [col for col in cols_to_clean if col != 'price_$']
        if 'price_$' in df.columns:
             cols_to_check_nans.append('price_$')

        df = df.dropna(subset=cols_to_check_nans)
        return df.reset_index(drop=True)

    def train(self, df):
        """Train RF and XGBoost on the entire provided dataframe."""
        df_clean = self._clean_data(df)
        
        # Fit the encoder on the district column
        district_encoded = self.district_encoder.fit_transform(df_clean[['district']])
        district_cols = self.district_encoder.get_feature_names_out(['district'])
        df_district = pd.DataFrame(district_encoded, columns=district_cols, index=df_clean.index)
        
        # Define and store feature sets
        X_location = pd.concat([df_district, df_clean[['latitude', 'longitude']]], axis=1)
        X_other = df_clean[['bedrooms', 'bathrooms', 'size_m2']]
        y = df_clean['price_$']
        
        # Store feature names to ensure consistency during prediction
        self.feature_names_loc = X_location.columns.tolist()
        self.feature_names_other = X_other.columns.tolist()

        # Train models
        self.rf_model.fit(X_location, y)
        self.xgb_model.fit(X_other, y)
        
        self.is_trained = True
        print("Model training is complete.")

    def predict(self, df_input):
        """Predict price for new property data."""
        if not self.is_trained:
            raise ValueError("Model is not trained. Load a trained instance before predicting.")
        
        df_clean = self._clean_data(df_input.copy())
        
        # Transform districts using the already-fitted encoder
        district_encoded = self.district_encoder.transform(df_clean[['district']])
        district_cols = self.district_encoder.get_feature_names_out(['district'])
        df_district = pd.DataFrame(district_encoded, columns=district_cols, index=df_clean.index)

        # Create feature sets and ensure column order matches training
        X_location = pd.concat([df_district, df_clean[['latitude', 'longitude']]], axis=1)
        X_location = X_location.reindex(columns=self.feature_names_loc, fill_value=0)
        
        X_other = df_clean[['bedrooms', 'bathrooms', 'size_m2']]
        X_other = X_other.reindex(columns=self.feature_names_other, fill_value=0)
        
        # Get predictions from both models
        pred_rf = self.rf_model.predict(X_location)
        pred_xgb = self.xgb_model.predict(X_other)
        
        # Return the ensemble average
        final_prediction = (pred_rf * 0.5 + pred_xgb * 0.5)
        return final_prediction.tolist()