import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor
from sklearn.preprocessing import OneHotEncoder
import joblib
import os # Import os for path handling

from sklearn.model_selection import train_test_split # <-- ADD THIS IMPORT
from sklearn.metrics import mean_absolute_error, r2_score # <-- ADD THIS IMPORT

class EnsemblePropertyPredictor:
    def __init__(self):
        # We will use one powerful model (XGBoost) that sees all features
        # This is often better than a complex ensemble for initial versions.
        self.model = XGBRegressor(n_estimators=200, learning_rate=0.1, max_depth=5, random_state=42, enable_categorical=True)
        
        # We will now have two encoders
        self.district_encoder = OneHotEncoder(handle_unknown='ignore', sparse_output=False)
        self.type_encoder = OneHotEncoder(handle_unknown='ignore', sparse_output=False)
        
        self.is_trained = False
        self.feature_names_in_ = None # Store all feature names

    def _clean_data(self, df):
        # ... (this method can remain the same) ...
        numeric_cols = ['bedrooms', 'bathrooms', 'size_m2']
        if 'price_$' in df.columns:
            numeric_cols.append('price_$')
            df[numeric_cols] = df[numeric_cols].apply(pd.to_numeric, errors='coerce')
            df = df.dropna(subset=numeric_cols)
        else:
            df[numeric_cols] = df[numeric_cols].apply(pd.to_numeric, errors='coerce')
            df = df.dropna(subset=numeric_cols)
        return df.reset_index(drop=True)

    def _prepare_features(self, df):
        """A new helper method to handle all feature engineering."""
        # Encode 'district'
        district_encoded = self.district_encoder.transform(df[['district']])
        df_district = pd.DataFrame(district_encoded, columns=self.district_encoder.get_feature_names_out(['district']))
        
        # Encode 'type'
        type_encoded = self.type_encoder.transform(df[['type']])
        df_type = pd.DataFrame(type_encoded, columns=self.type_encoder.get_feature_names_out(['type']))
        
        # Get the numeric features
        numeric_features = df[['bedrooms', 'bathrooms', 'size_m2']]
        
        # Combine all features into one final DataFrame
        return pd.concat([numeric_features, df_district, df_type], axis=1)

    def train(self, df):
        df_clean = self._clean_data(df.copy())
        
        # Fit the encoders on the training data
        self.district_encoder.fit(df_clean[['district']])
        self.type_encoder.fit(df_clean[['type']])
        
        # Prepare the full feature set
        X = self._prepare_features(df_clean)
        y = df_clean['price_$']
        
        self.feature_names_in_ = X.columns.tolist()
        
        # Train the single, more powerful model
        self.model.fit(X, y)
        
        self.is_trained = True
        print("✅ Model training complete.")

    def predict(self, df_input):
        if not self.is_trained:
            raise ValueError("Model is not trained.")
        
        df_clean = self._clean_data(df_input.copy())
        
        # Use the same feature preparation logic
        X = self._prepare_features(df_clean)
        
        # Reorder columns to match training order, just in case
        X = X.reindex(columns=self.feature_names_in_, fill_value=0)
        
        return self.model.predict(X)

    def evaluate(self, df_test):
        # ... (this method can remain exactly the same) ...
        if not self.is_trained:
            raise ValueError("Model is not trained. Cannot evaluate.")
        y_test = df_test['price_$']
        X_test = df_test.drop(columns=['price_$'])
        predictions = self.predict(X_test)
        mae = mean_absolute_error(y_test, predictions)
        r2 = r2_score(y_test, predictions)
        print("\n--- Model Evaluation Results ---")
        print(f"Mean Absolute Error (MAE): ${mae:,.2f}")
        print(f"R-squared (R²): {r2:.4f}")
        print("---------------------------------")
        print(f"Interpretation: On average, the model's price prediction is off by about ${mae:,.2f}.")
        print(f"Interpretation: The model explains {r2:.2%} of the variance in the property prices.")
        return mae, r2