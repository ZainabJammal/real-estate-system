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
        self.feature_names_district = None
        self.feature_names_other = None

    def _clean_data(self, df):
        numeric_cols = ['bedrooms', 'bathrooms', 'size_m2', 'price_$']
        df[numeric_cols] = df[numeric_cols].apply(pd.to_numeric, errors='coerce')
        df = df.dropna(subset=numeric_cols)
        return df.reset_index(drop=True)

    def train(self, df):
        df_clean = self._clean_data(df)

        # Encode district
        district_encoded = self.district_encoder.fit_transform(df_clean[['district']])
        district_cols = self.district_encoder.get_feature_names_out(['district'])
        df_district = pd.DataFrame(district_encoded, columns=district_cols, index=df_clean.index)

        # Define features and target
        X_district = df_district
        X_other = df_clean[['bedrooms', 'bathrooms', 'size_m2']]
        y = df_clean['price_$']

        # Save column names for prediction consistency
        self.feature_names_district = X_district.columns.tolist()
        self.feature_names_other = X_other.columns.tolist()

        # Train both models
        self.rf_model.fit(X_district, y)
        self.xgb_model.fit(X_other, y)

        self.is_trained = True
        print("✅ Model training complete.")

    def predict(self, df_input):
        if not self.is_trained:
            raise ValueError("Model is not trained.")

        df_clean = self._clean_data(df_input.copy())

        # Encode district
        district_encoded = self.district_encoder.transform(df_clean[['district']])
        df_district = pd.DataFrame(district_encoded, columns=self.feature_names_district, index=df_clean.index)

        # Match column order
        X_district = df_district.reindex(columns=self.feature_names_district, fill_value=0)
        X_other = df_clean[['bedrooms', 'bathrooms', 'size_m2']].reindex(columns=self.feature_names_other, fill_value=0)

        # Predict
        pred_rf = self.rf_model.predict(X_district)
        pred_xgb = self.xgb_model.predict(X_other)
        return (0.5 * pred_rf + 0.5 * pred_xgb).tolist()


# 🏗️ Train-and-save block
if __name__ == "__main__":
    try:
        df = pd.read_csv("C:\\Users\\user\\Documents\\Real Estate SPF\\real-estate-system\\real-estate-system\\Schema\\properties.csv")  # Ensure this file is available in the same directory
        model = EnsemblePropertyPredictor()
        model.train(df)
        joblib.dump(model, "C:\\Users\\user\\Documents\\Real Estate SPF\\real-estate-system\\real-estate-system\\property_price_model.joblib")
        print("✅ Model saved as 'property_price_model.joblib'")
    except Exception as e:
        print(f"❌ Failed to train/save model: {e}")
