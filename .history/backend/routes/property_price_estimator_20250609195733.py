import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor
from sklearn.preprocessing import OneHotEncoder
import joblib
import os # Import os for path handling

# This class is now in its own dedicated file, which is best practice.
class EnsemblePropertyPredictor:
    def __init__(self):
        self.rf_model = RandomForestRegressor(n_estimators=100, random_state=42)
        self.xgb_model = XGBRegressor(n_estimators=200, learning_rate=0.1, max_depth=5, random_state=42)
        self.district_encoder = OneHotEncoder(handle_unknown='ignore', sparse_output=False)
        self.is_trained = False
        self.feature_names_district = None
        self.feature_names_other = None

    def _clean_data(self, df):
        # The 'type' column is categorical and should not be converted to numeric
        numeric_cols = ['bedrooms', 'bathrooms', 'size_m2']
        # If 'price_$' exists (during training), clean it too.
        if 'price_$' in df.columns:
            numeric_cols.append('price_$')
            df[numeric_cols] = df[numeric_cols].apply(pd.to_numeric, errors='coerce')
            df = df.dropna(subset=numeric_cols)
        else: # During prediction, only clean the feature columns
            df[numeric_cols] = df[numeric_cols].apply(pd.to_numeric, errors='coerce')
            df = df.dropna(subset=numeric_cols)

        return df.reset_index(drop=True)
    
    def evaluate(self, X_test, y_test):
            """Evaluates the trained model on a hold-out test set."""
        if not self.is_trained:
            raise ValueError("Model is not trained. Cannot evaluate.")
            
        # The model's predict method expects a DataFrame with specific column names
        # so we need to combine the pre-processed test features back into one
        df_test = X_test.copy()
        
        # Make predictions on the test data
        predictions = self.predict(df_test)
        
        # Calculate performance metrics
        mae = mean_absolute_error(y_test, predictions)
        r2 = r2_score(y_test, predictions)
        
        print("\n--- Model Evaluation Results ---")
        print(f"Mean Absolute Error (MAE): ${mae:,.2f}")
        print(f"R-squared (R²): {r2:.4f}")
        print("---------------------------------")
        print(f"Interpretation: On average, the model's price prediction is off by about ${mae:,.2f}.")
        print(f"Interpretation: The model explains {r2:.2%} of the variance in the property prices.")
        
        return mae, r2

    def train(self, df):
        df_clean = self._clean_data(df.copy())
        district_encoded = self.district_encoder.fit_transform(df_clean[['district']])
        district_cols = self.district_encoder.get_feature_names_out(['district'])
        df_district = pd.DataFrame(district_encoded, columns=district_cols, index=df_clean.index)
        X_district = df_district
        X_other = df_clean[['bedrooms', 'bathrooms', 'size_m2']]
        y = df_clean['price_$']
        self.feature_names_district = X_district.columns.tolist()
        self.feature_names_other = X_other.columns.tolist()
        self.rf_model.fit(X_district, y)
        self.xgb_model.fit(X_other, y)
        self.is_trained = True
        print("✅ Model training complete.")

        

    def predict(self, df_input):
        if not self.is_trained:
            raise ValueError("Model is not trained.")
        df_clean = self._clean_data(df_input.copy())
        district_encoded = self.district_encoder.transform(df_clean[['district']])
        df_district = pd.DataFrame(district_encoded, columns=self.feature_names_district, index=df_clean.index)
        X_district = df_district.reindex(columns=self.feature_names_district, fill_value=0)
        X_other = df_clean[self.feature_names_other]
        pred_rf = self.rf_model.predict(X_district)
        pred_xgb = self.xgb_model.predict(X_other)
        return (0.5 * pred_rf + 0.5 * pred_xgb)
# ... (the EnsemblePropertyPredictor class definition remains the same) ...

# # 🏗️ Train-and-save block
# if __name__ == "__main__":
#     try:
#         # --- START OF CORRECTION ---

#         # Define paths relative to the project structure we know.
#         # This assumes the script is run from the 'backend' directory.
#         data_path = "../Schema/properties.csv"
#         model_save_path = "models/property_price_model.joblib"
        
#         # --- END OF CORRECTION ---

#         print(f"Reading data from: {data_path}")
#         df = pd.read_csv(data_path)
        
#         model = EnsemblePropertyPredictor()
#         model.train(df)
        
#         print(f"Saving model to: {model_save_path}")
#         joblib.dump(model, model_save_path)
        
#         print("✅ Model re-trained and saved successfully.")
        
#     except FileNotFoundError:
#         print(f"❌ CRITICAL ERROR: Could not find the dataset at '{data_path}'. Make sure you are running this script from the 'backend' directory.")
#     except Exception as e:
#         print(f"❌ Failed to train/save model: {e}")