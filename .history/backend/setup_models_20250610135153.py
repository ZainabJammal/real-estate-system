import sys
import os
sys.path.append(os.path.abspath(os.path.dirname(__file__)))
import pandas as pd
import joblib
import traceback
import numpy as np
from sklearn.model_selection import train_test_split
from model_downloader import download_model
from routes.property_price_estimator import EnsemblePropertyPredictor
from sklearn.model_selection import train_test_split # <-- ADD THIS IMPORT
from sklearn.metrics import mean_absolute_error, r2_score 
def setup():
    print("--- Running Model Setup ---")
    
    try:
        print("\n[SETUP] Step 1: Downloading models from Supabase...")
        download_model()
        print("[SETUP] Step 1 finished.\n")
    except Exception as e:
        print(f"[SETUP] ERROR during download: {e}")

    # # --- Part 2: Train on the new CURATED dataset ---
    # try:
    #     print("[SETUP] Step 2: Training on the manually curated dataset...")
        
    #     # --- POINT THE SCRIPT TO YOUR NEW FILE ---
    #     data_path = os.path.join(r"C:\Users\user\Documents\Real Estate SPF\real-estate-system\real-estate-system\Schema\properties_curated_rows.csv")
    #     model_save_path = os.path.join("models", "property_price_model.joblib")

    #     print(f"[SETUP] Reading curated data from: {data_path}")
    #     df = pd.read_csv(data_path)
    #     print(f"[SETUP] Using {len(df)} high-quality properties for training.")

    #     # The data is already clean, so we can go straight to training
    #     features = ['district', 'type', 'size_m2', 'bedrooms', 'bathrooms']
    #     target = 'price_$'
        
    #     X = df[features]
    #     y = np.log1p(df[target]) # Still use log transform, it's very effective

    #     X_train, X_test, y_train_log, y_test_log = train_test_split(X, y, test_size=0.2, random_state=42)

    #     df_train = X_train.copy()
    #     df_train[target] = y_train_log
        
    #     model = EnsemblePropertyPredictor()
    #     model.train(df_train)
        
    #     # Evaluate the model
    #     model.evaluate(X_test, y_test_log)
        
    #     print(f"\n[SETUP] Saving model to: {model_save_path}")
    #     joblib.dump(model, model_save_path)
        
    #     print("[SETUP] ✅ Model trained on curated data, evaluated, and saved successfully.")

    # try:
    #     print("\n[SETUP] Step 2: Training a new GEOSPATIAL model...")
        
    #     # Use the original, unfiltered dataset to get all locations
    #     data_path = os.path.join("..", "Schema", "properties.csv")
    #     model_save_path = os.path.join("models", "property_price_model.joblib")

    #     print(f"[SETUP] Reading data from: {data_path}")
    #     df = pd.read_csv(data_path)

    #     # --- DATA CLEANING FOR GEOSPATIAL MODEL ---
    #     # We only need these three columns and the price
    #     df_geo = df[['district', 'city', 'latitude', 'longitude', 'size_m2', 'price_$']].copy()

    #     # Drop any rows where these critical values are missing
    #     df_geo.dropna(inplace=True)
        
    #     # Optional: Remove extreme size/price outliers if they still exist
    #     price_cap = df_geo['price_$'].quantile(0.99)
    #     size_cap = df_geo['size_m2'].quantile(0.99)
    #     df_geo = df_geo[(df_geo['price_$'] < price_cap) & (df_geo['size_m2'] < size_cap)]
        
    #     print(f"[SETUP] Using {len(df_geo)} properties with valid geo-data for training.")
        
    #     # --- TRAINING AND EVALUATION ---
    #     features = ['latitude', 'longitude', 'size_m2']
    #     target = 'price_$'
        
    #     X = df_geo[features]
    #     y = np.log1p(df_geo[target])

    #     X_train, X_test, y_train_log, y_test_log = train_test_split(X, y, test_size=0.2, random_state=42)
        
    #     # For this, a RandomForest is often a great choice
    #     from sklearn.ensemble import RandomForestRegressor
    #     geo_model = RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1)
        
    #     print("[SETUP] Training RandomForest on geo-data...")
    #     geo_model.fit(X_train, y_train_log) # Train the new model
        
    #     # Evaluate
    #     y_pred_log = geo_model.predict(X_test)
    #     y_pred_dollars = np.expm1(y_pred_log)
    #     y_test_dollars = np.expm1(y_test_log)
        
    #     mae = mean_absolute_error(y_test_dollars, y_pred_dollars)
    #     r2 = r2_score(y_test_dollars, y_pred_dollars)
        
    #     print("\n--- Geospatial Model Evaluation Results ---")
    #     print(f"Mean Absolute Error (MAE): ${mae:,.2f}")
    #     print(f"R-squared (R²): {r2:.4f}")
    #     print("-----------------------------------------")
        
    #     print(f"\n[SETUP] Saving model to: {model_save_path}")
    #     joblib.dump(geo_model, model_save_path) # Save the new model
        
    #     print("[SETUP] ✅ Geospatial model trained, evaluated, and saved successfully.")

    try:
        print("[SETUP] Step 2: Training on the manually curated dataset...")
        
        data_path = os.path.join("..", "Schema", "properties.csv")
        model_save_path = os.path.join("models", "property_price_model.joblib")

        print(f"[SETUP] Reading curated data from: {data_path}")
        df = pd.read_csv(data_path)
        
        # Drop rows where city is missing, as it's now a critical feature
        df.dropna(subset=['city'], inplace=True)
        
        print(f"[SETUP] Using {len(df)} high-quality properties for training.")

        # --- Add 'city' to the list of features ---
        features = ['district', 'city', 'type', 'size_m2', 'bedrooms', 'bathrooms', 'longitude', ''] # <-- UPDATED
        target = 'price_$'
        
        X = df[features]
        y = np.log1p(df[target]) 

        X_train, X_test, y_train_log, y_test_log = train_test_split(X, y, test_size=0.2, random_state=42)

        df_train = X_train.copy()
        df_train[target] = y_train_log
        
        model = EnsemblePropertyPredictor()
        model.train(df_train)
        
        model.evaluate(X_test, y_test_log)
        
        print(f"\n[SETUP] Saving model to: {model_save_path}")
        joblib.dump(model, model_save_path)
        
        print("[SETUP] ✅ Model trained on curated data, evaluated, and saved successfully.")

    except Exception as e:
        print(f"❌ [SETUP] Failed to train/save local model: {e}")
        traceback.print_exc()

    print("--- Model Setup Complete ---")

if __name__ == "__main__":
    setup()