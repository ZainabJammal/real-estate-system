import sys
import os
sys.path.append(os.path.abspath(os.path.dirname(__file__)))
import pandas as pd
import joblib
import traceback
import numpy as np
from sklearn.model_selection import train_test_split
from model_downloader import download_model
from models.property_price_estimator import EnsemblePropertyPredictor

def setup():
    print("--- Running Model Setup ---")
    
    try:
        print("\n[SETUP] Step 1: Downloading models from Supabase...")
        download_model()
        print("[SETUP] Step 1 finished.\n")
    except Exception as e:
        print(f"[SETUP] ERROR during download: {e}")

    # --- Part 2: Train on the new CURATED dataset ---
    try:
        print("[SETUP] Step 2: Training on the manually curated dataset...")
        
        # --- POINT THE SCRIPT TO YOUR NEW FILE ---
        data_path = os.path.join("..", "Schema", "properties_curated.csv")
        model_save_path = os.path.join("models", "property_price_model.joblib")

        print(f"[SETUP] Reading curated data from: {data_path}")
        df = pd.read_csv(data_path)
        print(f"[SETUP] Using {len(df)} high-quality properties for training.")

        # The data is already clean, so we can go straight to training
        features = ['district', 'type', 'size_m2', 'bedrooms', 'bathrooms']
        target = 'price_$'
        
        X = df[features]
        y = np.log1p(df[target]) # Still use log transform, it's very effective

        X_train, X_test, y_train_log, y_test_log = train_test_split(X, y, test_size=0.2, random_state=42)

        df_train = X_train.copy()
        df_train[target] = y_train_log
        
        model = EnsemblePropertyPredictor()
        model.train(df_train)
        
        # Evaluate the model
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