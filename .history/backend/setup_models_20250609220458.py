import sys
import os
import traceback
from sklearn.model_selection import train_test_split
import pandas as pd
import joblib
import numpy as np


from model_downloader import download_model
from routes.property_price_estimator import EnsemblePropertyPredictor

sys.path.append(os.path.abspath(os.path.dirname(__file__)))
# --- END OF FIX ---



def setup():
    """
    This master function runs all setup steps:
    1. Downloads pre-trained models from Supabase.
    2. Trains the local property price estimation model.
    """
    print("--- Running Model Setup ---")
    
    # --- Part 1: Download pre-trained models ---
    try:
        print("\n[SETUP] Step 1: Downloading models from Supabase...")
        download_model()
        print("[SETUP] Step 1 finished.\n")
    except Exception as e:
        print(f"[SETUP] ERROR during download: {e}")
        traceback.print_exc()

    # --- Part 2: Train the local property price estimator ---
  # ... (imports and setup function start) ...

   # --- Part 2: Train & Evaluate the local property price estimator ---
    try:
        print("[SETUP] Step 2: Training & Evaluating with new feature engineering...")
        
        data_path = os.path.join("..", "Schema", "properties.csv")
        model_save_path = os.path.join("models", "property_price_model.joblib")

        print(f"[SETUP] Reading data from: {data_path}")
        df = pd.read_csv(data_path)

        # --- DATA CLEANING AND FEATURE ENGINEERING ---

        # 1. Filter for core residential types
        residential_types = ['Apartment', 'House/Villa', 'Chalet'] # Let's exclude 'Land' for now as it has no bed/bath
        print(f"\n[FILTER 1] Focusing on core residential types: {residential_types}")
        df_filtered = df[df['type'].isin(residential_types)].copy()
        
        # 2. Remove nonsensical data and outliers based on bed/bath counts
        print("\n[FILTER 2] Removing properties with unrealistic bedroom/bathroom counts...")
        original_count = len(df_filtered)
        df_filtered = df_filtered[(df_filtered['bedrooms'] < 10) & (df_filtered['bedrooms'] >= 0)]
        df_filtered = df_filtered[(df_filtered['bathrooms'] < 10) & (df_filtered['bathrooms'] >= 0)]
        df_filtered = df_filtered[df_filtered['size_m2'] > 20] # Remove tiny listings
        print(f"           Removed {original_count - len(df_filtered)} properties with unrealistic stats.")

        # 3. Create the new 'price_per_sqm' feature
        # We calculate this before removing price outliers to get a true sense of the market
        df_filtered['price_per_sqm'] = df_filtered['price_$'] / df_filtered['size_m2']
        
        # 4. Remove outliers based on price AND price_per_sqm
        print("\n[FILTER 3] Removing outliers based on price and price_per_sqm...")
        # Remove top 1% and bottom 1% to handle both extremes
        price_upper_bound = df_filtered['price_$'].quantile(0.99)
        price_lower_bound = df_filtered['price_$'].quantile(0.01)
        sqm_upper_bound = df_filtered['price_per_sqm'].quantile(0.99)
        sqm_lower_bound = df_filtered['price_per_sqm'].quantile(0.01)

        original_count = len(df_filtered)
        df_filtered = df_filtered[
            (df_filtered['price_$'] < price_upper_bound) & (df_filtered['price_$'] > price_lower_bound) &
            (df_filtered['price_per_sqm'] < sqm_upper_bound) & (df_filtered['price_per_sqm'] > sqm_lower_bound)
        ]
        print(f"           Removed {original_count - len(df_filtered)} price outliers.")
        print(f"           Final data size for training: {len(df_filtered)} properties.\n")

        # --- TRAINING AND EVALUATION ---
        # NOTE: We no longer use 'price_per_sqm' as a direct feature, but it helped us clean the data.
        features = ['district', 'type', 'size_m2', 'bedrooms', 'bathrooms']
        target = 'price_$'
        
        X = df_filtered[features]
        # Use log-transformed price for the target 'y'
        y = np.log1p(df_filtered[target])

        X_train, X_test, y_train_log, y_test_log = train_test_split(X, y, test_size=0.2, random_state=42)

        df_train = X_train.copy()
        df_train[target] = y_train_log # This is now log_price
        
        model = EnsemblePropertyPredictor() # We'll update this class next
        model.train(df_train)
        
        # Evaluate on the log-transformed test data
        model.evaluate(X_test, y_test_log)
        
        print(f"\n[SETUP] Saving model to: {model_save_path}")
        joblib.dump(model, model_save_path)
        
        print("[SETUP] ✅ Local property price model trained, evaluated, and saved successfully.")
        print("[SETUP] Step 2 finished.\n")



    except FileNotFoundError:
        print(f"❌ [SETUP] CRITICAL ERROR: Could not find the dataset at '{data_path}'. Make sure this path is correct.")
    except Exception as e:
        print(f"❌ [SETUP] Failed to train/save local model: {e}")
        traceback.print_exc()

    print("--- Model Setup Complete ---")


# This makes the script runnable from the command line
if __name__ == "__main__":
    setup()