import sys
import os
import traceback
from sklearn.model_selection import train_test_split
import pandas as pd
import joblib


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
        print("[SETUP] Step 2: Training & Evaluating local property price estimation model...")
        
        data_path = os.path.join("..", "Schema", "properties.csv")
        model_save_path = os.path.join("models", "property_price_model.joblib")

        print(f"[SETUP] Reading data from: {data_path}")
        df = pd.read_csv(data_path)

        # --- FILTERING STEP 1: FOCUS ON RESIDENTIAL TYPES ---
        residential_types = ['Apartment', 'House/Villa', 'Chalet', 'Land']
        print(f"\n[FILTER 1] Focusing model on these types: {residential_types}")
        df_filtered = df[df['type'].isin(residential_types)].copy()
        print(f"           Data size after type filter: {len(df_filtered)} properties.")

        # --- NEW FILTERING STEP 2: FOCUS ON DATA-RICH DISTRICTS ---
        # Set a threshold for how many properties a district must have to be included.
        min_properties_per_district = 0 # Let's say a district needs at least 50 listings.
        
        # Count properties per district
        district_counts = df_filtered['district'].value_counts()
        
        # Identify which districts meet the threshold
        districts_to_keep = district_counts[district_counts >= min_properties_per_district].index.tolist()
        
        print(f"\n[FILTER 2] Focusing on districts with at least {min_properties_per_district} properties.")
        print(f"           Districts to be used: {districts_to_keep}")
        
        # Filter the DataFrame to only include these districts.
        df_filtered = df_filtered[df_filtered['district'].isin(districts_to_keep)]
        print(f"           Data size after district filter: {len(df_filtered)} properties.")

        # --- FILTERING STEP 3: REMOVE PRICE OUTLIERS ---
        price_cap = df_filtered['price_$'].quantile(0.99)
        print(f"\n[FILTER 3] Identified price cap for remaining properties (99th percentile): ${price_cap:,.2f}")
        df_filtered = df_filtered[df_filtered['price_$'] < price_cap]
        print(f"           Final data size after price cap: {len(df_filtered)} properties.\n")
        
        # --- TRAINING AND EVALUATION ---
        features = ['district', 'type', 'size_m2', 'bedrooms', 'bathrooms']
        target = 'price_$'
        
        X = df_filtered[features]
        y = df_filtered[target]

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        df_train = X_train.copy()
        df_train[target] = y_train
        
        df_test = X_test.copy()
        df_test[target] = y_test
        
        model = EnsemblePropertyPredictor()
        model.train(df_train)
        model.evaluate(df_test)
        
        print(f"\n[SETUP] Saving model to: {model_save_path}")
        joblib.dump(model, model_save_path)
        
        print("[SETUP] ✅ Local property price model trained, evaluated, and saved successfully.")
        print("[SETUP] Step 2 finished.\n")

    # ... (except blocks) ...



    except FileNotFoundError:
        print(f"❌ [SETUP] CRITICAL ERROR: Could not find the dataset at '{data_path}'. Make sure this path is correct.")
    except Exception as e:
        print(f"❌ [SETUP] Failed to train/save local model: {e}")
        traceback.print_exc()

    print("--- Model Setup Complete ---")


# This makes the script runnable from the command line
if __name__ == "__main__":
    setup()