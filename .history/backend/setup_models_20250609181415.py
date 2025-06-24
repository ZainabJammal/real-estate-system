import pandas as pd
import joblib
import os
import traceback

# Step 1: Import the functions and classes we need
from model_downloader import download_model
from models.property_price_estimator import EnsemblePropertyPredictor

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
    try:
        print("[SETUP] Step 2: Training local property price estimation model...")
        
        # Define the paths relative to this script's location (the 'backend' folder)
        data_path = "../Schema/properties.csv"
        model_save_path = "models/property_price_model.joblib"

        print(f"[SETUP] Reading data from: {data_path}")
        df = pd.read_csv(data_path)
        
        # Instantiate the model from its class file
        model = EnsemblePropertyPredictor()
        
        # Train the model
        model.train(df)
        
        print(f"[SETUP] Saving model to: {model_save_path}")
        joblib.dump(model, model_save_path)
        
        print("[SETUP] ✅ Local property price model trained and saved successfully.")
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