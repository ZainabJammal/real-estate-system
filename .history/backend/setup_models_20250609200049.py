import sys
import os

# --- THIS IS THE FIX ---
# Add the current directory ('.') to Python's search path
# This allows it to find the 'models' and 'routes' packages.
sys.path.append(os.path.abspath(os.path.dirname(__file__)))
# --- END OF FIX ---

import pandas as pd
import joblib
import traceback

# Step 1: Import the functions and classes we need
from model_downloader import download_model
from routes.property_price_estimator import EnsemblePropertyPredictor

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
        # Using os.path.join for better cross-platform compatibility
        data_path = os.path.join("..", "Schema", "properties.csv")
        model_save_path = os.path.join("models", "property_price_model.joblib")

        print(f"[SETUP] Reading data from: {data_path}")
        df = pd.read_csv(data_path)

         
        # Define features (X) and target (y)
        features = ['district', 'type', 'size_m2', 'bedrooms', 'bathrooms']
        target = 'price_$'
        
        X = df[features]
        y = df[target]

        # Split the data: 80% for training, 20% for testing
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        # Combine training features and target for the model's train method
        df_train = X_train.copy()
        df_train[target] = y_train
        
        # --- END OF NEW EVALUATION LOGIC ---
        
        model = EnsemblePropertyPredictor()
        
        # Train the model ONLY on the training data
        model.train(df_train)
        
        # Now, evaluate the model on the unseen test data
        model.evaluate(X_test, y_test)
        
        # Finally, save the trained model
        print(f"[SETUP] Saving model to: {model_save_path}")
        joblib.dump(model, model_save_path)
        
        print("[SETUP] ✅ Local property price model trained, evaluated, and saved successfully.")
        print("[SETUP] Step 2 finished.\n")

        
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