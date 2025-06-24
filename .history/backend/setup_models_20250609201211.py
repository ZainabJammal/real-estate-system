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
     try:
        print("[SETUP] Step 2: Training & Evaluating local property price estimation model...")
        
        data_path = os.path.join("..", "Schema", "properties.csv")
        model_save_path = os.path.join("models", "property_price_model.joblib")

        print(f"[SETUP] Reading data from: {data_path}")
        df = pd.read_csv(data_path)
        
        features = ['district', 'type', 'size_m2', 'bedrooms', 'bathrooms']
        target = 'price_$'
        
        X = df[features]
        y = df[target]

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        # Create the DataFrame for training
        df_train = X_train.copy()
        df_train[target] = y_train
        
        # --- THIS IS THE FIX ---
        # Create the DataFrame for testing
        df_test = X_test.copy()
        df_test[target] = y_test
        # --- END OF FIX ---

        model = EnsemblePropertyPredictor()
        
        # 1. Train the model using only the training data
        model.train(df_train)
        
        # 2. Evaluate the model using the separate test data
        model.evaluate(df_test)
        
        # 3. Save the trained model
        print(f"[SETUP] Saving model to: {model_save_path}")
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