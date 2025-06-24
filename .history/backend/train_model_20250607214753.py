
import pandas as pd
import joblib
import asyncio
from db_connect import create_supabase  # Your Supabase connection function
from .Price_Estimation import EnsemblePropertyPredictor
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from db_connect import create_supabase


# Define table and columns to fetch for training
TABLE_NAME = "properties"
COLUMNS = ['district', 'price_$', 'latitude', 'longitude', 'bedrooms', 'bathrooms', 'size_m2']

async def train_and_save_model():
    """Fetches data, trains the ensemble model, and saves it to a file."""
    print("Connecting to Supabase to fetch training data...")
    supabase = await create_supabase()
    
    # Fetch all relevant data from the properties table
    response = await supabase.from_(TABLE_NAME).select(",".join(COLUMNS)).execute()
    
    if not response.data:
        print("Error: No data fetched from the database. Training cannot proceed.")
        return

    df = pd.DataFrame(response.data)
    print(f"Successfully fetched {len(df)} records for training.")

    # Instantiate the predictor
    predictor = EnsemblePropertyPredictor()

    # Train the model on the entire dataset
    print("Training the ensemble model... This may take a moment.")
    predictor.train(df)

    # Save the entire trained predictor object
    model_filename = 'property_price_estimator.joblib'
    joblib.dump(predictor, model_filename)
    print(f"✅ Model training complete. Predictor saved to {model_filename}")

if __name__ == "__main__":
    # To run this script, execute `python train_model.py` in your terminal
    asyncio.run(train_and_save_model())