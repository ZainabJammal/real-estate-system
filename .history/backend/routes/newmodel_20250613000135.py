import pandas as pd
import xgboost as xgb
import joblib
import os
from supabase import create_client, Client

# --- Configuration ---
# You'll get these from your Supabase project settings
SUPABASE_URL = os.environ.get("SUPABASE_URL") # Recommended to use environment variables
SUPABASE_KEY = os.environ.get("SUPABASE_KEY")

MODEL_PATH = 'forecast_model.joblib'
MODEL_COLS_PATH = 'model_columns.json'
TARGET_COL = 'transaction_value'
HORIZON_MONTHS = 60

# Initialize the Supabase client
# In a real app, you do this once when your server starts
try:
    supabase: Client = create_client(SUPABASE_URL, SUPABASE_KEY)
except Exception as e:
    print(f"Error initializing Supabase client: {e}")
    print("Please ensure SUPABASE_URL and SUPABASE_KEY are set as environment variables.")
    supabase = None


    