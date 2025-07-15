import os
import joblib
import pandas as pd
import numpy as np
import traceback
from quart import Blueprint, request, jsonify
from supabase import create_client, Client
from dotenv import load_dotenv

# This global dictionary will be populated by the init function
FORECAST_MODELS = {}
supabase = None

# --- NEW: An 'init' function to set up the blueprint and load everything ---
def init_forecasting_routes(model_dir):
    """Initializes the blueprint, loads models, and sets up the Supabase client."""
    
    global supabase # Use the global supabase variable
    forecasting_bp = Blueprint('forecasting', __name__)
    
    # --- Load Models ---
    try:
        print("--- Loading Forecasting Models ---")
        print(f"Attempting to load models from: {os.path.abspath(model_dir)}")
        
        FORECAST_MODELS['forecast_model'] = joblib.load(os.path.join(model_dir, 'forecast_model.joblib'))
        FORECAST_MODELS['forecast_columns'] = list(pd.read_json(os.path.join(model_dir, 'model_columns.json'), typ='series'))
        print("-> Forecasting models loaded successfully.")

    except FileNotFoundError as e:
        print(f"FATAL ERROR loading forecast models: {e}")
        # We don't need to return None, just leave the dictionary empty
    
    # --- Supabase Connection ---
    # Construct path to the .env file in the backend root
    backend_dir = os.path.dirname(model_dir) # Go up one level from 'models/forecasting'
    load_dotenv(os.path.join(backend_dir, '.env')) 
    
    SUPABASE_URL = os.environ.get("SUPABASE_URL")
    SUPABASE_KEY = os.environ.get("SUPABASE_KEY")
    if SUPABASE_URL and SUPABASE_KEY:
        supabase = create_client(SUPABASE_URL, SUPABASE_KEY)
        print("-> Supabase client for forecasting initialized.")
    else:
        print("-> Supabase credentials not found, forecasting endpoint will be disabled.")

    # Helper function
    def create_forecast_features(df):
        df_features = df.copy()
        df_features['year'] = df_features.index.year
        df_features['month_sin'] = np.sin(2 * np.pi * df_features.index.month / 12.0)
        df_features['month_cos'] = np.cos(2 * np.pi * df_features.index.month / 12.0)
        return df_features

    # --- Define the endpoint within the init function ---
    @forecasting_bp.route("/forecast/xgboost/<string:city_name>", methods=["GET"])
    async def forecast_with_xgboost(city_name: str):
        if not FORECAST_MODELS or not supabase:
            return jsonify({"error": "Forecasting service or database is not available."}), 503

        try:
            city_name_lower = city_name.lower()
            response = supabase.table('agg_trans').select('date').ilike('city', city_name_lower).order('date', desc=True).limit(1).execute()
            
            if not response.data:
                return jsonify({"error": f"No historical data for '{city_name}'."}), 404
            
            last_date_str = response.data[0]['date']
            parts = last_date_str.split('-')
            last_historical_date = pd.to_datetime('01-' + parts[1] + '-' + parts[0], format='%d-%b-%y')

            future_dates = pd.date_range(start=last_historical_date + pd.DateOffset(months=1), periods=60, freq='MS')
            future_df = pd.DataFrame(index=future_dates)
            future_df['city'] = city_name_lower
            features_for_pred = create_forecast_features(future_df)
            
            features_encoded = pd.get_dummies(features_for_pred, columns=['city'])
            features_aligned = features_encoded.reindex(columns=FORECAST_MODELS['forecast_columns'], fill_value=0)
            
            predictions = FORECAST_MODELS['forecast_model'].predict(features_aligned)

            forecast_json = [{'date': date.strftime('%Y-%m-%d'), 'predicted_value': float(val)} for date, val in zip(future_dates, predictions)]
            
            return jsonify({ "city_forecasted": city_name, "monthly_forecast": forecast_json })

        except Exception as e:
            traceback.print_exc()
            return jsonify({"error": "An internal server error occurred."}), 500
            
    return forecasting_bp