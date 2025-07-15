import os
import joblib
import pandas as pd
import numpy as np
import traceback
from quart import Blueprint, request, jsonify
from supabase import create_client, Client
from dotenv import load_dotenv
PRICE_MODELS = {}
# --- 1. SETUP AND MODEL LOADING ---
forecasting_bp = Blueprint('forecasting', __name__)

# SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
# BACKEND_DIR = os.path.dirname(SCRIPT_DIR)
# MODEL_DIR = os.path.join(BACKEND_DIR, 'forecasting_models')

def load_forecast_artifacts():
    """Loads all forecasting models and objects."""
    artifacts = {}
    try:
        print("--- Loading Forecasting Models ---")
        artifacts['forecast_model'] = joblib.load(os.path.join(MODEL_DIR, 'forecast_model.joblib'))
        artifacts['forecast_columns'] = list(pd.read_json(os.path.join(MODEL_DIR, 'model_columns.json'), typ='series'))
        return artifacts
    except FileNotFoundError as e:
        print(f"FATAL ERROR loading forecast models: {e}")
        return None

FORECAST_MODELS = load_forecast_artifacts()

# Supabase connection
load_dotenv(os.path.join(BACKEND_DIR, '.env')) # Make sure it finds the .env in the backend root
SUPABASE_URL = os.environ.get("SUPABASE_URL")
SUPABASE_KEY = os.environ.get("SUPABASE_KEY")
supabase = create_client(SUPABASE_URL, SUPABASE_KEY) if SUPABASE_URL and SUPABASE_KEY else None

# Helper functions...
def create_forecast_features(df):
    df_features = df.copy()
    df_features['year'] = df_features.index.year
    df_features['month_sin'] = np.sin(2 * np.pi * df_features.index.month / 12.0)
    df_features['month_cos'] = np.cos(2 * np.pi * df_features.index.month / 12.0)
    return df_features

# --- 2. API ENDPOINT ---

@forecasting_bp.route("/forecast/xgboost/<string:city_name>", methods=["GET"])
async def forecast_with_xgboost(city_name: str):
    """Generates a 5-year forecast for a given city."""
    if not FORECAST_MODELS or not supabase:
        return jsonify({"error": "Forecasting service or database is not available."}), 503

    try:
        # ... your existing forecasting logic goes here ...
        # (This part is already well-written)
        response = supabase.table('agg_trans').select('date').ilike('city', city_name).order('date', desc=True).limit(1).execute()
        if not response.data:
            return jsonify({"error": f"No historical data for '{city_name}'."}), 404
        
        last_date_str = response.data[0]['date']
        parts = last_date_str.split('-')
        last_historical_date = pd.to_datetime('01-' + parts[1] + '-' + parts[0], format='%d-%b-%y')

        future_dates = pd.date_range(start=last_historical_date + pd.DateOffset(months=1), periods=60, freq='MS')
        future_df = pd.DataFrame(index=future_dates)
        future_df['city'] = city_name.lower()
        features_for_pred = create_forecast_features(future_df)
        
        features_encoded = pd.get_dummies(features_for_pred, columns=['city'])
        features_aligned = features_encoded.reindex(columns=FORECAST_MODELS['forecast_columns'], fill_value=0)
        
        predictions = FORECAST_MODELS['forecast_model'].predict(features_aligned)

        forecast_json = [{'date': date.strftime('%Y-%m-%d'), 'predicted_value': float(val)} for date, val in zip(future_dates, predictions)]
        
        return jsonify({ "city_forecasted": city_name, "monthly_forecast": forecast_json })

    except Exception as e:
        traceback.print_exc()
        return jsonify({"error": "An internal server error occurred."}), 500