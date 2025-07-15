import os
import joblib
from sklearn.preprocessing import MinMaxScaler
import pandas as pd
import numpy as np
import traceback
from supabase import create_client, Client
from dotenv import load_dotenv

from quart import Blueprint, jsonify, request
from tensorflow.keras.models import load_model
from db_connect import create_supabase
from models import get_models_path, get_enc_paths

import warnings
warnings.filterwarnings("ignore", category=UserWarning, message="X does not have valid feature names")


# Create a Blueprint for your main routes
ml_routes = Blueprint('ml', __name__)


# Get models and encoders paths
trans_path, prop_path = get_models_path()
city_trans_enc_path, city_enc_path, dis_enc_path, prov_enc_path, type_enc_path = get_enc_paths()

# Import models and encoders
trans_model = joblib.load(trans_path)
prop_model = joblib.load(prop_path)
city_t_enc = joblib.load(city_trans_enc_path)
city_enc = joblib.load(city_enc_path)
dis_enc = joblib.load(dis_enc_path)
prov_enc = joblib.load(prov_enc_path)
type_enc = joblib.load(type_enc_path)

@ml_routes.route("/predict_transaction", methods=["GET","POST"])
async def predict_trans():
    """API endpoint for making predictions."""
    try:
        data = await request.get_json()
        
        # Convert input JSON to DataFrame
        input_data = pd.DataFrame([data])
        input_data["City"] = city_t_enc.transform([input_data["City"].iloc[0]])[0]

        # Convert data to float
        input_data = input_data.astype(float)
        # input_array = input_data.values.reshape(1, -1)

        # Make prediction
        prediction = trans_model.predict(input_data)
        print(prediction)

        return jsonify({"prediction": float(prediction[0])}) # Changed to prediction[0] for consistency

    except Exception as e:
        return jsonify({"error": str(e)}), 500
    
@ml_routes.route("/predict_property", methods=["POST"])
async def predict_prop():
    """API endpoint for making predictions."""
    try:
        data = await request.get_json()
        
        # Convert JSON to DataFrame
        input_data = pd.DataFrame([data])  

        # Encode categorical values correctly
        input_data["City"] = city_enc.transform([input_data["City"].iloc[0]])[0]
        input_data["District"] = dis_enc.transform([input_data["District"].iloc[0]])[0]
        input_data["Province"] = prov_enc.transform([input_data["Province"].iloc[0]])[0]
        input_data["Type"] = type_enc.transform([input_data["Type"].iloc[0]])[0]

        # Ensure the shape is correct
        input_data = input_data.astype(float)  # Convert to float for ML model
        input_array = input_data.values.reshape(1, -1)  # Ensure (1, n_features) shape

        print("Processed input:", input_array)  # Debugging

        # Make prediction
        prediction = prop_model.predict(input_array)
        print("Prediction:", prediction)

        return jsonify({"prediction": float(prediction[0])})

    except Exception as e:
        return jsonify({"error": str(e)}), 500




 
# Mayssoun's bit

# --- 1. CONFIGURATION AND DATABASE CONNECTION ---
# Load environment variables from .env file
load_dotenv() 

# Get Supabase credentials from environment variables
SUPABASE_URL: str = os.environ.get("SUPABASE_URL")
SUPABASE_KEY: str = os.environ.get("SUPABASE_KEY")

if not SUPABASE_URL or not SUPABASE_KEY:
    raise ValueError("FATAL ERROR: SUPABASE_URL and SUPABASE_KEY must be set in your .env file.")

# Initialize the Supabase client
supabase: Client = create_client(SUPABASE_URL, SUPABASE_KEY)
print("-> Supabase client initialized.")

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
BACKEND_DIR = os.path.dirname(SCRIPT_DIR)
# Define paths to the different model directories
PRICE_MODEL_DIR = os.path.join(BACKEND_DIR, 'final_models')
# FORECAST_MODEL_DIR = os.path.join(BACKEND_DIR, 'forecasting_models')

def load_all_artifacts():
    """Loads all necessary models and objects into memory when the app starts."""
    artifacts = {}
    try:
        print("--- Loading All ML Models and Artifacts ---")
        
        # --- Price Estimation Models ---
        # Note: We are only loading the models that were successfully created.
        print(f"Loading price models from: {PRICE_MODEL_DIR}")
        artifacts['model_apartment'] = joblib.load(os.path.join(PRICE_MODEL_DIR, 'model_apartment.joblib'))
        artifacts['model_general_fallback'] = joblib.load(os.path.join(PRICE_MODEL_DIR, 'model_general_fallback.joblib'))
        artifacts['kmeans_model'] = joblib.load(os.path.join(PRICE_MODEL_DIR, 'kmeans_model.joblib'))
        # Check if specialist models for office and shop exist before trying to load them
        office_model_path = os.path.join(PRICE_MODEL_DIR, 'model_office.joblib')
        shop_model_path = os.path.join(PRICE_MODEL_DIR, 'model_shop.joblib')
        if os.path.exists(office_model_path):
            artifacts['model_office'] = joblib.load(office_model_path)
        if os.path.exists(shop_model_path):
            artifacts['model_shop'] = joblib.load(shop_model_path)
        print("-> Price estimation models loaded.")

        # --- Forecasting Models ---
        # print(f"Loading forecasting models from: {FORECAST_MODEL_DIR}")
        # ML_artifacts['forecast_model'] = joblib.load(os.path.join(FORECAST_MODEL_DIR, 'forecast_model.joblib'))
        # ML_artifacts['forecast_columns'] = list(pd.read_json(os.path.join(FORECAST_MODEL_DIR, 'model_columns.json'), typ='series'))
        print("-> Forecasting models loaded.")
        
        print("--- All Artifacts Loaded Successfully ---")
        return ML_artifacts

    except FileNotFoundError as e:
        print(f"FATAL ERROR: Could not load a required model file. {e}")
        print("Please ensure all training scripts have been run and models are in the correct directories.")
        return None


# Transaction forecasting API 
# # Define paths to local model artifacts
# SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
# MODEL_PATH = os.path.join(SCRIPT_DIR, 'forecast_model.joblib')
# MODEL_COLS_PATH = os.path.join(SCRIPT_DIR, 'model_columns.json')

# Define global constants
# GROUPING_KEY must match the column name in your training data AND your database
GROUPING_KEY = 'city' 
TARGET_COL = 'transaction_value'
FORECAST_HORIZON_MONTHS = 60

# --- 2. HELPER FUNCTIONS (These must mirror your final training script) ---

def map_user_selection_to_city(selection_string: str) -> str:
    """
    Maps the user's dropdown selection to the specific custom region names
    used in the database, ensuring it's lowercase for matching.
    """
    if "Tripoli, Akkar" in selection_string:
        return "Tripoli"
    if "Baabda, Aley, Chouf" in selection_string:
        return "Baabda"
    if "Kesrouan, Jbeil" in selection_string:
        return "Kesrouan"
    return selection_string.lower()

def create_features(df):
    """Optimal features for a small dataset with a long forecast horizon."""
    df_features = df.copy()
    df_features['year'] = df_features.index.year
    df_features['month_sin'] = np.sin(2 * np.pi * df_features.index.month / 12.0)
    df_features['month_cos'] = np.cos(2 * np.pi * df_features.index.month / 12.0)
    df_features['is_december'] = (df_features.index.month == 12).astype(int)
    df_features['is_summer_peak'] = (df_features.index.month.isin([7, 8])).astype(int)
    return df_features

# --- 3. LOAD MODEL ARTIFACTS AT STARTUP ---
print("-> Loading local forecasting model artifacts...")
XGB_MODEL = joblib.load(MODEL_PATH)
XGB_MODEL_COLS = list(pd.read_json(MODEL_COLS_PATH, typ='series'))
print("--- Forecasting API is ready (connected to Supabase). ---")

# --- 4. API ENDPOINT (Simplified for "Best Guess" model) ---
@ml_routes.route("/forecast/xgboost/<string:user_selection>", methods=["GET"])
async def forecast_with_xgboost(user_selection: str):
    """
    Generates a 5-year 'Best Guess' forecast by predicting all future
    months in a single batch.
    """
    print(f"\nReceived 'Best Guess' forecast request for: '{user_selection}'")
    # if 'forecast_model' not in ML_ARTIFACTS or not supabase:
    #     return jsonify({"error": "Forecasting service or database is not available."}), 503


    try:
        city_name = map_user_selection_to_city(user_selection)
        print(f"-> Mapped to city: '{city_name}'")

        # step A: fetch histo data first
        print(f"-> Querying Supabase for '{city_name}' historical data...")
        response = supabase.table('agg_trans').select('*').ilike(GROUPING_KEY, city_name).order('date').execute()
        
          # --- START: NEW "SEE EVERYTHING" DEBUG BLOCK ---
        print("\n--- RAW SUPABASE RESPONSE DEBUG ---")
        print(f"Type of response object: {type(response)}")
        print(f"Response has 'data' attribute: {'data' in dir(response)}")
        
        # Check if the response contains data and print it
        if response.data:
            print(f"Number of records returned: {len(response.data)}")
            # Print the first record to see its structure
            print(f"First record received from Supabase: {response.data[0]}") 
        else:
            print("!!! CRITICAL: `response.data` is EMPTY or does not exist.")
            print(f"Full response object: {response}")
        
        print("-----------------------------------\n")
        # --- END OF DEBUG BLOCK ---

        if not response.data:
            return jsonify({"error": f"No historical data found for city '{city_name}' in the database."}), 404
        
        # Prepare the historical data DataFrame
        hist_df = pd.DataFrame(response.data)

        parts = hist_df['date'].str.split('-', expand=True)
        hist_df['date'] = pd.to_datetime('01-' + parts[1] + '-' + parts[0], format='%d-%b-%y')
        city_history_df = hist_df.set_index('date')

        # Standardize the grouping key column to lowercase to prevent case-mismatch
        city_history_df[GROUPING_KEY] = city_history_df[GROUPING_KEY].str.lower()


        # Step B: Create a DataFrame for all 60 future months

         # Find the last date from the historical data we just fetched.
        last_historical_date = hist_df['date'].iloc[-1].year

         # Start the forecast on the first day of the month AFTER the last historical date.
        start_year = last_historical_date + 1
        start_date = pd.to_datetime(f'{start_year}-01-01')
        future_dates = pd.date_range(start=start_date, periods=FORECAST_HORIZON_MONTHS, freq='MS')

        future_df = pd.DataFrame(index=future_dates)
        # Create features for the future DataFrame
        future_df[GROUPING_KEY] = city_name
        features_for_pred = create_features(future_df)

        
        # Step 4: One-hot encode the city and align columns
        features_encoded = pd.get_dummies(features_for_pred, columns=[GROUPING_KEY])
        features_aligned = features_encoded.reindex(columns=XGB_MODEL_COLS, fill_value=0)
        
        # Step 5: Make predictions for ALL 60 months at once
        predictions = XGB_MODEL.predict(features_aligned)
        print(f"-> Generated {len(predictions)} future predictions in a single batch.")

        # --- Part C: Format the Final JSON Response ---

        historical_json = hist_df.to_dict(orient='records')
        
        monthly_forecast_df = pd.DataFrame({'date': future_dates, 'predicted_value': predictions})
        yearly_forecast_df = monthly_forecast_df.resample('YE', on='date')['predicted_value'].sum().to_frame()
        # features_aligned = features_encoded.reindex(columns=ML_ARTIFACTS['forecast_columns'], fill_value=0)
        
        # predictions = ML_ARTIFACTS['forecast_model'].predict(features_aligned)

        monthly_json = monthly_forecast_df.to_dict(orient='records')
        yearly_json = yearly_forecast_df.reset_index().rename(columns={'date': 'year', 'predicted_value': 'total_value'}).to_dict(orient='records')
        for item in yearly_json: item['year'] = item['year'].year

        print(f"-> Successfully generated forecast for '{city_name}'.")
        return jsonify({
            "city_forecasted": user_selection,
            "historical_data": historical_json,
            "monthly_forecast": monthly_json,
            "yearly_forecast": yearly_json
        })

    except Exception as e:
        traceback.print_exc()
        return jsonify({"error": "An internal server error occurred."}), 500






# Price Estimstion API 
MODELS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'final_models')
SPECIALIST_TYPES = ['apartment', 'office', 'shop']
MODELS = {}

# --- 2. LOAD MODELS AT STARTUP ---

def load_models():
    """
    Loads all necessary model artifacts from disk into memory.
    This is done once when the application starts for performance.
    """
    print("--- Loading Price Estimator Models ---")
    try:
        # Load the essential K-Means model for location clustering
        kmeans_path = os.path.join(MODELS_DIR, 'kmeans_model.joblib')
        MODELS['kmeans'] = joblib.load(kmeans_path)
        print(f"-> Successfully loaded K-Means model from '{kmeans_path}'")

        # Load the specialist models
        for prop_type in SPECIALIST_TYPES:
            model_path = os.path.join(MODELS_DIR, f'model_{prop_type}.joblib')
            if os.path.exists(model_path):
                MODELS[prop_type] = joblib.load(model_path)
                print(f"-> Successfully loaded specialist model for '{prop_type}'")
            else:
                print(f"!! WARNING: Specialist model for '{prop_type}' not found at '{model_path}'")

        # Load the general fallback model
        fallback_path = os.path.join(MODELS_DIR, 'model_general_fallback.joblib')
        MODELS['general_fallback'] = joblib.load(fallback_path)
        print(f"-> Successfully loaded General Fallback model from '{fallback_path}'")
        
        print("--- All Price Estimator Models Loaded ---")

    except FileNotFoundError as e:
        print(f"FATAL ERROR: Could not find a required model file: {e}")
        print("Please ensure the 'final_models' directory is correctly placed and contains all necessary .joblib files.")
        # In a real app, you might want to exit or prevent the server from starting
        raise e
    except Exception as e:
        print(f"An unexpected error occurred during model loading: {e}")
        traceback.print_exc()
        raise e

# Load the models right away when the script is imported
load_models()


# --- 3. API ENDPOINT ---
# In your ml_routes.py file

    @ml_routes.route("/predict_price", methods=["POST"])
    async def predict_price():
        if not artifacts:
            return jsonify({"error": "Models are not available due to a startup error."}), 503

        try:
            # --- DEBUG CHECKPOINT 1: Request Received ---
            print("\n[DEBUG] 1. Received request for /predict_price.")

            data = await request.get_json()
            input_df = pd.DataFrame([data])
            
            property_type = input_df['type'].iloc[0].lower().replace('/', '_')
            print(f"[DEBUG] 2. Property type identified as: '{property_type}'")
            
            # --- Step 1: Pre-process the input ---
            kmeans_model = artifacts['kmeans_model']
            input_df['location_cluster'] = kmeans_model.predict(input_df[['latitude', 'longitude']]).astype(str)
            print("[DEBUG] 3. Assigned location_cluster using K-Means model.")
            
            # --- Step 2: The "Router" ---
            model_to_use = None
            is_specialist = False

            if property_type == 'apartment' and 'model_apartment' in artifacts:
                print("[DEBUG] 4. Selecting 'Specialist Apartment Model'.")
                model_to_use = ML_ARTIFACTS['model_apartment']
                is_specialist = True
            elif property_type == 'office' and 'model_office' in ML_ARTIFACTS:
                print("[DEBUG] 4. Selecting 'Specialist Office Model'.")
                model_to_use = ML_ARTIFACTS['model_office']
                is_specialist = True
            elif property_type == 'shop' and 'model_shop' in ML_ARTIFACTS:
                print("[DEBUG] 4. Selecting 'Specialist Shop Model'.")
                model_to_use = ML_ARTIFACTS['model_shop']
                is_specialist = True
            else:
                print(f"[DEBUG] 4. Selecting 'General Fallback Model' for type '{property_type}'.")
                model_to_use = ML_ARTIFACTS['model_general_fallback']

            # --- Step 3: Make the prediction ---
            print("[DEBUG] 5. Calling model.predict()...")
            prediction_log = model_to_use.predict(input_df)
            print(f"[DEBUG] 6. Raw prediction (log scale) successful: {prediction_log}")
            
            prediction_dollars = np.expm1(prediction_log)[0]
            
            # --- Step 4: Format and return the response ---
            if is_specialist:
                response_data = {
                    "status": "high_confidence",
                    "prediction": round(prediction_dollars, -3)
                }
            else:
                mae_fallback = 224000 
                lower_bound = max(0, prediction_dollars - mae_fallback)
                upper_bound = prediction_dollars + mae_fallback
                response_data = {
                    "status": "low_confidence_range",
                    "estimated_center": round(prediction_dollars, -3),
                    "estimated_range": [round(lower_bound, -3), round(upper_bound, -3)]
                }
            
            print(f"[DEBUG] 7. Sending final JSON response.")
            return jsonify(response_data)

        except Exception as e:
            # This will catch any errors that happen during the process
            print(f"!!! ERROR during prediction: {e}")
            traceback.print_exc()
            return jsonify({"error": "An internal server error occurred.", "details": str(e)}), 500
        
