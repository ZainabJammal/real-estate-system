import os
import joblib
from sklearn.preprocessing import MinMaxScaler
import pandas as pd
import numpy as np
import traceback
from supabase import create_client, Client
from dotenv import load_dotenv
from models import get_models_path, get_enc_paths
from quart import Blueprint, jsonify, request
from tensorflow.keras.models import load_model
from db_connect import create_supabase




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
SUPABASE_URL: str = os.environ.get("SUPABASE_URL")
SUPABASE_KEY: str = os.environ.get("SUPABASE_KEY")

if not SUPABASE_URL or not SUPABASE_KEY:
    raise ValueError("FATAL ERROR: SUPABASE_URL and SUPABASE_KEY must be set in your .env file.")

# Initialize the Supabase client
supabase: Client = create_client(SUPABASE_URL, SUPABASE_KEY)
print("-> Supabase client initialized.")

 

# SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
# BACKEND_DIR = os.path.dirname(SCRIPT_DIR)

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
ROUTES_DIR = os.path.dirname(SCRIPT_DIR)
BACKEND_DIR= os.path.dirname(ROUTES_DIR)
FORECAST_MODEL_DIR = os.path.join(BACKEND_DIR, 'forecasting_models')

# SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(SCRIPT_DIR, 'forecast_model.joblib')
MODEL_COLS_PATH = os.path.join(SCRIPT_DIR, 'model_columns.json')

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

# 
#         # --- Load the specific model you trained ---
#         model_path = os.path.join(model_dir, 'model_apartment_advanced.joblib')
        
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
  
        print("\n--- RAW SUPABASE RESPONSE DEBUG ---")
        print(f"Type of response object: {type(response)}")
        print(f"Response has 'data' attribute: {'data' in dir(response)}")
        
        # Check if the response contains data and print it
        if response.data:
            print(f"Number of records returned: {len(response.data)}")
            print(f"First record received from Supabase: {response.data[0]}") 
        else:
            print("!!! CRITICAL: `response.data` is EMPTY or does not exist.")
            print(f"Full response object: {response}")
        
        print("-----------------------------------\n")

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

