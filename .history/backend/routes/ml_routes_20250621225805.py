import os
import joblib
from sklearn.preprocessing import MinMaxScaler
import pandas as pd

from quart import Blueprint, jsonify, request, Response
from tensorflow.keras.models import load_model

from db_connect import create_supabase
from models import get_models_path, get_enc_paths


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

# 
# Mayssoun's bit ==============================================================================
import pandas as pd
import numpy as np
import joblib
import os
import traceback
from supabase import create_client, Client
from dotenv import load_dotenv

# --- 1. CONFIGURATION AND DATABASE CONNECTION ---
# Load environment variables from .env file
load_dotenv() 

# Get Supabase credentials from environment variables
SUPABASE_URL: str = os.environ.get("SUPABASE_URL")
SUPABASE_KEY: str = os.environ.get("SUPABASE_KEY")

# Check if credentials are provided
if not SUPABASE_URL or not SUPABASE_KEY:
    raise ValueError("FATAL ERROR: SUPABASE_URL and SUPABASE_KEY must be set in your .env file.")

# Initialize the Supabase client once when the API starts
supabase: Client = create_client(SUPABASE_URL, SUPABASE_KEY)
print("-> Supabase client initialized.")

# Define paths to local model artifacts
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(SCRIPT_DIR, 'forecast_model.joblib')
MODEL_COLS_PATH = os.path.join(SCRIPT_DIR, 'model_columns.json')

# Define global constants
GROUPING_KEY = 'city' # This is the name of your region column in Supabase
TARGET_COL = 'transaction_value'
FORECAST_HORIZON_MONTHS = 60

# --- 2. HELPER FUNCTIONS (These must mirror your final training script) ---

def map_user_selection_to_region(selection_string: str) -> str:
    """
    Intelligently maps the user's dropdown selection to the 
    lowercase region name used in the model.
    """
    # Use .lower() to make the check case-insensitive
    lowered_selection = selection_string.lower()
    
    if "tripoli" in lowered_selection and "akkar" in lowered_selection:
        return "tripoli"
    if "baabda" in lowered_selection and "aley" in lowered_selection:
        return "baabda"
    if "kesrouan" in lowered_selection and "jbeil" in lowered_selection:
        return "kesrouane"
    # For all other cases (like "Beirut"), convert the selection to lowercase.
    return selection_string.lower()

def create_features(df):
    """
    This MUST be an exact copy of the final, successful create_features 
    function from your training script.
    """
    df_features = df.copy()
    
    # Cyclical Features
    df_features['month_sin'] = np.sin(2 * np.pi * df_features.index.month / 12.0)
    df_features['month_cos'] = np.cos(2 * np.pi * df_features.index.month / 12.0)
    
    # Holiday/Event Flags
    df_features['is_december'] = (df_features.index.month == 12).astype(int)
    df_features['is_summer_peak'] = (df_features.index.month.isin([7, 8])).astype(int)

    # Core Time Features
    df_features['year'] = df_features.index.year
    df_features['quarter'] = df_features.index.quarter
    
    # Lag and Rolling Features
    df_features['lag_1'] = df_features.groupby(GROUPING_KEY)[TARGET_COL].shift(1)
    df_features['lag_3'] = df_features.groupby(GROUPING_KEY)[TARGET_COL].shift(3)
    df_features['lag_12'] = df_features.groupby(GROUPING_KEY)[TARGET_COL].shift(12)
    df_features['rolling_mean_3'] = df_features.groupby(GROUPING_KEY)[TARGET_COL].shift(1).rolling(window=3).mean()
    df_features['rolling_std_3'] = df_features.groupby(GROUPING_KEY)[TARGET_COL].shift(1).rolling(window=3).std()

    if 'month' in df_features.columns:
        df_features = df_features.drop('month', axis=1)

    return df_features

# --- 3. LOAD MODEL ARTIFACTS AT STARTUP ---
print("-> Loading local forecasting model artifacts...")
XGB_MODEL = joblib.load(MODEL_PATH)
XGB_MODEL_COLS = list(pd.read_json(MODEL_COLS_PATH, typ='series'))
# *** Notice: We are NOT loading any large CSV files into memory. ***
print("--- Forecasting API is ready (connected to Supabase). ---")


# --- 4. API ENDPOINT (Refactored for Supabase) ---
@ml_routes.route("/forecast/xgboost/<string:user_selection>", methods=["GET"])
async def forecast_with_xgboost(user_selection: str):
    """
    Generates a 5-year forecast by fetching targeted historical data from Supabase
    and using a robust recursive prediction method.
    """
    print(f"\nReceived forecast request for user selection: '{user_selection}'")
    try:
        # Step 1: Map user selection to the lowercase region name
        region_name = map_user_selection_to_region(user_selection)
        print(f"-> Mapped to region: '{region_name}'")

        # Step 2: Fetch historical data FOR THAT REGION ONLY from Supabase
        print(f"-> Querying Supabase for '{region_name}' history...")
        response = supabase.table('regional_transactions').select('*').eq(GROUPING_KEY, region_name).order('date').execute()
        
        if not response.data:
            return jsonify({"error": f"No historical data found for region '{region_name}' in the database."}), 404
        
        # Convert query result to a clean DataFrame
        region_history_df = pd.DataFrame(response.data)
        region_history_df['date'] = pd.to_datetime(region_history_df['date'])
        region_history_df = region_history_df.set_index('date').drop(columns=['id', 'created_at'], errors='ignore')
        # Ensure the column name is what the model expects
        region_history_df.rename(columns={'region': GROUPING_KEY}, inplace=True, errors='ignore')

        # Step 3: The robust recursive forecasting loop
        future_df_dynamic = region_history_df.copy()
        last_date = future_df_dynamic.index.max()
        future_dates = pd.date_range(start=last_date + pd.DateOffset(months=1), periods=FORECAST_HORIZON_MONTHS, freq='MS')
        
        for date in future_dates:
            features_df = create_features(future_df_dynamic)
            current_features = features_df.tail(1)
            current_features_encoded = pd.get_dummies(current_features, columns=[GROUPING_KEY])
            current_features_aligned = current_features_encoded.reindex(columns=XGB_MODEL_COLS, fill_value=0)
            
            prediction = XGB_MODEL.predict(current_features_aligned)[0]
            
            future_df_dynamic.loc[date] = {GROUPING_KEY: region_name, TARGET_COL: float(prediction)}

        # Step 4: Format the output for the frontend
        forecast_results = future_df_dynamic.tail(FORECAST_HORIZON_MONTHS)
        
        monthly_forecast_df = pd.DataFrame({
            'date': forecast_results.index,
            'predicted_value': forecast_results[TARGET_COL].values
        })
        yearly_forecast_df = monthly_forecast_df.resample('YE', on='date')['predicted_value'].sum().to_frame()

        historical_json = region_history_df.reset_index().to_dict(orient='records')
        monthly_json = monthly_forecast_df.to_dict(orient='records')
        yearly_json = yearly_forecast_df.reset_index().rename(columns={'date': 'year', 'predicted_value': 'total_value'}).to_dict(orient='records')
        for item in yearly_json: item['year'] = item['year'].year

        print(f"-> Successfully generated forecast for region '{region_name}'.")
        return jsonify({
            "city_forecasted": user_selection, # Send back the original selection
            "historical_data": historical_json,
            "monthly_forecast": monthly_json,
            "yearly_forecast": yearly_json
        })

    except Exception as e:
        traceback.print_exc()
        return jsonify({"error": "An internal server error occurred."}), 500