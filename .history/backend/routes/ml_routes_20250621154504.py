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

# ==============================================================================
#                 MAYSNOUN'S BIT - REFACTORED FOR PRODUCTION
# ==============================================================================

import json

# --- 1. CONFIGURATION AND PATHING (Corrected for Production) ---
# This assumes your script is in a subfolder like '.../backend/routes/'
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT_DIR = os.path.dirname(os.path.dirname(SCRIPT_DIR)) # Goes up two levels to the project root

# Define paths to your artifacts in their proper folders
# --- 1. CONFIGURATION AND PATHING (Temporary Fix for Current Layout) ---
# All files are assumed to be in the same directory as this script.
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

# Define paths to all files, assuming they are in the same folder as this script
DATA_PATH = os.path.join(SCRIPT_DIR, 'transactions.csv')
CONFIG_PATH = os.path.join(SCRIPT_DIR, 'region_mapping.json')
XGB_MODEL_PATH = os.path.join(SCRIPT_DIR, 'forecast_model.joblib')
XGB_COLS_PATH = os.path.join(SCRIPT_DIR, 'model_columns.json')

# Define global constants
REGION_COL = 'region'
TARGET_COL = 'transaction_value'
FORECAST_HORIZON_MONTHS = 60

# --- 2. HELPER FUNCTIONS (Must be IDENTICAL to training script) ---

def load_and_preprocess_data(filepath, mapping_filepath):
    """
    Loads data, normalizes cities into regions using a mapping file,
    aggregates values by region, and performs date preprocessing.
    """
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"Data file not found at {filepath}")
    if not os.path.exists(mapping_filepath):
        raise FileNotFoundError(f"Region mapping file not found at {mapping_filepath}")

    df = pd.read_csv(filepath)
    df['city'] = df['city'].str.split(',').apply(lambda x: [c.strip() for c in x])
    df = df.explode('city')

    with open(mapping_filepath, 'r') as f:
        region_map = json.load(f)

    reverse_map = {}
    for region_name, cities in region_map.items():
        for city in cities:
            reverse_map[city] = region_name if region_name != "_DEFAULT_" else city

    df[REGION_COL] = df['city'].map(reverse_map)
    df[REGION_COL].fillna(df['city'], inplace=True)

    parts = df['date'].str.split('-', expand=True)
    df['date'] = pd.to_datetime('01-' + parts[1] + '-' + parts[0], format='%d-%b-%y')

    regional_df = df.groupby(['date', REGION_COL])[[TARGET_COL]].sum().reset_index()
    return regional_df.set_index('date')


def create_features(df):
    """
    Creates all time series features needed for the model.
    MUST BE IDENTICAL TO THE FUNCTION IN THE TRAINING SCRIPT.
    """
    df_features = df.copy()
    df_features['year'] = df_features.index.year
    df_features['quarter'] = df_features.index.quarter
    
    
    # Holiday/Event flags
    df_features['is_december'] = (df_features.index.month == 12).astype(int)
    # df_features['is_summer_peak'] = (df_features.index.month.isin([7, 8])).astype(int)

    # Add any crisis flags you used in training
    # df_features['is_economic_crisis_period'] = (df_features.index >= '2019-10-01').astype(int)

    # Lag and Rolling Features (grouped by REGION)
    df_features['lag_1'] = df_features.groupby(REGION_COL)[TARGET_COL].shift(1)
    df_features['lag_3'] = df_features.groupby(REGION_COL)[TARGET_COL].shift(3)
    df_features['lag_12'] = df_features.groupby(REGION_COL)[TARGET_COL].shift(12)
    df_features['rolling_mean_3'] = df_features.groupby(REGION_COL)[TARGET_COL].shift(1).rolling(window=3).mean()
    df_features['rolling_std_3'] = df_features.groupby(REGION_COL)[TARGET_COL].shift(1).rolling(window=3).std()
    
    return df_features

# --- 3. LOAD ARTIFACTS AT API STARTUP ---

print("-> Loading artifacts for XGBoost forecasting model...")

XGB_MODEL = joblib.load(XGB_MODEL_PATH)
XGB_MODEL_COLS = list(pd.read_json(XGB_COLS_PATH, typ='series'))

# Load the region mapping to be used by the endpoint
with open(CONFIG_PATH, 'r') as f:
    REGION_MAPPING_DATA = json.load(f)

# Build the reverse map (city -> region) once for efficiency
CITY_TO_REGION_MAP = {}
for region_name, cities in REGION_MAPPING_DATA.items():
    for city in cities:
        CITY_TO_REGION_MAP[city] = region_name if region_name != "_DEFAULT_" else city

# Load all historical data, now aggregated by REGION
ALL_HISTORICAL_DATA = load_and_preprocess_data(DATA_PATH, CONFIG_PATH)

print("--- XGBoost Forecasting Initialization Complete. API is ready. ---")


# --- 4. API ENDPOINT (REFACTORED) ---

@ml_routes.route("/forecast/xgboost/<string:city_name>", methods=["GET"])
async def forecast_with_xgboost(city_name: str):
    """
    Generates a forecast for a given city. It intelligently handles both
    single city names and combined strings (e.g., 'Baabda, Aley, Chouf') by
    mapping them to their correct forecast region.
    """
    print(f"\nReceived XGBoost forecast request for: '{city_name}'")
    try:
        # --- START: INTELLIGENT REGION MAPPING LOGIC ---
        
        # Step 1: Try a direct lookup first. This works for single cities like "Beirut".
        region_name = CITY_TO_REGION_MAP.get(city_name)

        # Step 2: If direct lookup fails, assume it's a combined string.
        if not region_name:
            print(f"-> Direct lookup for '{city_name}' failed. Treating as a combined string.")
            # Take the first city from the comma-separated string, and strip any whitespace.
            first_city_in_string = city_name.split(',')[0].strip()
            # Now, look up the region for that first city.
            region_name = CITY_TO_REGION_MAP.get(first_city_in_string)

        # Step 3: If it's still not found, then the city is truly unknown.
        if not region_name:
            return jsonify({"error": f"City '{city_name}' or its components could not be found in the region mapping."}), 404
        
        # --- END: INTELLIGENT REGION MAPPING LOGIC ---

        print(f"-> Mapped input to region '{region_name}'.")

        # The rest of the function from here onwards is exactly the same...
        region_history = ALL_HISTORICAL_DATA[ALL_HISTORICAL_DATA[REGION_COL] == region_name].copy()
        
        if region_history.empty:
            return jsonify({"error": f"No historical data found for region '{region_name}'."}), 404

        last_date = region_history.index.max()
        future_dates = pd.date_range(start=last_date + pd.DateOffset(months=1), periods=FORECAST_HORIZON_MONTHS, freq='MS')
        
        future_predictions = []
        for date in future_dates:
            temp_df = pd.DataFrame(index=[date], data={REGION_COL: region_name})
            combined_df = pd.concat([region_history, temp_df])
            features_df = create_features(combined_df)
            current_features = features_df.tail(1)
            current_features_encoded = pd.get_dummies(current_features, columns=[REGION_COL])
            current_features_aligned = current_features_encoded.reindex(columns=XGB_MODEL_COLS, fill_value=0)
            prediction = XGB_MODEL.predict(current_features_aligned)[0]
            future_predictions.append(float(prediction))
            region_history.loc[date] = {REGION_COL: region_name, TARGET_COL: float(prediction)}

        monthly_forecast_df = pd.DataFrame({'date': future_dates, 'predicted_value': future_predictions})
        yearly_forecast_df = monthly_forecast_df.resample('YE', on='date').sum()

        historical_json = region_history.iloc[:-FORECAST_HORIZON_MONTHS].reset_index().to_dict(orient='records')
        monthly_json = monthly_forecast_df.to_dict(orient='records')
        yearly_json = yearly_forecast_df.reset_index().rename(columns={'date': 'year', 'predicted_value': 'total_value'}).to_dict(orient='records')
        for item in yearly_json: item['year'] = item['year'].year

        print(f"-> Successfully generated forecast for region '{region_name}'.")
        
        # We now return the original requested name AND the region name for clarity
        return jsonify({
            "city_requested": city_name,
            "region_forecasted": region_name,
            "historical_data": historical_json,
            "monthly_forecast": monthly_json,
            "yearly_forecast": yearly_json
        })

    except Exception as e:
        import traceback
        traceback.print_exc()
        return jsonify({"error": "An internal server error occurred."}), 500j