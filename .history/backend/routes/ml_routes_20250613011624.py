import traceback
import asyncio
import os
import joblib
import pickle
from sklearn.preprocessing import MinMaxScaler
import pandas as pd
import numpy as np
from quart import Blueprint, jsonify, request, Response
from datetime import datetime # For date calculations
from dateutil.relativedelta import relativedelta # For easy date arithmetic 
from tensorflow.keras.models import load_model
import json
from db_connect import create_supabase
from models import get_models_path, get_enc_paths
from routes import run_model
# from .property_price_estimator import EnsemblePropertyPredictor
# from .forecasting_lstm import ModelEvaluator
# from .train_master_model import MasterModelTrainer


# Create a Blueprint for your main routes
ml_routes = Blueprint('ml', __name__)

# lstm_routes = Blueprint('lstm', __name__)

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


# mayssoun's bit
### START OF XGBOOST FORECASTING LOGIC ###
# --- Configuration for the NEW XGBoost Forecasting Model ---
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
CSV_FILE_PATH = os.path.join(SCRIPT_DIR, 'transactions.csv') # Assuming the CSV is here
XGB_MODEL_PATH = os.path.join(SCRIPT_DIR, 'forecast_model.joblib')
XGB_COLS_PATH = os.path.join(SCRIPT_DIR, 'model_columns.json')
TARGET_COL = 'transaction_value'
FORECAST_HORIZON_MONTHS = 60

# ==============================================================================
# 2. HELPER FUNCTIONS FOR XGBOOST FORECASTING
#    (These are copied directly from our tested scripts)
# ==============================================================================

def load_and_preprocess_data(filepath):
    """Loads and prepares the historical transaction data from the CSV."""
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"FATAL: The data file was not found at {filepath}. The API cannot start.")
    
    df = pd.read_csv(filepath)
    parts = df['date'].str.split('-', expand=True)
    df['date_str'] = '01-' + parts[1] + '-' + parts[0]
    df['date'] = pd.to_datetime(df['date_str'], format='%d-%b-%y')
    df = df.set_index('date').drop(columns=['id', 'date_str'])
    df.sort_index(inplace=True)
    return df

def create_features(df):
    """Creates time series features needed for the model."""
    df['month'] = df.index.month
    df['year'] = df.index.year
    df['quarter'] = df.index.quarter
    df['lag_1'] = df.groupby('city')[TARGET_COL].shift(1)
    df['lag_3'] = df.groupby('city')[TARGET_COL].shift(3)
    df['lag_12'] = df.groupby('city')[TARGET_COL].shift(12)
    df['rolling_mean_3'] = df.groupby('city')[TARGET_COL].shift(1).rolling(window=3).mean()
    df['rolling_std_3'] = df.groupby('city')[TARGET_COL].shift(1).rolling(window=3).std()
    return df

print("-> Loading XGBoost forecasting model and historical data...")
if not os.path.exists(XGB_MODEL_PATH):
    raise FileNotFoundError(f"FATAL: XGBoost model not found at {XGB_MODEL_PATH}. Run the training script first.")
    
XGB_MODEL = joblib.load(XGB_MODEL_PATH)
XGB_MODEL_COLS = list(pd.read_json(XGB_COLS_PATH, typ='series'))
ALL_HISTORICAL_DATA = load_and_preprocess_data(CSV_FILE_PATH)

print("--- Initialization Complete. API is ready. ---")
@ml_routes.route("/forecast/xgboost/<string:city_name>", methods=["GET"])
async def forecast_with_xgboost(city_name: str):
    """
    Fast and efficient endpoint to generate a 5-year forecast using the pre-trained XGBoost model.
    """
    print(f"\nReceived XGBoost forecast request for city: {city_name}")
    try:
        available_cities = ALL_HISTORICAL_DATA['city'].unique()
        if city_name not in available_cities:
            return jsonify({"error": f"City '{city_name}' not found. Available cities are: {list(available_cities)}"}), 404

        city_history = ALL_HISTORICAL_DATA[ALL_HISTORICAL_DATA['city'] == city_name].copy()
        
        last_date = city_history.index.max()
        future_dates = pd.date_range(start=last_date + pd.DateOffset(months=1), periods=FORECAST_HORIZON_MONTHS, freq='MS')
        
        future_predictions = []
        for date in future_dates:
            temp_df = pd.DataFrame(index=[date], data={'city': city_name})
            combined_df = pd.concat([city_history, temp_df])
            features_df = create_features(combined_df)
            current_features = features_df.tail(1)
            current_features_encoded = pd.get_dummies(current_features, columns=['city'], prefix='city')
            current_features_aligned = current_features_encoded.reindex(columns=XGB_MODEL_COLS, fill_value=0)
            prediction = XGB_MODEL.predict(current_features_aligned)[0]
            future_predictions.append(float(prediction))
            city_history.loc[date] = {'city': city_name, TARGET_COL: float(prediction)}

        monthly_forecast_df = pd.DataFrame({'date': future_dates, 'predicted_value': future_predictions})
        yearly_forecast_df = monthly_forecast_df.resample('Y', on='date').sum()

        historical_json = city_history.iloc[:-FORECAST_HORIZON_MONTHS].reset_index().to_dict(orient='records')
        monthly_json = monthly_forecast_df.to_dict(orient='records')
        yearly_json = yearly_forecast_df.reset_index().rename(columns={'date': 'year', 'predicted_value': 'total_value'}).to_dict(orient='records')
        for item in yearly_json: item['year'] = item['year'].year

        print(f"-> Successfully generated XGBoost forecast for '{city_name}'.")
        return jsonify({
            "city_forecasted": city_name,
            "historical_data": historical_json,
            "monthly_forecast": monthly_json,
            "yearly_forecast": yearly_json
        })

    except Exception as e:
        import traceback
        traceback.print_exc()
        return jsonify({"error": "An internal server error occurred."}), 500












# backend_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
# artifacts_path = os.path.join(backend_dir, "model_artifacts")

# # --- Function to load artifacts for a given granularity ---
# # We use a dictionary to cache loaded models so we don't reload from disk on every request
# LOADED_ARTIFACTS = {}

# def get_artifacts(granularity: str):
#     """Loads and caches the model, scaler, and features for a given granularity."""
#     if granularity in LOADED_ARTIFACTS:
#         return LOADED_ARTIFACTS[granularity]

#     print(f"Loading artifacts for granularity: {granularity} for the first time...")
#     suffix = "monthly" if granularity == 'M' else "yearly"
    
#     # 1. Load Model
#     model_path = os.path.join(artifacts_path, f"master_model_{suffix}.h5")
#     model = load_model(model_path)
    
#     # 2. Load Scaler - THIS IS THE CRITICAL FIX
#     scaler_path = os.path.join(artifacts_path, f"master_scaler_{suffix}.pkl")
#     with open(scaler_path, 'rb') as f:
#         scaler = pickle.load(f)
        
#     # 3. Load Feature List
#     features_path = os.path.join(artifacts_path, f"model_features_{suffix}.json")
#     with open(features_path, 'r') as f:
#         features = json.load(f)

#     artifacts = {
#         "model": model,
#         "scaler": scaler,
#         "features": features,
#         "look_back": model.input_shape[1]
#     }
#     LOADED_ARTIFACTS[granularity] = artifacts
#     print(f"✅ Artifacts for '{granularity}' loaded and cached.")
#     return artifacts
# @ml_routes.route("/api/forecast", methods=["POST"])
# async def forecast_route():
#     print("\n--- FORECAST REQUEST RECEIVED ---")
#     try:
#         # --- Step 1: Parse User Input and Validate ---
#         print("[DEBUG] Step 1: Parsing user input...")
#         data = await request.get_json()
#         city_name = data.get('city')
#         granularity = data.get('granularity') # 'M' or 'Y'

#         # --- Validate the required inputs first ---
#         if not all([city_name, granularity]):
#             print("[ERROR] Missing required fields.")
#             return jsonify({"error": "Missing required fields: 'city' and 'granularity'."}), 400
        
#         if granularity not in ['M', 'Y']:
#             print(f"[ERROR] Invalid granularity: {granularity}")
#             return jsonify({"error": "Invalid granularity. Must be 'M' for Monthly or 'Y' for Yearly."}), 400

#         # --- Set the forecast periods based on the validated granularity ---
#         if granularity == 'Y':
#             future_periods = 5  # Forecast next 5 years
#         else: # This covers the 'M' case
#             future_periods = 60 # Forecast next 60 months (5 years)
        
#         print(f"[DEBUG] Params: city={city_name}, granularity={granularity}, periods={future_periods}")
#         # --- 2. Load the Correct Artifacts ---
#         print("[DEBUG] Step 2: Loading model artifacts...")
#         artifacts = get_artifacts(granularity)
#         model = artifacts['model']
#         scaler = artifacts['scaler']
#         feature_columns = artifacts['features']
#         look_back = artifacts['look_back']
#         print("[DEBUG] Artifacts loaded successfully.")

#         # --- 3. Fetch Required Historical Data from Database ---
#         print("[DEBUG] Step 3: Fetching data from Supabase...")
#         supabase = await create_supabase()
        
#         # PLEASE DOUBLE-CHECK THESE COLUMN NAMES
#         db_date_col = "date"
#         db_value_col = "transaction_value"
#         db_city_col = "city"

#         print(f"[DEBUG] Querying table 'transactions' for city '{city_name}'...")
#         response = await supabase.table('transactions').select(f"{db_date_col}, {db_value_col}").eq(db_city_col, city_name).order(db_date_col, desc=False).execute()
#         print("[DEBUG] Supabase query executed.")
        
#         if not response.data or len(response.data) < look_back:
#             print(f"[ERROR] Not enough data for {city_name}. Found {len(response.data) if response.data else 0} records, need {look_back}.")
#             return jsonify({"error": f"Not enough historical data for {city_name} to make a forecast. Need at least {look_back} data points."}), 404
        
#         print(f"[DEBUG] Found {len(response.data)} records for {city_name}.")
        
#         # --- Data Resampling ---
#         print("[DEBUG] Step 3.5: Resampling data...")
#         df = pd.DataFrame(response.data)
#         df['date'] = pd.to_datetime(df[db_date_col])
#         df_resampled = df.set_index('date').resample(granularity)[db_value_col].sum().reset_index()
        
#         historical_sequence = df_resampled.tail(look_back)
#         last_date = historical_sequence['date'].max()
#         print("[DEBUG] Data resampled successfully.")

#         # --- 4. Prepare the Input Sequence for the Model ---
#         print("[DEBUG] Step 4: Preparing input sequence...")
#         input_df = pd.DataFrame(columns=feature_columns)
#         input_df[db_value_col] = historical_sequence[db_value_col].values
#         input_df.fillna(0, inplace=True)
        
#         city_col_name = f"city_{city_name}"
#         if city_col_name in input_df.columns:
#             input_df[city_col_name] = 1
#         else:
#             print(f"[ERROR] City '{city_name}' not found in model features.")
#             return jsonify({"error": f"City '{city_name}' was not found in the trained model."}), 400
        
#         scaled_input = scaler.transform(input_df[feature_columns])
#         current_input = scaled_input.reshape((1, look_back, len(feature_columns)))
#         print("[DEBUG] Input sequence prepared.")

#         # --- 5. Iteratively Predict the Future ---
#         print("[DEBUG] Step 5: Starting prediction loop...")
#         future_predictions_scaled = []
#         for i in range(future_periods):
#             pred_scaled = model.predict(current_input, verbose=0)[0, 0]
#             future_predictions_scaled.append(pred_scaled)
            
#             new_step = current_input[0, -1, :].copy()
#             new_step[0] = pred_scaled
#             new_step = new_step.reshape(1, 1, len(feature_columns))
#             current_input = np.append(current_input[:, 1:, :], new_step, axis=1)
#         print(f"[DEBUG] Prediction loop finished after {future_periods} iterations.")

#         # --- 6. Inverse Transform ---
#         print("[DEBUG] Step 6: Inverse transforming predictions...")
#         dummy_array = np.zeros((len(future_predictions_scaled), len(feature_columns)))
#         dummy_array[:, 0] = future_predictions_scaled
#         final_predictions = scaler.inverse_transform(dummy_array)[:, 0]
#         print("[DEBUG] Inverse transform complete.")

    
#         print("[DEBUG] Step 7: Formatting final output...")
#         future_dates = []
#         current_date = last_date
#         for _ in range(future_periods):
#             if granularity == 'M':
#                 current_date = current_date + pd.DateOffset(months=1) + pd.tseries.offsets.MonthEnd(0)
#             elif granularity == 'Y':
#                 current_date = current_date + pd.DateOffset(years=1) + pd.tseries.offsets.YearEnd(0)
#             future_dates.append(current_date)
        
#         # This is the variable that holds the data we want to send back.
#         forecast_result = [
#             {"date": date.strftime('%Y-%m-%d'), "predicted_value": round(value, 2)}
#             for date, value in zip(future_dates, final_predictions)
#         ]

#         print("[DEBUG] Final output formatted.")
#         print("--- FORECAST REQUEST COMPLETED SUCCESSFULLY ---")
        
#         # --- THIS IS THE FIX ---
#         # Return the 'forecast_result' variable which contains the list of predictions.
#         return jsonify({"forecast": forecast_result})

#     except Exception as e:
#         # ... your error handling block ...
#         print("\n--- ❌ AN UNHANDLED EXCEPTION OCCURRED! ---")
#         traceback.print_exc()
#         return jsonify({"error": f"An internal server error occurred. Check the server logs for details."}), 500


# Estimation of Current Property Price


# --- Helper function to fetch unique, non-empty values for a column ---
# # This function is the robust one from our previous step. It is correct.
# async def fetch_distinct_values(supabase_client, table_name, column_name):
#     """
#     Fetches sorted, unique, non-null values from a specified column.
#     This version robustly handles mixed numeric types and bad data.
#     """
#     try:
#         res = await supabase_client.from_(table_name) \
#                                 .select(column_name) \
#                                 .not_.is_(column_name, "null") \
#                                 .execute()

#         if not res.data:
#             return []

#         raw_values = [item[column_name] for item in res.data]
#         is_numeric_column = column_name in ["bedrooms", "bathrooms"]
        
#         processed_values = set()
#         if is_numeric_column:
#             for val in raw_values:
#                 try:
#                     num_val = float(val)
#                     processed_values.add(int(num_val))
#                 except (ValueError, TypeError):
#                     continue
#         else:
#             for val in raw_values:
#                 if isinstance(val, str) and val.strip():
#                     processed_values.add(val.strip())
#                 elif val is not None:
#                      processed_values.add(val)
        
#         return sorted(list(processed_values))

#     except Exception as e:
#         print(f"ERROR fetching distinct values for '{column_name}': {e}")
#         import traceback
#         traceback.print_exc()
#         return []


# # === Optional: Serve Property Input Options (DYNAMICALLY FROM DATABASE) ===
# # This route is also correct from our previous steps.
# @ml_routes.route("/api/property_input_options")
# async def get_property_input_options():
#     supabase = None
#     table_name = "properties"
#     try:
#         supabase = await create_supabase()
#         results = await asyncio.gather(
#             fetch_distinct_values(supabase, table_name, "district"),
#             fetch_distinct_values(supabase, table_name, "type"),
#             fetch_distinct_values(supabase, table_name, "bedrooms"),
#             fetch_distinct_values(supabase, table_name, "bathrooms"),
#             supabase.from_(table_name).select("size_m2").not_.is_("size_m2", "null").order("size_m2", desc=False).limit(1).execute(),
#             supabase.from_(table_name).select("size_m2").not_.is_("size_m2", "null").order("size_m2", desc=True).limit(1).execute()
#         )
#         districts, types, bedroom_options, bathroom_options, min_size_res, max_size_res = results
#         min_size = min_size_res.data[0]['size_m2'] if min_size_res.data else 30
#         max_size = max_size_res.data[0]['size_m2'] if max_size_res.data else 1000

#         return jsonify({
#             "districts": districts,
#             "types": types,
#             "bedroom_options": bedroom_options,
#             "bathroom_options": bathroom_options,
#             "size_range": {"min": int(min_size), "max": int(max_size)}
#         })

#     except Exception as e:
#         print(f"ERROR in /api/property_input_options: {e}")
#         traceback.print_exc()
#         return jsonify({"error": "Failed to fetch filter options from the database: " + str(e)}), 500
#     finally:
#         if supabase:
#             pass


# price_estimation_model = None
# # <<< FIX 2: USE A RELIABLE, RELATIVE PATH >>>
# # This is more portable and avoids Windows backslash issues.
# # It assumes your script is run from the 'backend' directory.
# MODEL_PATH = "models/property_price_model.joblib"


# # === API Endpoint to Estimate Property Price ===
# @ml_routes.route("/api/estimate_property_price", methods=["POST"])
# async def estimate_price():
#     global price_estimation_model
#     try:
#         if price_estimation_model is None:
#             print(f"Attempting to load property estimation model from: {os.path.abspath(MODEL_PATH)}")
#             if not os.path.exists(MODEL_PATH):
#                 raise FileNotFoundError(f"Model file not found at '{MODEL_PATH}'. Check path relative to server startup directory.")
            
#             price_estimation_model = joblib.load(MODEL_PATH)
#             print("Property estimation model loaded successfully.")

#         data = await request.json

#         # <<< FIX 3: USE CORRECT, CAPITALIZED JSON KEYS TO MATCH FRONTEND >>>
#         df_input = pd.DataFrame([{
#             "district": data.get("District"),
#             "type": data.get("Type"),
#             "size_m2": data.get("Size_m2"),
#             "bedrooms": data.get("Num_Bedrooms"),
#             "bathrooms": data.get("Num_Bathrooms")
#         }])

#         prediction = price_estimation_model.predict(df_input)
#         return jsonify({"prediction": round(float(prediction[0]), 2)})

#     except Exception as e:
#         print("--- ERROR IN /api/estimate_property_price ---")
#         traceback.print_exc()
#         print("--- END OF ERROR ---")
#         return jsonify({"error": str(e)}), 500
    


# # Cache for loaded artifacts
# LOADED_ARTIFACTS = {}

# def get_artifacts(granularity: str):
#     """Loads and caches the model, scaler, and features for a given granularity."""
#     if granularity in LOADED_ARTIFACTS:
#         return LOADED_ARTIFACTS[granularity]

#     print(f"Loading artifacts for granularity: {granularity} for the first time...")
#     suffix = "monthly" if granularity == 'M' else "yearly"
    
#     # Use the model_artifacts directory created by train_master_model.py
#     artifacts_path = "model_artifacts"
    
#     # 1. Load Model
#     model_path = os.path.join(artifacts_path, f"master_model_{suffix}.h5")
#     model = load_model(model_path)
    
#     # 2. Load Scaler
#     scaler_path = os.path.join(artifacts_path, f"master_scaler_{suffix}.pkl")
#     with open(scaler_path, 'rb') as f:
#         scaler = pickle.load(f)
        
#     # 3. Load Feature List
#     features_path = os.path.join(artifacts_path, f"model_features_{suffix}.json")
#     with open(features_path, 'r') as f:
#         features = json.load(f)

#     artifacts = {
#         "model": model,
#         "scaler": scaler,
#         "features": features,
#         "look_back": model.input_shape[1]
#     }
#     LOADED_ARTIFACTS[granularity] = artifacts
#     print(f"✅ Artifacts for '{granularity}' loaded and cached.")
#     return artifacts

# @ml_routes.route("/api/historical_data", methods=["POST"])
# async def get_historical_data():
#     """
#     Fetch historical data for a specific city.
#     """
#     print("\n--- HISTORICAL DATA REQUEST RECEIVED ---")
#     try:
#         data = await request.get_json()
#         city_name = data.get('city')
#         granularity = data.get('granularity', 'Y')
#         start_year = data.get('start_year', 2012)
#         end_year = data.get('end_year', 2016)
        
#         if not city_name:
#             return jsonify({"error": "City name is required"}), 400
        
#         print(f"[DEBUG] Fetching historical data for {city_name}, {granularity}, {start_year}-{end_year}")
        
#         # Fetch data from Supabase
#         supabase = await create_supabase()
#         response = await supabase.table('transactions').select("date, transaction_value").eq("city", city_name).execute()
        
#         if not response.data:
#             return jsonify({"historical": []}), 200
        
#         # Process data
#         df = pd.DataFrame(response.data)
#         df['date'] = pd.to_datetime(df['date'])
        
#         # Filter by year range
#         df = df[(df['date'].dt.year >= start_year) & (df['date'].dt.year <= end_year)]
        
#         # Resample by granularity
#         df_resampled = df.set_index('date').resample(granularity)['transaction_value'].sum().reset_index()
        
#         # Format response
#         historical_data = [
#             {"date": row['date'].strftime('%Y-%m-%d'), "transaction_value": row['transaction_value']}
#             for _, row in df_resampled.iterrows()
#         ]
        
#         print(f"[DEBUG] Returning {len(historical_data)} historical data points")
#         return jsonify({"historical": historical_data})
        
#     except Exception as e:
#         print(f"\n--- ❌ HISTORICAL DATA ERROR ---")
#         traceback.print_exc()
#         return jsonify({"error": f"An internal server error occurred: {str(e)}"}), 500

# @ml_routes.route("/api/all_historical_data", methods=["POST"])
# async def get_all_historical_data():
#     """
#     Fetch historical data for all cities.
#     """
#     print("\n--- ALL HISTORICAL DATA REQUEST RECEIVED ---")
#     try:
#         data = await request.get_json()
#         granularity = data.get('granularity', 'Y')
#         start_year = data.get('start_year', 2012)
#         end_year = data.get('end_year', 2016)
        
#         print(f"[DEBUG] Fetching all historical data, {granularity}, {start_year}-{end_year}")
        
#         # Fetch data from Supabase
#         supabase = await create_supabase()
#         response = await supabase.table('transactions').select("date, city, transaction_value").execute()
        
#         if not response.data:
#             return jsonify({"historical": {}}), 200
        
#         # Process data
#         df = pd.DataFrame(response.data)
#         df['date'] = pd.to_datetime(df['date'])
        
#         # Filter by year range
#         df = df[(df['date'].dt.year >= start_year) & (df['date'].dt.year <= end_year)]
        
#         # Resample by granularity for each city
#         df_grouped = df.set_index('date').groupby('city')['transaction_value'].resample(granularity).sum()
#         df_resampled = df_grouped.reset_index()
        
#         # Group by city
#         historical_by_city = {}
#         for city in df_resampled['city'].unique():
#             city_data = df_resampled[df_resampled['city'] == city]
#             historical_by_city[city] = [
#                 {"date": row['date'].strftime('%Y-%m-%d'), "transaction_value": row['transaction_value']}
#                 for _, row in city_data.iterrows()
#             ]
        
#         print(f"[DEBUG] Returning historical data for {len(historical_by_city)} cities")
#         return jsonify({"historical": historical_by_city})
        
#     except Exception as e:
#         print(f"\n--- ❌ ALL HISTORICAL DATA ERROR ---")
#         traceback.print_exc()
#         return jsonify({"error": f"An internal server error occurred: {str(e)}"}), 500

# @ml_routes.route("/api/lstm_forecast", methods=["POST"])
# async def lstm_forecast():
#     """
#     LSTM forecasting endpoint that predicts next 5 years of transaction values for all cities.
#     """
#     print("\n--- LSTM FORECAST REQUEST RECEIVED ---")
#     try:
#         # Parse request data
#         data = await request.get_json()
#         granularity = data.get('granularity', 'Y')  # Default to yearly
        
#         # Validate granularity
#         if granularity not in ['Y', 'M']:
#             return jsonify({"error": "Invalid granularity. Must be 'Y' for Yearly or 'M' for Monthly."}), 400
        
#         # Set forecast periods
#         future_periods = 5 if granularity == 'Y' else 60  # 5 years or 60 months
        
#         print(f"[DEBUG] LSTM Forecast - granularity={granularity}, periods={future_periods}")
        
#         # Load artifacts
#         artifacts = get_artifacts(granularity)
#         model = artifacts['model']
#         scaler = artifacts['scaler']
#         feature_columns = artifacts['features']
#         look_back = artifacts['look_back']
        
#         # Get all cities from feature columns
#         city_columns = [col for col in feature_columns if col.startswith('city_')]
#         cities = [col.replace('city_', '') for col in city_columns]
        
#         print(f"[DEBUG] Found {len(cities)} cities in model: {cities}")
        
#         # Fetch data from Supabase
#         supabase = await create_supabase()
#         response = await supabase.table('transactions').select("date, city, transaction_value").execute()
        
#         if not response.data:
#             return jsonify({"error": "No data found in transactions table."}), 404
        
#         # Process data
#         df = pd.DataFrame(response.data)
#         df['date'] = pd.to_datetime(df['date'])
        
#         # Resample data by granularity
#         df_grouped = df.set_index('date').groupby('city')['transaction_value'].resample(granularity).sum()
#         df_resampled = df_grouped.reset_index()
        
#         # Generate forecasts for all cities
#         all_forecasts = {}
        
#         for city in cities:
#             print(f"[DEBUG] Generating forecast for city: {city}")
            
#             # Get historical data for this city
#             city_data = df_resampled[df_resampled['city'] == city].sort_values('date')
            
#             if len(city_data) < look_back:
#                 print(f"[WARNING] Not enough data for {city}. Skipping.")
#                 continue
            
#             # Get last look_back periods
#             historical_sequence = city_data.tail(look_back)
#             last_date = historical_sequence['date'].max()
            
#             # Prepare input for model
#             input_df = pd.DataFrame(columns=feature_columns)
#             input_df['transaction_value'] = historical_sequence['transaction_value'].values
#             input_df.fillna(0, inplace=True)
            
#             # Set city one-hot encoding
#             city_col_name = f"city_{city}"
#             if city_col_name in feature_columns:
#                 input_df[city_col_name] = 1
            
#             # Scale input
#             scaled_input = scaler.transform(input_df[feature_columns])
#             current_input = scaled_input.reshape((1, look_back, len(feature_columns)))
            
#             # Generate predictions
#             future_predictions_scaled = []
#             for i in range(future_periods):
#                 pred_scaled = model.predict(current_input, verbose=0)[0, 0]
#                 future_predictions_scaled.append(pred_scaled)
                
#                 # Update input for next prediction
#                 new_step = current_input[0, -1, :].copy()
#                 new_step[0] = pred_scaled
#                 # Maintain city encoding
#                 if city_col_name in feature_columns:
#                     city_col_index = feature_columns.index(city_col_name)
#                     new_step[city_col_index] = 1
                
#                 new_step = new_step.reshape(1, 1, len(feature_columns))
#                 current_input = np.append(current_input[:, 1:, :], new_step, axis=1)
            
#             # Inverse transform predictions
#             dummy_array = np.zeros((len(future_predictions_scaled), len(feature_columns)))
#             dummy_array[:, 0] = future_predictions_scaled
#             final_predictions = scaler.inverse_transform(dummy_array)[:, 0]
            
#             # Generate future dates
#             future_dates = []
#             current_date = last_date
#             for _ in range(future_periods):
#                 if granularity == 'M':
#                     current_date = current_date + pd.DateOffset(months=1) + pd.tseries.offsets.MonthEnd(0)
#                 elif granularity == 'Y':
#                     current_date = current_date + pd.DateOffset(years=1) + pd.tseries.offsets.YearEnd(0)
#                 future_dates.append(current_date)
            
#             # Format forecast for this city
#             city_forecast = [
#                 {"date": date.strftime('%Y-%m-%d'), "predicted_value": round(value, 2)}
#                 for date, value in zip(future_dates, final_predictions)
#             ]
            
#             all_forecasts[city] = city_forecast
        
#         print(f"[DEBUG] Generated forecasts for {len(all_forecasts)} cities")
#         print("--- LSTM FORECAST REQUEST COMPLETED SUCCESSFULLY ---")
        
#         return jsonify({
#             "granularity": granularity,
#             "forecast_periods": future_periods,
#             "forecasts": all_forecasts
#         })
        
#     except Exception as e:
#         print("\n--- ❌ LSTM FORECAST ERROR ---")
#         traceback.print_exc()
#         return jsonify({"error": f"An internal server error occurred: {str(e)}"}), 500

