import traceback
import asyncio
import os
import joblib
import 
from sklearn.preprocessing import MinMaxScaler
import pandas as pd
import numpy as np
from quart import Blueprint, jsonify, request, Response
from datetime import datetime # For date calculations
from dateutil.relativedelta import relativedelta # For easy date arithmetic 
from tensorflow.keras.models import load_model
import json
from db_connect import create_supabase
from model_downloader import get_models_path, get_enc_paths
# from .property_price_estimator import EnsemblePropertyPredictor
from .forecasting_lstm import ModelEvaluator


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


# mayssoun's bit
backend_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
artifacts_path = os.path.join(backend_dir, "model_artifacts")

# --- Function to load artifacts for a given granularity ---
# We use a dictionary to cache loaded models so we don't reload from disk on every request
LOADED_ARTIFACTS = {}

def get_artifacts(granularity: str):
    """Loads and caches the model, scaler, and features for a given granularity."""
    if granularity in LOADED_ARTIFACTS:
        return LOADED_ARTIFACTS[granularity]

    print(f"Loading artifacts for granularity: {granularity} for the first time...")
    suffix = "monthly" if granularity == 'M' else "yearly"
    
    # 1. Load Model
    model_path = os.path.join(artifacts_path, f"master_model_{suffix}.h5")
    model = load_model(model_path)
    
    # 2. Load Scaler - THIS IS THE CRITICAL FIX
    scaler_path = os.path.join(artifacts_path, f"master_scaler_{suffix}.pkl")
    with open(scaler_path, 'rb') as f:
        scaler = pickle.load(f)
        
    # 3. Load Feature List
    features_path = os.path.join(artifacts_path, f"model_features_{suffix}.json")
    with open(features_path, 'r') as f:
        features = json.load(f)

    artifacts = {
        "model": model,
        "scaler": scaler,
        "features": features,
        "look_back": model.input_shape[1]
    }
    LOADED_ARTIFACTS[granularity] = artifacts
    print(f"✅ Artifacts for '{granularity}' loaded and cached.")
    return artifacts

# ===================================================================
#                       2. THE API ENDPOINT
# ===================================================================

@ml_routes.route("/api/forecast", methods=["POST"])
async def forecast_route():
    """
    Receives a city, granularity, and periods to forecast, then returns a time series prediction.
    """
    try:
        # --- 1. Get User Input ---
        data = await request.get_json()
        city_name = data.get('city')
        granularity = data.get('granularity') # 'M' or 'Y'
        future_periods = data.get('periods', 12) # Default to 12 periods if not provided

        if not all([city_name, granularity]):
            return jsonify({"error": "Missing required fields: 'city' and 'granularity'."}), 400
        if granularity not in ['M', 'Y']:
            return jsonify({"error": "Invalid granularity. Must be 'M' for Monthly or 'Y' for Yearly."}), 400

        # --- 2. Load the Correct Artifacts ---
        try:
            artifacts = get_artifacts(granularity)
            model = artifacts['model']
            scaler = artifacts['scaler']
            feature_columns = artifacts['features']
            look_back = artifacts['look_back']
        except FileNotFoundError:
            return jsonify({"error": f"No trained model found for granularity '{granularity}'."}), 404

        # --- 3. Fetch Required Historical Data from Database ---
        supabase = get_supabase_client()
        response = supabase.table('transactions').select("created_at, transaction_value").eq('city', city_name).order('created_at', desc=False).execute()
        
        if not response.data or len(response.data) < look_back:
            return jsonify({"error": f"Not enough historical data for {city_name} to make a forecast. Need at least {look_back} data points."}), 404
        
        # Resample data to the correct granularity
        df = pd.DataFrame(response.data)
        df['date'] = pd.to_datetime(df['created_at'])
        df_resampled = df.set_index('date').resample(granularity)['transaction_value'].sum().reset_index()
        
        # Get the last 'look_back' records to form the input sequence
        historical_sequence = df_resampled.tail(look_back)
        last_date = historical_sequence['date'].max()


        # --- 4. Prepare the Input Sequence for the Model ---
        input_df = pd.DataFrame(columns=feature_columns)
        input_df['transaction_value'] = historical_sequence['transaction_value'].values
        input_df.fillna(0, inplace=True)
        
        city_col_name = f"city_{city_name}"
        if city_col_name in input_df.columns:
            input_df[city_col_name] = 1
        else:
            return jsonify({"error": f"City '{city_name}' was not found in the trained model."}), 400
        
        # Scale the data and reshape for LSTM
        scaled_input = scaler.transform(input_df[feature_columns])
        current_input = scaled_input.reshape((1, look_back, len(feature_columns)))

        # --- 5. Iteratively Predict the Future ---
        future_predictions_scaled = []
        for _ in range(future_periods):
            pred_scaled = model.predict(current_input, verbose=0)[0, 0]
            future_predictions_scaled.append(pred_scaled)
            
            new_step = current_input[0, -1, :].copy()
            new_step[0] = pred_scaled
            new_step = new_step.reshape(1, 1, len(feature_columns))
            current_input = np.append(current_input[:, 1:, :], new_step, axis=1)

        # --- 6. Inverse Transform the Predictions to get Real Values ---
        dummy_array = np.zeros((len(future_predictions_scaled), len(feature_columns)))
        dummy_array[:, 0] = future_predictions_scaled
        final_predictions = scaler.inverse_transform(dummy_array)[:, 0]

        # --- 7. Format the Output for the Frontend ---
        # Create future dates for the forecast
        future_dates = []
        current_date = last_date
        for _ in range(future_periods):
            if granularity == 'M':
                # Move to the next month's end
                current_date = current_date + pd.DateOffset(months=1) + pd.tseries.offsets.MonthEnd(0)
            elif granularity == 'Y':
                # Move to the next year's end
                current_date = current_date + pd.DateOffset(years=1) + pd.tseries.offsets.YearEnd(0)
            future_dates.append(current_date)
        
        forecast_result = [
            {"date": date.strftime('%Y-%m-%d'), "predicted_value": round(value, 2)}
            for date, value in zip(future_dates, final_predictions)
        ]

        return jsonify({"forecast": forecast_result})

    except Exception as e:
        # This will catch any error and return a helpful message
        return jsonify({"error": f"An internal error occurred: {str(e)}"}), 500

    


# Estimation of Current Property Price


# --- Helper function to fetch unique, non-empty values for a column ---
# This function is the robust one from our previous step. It is correct.
async def fetch_distinct_values(supabase_client, table_name, column_name):
    """
    Fetches sorted, unique, non-null values from a specified column.
    This version robustly handles mixed numeric types and bad data.
    """
    try:
        res = await supabase_client.from_(table_name) \
                                .select(column_name) \
                                .not_.is_(column_name, "null") \
                                .execute()

        if not res.data:
            return []

        raw_values = [item[column_name] for item in res.data]
        is_numeric_column = column_name in ["bedrooms", "bathrooms"]
        
        processed_values = set()
        if is_numeric_column:
            for val in raw_values:
                try:
                    num_val = float(val)
                    processed_values.add(int(num_val))
                except (ValueError, TypeError):
                    continue
        else:
            for val in raw_values:
                if isinstance(val, str) and val.strip():
                    processed_values.add(val.strip())
                elif val is not None:
                     processed_values.add(val)
        
        return sorted(list(processed_values))

    except Exception as e:
        print(f"ERROR fetching distinct values for '{column_name}': {e}")
        import traceback
        traceback.print_exc()
        return []


# === Optional: Serve Property Input Options (DYNAMICALLY FROM DATABASE) ===
# This route is also correct from our previous steps.
@ml_routes.route("/api/property_input_options")
async def get_property_input_options():
    supabase = None
    table_name = "properties"
    try:
        supabase = await create_supabase()
        results = await asyncio.gather(
            fetch_distinct_values(supabase, table_name, "district"),
            fetch_distinct_values(supabase, table_name, "type"),
            fetch_distinct_values(supabase, table_name, "bedrooms"),
            fetch_distinct_values(supabase, table_name, "bathrooms"),
            supabase.from_(table_name).select("size_m2").not_.is_("size_m2", "null").order("size_m2", desc=False).limit(1).execute(),
            supabase.from_(table_name).select("size_m2").not_.is_("size_m2", "null").order("size_m2", desc=True).limit(1).execute()
        )
        districts, types, bedroom_options, bathroom_options, min_size_res, max_size_res = results
        min_size = min_size_res.data[0]['size_m2'] if min_size_res.data else 30
        max_size = max_size_res.data[0]['size_m2'] if max_size_res.data else 1000

        return jsonify({
            "districts": districts,
            "types": types,
            "bedroom_options": bedroom_options,
            "bathroom_options": bathroom_options,
            "size_range": {"min": int(min_size), "max": int(max_size)}
        })

    except Exception as e:
        print(f"ERROR in /api/property_input_options: {e}")
        traceback.print_exc()
        return jsonify({"error": "Failed to fetch filter options from the database: " + str(e)}), 500
    finally:
        if supabase:
            pass


price_estimation_model = None
# <<< FIX 2: USE A RELIABLE, RELATIVE PATH >>>
# This is more portable and avoids Windows backslash issues.
# It assumes your script is run from the 'backend' directory.
MODEL_PATH = "models/property_price_model.joblib"


# === API Endpoint to Estimate Property Price ===
@ml_routes.route("/api/estimate_property_price", methods=["POST"])
async def estimate_price():
    global price_estimation_model
    try:
        if price_estimation_model is None:
            print(f"Attempting to load property estimation model from: {os.path.abspath(MODEL_PATH)}")
            if not os.path.exists(MODEL_PATH):
                raise FileNotFoundError(f"Model file not found at '{MODEL_PATH}'. Check path relative to server startup directory.")
            
            price_estimation_model = joblib.load(MODEL_PATH)
            print("Property estimation model loaded successfully.")

        data = await request.json

        # <<< FIX 3: USE CORRECT, CAPITALIZED JSON KEYS TO MATCH FRONTEND >>>
        df_input = pd.DataFrame([{
            "district": data.get("District"),
            "type": data.get("Type"),
            "size_m2": data.get("Size_m2"),
            "bedrooms": data.get("Num_Bedrooms"),
            "bathrooms": data.get("Num_Bathrooms")
        }])

        prediction = price_estimation_model.predict(df_input)
        return jsonify({"prediction": round(float(prediction[0]), 2)})

    except Exception as e:
        print("--- ERROR IN /api/estimate_property_price ---")
        traceback.print_exc()
        print("--- END OF ERROR ---")
        return jsonify({"error": str(e)}), 500