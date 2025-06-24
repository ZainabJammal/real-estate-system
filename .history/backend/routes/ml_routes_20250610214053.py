import traceback
import asyncio
import os
import joblib
from sklearn.preprocessing import MinMaxScale
import pandas as pd
import numpy as np
from quart import Blueprint, jsonify, request, Response
from datetime import datetime # For date calculations
from dateutil.relativedelta import relativedelta # For easy date arithmetic 
from tensorflow.keras.models import load_model
r
import json
from db_connect import create_supabase
from model_downloader import get_models_path, get_enc_paths
from .property_price_estimator import EnsemblePropertyPredictor
from .forecasting_lstm import LSTMPredictor

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
@ml_routes.route("/forecast_transaction", methods=["GET","POST"])
async def forecasting_transaction():
    try:
        data = await request.get_json()
        input_data = pd.DataFrame([data])
        input_data["City"] = city_t_enc.transform([input_data["City"].iloc[0]])[0]
        input_data = input_data.astype(float)
        prediction = trans_model.predict(input_data)
        return jsonify({"prediction": float(prediction[0])})
    except Exception as e:
        return jsonify({"error": str(e)}), 500

# --- LOAD THE MODEL AND FEATURES ONCE WHEN THE APP STARTS ---
print("Loading master LSTM model...")
MASTER_MODEL = load_model("master_transaction_model.h5")
print("Model loaded.")

print("Loading model feature list...")
with open('model_features.json', 'r') as f:
    MODEL_FEATURES = json.load(f)
print(f"Features loaded: {MODEL_FEATURES}")

# Create a new scaler instance, we will use it for predictions
SCALER = MinMaxScaler(feature_range=(0, 1))

dummy_data_for_scaler = np.zeros((2, len(MODEL_FEATURES)))
SCALER.fit(dummy_data_for_scaler)


@ml_routes.route("/api/predict_transaction_timeseries", methods=["POST"])
async def predict_transaction_timeseries_route():
    try:
        data = await request.get_json()
        city_name = data.get('city_name')
        # The API now expects the historical values to be passed in
        historical_values = data.get('historical_values') # e.g., a list of the last 12 values
        
        look_back = 12 # Must match the trained model
        
        if len(historical_values) < look_back:
            return jsonify({"error": f"You must provide at least {look_back} historical values."}), 400

        # --- PREPARE THE INPUT FOR THE MULTIVARIATE MODEL ---
        # 1. Create a DataFrame for the input sequence
        df_input = pd.DataFrame(columns=MODEL_FEATURES)
        df_input['value'] = historical_values[-look_back:]
        
        # 2. Fill the one-hot encoded columns
        for col in MODEL_FEATURES:
            if col.startswith('city_'):
                # Set the column for the requested city to 1, others to 0
                if col == f"city_{city_name}":
                    df_input[col] = 1
                else:
                    df_input[col] = 0

        # 3. Scale and reshape the data
        input_sequence = SCALER.transform(df_input.values)
        input_sequence = input_sequence.reshape(1, look_back, len(MODEL_FEATURES))

        # 4. Predict
        prediction_scaled = MASTER_MODEL.predict(input_sequence)
        
        # 5. Inverse transform the prediction
        # We need to create a dummy array with the same shape as the scaler expects
        dummy_for_inverse = np.zeros((1, len(MODEL_FEATURES)))
        dummy_for_inverse[0, 0] = prediction_scaled[0, 0] # Put prediction in the 'value' column
        prediction_unscaled = SCALER.inverse_transform(dummy_for_inverse)

        final_prediction = prediction_unscaled[0, 0]

        return jsonify({"forecast": {"city": city_name, "next_value": float(final_prediction)}})

    except Exception as e:
        # ... your error handling ...
        return jsonify({"error": str(e)}), 500



    


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