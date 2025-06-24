import traceback
import asyncio
import os
import joblib
import pandas as pd
from db_connect import create_supabase

from models import get_models_path, get_enc_paths

from quart import Blueprint, jsonify, request, Response

from datetime import datetime # For date calculations
from dateutil.relativedelta import relativedelta # For easy date arithmetic 

from .forecasting_lstm import LSTMPredictor, fetch_and_prepare_transaction_data  # Import your LSTM predictor class

# from .Price_Estimation import EnsemblePropertyPredictor


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

@ml_routes.route("/forecast_property", methods=["POST"])
async def forecasting_property():
    try:
        data = await request.get_json()
        input_data = pd.DataFrame([data])  
        input_data["City"] = city_enc.transform([input_data["City"].iloc[0]])[0]
        input_data = input_data.astype(float)
        input_array = input_data.values.reshape(1, -1)
        prediction = prop_model.predict(input_array)
        return jsonify({"prediction": float(prediction[0])})
    except Exception as e:
        return jsonify({"error": str(e)}), 500

# --- Helper for fetching distinct values for filters (if needed for transactions) ---
async def fetch_distinct_transaction_cities(supabase_client):
    table_name = "transactions"
    column_name = "city" # Assuming 'city' column in transactions table
    res = await supabase_client.from_(table_name) \
                        .select(column_name) \
                        .neq(column_name, "is.null") \
                        .neq(column_name, "") \
                        .execute()
    if res.data:
        return sorted(list(set(item[column_name] for item in res.data if item[column_name])))
    return []

@ml_routes.route("/api/transaction_filters", methods=["GET"])
async def get_transaction_filters():
    supabase = await create_supabase()
    try:
        cities = await fetch_distinct_transaction_cities(supabase)
        return jsonify({
            "cities": cities,
        })
    except Exception as e:
        print(f"Error in /api/transaction_filters: {e}")
        return jsonify({"error": "Failed to fetch transaction filters: " + str(e)}), 500
    finally:
        if supabase:
            pass # Close client if needed

@ml_routes.route("/api/predict_transaction_timeseries", methods=["POST"])
async def predict_transaction_timeseries_route():
    supabase = None  # Initialize for finally block
    try:
        data = await request.get_json()
        city_name = data.get('city_name')
        granularity = data.get('granularity', 'M')

        value_column_for_lstm = "value"

        supabase = await create_supabase()

        print(f"DEBUG: Calling fetch_and_prepare_transaction_data with city: {city_name}")
        df_history_monthly = await fetch_and_prepare_transaction_data(
            supabase_client=supabase,
            city_name=city_name
        )
        print(f"DEBUG: df_history_monthly empty: {df_history_monthly.empty}, length: {len(df_history_monthly)}")

        if df_history_monthly.empty:
            return jsonify({"error": "No historical transaction data found for the specified city after preparation."}), 400

        look_back = 12
        epochs = 10
        batch_size = 1
        periods_to_predict_monthly = 60

        if len(df_history_monthly) < look_back + 1:
            return jsonify({"error": f"Insufficient monthly transaction data for LSTM. Need at least {look_back + 1} data points, got {len(df_history_monthly)}."}), 400

        lstm_predictor = LSTMPredictor(look_back=look_back)

        print(f"DEBUG: Training LSTM for Transactions: City='{city_name}'")
        lstm_predictor.train(df_history_monthly, value_column=value_column_for_lstm, epochs=epochs, batch_size=batch_size)

        print(f"DEBUG: Predicting with LSTM for {periods_to_predict_monthly} periods...")
        predictions_list_monthly = lstm_predictor.predict(df_history_monthly, future_periods=periods_to_predict_monthly, value_column=value_column_for_lstm)

        if not predictions_list_monthly:
            print("ERROR: No predictions returned by LSTM.")
            return jsonify({"error": "LSTM model failed to generate predictions."}), 500

        print(f"DEBUG: predictions_list_monthly (first 5): {predictions_list_monthly[:5]}")

        if not isinstance(df_history_monthly, pd.DataFrame) or df_history_monthly.empty or 'date' not in df_history_monthly.columns:
            print(f"CRITICAL ERROR: df_history_monthly is invalid.")
            return jsonify({"error": "Internal error: Historical data became invalid before date generation."}), 500

        last_historical_date_monthly = df_history_monthly['date'].iloc[-1]
        future_dates_monthly = pd.date_range(start=last_historical_date_monthly + pd.DateOffset(months=1),
                                             periods=periods_to_predict_monthly,
                                             freq='ME')

        df_forecast_monthly = pd.DataFrame({'date': future_dates_monthly, value_column_for_lstm: predictions_list_monthly})

        granularity_map_pandas = {'M': 'ME', 'Y': 'YE'}
        if granularity.upper() not in granularity_map_pandas:
            return jsonify({"error": f"Invalid granularity '{granularity}'. Choose 'M', or 'Y'."}), 400

        pandas_freq = granularity_map_pandas[granularity.upper()]

        df_history_agg = df_history_monthly.set_index('date')[value_column_for_lstm].resample(pandas_freq).mean().reset_index()
        df_forecast_agg = df_forecast_monthly.set_index('date')[value_column_for_lstm].resample(pandas_freq).mean().reset_index()

        if not isinstance(df_history_agg, pd.DataFrame) or not isinstance(df_forecast_agg, pd.DataFrame):
            return jsonify({"error": "Internal error: Aggregation failed."}), 500

        response_payload = {
            'historical': [
                {'ds': r['date'].strftime('%Y-%m-%d'), 'y': r[value_column_for_lstm]}
                for i, r in df_history_agg.iterrows() if pd.notnull(r[value_column_for_lstm])
            ],
            'forecast': [
                {'ds': r['date'].strftime('%Y-%m-%d'), 'y': r[value_column_for_lstm]}
                for i, r in df_forecast_agg.iterrows() if pd.notnull(r[value_column_for_lstm])
            ]
        }
        print(f"DEBUG: Successfully prepared response_payload. Returning to client.")
        return jsonify(response_payload)
    except Exception as e:
        print(f"ERROR in predict_transaction_timeseries_route: {e}")
        traceback.print_exc()
        return jsonify({"error": "An internal server error occurred: " + str(e)}), 500
    

# Estimation of Current Property Price

# --- Helper function to fetch unique, non-empty values for a column ---
async def fetch_distinct_values(supabase_client, table_name, column_name):
    """Fetches sorted, unique, non-null values from a specified column."""
    try:
        res = await supabase_client.from_(table_name) \
                                .select(column_name) \
                                .neq(column_name, "is.null") \
                                .execute()
        if res.data:
            # Use a set for efficient uniqueness, then sort for consistent UI
            distinct_items = sorted(list(set(item[column_name] for item in res.data if item[column_name])))
            # If the column is numeric (like bedrooms), convert from float to int
            if distinct_items and isinstance(distinct_items[0], float):
                return [int(item) for item in distinct_items]
            return distinct_items
        return []
    except Exception as e:
        print(f"Error fetching distinct values for {column_name}: {e}")
        return [] # Return empty list on error to prevent crashing

# === Optional: Serve Property Input Options (DYNAMICALLY FROM DATABASE) ===
@ml_routes.route("/api/property_input_options")
async def get_property_input_options():
    supabase = None
    table_name = "properties" # IMPORTANT: Change this if your table has a different name
    try:
        supabase = await create_supabase()

        # We use asyncio.gather to run all these database queries concurrently for performance
        results = await asyncio.gather(
            fetch_distinct_values(supabase, table_name, "district"),
            fetch_distinct_values(supabase, table_name, "type"),
            fetch_distinct_values(supabase, table_name, "bedrooms"),
            fetch_distinct_values(supabase, table_name, "bathrooms"),
            # For min/max, we fetch the single smallest/largest non-null value
            supabase.from_(table_name).select("size_m2").neq("size_m2", "is.null").order("size_m2", desc=False).limit(1).execute(),
            supabase.from_(table_name).select("size_m2").neq("size_m2", "is.null").order("size_m2", desc=True).limit(1).execute()
        )

        # Unpack the results from asyncio.gather
        districts, types, bedroom_options, bathroom_options, min_size_res, max_size_res = results

        # Process the min/max results
        min_size = min_size_res.data[0]['size_m2'] if min_size_res.data else 30
        max_size = max_size_res.data[0]['size_m2'] if max_size_res.data else 1000

        # Assemble the final JSON response
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
        # Return a meaningful error to the frontend
        return jsonify({"error": "Failed to fetch filter options from the database: " + str(e)}), 500
    finally:
        # The Supabase client from this library doesn't require explicit closing,
        # but the `finally` block is good practice for resource management.
        if supabase:
            pass

# === Define model variable (will be loaded lazily) ===
# ... the rest of your file remains the same ...
price_estimation_model = None
MODEL_PATH = "models/property_price_model.joblib"

# === API Endpoint to Estimate Property Price ===
@ml_routes.route("/api/estimate_property_price", methods=["POST"])
async def estimate_price():
    global price_estimation_model
    try:
        # Lazy-load the model if it hasn't been loaded yet
        if price_estimation_model is None:
            print(f"Attempting to load property estimation model from: {os.path.abspath(MODEL_PATH)}")
            if not os.path.exists(MODEL_PATH):
                # Now, this error will be correctly sent to the frontend if the file is missing
                raise FileNotFoundError(f"Model file not found at '{MODEL_PATH}'. Check path relative to server startup directory.")
            
            price_estimation_model = joblib.load(MODEL_PATH)
            print("Property estimation model loaded successfully.")

        data = await request.json

        # Create a DataFrame from the input. The model is expected to be a scikit-learn
        # Pipeline that handles string-to-numeric conversion (e.g., one-hot encoding) internally.
        df_input = pd.DataFrame([{
            "district": data.get("district"),
            "type": data.get("type", "Apartment"),
            "size_m2": data.get("size_m2"),
            "bedrooms": data.get("bedrooms"),
            "bathrooms": data.get("bathrooms")
        }])

        # The model's .predict() method should return a numpy array
        prediction = price_estimation_model.predict(df_input)

        # Extract the single prediction value and return it as JSON
        return jsonify({"prediction": round(float(prediction[0]), 2)})

    except Exception as e:
        # Log the full error to the console for easier debugging
        print("--- ERROR IN /api/estimate_property_price ---")
        traceback.print_exc()
        print("--- END OF ERROR ---")
        return jsonify({"error": str(e)}), 500