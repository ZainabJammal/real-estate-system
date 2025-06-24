import traceback
import asyncio
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

        return jsonify({"prediction": float(prediction)})

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
        print("DEBUG: Final return triggered.")
        return jsonify(response_payload)
      print(f"DEBUG: Successfully prepared response_payload. Returning to client.")
        return jsonify(response_payload)
    except Exception as e:
        print(f"Exception occurred in predict_transaction_timeseries_route: {e}")
        traceback.print_exc()
        return jsonify({"error": str(e)}), 500

# # Current Price estimation
# # --- MODEL LOADING ---
# # Load the entire trained predictor object created by train_model.py

# # --- MODEL LOADING ---
# try:
#     predictor: EnsemblePropertyPredictor = joblib.load('property_price_estimator.joblib')
#     print("✅ Successfully loaded property price estimator model.")
# except FileNotFoundError:
#     predictor = None
#     print("❌ FATAL: 'property_price_estimator.joblib' not found.")
#     print("   Run `python train_model.py` to generate the model file before starting the server.")

# # --- DATABASE CONFIG ---
# PROPERTIES_TABLE_NAME = "properties"

# # --- HELPER FUNCTIONS (IMPROVED) ---
# async def fetch_distinct_values(supabase, column_name):
#     """Fetches unique, non-null, non-empty string values for a column."""
#     res = await supabase.from_(PROPERTIES_TABLE_NAME).select(column_name).neq(column_name, "is.null").neq(column_name, "").execute()
#     if res.data:
#         # Filter out None or empty strings again just in case DB returns them
#         return sorted(list(set(
#             item[column_name] for item in res.data 
#             if item.get(column_name) and str(item[column_name]).strip()
#         )))
#     return []

# def is_float(value):
#     """Helper to check if a value can be converted to a float."""
#     if value is None:
#         return False
#     try:
#         float(value)
#         return True
#     except (ValueError, TypeError):
#         return False

# async def fetch_numerical_options(supabase, column_name, is_int=True):
#     """More robust function to fetch numerical options, ignoring non-numeric values."""
#     res = await supabase.from_(PROPERTIES_TABLE_NAME).select(column_name).neq(column_name, "not.null").execute()
#     if res.data:
#         # **CRITICAL FIX**: Filter for valid numbers BEFORE processing
#         valid_numbers = [item[column_name] for item in res.data if is_float(item.get(column_name))]
        
#         if not valid_numbers:
#             return [] if is_int else {}
        
#         if is_int:
#             # Convert to float first, then to int to handle "3.0" style strings
#             return sorted(list(set(int(float(v)) for v in valid_numbers)))
#         else:
#             float_values = [float(v) for v in valid_numbers]
#             return {"min": min(float_values), "max": max(float_values)}
            
#     return [] if is_int else {}

# async def get_avg_coords_for_district(supabase, district_name):
#     """Fetches the average latitude and longitude for a given district."""
#     res = await supabase.from_(PROPERTIES_TABLE_NAME).select("latitude, longitude").eq("district", district_name).neq("latitude", "is.null").execute()
#     if res.data:
#         df = pd.DataFrame(res.data)
#         df['latitude'] = pd.to_numeric(df['latitude'], errors='coerce')
#         df['longitude'] = pd.to_numeric(df['longitude'], errors='coerce')
#         df.dropna(inplace=True)
#         if not df.empty:
#             return df.latitude.mean(), df.longitude.mean()
#     return None, None

# # --- API ENDPOINTS ---
# @ml_routes.route("/api/property_input_options", methods=["GET"])
# async def get_property_input_options():
#     supabase = await create_supabase()
#     try:
#         # **CRITICAL FIX**: Ensure these column names match your database schema EXACTLY.
#         district = await fetch_distinct_values(supabase, "district")
#         type = await fetch_distinct_values(supabase, "type")
#         size_range = await fetch_numerical_options(supabase, "size_m2", is_int=False)
#         bedroom_options = await fetch_numerical_options(supabase, "bedrooms", is_int=True) # Corrected name
#         bathroom_options = await fetch_numerical_options(supabase, "bathrooms", is_int=True) # Corrected name

#         return jsonify({
#             "district": district,
#             "type": type,
#             "size_range": size_range or {"min": 20, "max": 1000},
#             "bedroom_options": bedroom_options or [0, 1, 2, 3, 4, 5],
#             "bathroom_options": bathroom_options or [0, 1, 2, 3, 4],
#         })
#     except Exception as e:
#         print(f"CRITICAL ERROR in /api/property_input_options: {e}")
#         traceback.print_exc()
#         return jsonify({"error": "Failed to fetch property input options from the server."}), 500

# @ml_routes.route("/api/estimate_property_price", methods=["POST"])
# async def estimate_property_price():
#     if not predictor or not predictor.is_trained:
#         return jsonify({"error": "Model is not available. Please check server logs."}), 503

#     supabase = await create_supabase()
#     try:
#         data = await request.get_json()
#         required = ["district", "type", "size_m2", "bedrooms", "bathrooms"]
#         if not all(k in data for k in required):
#             return jsonify({"error": f"Missing one or more required fields: {required}"}), 400

#         latitude, longitude = await get_avg_coords_for_district(supabase, data["district"])
#         if latitude is None:
#             return jsonify({"error": f"Cannot determine location for district '{data['district']}'."}), 400

#         feature_dict = {
#             "district": data["district"],
#             "type": data["type"],
#             "size_m2": float(data["size_m2"]),
#             "bedrooms": int(data["bedrooms"]),
#             "bathrooms": int(data["bathrooms"]),
#             "latitude": latitude,
#             "longitude": longitude,
#         }
        
#         if feature_dict["type"].lower() == "land":
#             feature_dict["bedrooms"] = 0
#             feature_dict["bathrooms"] = 0
        
#         input_df = pd.DataFrame([feature_dict])
#         prediction = predictor.predict(input_df)
        
#         return jsonify({"prediction": float(prediction[0])})

#     except Exception as e:
#         print(f"Error in /api/estimate_property_price: {e}")
#         traceback.print_exc()
#         return jsonify({"error": "An unexpected error occurred during estimation."}), 500