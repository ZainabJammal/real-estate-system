import joblib
import pandas as pd

from models import get_models_path, get_enc_paths

from quart import Blueprint, jsonify, request, Response
from db_connect import create_supabase # For connecting to Supabase
from datetime import datetime # For date calculations
from dateutil.relativedelta import relativedelta # For easy date arithmetic (e.g., 5 years ago)

import traceback
import asyncio
from .forecasting_lstm import LSTMPredictor, fetch_and_prepare_transaction_data  # Import your LSTM predictor class

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



# mayssoun's bit-- TimeSeries Forecasting for Transactions
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
        return
        jsonify(response_payload)

    except Exception as e:
        print(f"Exception occurred in predict_transaction_timeseries_route: {e}")
    traceback.print_exc()
    return jsonify({"error": str(e)}), 500



# Current Price estimation
# --- Supabase Configuration ---
PROPERTIES_TABLE_NAME = "properties" # ADJUST IF YOUR TABLE NAME IS DIFFERENT

# --- Helper Functions to Fetch Data from Supabase ---
async def fetch_distinct_values(supabase_client, column_name, table_name= PROPERTIES_TABLE_NAME):
    """Fetches unique, non-null, non-empty string values for a column."""
    try:
        res = await supabase_client.from_(table_name) \
                            .select(column_name, count='exact') \
                            .neq(column_name, "is.null") \
                            .neq(str(column_name), "") \
                            .execute()
        if res.data:
            # Filter out None or empty strings again just in case DB returns them despite neq
            distinct_items = sorted(list(set(
                item[column_name] for item in res.data 
                if item[column_name] is not None and str(item[column_name]).strip() != ""
            )))
            print(f"Fetched {len(distinct_items)} distinct values for {column_name}: {distinct_items[:5]}...") # Log first 5
            return distinct_items
        print(f"No data or error for distinct values of {column_name}")
        return []
    except Exception as e:
        print(f"Error fetching distinct values for {column_name} from {table_name}: {e}")
        traceback.print_exc()
        return []

async def fetch_numerical_options(supabase_client, column_name, table_name=PROPERTIES_TABLE_NAME, is_int=True):
    """Fetches unique numerical values, or min/max for a float column."""
    try:
        res = await supabase_client.from_(table_name) \
                            .select(column_name) \
                            .neq(column_name, "is.null") \
                            .execute()
        if res.data:
            values = [item[column_name] for item in res.data if item[column_name] is not None]
            if not values:
                print(f"No numerical values found for {column_name}")
                return []
            
            if is_int: # For bedroom/bathroom counts (distinct options)
                # Convert to float first to handle potential strings like "3.0", then to int
                processed_values = sorted(list(set(int(float(v)) for v in values)))
                print(f"Fetched integer options for {column_name}: {processed_values[:5]}...")
                return processed_values
            else: # For size_m2 (min/max range)
                float_values = [float(v) for v in values]
                min_val, max_val = min(float_values), max(float_values)
                print(f"Fetched float range for {column_name}: min={min_val}, max={max_val}")
                return {"min": min_val, "max": max_val}
        print(f"No data or error for numerical options of {column_name}")
        return [] if is_int else {"min": 0, "max": 1000} # Default range if no data
    except Exception as e:
        print(f"Error fetching numerical options for {column_name} from {table_name}: {e}")
        traceback.print_exc()
        return [] if is_int else {"min": 0, "max": 1000}


# --- API Endpoint to Get Filter Options ---
@ml_routes.route("/api/property_input_options", methods=["GET"])
async def get_property_input_options():
    supabase = None
    try:
        supabase = await create_supabase()
        
        # Adjust column names here if they are different in your Supabase table
      
        districts = await fetch_distinct_values(supabase, "district")
        types = await fetch_distinct_values(supabase, "type")
        
        # For size_m2, get min/max range
        size_range = await fetch_numerical_options(supabase, "size_m2", is_int=False)

        # For bedrooms/bathrooms, get distinct counts as options
        bedroom_options = await fetch_numerical_options(supabase, "num_bedrooms", is_int=True)
        if not bedroom_options: bedroom_options = [0, 1, 2, 3, 4, 5] # Default if no data

        bathroom_options = await fetch_numerical_options(supabase, "num_bathrooms", is_int=True)
        if not bathroom_options: bathroom_options = [0, 1, 2, 3, 4] # Default if no data

        options_payload = {
            "districts": districts,
            "types": types,
            "size_range": size_range, # Should be like {"min": X, "max": Y}
            "bedroom_options": bedroom_options,
            "bathroom_options": bathroom_options,
        }
        print(f"Returning property input options: {options_payload}")
        return jsonify(options_payload)
        
    except Exception as e:
        print(f"Critical error in /api/property_input_options: {e}")
        traceback.print_exc()
        return jsonify({"error": "Failed to fetch property input options: " + str(e)}), 500
    finally:
        if supabase:
            # If your create_supabase() returns a client that needs explicit closing:
            # await supabase.aclose() # Or similar, depending on your Supabase client library
            pass


# --- API Endpoint for Property Price Estimation ---
@ml_routes.route("/api/estimate_property_price", methods=["POST"])
async def estimate_property_price():
    """API endpoint for making property price estimations based on detailed features."""
    try:
        data = await request.get_json()
        print(f"Received data for price estimation: {data}")

        # Basic validation for required fields
        required_fields = ["City", "District", "Province", "Type", "Size_m2"]
        for field in required_fields:
            if field not in data or data[field] is None or str(data[field]).strip() == "":
                return jsonify({"error": f"Missing or empty required field: {field}"}), 400
        
        # Prepare features dictionary
        feature_dict = {
            "District": data.get("District"),
            "Type": data.get("Type"),
            "Size_m2": float(data.get("Size_m2", 0)), # Default to 0 if missing, though validated above
            # Bedrooms/Bathrooms can be missing if type is Land, default to 0
            "Num_Bedrooms": int(float(data.get("Num_Bedrooms", 0))), # Convert to float then int to handle "2.0"
            "Num_Bathrooms": int(float(data.get("Num_Bathrooms", 0))),
        }

        # Conditional logic for "Land" type
        if feature_dict["Type"] == "Land":
            feature_dict["Num_Bedrooms"] = 0
            feature_dict["Num_Bathrooms"] = 0
        
        # Create DataFrame from the single data point
        input_df = pd.DataFrame([feature_dict])

        # Encode categorical features
        # Ensure inputs to transform are iterables (e.g., list or pd.Series)
        try:
            input_df["District"] = dis_enc.transform(input_df["District"])[0]
            input_df["Province"] = prov_enc.transform(input_df["Province"])[0]
            input_df["Type"] = type_enc.transform(input_df["Type"])[0]
        except Exception as enc_error:
            print(f"Error during encoding: {enc_error}")
            traceback.print_exc()
            # Identify which feature caused the error if possible
            # This often happens if a value from DB/frontend is not in the encoder's known classes
            return jsonify({"error": f"Encoding error: {enc_error}. Check if all selected values are known to the model/encoders."}), 400

        # Reorder columns to match model's expected input feature order
        # And ensure all are float for the model
        try:
            input_df_ordered = input_df[PROP_MODEL_FEATURE_ORDER]
        except KeyError as ke:
            print(f"KeyError during column reordering: {ke}. Expected features: {PROP_MODEL_FEATURE_ORDER}, DataFrame columns: {input_df.columns.tolist()}")
            return jsonify({"error": f"Feature mismatch error: {ke}. Ensure all model features are provided."}), 500
            
        input_df_ordered = input_df_ordered.astype(float)
        
        # Convert to NumPy array for prediction
        input_array = input_df_ordered.values

        print(f"Processed input array for prop_model: {input_array}")

        # Make prediction
        prediction = prop_model.predict(input_array)
        print(f"Raw prediction from prop_model: {prediction}")

        return jsonify({"prediction": float(prediction[0])})

    except ValueError as ve: # Catch issues like float conversion for Size_m2
        print(f"ValueError in /api/estimate_property_price: {ve}")
        traceback.print_exc()
        return jsonify({"error": f"Invalid input value: {ve}"}), 400
    except Exception as e:
        print(f"General error in /api/estimate_property_price: {e}")
        traceback.print_exc()
        return jsonify({"error": f"An unexpected error occurred: {str(e)}"}), 500