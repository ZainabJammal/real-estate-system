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

@ml_routes.route("/estimate_price", methods=["GET","POST"])
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


# @ml_routes.route("/api/estimate_property_price", methods=["POST"])
# async def estimate_property_price():
#     """API endpoint for making property price estimations based on detailed features."""
#     try:
#         data = await request.get_json()

#         # Prepare features dictionary
#         feature_dict = {
#             "City": data.get("City"),
#             "District": data.get("District"),
#             "Province": data.get("Province"),
#             "Type": data.get("Type"),
#             "Size_m2": float(data.get("Size_m2", 0)),
#             "Num_Bedrooms": int(data.get("Num_Bedrooms", 0)),
#             "Num_Bathrooms": int(data.get("Num_Bathrooms", 0)),
#         }

#         # Conditional logic for "Land" type
#         if feature_dict["Type"] == "Land":
#             feature_dict["Num_Bedrooms"] = 0
#             feature_dict["Num_Bathrooms"] = 0
        
#         # Create DataFrame from the single data point
#         input_df = pd.DataFrame([feature_dict])

#         # Encode categorical features
#         # Ensure inputs to transform are iterables (e.g., list)
#         input_df["City"] = city_enc.transform(input_df["City"])[0]
#         input_df["District"] = dis_enc.transform(input_df["District"])[0]
#         input_df["Province"] = prov_enc.transform(input_df["Province"])[0]
#         input_df["Type"] = type_enc.transform(input_df["Type"])[0]

#         # Reorder columns to match model's expected input feature order
#         # And ensure all are float for the model
#         input_df_ordered = input_df[PROP_MODEL_FEATURE_ORDER]
#         input_df_ordered = input_df_ordered.astype(float)
        
#         # Convert to NumPy array for prediction
#         input_array = input_df_ordered.values

#         print("Processed input for /api/estimate_property_price:", input_array)

#         # Make prediction
#         prediction = prop_model.predict(input_array)
#         print("Prediction for /api/estimate_property_price:", prediction)

#         return jsonify({"prediction": float(prediction[0])})

#     except Exception as e:
#         print(f"Error in /api/estimate_property_price: {e}")
#         traceback.print_exc()
#         return jsonify({"error": str(e)}), 500
