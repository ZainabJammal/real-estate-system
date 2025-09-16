import json
import os
import joblib
from sklearn.preprocessing import MinMaxScaler
import pandas as pd
import numpy as np
import traceback
from supabase import create_client, Client
from dotenv import load_dotenv
from models import get_models_path, get_enc_paths
from quart import Blueprint, jsonify, request,  current_app
from statsmodels.tsa.holtwinters import ExponentialSmoothing
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

@ml_routes.route("/ml/city_circles", methods=["GET"])
async def city_circles():
    try:
        supabase = current_app.supabase
        result = await supabase.table("city_prices").select("city, listings_count, latitude, longitude").execute()
        data = result.data
        return jsonify(data)
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@ml_routes.route("/ml/city_price_trend", methods=["GET"])
async def city_price_trend():
    try:
        supabase = current_app.supabase

        # Call the SQL function we created
        result = await supabase.rpc("city_price_trend_medians").select("*").execute()
        query_result = result.data

        final_result = []

        def get_coordinates_for_city(city_name):
            # Static mapping — update with real coordinates!
            coords = {
                "Beirut": (33.8965, 35.4829),
                "Bekaa": (33.9105, 35.9631),
                "Baabda, Aley, Chouf": (33.7779, 35.5737),
                "Kesrouan, Jbeil": (34.1145, 35.6634), 
                "Tripoli, Akkar": (34.4887, 35.9544)
            }

            if not city_name:
                return (None, None)
            
            return coords.get(city_name.strip(), (None, None))

        for row in query_result:
            p2015 = row["median_price_2015"]
            p2016 = row["median_price_2016"]
            city = row["city"]

            if p2015 is None or p2016 is None:
                continue

            change = p2016 - p2015
            percent = (change / p2015) * 100

            direction = "neutral"
            if percent > 0.5: #use small threshold to avoid "up" for tiny changes
                direction = "up"
            elif percent < -0.5:
                direction = "down"

            lat, lng = get_coordinates_for_city(city)
            if lat is None or lng is None:
                continue

            final_result.append({
                "city": city,
                "direction": direction,
                "change_percent": round(percent, 2),
                "current_price": p2016, #price for most recent period
                "latitude": lat,
                "longitude": lng,
            })

        return jsonify(final_result)

    except Exception as e:
        import traceback
        traceback.print_exc()
        return jsonify({"error": str(e)}), 500


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

# # --- 1. CONFIGURATION AND DATABASE CONNECTION ---
# # Load environment variables from .env file
# load_dotenv() 
# SUPABASE_URL: str = os.environ.get("SUPABASE_URL")
# SUPABASE_KEY: str = os.environ.get("SUPABASE_KEY")

# if not SUPABASE_URL or not SUPABASE_KEY:
#     raise ValueError("FATAL ERROR: SUPABASE_URL and SUPABASE_KEY must be set in your .env file.")

# # Initialize the Supabase client
# supabase: Client = create_client(SUPABASE_URL, SUPABASE_KEY)
# print("-> Supabase client initialized.")

# BACKEND_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


# FORECAST_MODEL_DIR = os.path.join(BACKEND_DIR, 'forecasting_models')
# MODEL_PATH = os.path.join(FORECAST_MODEL_DIR, 'forecast_model.joblib')
# MODEL_COLS_PATH = os.path.join(FORECAST_MODEL_DIR, 'model_columns.json') 

# # Add a print statement to verify the path during startup.
# print(f"--- [DEBUG] Attempting to load model from: {MODEL_PATH}")

# # Define global constants
# # GROUPING_KEY must match the column name in your training data AND your database
# GROUPING_KEY = 'city' 
# TARGET_COL = 'transaction_value'
# FORECAST_HORIZON_MONTHS = 60

# # --- 2. HELPER FUNCTIONS (These must mirror your final training script) ---

# def map_user_selection_to_city(selection_string: str) -> str:
#     """
#     Maps the user's dropdown selection to the specific custom region names
#     used in the database, ensuring it's lowercase for matching.
#     """
#     if "Tripoli, Akkar" in selection_string:
#         return "Tripoli"
#     if "Baabda, Aley, Chouf" in selection_string:
#         return "Baabda"
#     if "Kesrouan, Jbeil" in selection_string:
#         return "Kesrouan"
#     return selection_string.lower()

# def create_features(df):
#     """Optimal features for a small dataset with a long forecast horizon."""
#     df_features = df.copy()
#     df_features['year'] = df_features.index.year
#     df_features['month_sin'] = np.sin(2 * np.pi * df_features.index.month / 12.0)
#     df_features['month_cos'] = np.cos(2 * np.pi * df_features.index.month / 12.0)
#     df_features['is_december'] = (df_features.index.month == 12).astype(int)
#     df_features['is_summer_peak'] = (df_features.index.month.isin([7, 8])).astype(int)
#     return df_features

# # --- 3. LOAD MODEL ARTIFACTS AT STARTUP ---
# print("-> Loading local forecasting model artifacts...")   
# XGB_MODEL = joblib.load(MODEL_PATH)
# XGB_MODEL_COLS = list(pd.read_json(MODEL_COLS_PATH, typ='series'))
# print("--- Forecasting API is ready (connected to Supabase). ---")

# # --- 4. API ENDPOINT (Simplified for "Best Guess" model) ---
# @ml_routes.route("/forecast/xgboost/<string:user_selection>", methods=["GET"])
# async def forecast_with_xgboost(user_selection: str):
#     """
#     Generates a 5-year 'Best Guess' forecast by predicting all future
#     months in a single batch.
#     """
#     print(f"\nReceived 'Best Guess' forecast request for: '{user_selection}'")
#     # if 'forecast_model' not in ML_ARTIFACTS or not supabase:
#     #     return jsonify({"error": "Forecasting service or database is not available."}), 503


#     try:
#         city_name = map_user_selection_to_city(user_selection)
#         print(f"-> Mapped to city: '{city_name}'")

#         # step A: fetch histo data first
#         print(f"-> Querying Supabase for '{city_name}' historical data...")
#         response = supabase.table('agg_trans').select('*').ilike(GROUPING_KEY, city_name).order('date').execute()
  
#         print("\n--- RAW SUPABASE RESPONSE DEBUG ---")
#         print(f"Type of response object: {type(response)}")
#         print(f"Response has 'data' attribute: {'data' in dir(response)}")
        
#         # Check if the response contains data and print it
#         if response.data:
#             print(f"Number of records returned: {len(response.data)}")
#             print(f"First record received from Supabase: {response.data[0]}") 
#         else:
#             print("!!! CRITICAL: `response.data` is EMPTY or does not exist.")
#             print(f"Full response object: {response}")
        
#         print("-----------------------------------\n")

#         if not response.data:
#             return jsonify({"error": f"No historical data found for city '{city_name}' in the database."}), 404
        
#         # Prepare the historical data DataFrame
#         hist_df = pd.DataFrame(response.data)

#         parts = hist_df['date'].str.split('-', expand=True)
#         hist_df['date'] = pd.to_datetime('01-' + parts[1] + '-' + parts[0], format='%d-%b-%y')
#         city_history_df = hist_df.set_index('date')

#         # Standardize the grouping key column to lowercase to prevent case-mismatch
#         city_history_df[GROUPING_KEY] = city_history_df[GROUPING_KEY].str.lower()


#         # Step B: Create a DataFrame for all 60 future months

#          # Find the last date from the historical data we just fetched.
#         last_historical_date = hist_df['date'].iloc[-1].year

#          # Start the forecast on the first day of the month AFTER the last historical date.
#         start_year = last_historical_date + 1
#         start_date = pd.to_datetime(f'{start_year}-01-01')
#         future_dates = pd.date_range(start=start_date, periods=FORECAST_HORIZON_MONTHS, freq='MS')

#         future_df = pd.DataFrame(index=future_dates)
#         # Create features for the future DataFrame
#         future_df[GROUPING_KEY] = city_name
#         features_for_pred = create_features(future_df)

        
#         # Step 4: One-hot encode the city and align columns
#         features_encoded = pd.get_dummies(features_for_pred, columns=[GROUPING_KEY])
#         features_aligned = features_encoded.reindex(columns=XGB_MODEL_COLS, fill_value=0)
        
#         # Step 5: Make predictions for ALL 60 months at once
#         predictions = XGB_MODEL.predict(features_aligned)
#         print(f"-> Generated {len(predictions)} future predictions in a single batch.")

#         # --- Part C: Format the Final JSON Response ---

#         historical_json = hist_df.to_dict(orient='records')
        
#         monthly_forecast_df = pd.DataFrame({'date': future_dates, 'predicted_value': predictions})
#         yearly_forecast_df = monthly_forecast_df.resample('YE', on='date')['predicted_value'].sum().to_frame()

#         monthly_json = monthly_forecast_df.to_dict(orient='records')
#         yearly_json = yearly_forecast_df.reset_index().rename(columns={'date': 'year', 'predicted_value': 'total_value'}).to_dict(orient='records')
#         for item in yearly_json: item['year'] = item['year'].year

#         print(f"-> Successfully generated forecast for '{city_name}'.")
#         return jsonify({
#             "city_forecasted": user_selection,
#             "historical_data": historical_json,
#             "monthly_forecast": monthly_json,
#             "yearly_forecast": yearly_json
#         })

#     except Exception as e:
#         traceback.print_exc()
#         return jsonify({"error": "An internal server error occurred."}), 500



load_dotenv()
SUPABASE_URL: str = os.environ.get("SUPABASE_URL")
SUPABASE_KEY: str = os.environ.get("SUPABASE_KEY")

if not SUPABASE_URL or not SUPABASE_KEY:
    raise ValueError("FATAL ERROR: SUPABASE_URL and SUPABASE_KEY must be set in your .env file.")

supabase: Client = create_client(SUPABASE_URL, SUPABASE_KEY)
print("-> Supabase client initialized.")

try:
    SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
    
    BACKEND_DIR = os.path.dirname(SCRIPT_DIR)
except NameError:
    
    BACKEND_DIR = os.getcwd()

# Now, build the path to the models folder from this stable 'backend' directory.
# VERIFY THIS FOLDER NAME! In your error it was 'timeseries_models'. In your code it is 'forecasting_models'.
# I will use 'forecasting_models' as it matches your training script.
FORECAST_MODEL_DIR = os.path.join(BACKEND_DIR, 'timeseries_models')

# Define the final paths to your artifacts
MEAN_MODEL_PATH = os.path.join(FORECAST_MODEL_DIR, 'mean_forecast_model.joblib')
# P10_MODEL_PATH = os.path.join(FORECAST_MODEL_DIR, 'p10_forecast_model.joblib')
# P90_MODEL_PATH = os.path.join(FORECAST_MODEL_DIR, 'p90_forecast_model.joblib')
FEATURES_PATH = os.path.join(FORECAST_MODEL_DIR, 'model_features.json')

# --- Add this diagnostic print statement ---
print("--- [STARTUP DEBUG] ---")
print(f"SCRIPT_DIR resolved to: {SCRIPT_DIR}")
print(f"BACKEND_DIR resolved to: {BACKEND_DIR}")
print(f"Attempting to load models from: {FORECAST_MODEL_DIR}")
print(f"Full path to mean model: {MEAN_MODEL_PATH}")
print("-----------------------")
# --- END OF FIX ---

# Define global constants
GROUPING_KEY = 'city' 
TARGET_COL = 'transaction_value'
FORECAST_HORIZON_MONTHS = 60
# --- 3. HELPER FUNCTIONS (Must mirror the training script) ---

def map_user_selection_to_city(selection_string: str) -> str:
    """Maps user selection to the lowercase city name 
    used in training."""
   
    if "Tripoli, Akkar" in selection_string:
         return "Tripoli"
    if "Baabda, Aley, Chouf" in selection_string:
         return "Baabda"
    if "Kesrouan, Jbeil" in selection_string:
         return "Kesrouan"
    return selection_string.lower()

# --- FIX 2: Use the FINAL `create_features` function from your training script ---
def create_features(df):
    """
    Creates the final, non-leaky feature set for the model.
    This MUST be identical to the function used in training.
    """
    df_features = df.copy()
    
    df_features['year'] = df_features.index.year
    df_features['month_sin'] = np.sin(2 * np.pi * df_features.index.month / 12.0)
    df_features['month_cos'] = np.cos(2 * np.pi * df_features.index.month / 12.0)
    df_features['is_december'] = (df_features.index.month == 12).astype(int)
    df_features['is_summer_peak'] = (df_features.index.month.isin([7, 8])).astype(int)
    # Critical: Create the lag features the model expects
    for lag in [1, 2, 3, 12]:
        df_features[f'lag_{lag}'] = df_features.groupby(GROUPING_KEY)[TARGET_COL].shift(lag)
        
    # Important: Fill NaNs created by lags
    df_features.fillna(0, inplace=True)
    return df_features

# --- 4. LOAD MODEL ARTIFACTS AT STARTUP ---
print("-> Loading forecasting model artifacts...")   
MEAN_MODEL = joblib.load(MEAN_MODEL_PATH)
# P10_MODEL = joblib.load(P10_MODEL_PATH)
# P90_MODEL = joblib.load(P90_MODEL_PATH)
MODEL_FEATURES = pd.read_json(FEATURES_PATH, typ='series').tolist()
print("--- Forecasting API is ready. ---")

# --- 5. THE NEW FORECASTING LOGIC ---
def generate_future_forecasts(city_name: str, historical_data: pd.DataFrame) -> pd.DataFrame:
    """
    Generates a 5-year forecast iteratively.
    This function uses the historical data to predict future values
    month by month, feeding back the predictions into the historical data.
    """
    
    try:
        city_name = map_user_selection_to_city(user_selection)
        print(f"-> Mapped to city: '{city_name}'")
    print(f"-> Generating future forecasts for city: {city_name}")
    future_predictions = []
    
    current_data = historical_data.copy() # Start with the full, real history
    last_date = current_data.index.max()
    print(f"-> Starting iterative forecast for '{city_name}' from: {last_date + pd.DateOffset(months=1)}")

    for i in range(FORECAST_HORIZON_MONTHS):
        next_date = last_date + pd.DateOffset(months=i + 1)
        features_encoded = pd.get_dummies(features_for_pred, columns=[GROUPING_KEY])
        X_future = features_encoded.reindex(columns=MODEL_FEATURES, fill_value=0).iloc[-1:]
        # A. *** THE FIX ***
        # Create a temporary row for the next date so we can generate its features correctly.
        # We then append it to our current history to calculate the lags.
        temp_row = pd.DataFrame([{'date': next_date, TARGET_COL: 0, GROUPING_KEY: city_name}]).set_index('date')
        data_for_features = pd.concat([current_data, temp_row])
        features_with_lags = create_features(data_for_features)
        
        # B. Prepare the single feature row for the future date we want to predict
        future_row_features = pd.get_dummies(features_with_lags, columns=[GROUPING_KEY])
        future_row_aligned = future_row_features.reindex(columns=MODEL_FEATURES, fill_value=0)
        X_future = future_row_aligned.iloc[-1:] # Select the very last row

        # C. Predict using ONLY the mean model
        pred_mean = MEAN_MODEL.predict(X_future)[0]
        
        # D. Store the simplified result
        future_predictions.append({
            'date': next_date,
            'mean_forecast': float(pred_mean),
        })

        # E. CRITICAL: Feed the prediction back into our data history for the next loop
        new_row = pd.DataFrame([{
            'date': next_date,
            TARGET_COL: pred_mean,
            GROUPING_KEY: city_name
        }]).set_index('date')
        
        current_data = pd.concat([current_data, new_row])

    print("-> Iterative forecast generation complete.")
    return pd.DataFrame(future_predictions).set_index('date')

@ml_routes.route("/forecast/xgboost/<string:user_selection>", methods=["GET"])
async def forecast_with_xgboost(user_selection: str):
    """
    Fetches historical data and uses an iterative process to generate a robust 5-year forecast.
    """
    print(f"\nReceived forecast request for: '{user_selection}'")
    try:
        city_name = map_user_selection_to_city(user_selection)
        print(f"-> Mapped to city: '{city_name}'")

        # Step A: Fetch all historical data for the chosen city
        response = supabase.table('merged_trans').select('date, transaction_value, city').ilike(GROUPING_KEY, city_name).order('date').execute()
        print("\n--- RAW SUPABASE RESPONSE DEBUG ---")
#         print(f"Type of response object: {type(response)}")
#         print(f"Response has 'data' attribute: {'data' in dir(response)}")
        


        if response.data:
#             print(f"Number of records returned: {len(response.data)}")
#             print(f"First record received from Supabase: {response.data[0]}") 
#         else:
            return jsonify({"error": f"No historical data found for '{city_name}'."}), 404
        
        # Prepare the historical DataFrame
        hist_df = pd.DataFrame(response.data)
        hist_df['date'] = pd.to_datetime(hist_df['date'])
        hist_df.set_index('date', inplace=True)
        hist_df[GROUPING_KEY] = hist_df[GROUPING_KEY].str.lower()
        print(f"-> Fetched {len(hist_df)} historical records for '{city_name}'.")

        # Step B: Generate future forecasts iteratively
        forecast_df = generate_future_forecasts(city_name, hist_df)

        # Step C: Format the response
        historical_json = hist_df.reset_index().to_dict(orient='records')
        
        monthly_forecast_df = forecast_df.reset_index()
        # Resample on the DataFrame's DatetimeIndex for yearly summary
        yearly_forecast_df = forecast_df.resample('YE').sum()

        monthly_json = monthly_forecast_df.to_dict(orient='records')
        yearly_json = yearly_forecast_df.reset_index().rename(columns={'date': 'year'}).to_dict(orient='records')
        # Extract just the year number for the frontend
        for item in yearly_json: item['year'] = item['year'].year

        print(f"-> Successfully generated forecast for '{city_name}'.")
        return jsonify({
            "city_forecasted": user_selection,
            "historical_data": historical_json,
            "monthly_forecast": monthly_json,
            "yearly_forecast": yearly_json
        })

    except Exception as e:
        # Using traceback provides much more detail for debugging
        traceback.print_exc()
        return jsonify({"error": f"An internal server error occurred: {str(e)}"}), 500