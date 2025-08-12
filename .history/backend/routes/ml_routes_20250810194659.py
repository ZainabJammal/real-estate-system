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



load_dotenv()

# Setup robust paths to model files
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
BACKEND_DIR = os.path.dirname(SCRIPT_DIR)
OUTPUT_DIR = os.path.join(BACKEND_DIR, 'forecasting_models')

MODEL_FILE_PATH = os.path.join(OUTPUT_DIR, 'forecast_model.joblib')
MODEL_COLS_PATH = os.path.join(OUTPUT_DIR, 'model_columns.json')

# Define global constants matching the training script
GROUPING_KEY = 'city'
TARGET_COL = 'transaction_value'
FORECAST_HORIZON_MONTHS = 60 # 5 years

# --- 2. LOAD MODEL ARTIFACTS AT STARTUP ---
# This is efficient as it's done only once when the app starts
print("-> Loading forecasting model artifacts...")
try:
    if not os.path.exists(MODEL_FILE_PATH):
        raise FileNotFoundError(f"Model file not found at: {MODEL_FILE_PATH}")
    if not os.path.exists(MODEL_COLS_PATH):
        raise FileNotFoundError(f"Model columns file not found at: {MODEL_COLS_PATH}")
        
    MODEL = joblib.load(MODEL_FILE_PATH)
    MODEL_COLS = list(pd.read_json(MODEL_COLS_PATH, typ='series'))
    print("-> Forecasting model and columns loaded successfully.")
except Exception as e:
    print(f"FATAL ERROR: Could not load model artifacts. {e}")
    MODEL = None # Set to None to prevent app from running with a broken model

# --- 3. DATABASE CONNECTION ---
SUPABASE_URL = os.environ.get("SUPABASE_URL")
SUPABASE_KEY = os.environ.get("SUPABASE_KEY")

if not SUPABASE_URL or not SUPABASE_KEY or not MODEL:
    print("FATAL: Supabase credentials or a loaded model is missing. API will not be functional.")
    supabase = None
else:
    supabase: Client = create_client(SUPABASE_URL, SUPABASE_KEY)
    print("--- Forecasting API is ready (connected to Supabase). ---")


# --- 4. HELPER FUNCTION (Must PERFECTLY match training script logic) ---

def map_user_selection_to_city(selection_string: str) -> str:
    """
    Maps the user's dropdown selection to the specific custom region names
    used in the database, ensuring it's lowercase for matching.
    """
    if "Tripoli, Akkar" in selection_string:
        return "Tripoli"
    if "Baabda, Aley, Chouf" in selection_string:
        return "Baabda"
    if "Kesrouan, Jbeil" in selection_string:
        return "Kesrouan"
    return selection_string.lower()
def create_features_for_prediction(df, city_avg_value, monthly_avg_map):
    """
    Creates the feature set for prediction. This must mirror the final training script.
    It takes pre-computed averages as input since future data has no target value.
    """
    df_features = df.copy()
    
    # Time-based features (can be created directly from the future dates)
    df_features['year'] = df_features.index.year
    df_features['month'] = df_features.index.month
    df_features['quarter'] = df_features.index.quarter
    df_features['month_sin'] = np.sin(2 * np.pi * df_features.index.month/12)
    df_features['month_cos'] = np.cos(2 * np.pi * df_features.index.month/12)
    df_features['is_december'] = (df_features.index.month == 12).astype(int)
    df_features['is_summer'] = df_features.index.month.isin([6,7,8]).astype(int)
    df_features['is_quarter_end'] = df_features.index.is_quarter_end.astype(int)
    df_features['post_2013'] = (df_features.index.year > 2013).astype(int)
    
    # Features requiring historical context (passed as arguments)
    df_features['city_avg'] = city_avg_value
    df_features['monthly_avg'] = df_features['month'].map(monthly_avg_map)
    
    return df_features

# --- 5. API ENDPOINT ---
@ml_routes.route("/forecast/xgboost/<string:city_name>", methods=["GET"])
async def forecast_with_xgboost(city_name: str):
    """
    Generates a 5-year forecast for a given city.
    """
    city_name = city_name.lower().strip()
    print(f"\nReceived forecast request for city: '{city_name}'")

    if not supabase or not MODEL:
        return jsonify({"error": "Server is not configured correctly. Check logs."}), 500

    try:
        # === STEP 1: Fetch all historical data for the selected city ===
        print(f"-> Step 1: Querying Supabase for '{city_name}' historical data...")
        response = supabase.table('agg_trans').select(f'date, {TARGET_COL}').ilike(GROUPING_KEY, city_name).order('date').execute()
        
        if not response.data:
            return jsonify({"error": f"No historical data found for city '{city_name}'."}), 404

        hist_df = pd.DataFrame(response.data)
        # Clean and prepare historical data
        hist_df['date'] = pd.to_datetime(hist_df['date'], format='%d-%b-%y')
        hist_df.set_index('date', inplace=True)
        hist_df.sort_index(inplace=True)
        print(f"-> Found {len(hist_df)} historical records for '{city_name}'.")

        # === STEP 2: Calculate historical averages needed for feature engineering ===
        print("-> Step 2: Calculating historical averages for feature creation...")
        city_avg_value = hist_df[TARGET_COL].mean()
        monthly_avg_map = hist_df.groupby(hist_df.index.month)[TARGET_COL].mean()

        # === STEP 3: Create a future DataFrame for the forecast horizon ===
        print(f"-> Step 3: Creating future dataframe for {FORECAST_HORIZON_MONTHS} months...")
        last_historical_date = hist_df.index.max()
        future_dates = pd.date_range(start=last_historical_date + pd.DateOffset(months=1), periods=FORECAST_HORIZON_MONTHS, freq='MS')
        future_df = pd.DataFrame(index=future_dates)
        future_df[GROUPING_KEY] = city_name

        # === STEP 4: Generate features for the future DataFrame ===
        print("-> Step 4: Engineering features for future dates...")
        features_for_pred = create_features_for_prediction(future_df, city_avg_value, monthly_avg_map)

        # === STEP 5: One-hot encode and align columns to match model's training data ===
        print("-> Step 5: One-hot encoding and aligning columns...")
        features_encoded = pd.get_dummies(features_for_pred, columns=[GROUPING_KEY])
        # .reindex is crucial to ensure the dataframe has the exact same columns in the same order as the model was trained on
        features_aligned = features_encoded.reindex(columns=MODEL_COLS, fill_value=0)
        
        # === STEP 6: Make predictions ===
        print(f"-> Step 6: Generating {len(features_aligned)} predictions...")
        predictions = MODEL.predict(features_aligned)

        # === STEP 7: Format the JSON response ===
        print("-> Step 7: Formatting final JSON response...")
        historical_json = hist_df.reset_index().to_dict(orient='records')
        
        monthly_forecast_df = pd.DataFrame({'date': future_dates, 'predicted_value': predictions})
        yearly_forecast_df = monthly_forecast_df.resample('YE', on='date')['predicted_value'].sum().to_frame()

        # Convert timestamps to strings for JSON compatibility
        for record in historical_json:
            record['date'] = record['date'].strftime('%Y-%m-%d')
        
        monthly_json = monthly_forecast_df.to_dict(orient='records')
        for record in monthly_json:
            record['date'] = record['date'].strftime('%Y-%m-%d')
            
        yearly_json = yearly_forecast_df.reset_index().rename(columns={'date': 'year', 'predicted_value': 'total_value'}).to_dict(orient='records')
        for item in yearly_json:
            item['year'] = item['year'].year

        print(f"-> Successfully completed forecast for '{city_name}'.")
        return jsonify({
            "city_forecasted": city_name,
            "historical_data": historical_json,
            "monthly_forecast": monthly_json,
            "yearly_forecast": yearly_json
        })

    except Exception as e:
        print(f"!!! AN ERROR OCCURRED for city '{city_name}' !!!")
        traceback.print_exc()
        return jsonify({"error": "An internal server error occurred.", "details": str(e)}), 500