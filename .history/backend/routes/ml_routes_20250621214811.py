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


# mayssoun's bit

# # ------------------- transactions forecasting/ Start by loading the model and data -------------------
# # ------------------- Configuration for the NEW XGBoost Forecasting Model ---
# SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
# CSV_FILE_PATH = os.path.join(SCRIPT_DIR, 'transactions.csv') # Assuming the CSV is here
# XGB_MODEL_PATH = os.path.join(SCRIPT_DIR, 'forecast_model.joblib')
# XGB_COLS_PATH = os.path.join(SCRIPT_DIR, 'model_columns.json')
# TARGET_COL = 'transaction_value'
# FORECAST_HORIZON_MONTHS = 60

# def load_and_preprocess_data(filepath):
#     """Loads and prepares the historical transaction data from the CSV."""
#     if not os.path.exists(filepath):
#         raise FileNotFoundError(f"FATAL: The data file was not found at {filepath}. The API cannot start.")
    
#     df = pd.read_csv(filepath)
#     parts = df['date'].str.split('-', expand=True)
#     df['date_str'] = '01-' + parts[1] + '-' + parts[0]
#     df['date'] = pd.to_datetime(df['date_str'], format='%d-%b-%y')
#     df = df.set_index('date').drop(columns=['id', 'date_str'])
#     df.sort_index(inplace=True)
#     return df

# def create_features(df):
#     """Creates time series features needed for the model."""
#     df['month'] = df.index.month
#     df['year'] = df.index.year
#     df['quarter'] = df.index.quarter
#     df['lag_1'] = df.groupby('city')[TARGET_COL].shift(1)
#     df['lag_3'] = df.groupby('city')[TARGET_COL].shift(3)
#     df['lag_12'] = df.groupby('city')[TARGET_COL].shift(12)
#     df['rolling_mean_3'] = df.groupby('city')[TARGET_COL].shift(1).rolling(window=3).mean()
#     df['rolling_std_3'] = df.groupby('city')[TARGET_COL].shift(1).rolling(window=3).std()
#     return df

# print("-> Loading XGBoost forecasting model and historical data...")
# if not os.path.exists(XGB_MODEL_PATH):
#     raise FileNotFoundError(f"FATAL: XGBoost model not found at {XGB_MODEL_PATH}. Run the training script first.")
    
# XGB_MODEL = joblib.load(XGB_MODEL_PATH)
# XGB_MODEL_COLS = list(pd.read_json(XGB_COLS_PATH, typ='series'))
# ALL_HISTORICAL_DATA = load_and_preprocess_data(CSV_FILE_PATH)

# print("--- Initialization Complete. API is ready. ---")
# @ml_routes.route("/forecast/xgboost/<string:city_name>", methods=["GET"])
# async def forecast_with_xgboost(city_name: str):
#     """
#     Fast and efficient endpoint to generate a 5-year forecast using the pre-trained XGBoost model.
#     """
#     print(f"\nReceived XGBoost forecast request for city: {city_name}")
#     try:
#         available_cities = ALL_HISTORICAL_DATA['city'].unique()
#         if city_name not in available_cities:
#             return jsonify({"error": f"City '{city_name}' not found. Available cities are: {list(available_cities)}"}), 404

#         city_history = ALL_HISTORICAL_DATA[ALL_HISTORICAL_DATA['city'] == city_name].copy()
        
#         last_date = city_history.index.max()
#         future_dates = pd.date_range(start=last_date + pd.DateOffset(months=1), periods=FORECAST_HORIZON_MONTHS, freq='MS')
        
#         future_predictions = []
#         for date in future_dates:
#             temp_df = pd.DataFrame(index=[date], data={'city': city_name})
#             combined_df = pd.concat([city_history, temp_df])
#             features_df = create_features(combined_df)
#             current_features = features_df.tail(1)
#             current_features_encoded = pd.get_dummies(current_features, columns=['city'], prefix='city')
#             current_features_aligned = current_features_encoded.reindex(columns=XGB_MODEL_COLS, fill_value=0)
#             prediction = XGB_MODEL.predict(current_features_aligned)[0]
#             future_predictions.append(float(prediction))
#             city_history.loc[date] = {'city': city_name, TARGET_COL: float(prediction)}

#         monthly_forecast_df = pd.DataFrame({'date': future_dates, 'predicted_value': future_predictions})
#         yearly_forecast_df = monthly_forecast_df.resample('YE', on='date').sum()

#         historical_json = city_history.iloc[:-FORECAST_HORIZON_MONTHS].reset_index().to_dict(orient='records')
#         monthly_json = monthly_forecast_df.to_dict(orient='records')
#         yearly_json = yearly_forecast_df.reset_index().rename(columns={'date': 'year', 'predicted_value': 'total_value'}).to_dict(orient='records')
#         for item in yearly_json: item['year'] = item['year'].year

#         print(f"-> Successfully generated XGBoost forecast for '{city_name}'.")
#         return jsonify({
#             "city_forecasted": city_name,
#             "historical_data": historical_json,
#             "monthly_forecast": monthly_json,
#             "yearly_forecast": yearly_json
#         })

#     except Exception as e:
#         import traceback
#         traceback.print_exc()
#         return jsonify({"error": "An internal server error occurred."}), 500





# ==============================================================================
#                 MAYSNOUN'S BIT - FINAL PRODUCTION-READY API
# ==============================================================================
import pandas as pd
import numpy as np
import joblib
import os

# --- 1. CONFIGURATION (All files are in the same directory as this script) ---
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
HISTORICAL_DATA_PATH = os.path.join(SCRIPT_DIR, 'agg_trans.csv')
MODEL_PATH = os.path.join(SCRIPT_DIR, 'forecast_model.joblib')
MODEL_COLS_PATH = os.path.join(SCRIPT_DIR, 'model_columns.json')

# Define column names and forecast horizon
GROUPING_KEY = 'city' # This is the name of the region column in your agg_trans.csv
TARGET_COL = 'transaction_value'
FORECAST_HORIZON_MONTHS = 60


print("-> Loading final forecasting model and artifacts...")
XGB_MODEL = joblib.load(MODEL_PATH)
XGB_MODEL_COLS = list(pd.read_json(MODEL_COLS_PATH, typ='series'))

# # Load and prepare all historical data once
# df_hist = pd.read_csv(HISTORICAL_DATA_PATH)
# parts = df_hist['date'].str.split('-', expand=True)
# df_hist['date'] = pd.to_datetime('01-' + parts[1] + '-' + parts[0], format='%d-%b-%y')
# ALL_HISTORICAL_DATA = df_hist.set_index('date')
# print("--- Forecasting API is ready. ---")

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




# --- 2. HELPER FUNCTIONS (Must mirror the logic from the final training script) ---

def map_user_selection_to_region(selection_string: str) -> str:
    """
    Intelligently maps the user's dropdown selection to the 
    region name used in the model.
    """
    # This logic handles your exact requirement.
    if "Tripoli, Akkar" in selection_string:
        return "tripoli"
    if "Baabda, Aley, Chouf" in selection_string:
        return "baabda"
    if "Kesrouan, Jbeil" in selection_string:
        return "kesrouane"
    # For all other cases (like "Beirut"), the selection string is the region name.
    return selection_string.lower()


def create_features(df):
    """
    This MUST be an exact copy of the final, successful create_features 
    function from your training script.
    """
    df_features = df.copy()
    
    # Cyclical Features
    df_features['month_sin'] = np.sin(2 * np.pi * df_features.index.month / 12.0)
    df_features['month_cos'] = np.cos(2 * np.pi * df_features.index.month / 12.0)
    
    # Holiday/Event Flags
    df_features['is_december'] = (df_features.index.month == 12).astype(int)
    df_features['is_summer_peak'] = (df_features.index.month.isin([7, 8])).astype(int)

    # Core Time Features
    df_features['year'] = df_features.index.year
    df_features['quarter'] = df_features.index.quarter
    
    # Lag and Rolling Features
    df_features['lag_1'] = df_features.groupby(GROUPING_KEY)[TARGET_COL].shift(1)
    df_features['lag_3'] = df_features.groupby(GROUPING_KEY)[TARGET_COL].shift(3)
    df_features['lag_12'] = df_features.groupby(GROUPING_KEY)[TARGET_COL].shift(12)
    df_features['rolling_mean_3'] = df_features.groupby(GROUPING_KEY)[TARGET_COL].shift(1).rolling(window=3).mean()
    df_features['rolling_std_3'] = df_features.groupby(GROUPING_KEY)[TARGET_COL].shift(1).rolling(window=3).std()

    if 'month' in df_features.columns:
        df_features = df_features.drop('month', axis=1)

    return df_features

# --- 3. LOAD ARTIFACTS AT API STARTUP ---


# --- 4. API ENDPOINT (Final, Robust Version) ---
@ml_routes.route("/forecast/xgboost/<string:user_selection>", methods=["GET"])
async def forecast_with_xgboost(user_selection: str):
    """
    Generates a 5-year forecast. It maps the user's selection to the correct
    region and uses a robust recursive method to predict.
    """
    print(f"\nReceived forecast request for user selection: '{user_selection}'")
    try:
        # Step 1: Map the user's dropdown choice to the model's region name.
        region_name = map_user_selection_to_region(user_selection)
        print(f"-> Mapped to region: '{region_name}'")

        # Step 2: Get the historical data for that specific region.
        region_history = ALL_HISTORICAL_DATA[ALL_HISTORICAL_DATA[GROUPING_KEY] == region_name].copy()
        
        if region_history.empty:
            return jsonify({"error": f"No historical data found for region '{region_name}'."}), 404

        # Step 3: The corrected recursive forecasting loop.
        # This will be used to build the future predictions step-by-step.
        future_df_dynamic = region_history.copy()
        
        last_date = future_df_dynamic.index.max()
        future_dates = pd.date_range(start=last_date + pd.DateOffset(months=1), periods=FORECAST_HORIZON_MONTHS, freq='MS')
        
        for date in future_dates:
            # Create features for all known data up to this point
            features_df = create_features(future_df_dynamic)
            
            # Isolate the last row which contains the features for our prediction
            current_features = features_df.tail(1)
            
            # One-hot encode and align columns to match the model's training data
            current_features_encoded = pd.get_dummies(current_features, columns=[GROUPING_KEY])
            current_features_aligned = current_features_encoded.reindex(columns=XGB_MODEL_COLS, fill_value=0)
            
            # Make the prediction
            prediction = XGB_MODEL.predict(current_features_aligned)[0]
            
            # Add the prediction to our dynamic history so it can be used for the next step's lags
            future_df_dynamic.loc[date] = {GROUPING_KEY: region_name, TARGET_COL: float(prediction)}

        # Step 4: Format the output for the frontend
        forecast_results = future_df_dynamic.tail(FORECAST_HORIZON_MONTHS)
        
        monthly_forecast_df = pd.DataFrame({
            'date': forecast_results.index,
            'predicted_value': forecast_results[TARGET_COL].values
        })
        yearly_forecast_df = monthly_forecast_df.resample('YE', on='date')['predicted_value'].sum().to_frame()

        historical_json = region_history.reset_index().to_dict(orient='records')
        monthly_json = monthly_forecast_df.to_dict(orient='records')
        yearly_json = yearly_forecast_df.reset_index().rename(columns={'date': 'year', 'predicted_value': 'total_value'}).to_dict(orient='records')
        for item in yearly_json: item['year'] = item['year'].year

        print(f"-> Successfully generated forecast for region '{region_name}'.")
        return jsonify({
            "city_forecasted": user_selection, # Send back the original selection
            "historical_data": historical_json,
            "monthly_forecast": monthly_json,
            "yearly_forecast": yearly_json
        })

    except Exception as e:
        import traceback
        traceback.print_exc()
        return jsonify({"error": "An internal server error occurred."}), 500



