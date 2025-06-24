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

# ------------------- transactions forecasting
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
        yearly_forecast_df = monthly_forecast_df.resample('YE', on='date').sum()

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









