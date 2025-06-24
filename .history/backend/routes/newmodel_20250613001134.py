import pandas as pd
import xgboost as xgb
import joblib
import os
import json
import pickle
import asyncio
import traceback
import matplotlib.pyplot as plt
from dotenv import load_dotenv
from tensorflow.keras.models import load_model
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error
from supabase import create_client, Client
# --- Configuration ---
# You'll get these from your Supabase project settings
def get_supabase_client() -> Client: # The return type hint should be Client
    """Connects to Supabase using credentials from .env file."""
    load_dotenv()
    url = os.environ.get("SUPABASE_URL")
    key = os.environ.get("SUPABASE_KEY")
    if not url or not key:
        raise ValueError("Supabase URL and Key must be set in your .env file.")
    # This creates the SYNCHRONOUS client
    return create_client(url, key)

MODEL_PATH = 'forecast_model.joblib'
MODEL_COLS_PATH = 'model_columns.json'
TARGET_COL = 'transaction_value'
HORIZON_MONTHS = 60

# Initialize the Supabase client
# In a real app, you do this once when your server starts
try:
    supabase: Client = get_supabase_client()
except Exception as e:
    print(f"Error initializing Supabase client: {e}")
    print("Please ensure SUPABASE_URL and SUPABASE_KEY are set as environment variables.")
    supabase = None


    
def load_and_preprocess_data(filepath):
    """Loads the raw CSV, cleans the date column, and sets a proper datetime index."""
    print("Loading and preprocessing data...")
    df = pd.read_csv(filepath)
    
    # Create a full date string that pandas can understand (e.g., '01-Jan-11')
    df['date_str'] = '01-' + df['date'].str.replace('-', '-')
    # Convert to datetime, correctly interpreting two-digit years like '11' as 2011
    df['date'] = pd.to_datetime(df['date_str'], format='%d-%b-%y')
    
    # Set the date as the index and sort it
    df = df.set_index('date')
    df.sort_index(inplace=True)
    
    # Drop unnecessary columns
    df = df.drop(columns=['id', 'date_str'])
    
    print("Data loaded successfully.")
    return df

def create_features(df):
    """Creates time series features from a datetime index."""
    df['month'] = df.index.month
    df['year'] = df.index.year
    df['quarter'] = df.index.quarter
    
    # Lag features (past values). We group by city to ensure lags are not from other cities.
    df['lag_1'] = df.groupby('city')[TARGET_COL].shift(1)
    df['lag_3'] = df.groupby('city')[TARGET_COL].shift(3)
    df['lag_12'] = df.groupby('city')[TARGET_COL].shift(12) # Value from same month, previous year

    # Rolling window features (recent trends)
    df['rolling_mean_3'] = df.groupby('city')[TARGET_COL].shift(1).rolling(window=3).mean()
    df['rolling_std_3'] = df.groupby('city')[TARGET_COL].shift(1).rolling(window=3).std()
    
    return df

def train_and_save_model(df):
    """Trains a global XGBoost model on all historical data and saves it."""
    print("Starting model training...")
    
    # 1. Create features
    df_feat = create_features(df.copy())
    
    # 2. One-Hot Encode the 'city' column
    df_final = pd.get_dummies(df_feat, columns=['city'], prefix='city')

    # 3. Prepare data for training
    # Drop rows with NaN values created by lags/rolling windows
    df_final = df_final.dropna()
    
    FEATURES = [col for col in df_final.columns if col != TARGET_COL]
    TARGET = TARGET_COL
    
    X, y = df_final[FEATURES], df_final[TARGET]

    # Save the column layout so we can use it during prediction
    pd.Series(X.columns).to_json(MODEL_COLS_PATH)
    
    # 4. Train the XGBoost model
    # Using parameters that work well for time series
    reg = xgb.XGBRegressor(
        n_estimators=1000,
        learning_rate=0.01,
        max_depth=3,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42,
        objective='reg:squarederror'
    )
    
    reg.fit(X, y, verbose=False) # verbose=True to see training progress
    
    # 5. Save the trained model to a file
    joblib.dump(reg, MODEL_PATH)
    print(f"Model trained and saved to '{MODEL_PATH}'")

def get_historical_data_from_supabase(city_to_forecast: str) -> pd.DataFrame:
    """Queries Supabase to get all historical data for a specific city."""
    print(f"Querying Supabase for historical data for '{city_to_forecast}'...")
    if not supabase:
        raise ConnectionError("Supabase client is not initialized.")

    # Fetch data from the 'transactions' table
    response = supabase.table('transactions') \
                     .select('date', 'city', 'transaction_value') \
                     .eq('city', city_to_forecast) \
                     .order('date', desc=False) \
                     .execute()

    if not response.data:
        raise ValueError(f"No data found for city: {city_to_forecast}")

    # Convert the list of dictionaries to a pandas DataFrame
    df = pd.DataFrame(response.data)
    
    # --- IMPORTANT: Re-apply the same date processing as in training ---
    df['date_str'] = '01-' + df['date'].str.replace('-', '-')
    df['date'] = pd.to_datetime(df['date_str'], format='%d-%b-%y')
    df = df.set_index('date').drop(columns=['date_str'])
    df.sort_index(inplace=True)
    
    return df

def predict_future(city_to_forecast: str):
    """
    Main prediction function for the API. It gets data from Supabase,
    loads the model, and performs the forecast.
    """
    print(f"\n--- Starting 5-Year Forecast for: {city_to_forecast} ---")
    
    # --- API Step 1: Load the pre-trained model and column info ---
    model = joblib.load(MODEL_PATH)
    model_cols = list(pd.read_json(MODEL_COLS_PATH, typ='series'))

    # --- API Step 2: Fetch the complete historical data for the chosen city ---
    historical_df = get_historical_data_from_supabase(city_to_forecast)
    
    # --- The rest of the recursive forecasting logic is THE SAME as before ---
    last_date = historical_df.index.max()
    future_dates = pd.date_range(start=last_date + pd.offsets.MonthBegin(1), periods=HORIZON_MONTHS, freq='MS')
    
    future_predictions = []
    history_for_features = historical_df.copy()

    for date in future_dates:
        # (This loop is identical to the previous script)
        temp_df = pd.DataFrame(index=[date])
        temp_df['city'] = city_to_forecast
        combined_df = pd.concat([history_for_features, temp_df])
        
        # We need the create_features function from the original script here
        # For brevity, assuming it's available.
        features_df = create_features(combined_df) # Assuming create_features is defined
        current_features = features_df.tail(1)
        current_features_encoded = pd.get_dummies(current_features, columns=['city'], prefix='city')
        current_features_aligned = current_features_encoded.reindex(columns=model_cols, fill_value=0)
        
        prediction = model.predict(current_features_aligned)[0]
        future_predictions.append(prediction)
        
        history_for_features.loc[date] = {'city': city_to_forecast, TARGET_COL: prediction}

    # --- Format Output ---
    predictions_df = pd.DataFrame({'date': future_dates, 'predicted_value': future_predictions})
    predictions_df.set_index('date', inplace=True)
    yearly_predictions = predictions_df.resample('Y').sum()
    yearly_predictions['year'] = yearly_predictions.index.year
    
    return predictions_df, yearly_predictions

# # Helper function needed by predict_future
# def create_features(df):
#     df['month'] = df.index.month
#     df['year'] = df.index.year
#     df['quarter'] = df.index.quarter
#     df['lag_1'] = df.groupby('city')[TARGET_COL].shift(1)
#     df['lag_3'] = df.groupby('city')[TARGET_COL].shift(3)
#     df['lag_12'] = df.groupby('city')[TARGET_COL].shift(12)
#     df['rolling_mean_3'] = df.groupby('city')[TARGET_COL].shift(1).rolling(window=3).mean()
#     df['rolling_std_3'] = df.groupby('city')[TARGET_COL].shift(1).rolling(window=3).std()
#     return df

# You would not run this __main__ block on the server.
# This is just for testing. The server would import and use predict_future in an API endpoint.
if __name__ == "__main__":
    try:
        USER_CHOICE_CITY = "Beirut"
        monthly_forecast, yearly_forecast = predict_future(USER_CHOICE_CITY)
        
        if monthly_forecast is not None:
            print("\n--- Monthly Forecast (First 12 Months) ---")
            print(monthly_forecast.head(12).to_string(formatters={'predicted_value': '{:,.2f}'.format}))
            
            print("\n--- Yearly Forecast Summary ---")
            print(yearly_forecast[['year', 'predicted_value']].to_string(index=False, formatters={'predicted_value': '{:,.2f}'.format}))
      except Exception as e:
      print(f"\n--- ❌ An error occurred during evaluation! ---")
      traceback.print_exc()