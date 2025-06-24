import pandas as pd
import xgboost as xgb
import joblib
import os
from supabase import create_client, Client

# --- Configuration ---
# You'll get these from your Supabase project settings
SUPABASE_URL = os.environ.get("SUPABASE_URL") # Recommended to use environment variables
SUPABASE_KEY = os.environ.get("SUPABASE_KEY")

MODEL_PATH = 'forecast_model.joblib'
MODEL_COLS_PATH = 'model_columns.json'
TARGET_COL = 'transaction_value'
HORIZON_MONTHS = 60

# Initialize the Supabase client
# In a real app, you do this once when your server starts
try:
    supabase: Client = create_client(SUPABASE_URL, SUPABASE_KEY)
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

    