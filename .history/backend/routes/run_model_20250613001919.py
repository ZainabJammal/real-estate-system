import pandas as pd
import xgboost as xgb
import joblib
import os

# ==============================================================================
# 1. CONFIGURATION AND PATH SETUP
#    (This section makes the script robust, preventing FileNotFoundError)
# ==============================================================================

# Get the absolute path of the directory where the script is located
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

# Define absolute paths for all files to ensure they are found correctly
CSV_FILE_PATH = os.path.join(SCRIPT_DIR, 'transactions.csv')
MODEL_FILE_PATH = os.path.join(SCRIPT_DIR, 'forecast_model.joblib')
MODEL_COLS_PATH = os.path.join(SCRIPT_DIR, 'model_columns.json')

TARGET_COL = 'transaction_value'
FORECAST_HORIZON_MONTHS = 60 # 5 years

# ==============================================================================
# 2. DATA PREPARATION FUNCTIONS
# ==============================================================================

def load_and_preprocess_data(filepath):
    """Loads the raw CSV, cleans the date column, and sets a proper datetime index."""
    print("-> Loading and preprocessing data...")
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"The data file was not found at the expected path: {filepath}")
    
    df = pd.read_csv(filepath)
    df['date_str'] = '01-' + df['date'].str.replace('-', '-')
    df['date'] = pd.to_datetime(df['date_str'], format='%d-%b-%y')
    df = df.set_index('date').drop(columns=['id', 'date_str'])
    df.sort_index(inplace=True)
    print("-> Data loaded successfully.")
    return df

def create_features(df):
    """Creates time series features from a datetime index."""
    df['month'] = df.index.month
    df['year'] = df.index.year
    df['quarter'] = df.index.quarter
    df['lag_1'] = df.groupby('city')[TARGET_COL].shift(1)
    df['lag_3'] = df.groupby('city')[TARGET_COL].shift(3)
    df['lag_12'] = df.groupby('city')[TARGET_COL].shift(12)
    df['rolling_mean_3'] = df.groupby('city')[TARGET_COL].shift(1).rolling(window=3).mean()
    df['rolling_std_3'] = df.groupby('city')[TARGET_COL].shift(1).rolling(window=3).std()
    return df

# ==============================================================================
# 3. MODEL TRAINING FUNCTION
# ==============================================================================

def train_and_save_model(df):
    """Trains a global XGBoost model on all historical data and saves it."""
    print("\n--- [TRAINING MODE] ---")
    print("-> Model file not found. Training a new model...")
    
    df_feat = create_features(df.copy())
    df_final = pd.get_dummies(df_feat, columns=['city'], prefix='city')
    df_final = df_final.dropna()
    
    FEATURES = [col for col in df_final.columns if col != TARGET_COL]
    X, y = df_final[FEATURES], df_final[TARGET_COL]
    
    pd.Series(X.columns).to_json(MODEL_COLS_PATH)
    
    reg = xgb.XGBRegressor(
        n_estimators=1000, learning_rate=0.01, max_depth=3,
        subsample=0.8, colsample_bytree=0.8, random_state=42,
        objective='reg:squarederror'
    )
    
    print("-> Fitting the XGBoost model...")
    reg.fit(X, y, verbose=False)
    joblib.dump(reg, MODEL_FILE_PATH)
    print(f"-> Model trained and saved to '{MODEL_FILE_PATH}'")
    print("--- [TRAINING COMPLETE] ---\n")

# ==============================================================================
# 4. PREDICTION FUNCTION
# ==============================================================================

def generate_forecast(city_to_forecast: str, historical_df: pd.DataFrame):
    """Loads a pre-trained model and predicts future values for a specific city."""
    print(f"--- [PREDICTION MODE] for city: {city_to_forecast} ---")
    
    model = joblib.load(MODEL_FILE_PATH)
    model_cols = list(pd.read_json(MODEL_COLS_PATH, typ='series'))
    
    # **FIXED LOGIC**: Use the provided historical_df instead of fetching new data
    city_history = historical_df[historical_df['city'] == city_to_forecast].copy()
    
    last_date = city_history.index.max()
    future_dates = pd.date_range(start=last_date + pd.offsets.MonthBegin(1), periods=FORECAST_HORIZON_MONTHS, freq='MS')
    
    future_predictions = []
    
    print(f"-> Generating forecast for the next {FORECAST_HORIZON_MONTHS} months recursively...")
    for date in future_dates:
        temp_df = pd.DataFrame(index=[date], data={'city': city_to_forecast})
        combined_df = pd.concat([city_history, temp_df])
        features_df = create_features(combined_df)
        
        current_features = features_df.tail(1)
        current_features_encoded = pd.get_dummies(current_features, columns=['city'], prefix='city')
        current_features_aligned = current_features_encoded.reindex(columns=model_cols, fill_value=0)
        
        prediction = model.predict(current_features_aligned)[0]
        future_predictions.append(float(prediction))
        
        city_history.loc[date] = {'city': city_to_forecast, TARGET_COL: float(prediction)}

    predictions_df = pd.DataFrame({'date': future_dates, 'predicted_value': future_predictions})
    predictions_df.set_index('date', inplace=True)
    yearly_predictions = predictions_df.resample('Y').sum()
    yearly_predictions['year'] = yearly_predictions.index.year
    
    return predictions_df, yearly_predictions

# ==============================================================================
# 5. MAIN EXECUTION BLOCK
# ==============================================================================
if __name__ == "__main__":
    # Load all data from the local CSV file
    all_historical_data = load_and_preprocess_data(CSV_FILE_PATH)
    
    # Check if the model has been trained. If not, train it.
    if not os.path.exists(MODEL_FILE_PATH):
        train_and_save_model(all_historical_data)

    # --- CHOOSE WHICH CITY TO FORECAST ---
    USER_CHOICE_CITY = "Beirut"
    
    # Generate the forecast for the chosen city
    monthly_forecast, yearly_forecast = generate_forecast(
        city_to_forecast=USER_CHOICE_CITY,
        historical_df=all_historical_data
    )
    
    # Print the results
    print("\n\n--- FORECAST RESULTS ---")
    print("\n--- Monthly Forecast (First 12 Months) ---")
    print(monthly_forecast.head(12).to_string(formatters={'predicted_value': '{:,.2f}'.format}))
    
    print("\n--- Yearly Forecast Summary ---")
    print(yearly_forecast[['year', 'predicted_value']].to_string(index=False, formatters={'predicted_value': '{:,.2f}'.format}))