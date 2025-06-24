import pandas as pd
import xgboost as xgb
import joblib
import os
import matplotlib.pyplot as plt 



# 01 Configuration
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
CSV_FILE_PATH = os.path.join(SCRIPT_DIR, 'transactions.csv')
MODEL_FILE_PATH = os.path.join(SCRIPT_DIR, 'forecast_model.joblib')
MODEL_COLS_PATH = os.path.join(SCRIPT_DIR, 'model_columns.json')
TARGET_COL = 'transaction_value'
FORECAST_HORIZON_MONTHS = 60


# 02 data preparation
def load_and_preprocess_data(filepath):
    print("-> Loading and preprocessing data...")
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"The data file was not found at the expected path: {filepath}")
    # data parsing 
    df = pd.read_csv(filepath)
    parts = df['date'].str.split('-', expand=True)
    df['date_str'] = '01-' + parts[1] + '-' + parts[0]
    df['date'] = pd.to_datetime(df['date_str'], format='%d-%b-%y')
    
    df = df.set_index('date').drop(columns=['id', 'date_str'])
    df.sort_index(inplace=True)
    print("-> Data loaded successfully.")
    return df

def create_features(df):
    df['month'] = df.index.month
    df['year'] = df.index.year
    df['quarter'] = df.index.quarter
    df['lag_1'] = df.groupby('city')[TARGET_COL].shift(1)
    df['lag_3'] = df.groupby('city')[TARGET_COL].shift(3)
    df['lag_12'] = df.groupby('city')[TARGET_COL].shift(12)
    df['rolling_mean_3'] = df.groupby('city')[TARGET_COL].shift(1).rolling(window=3).mean()
    df['rolling_std_3'] = df.groupby('city')[TARGET_COL].shift(1).rolling(window=3).std()
    return df


# 03 model training + encoding cities
def train_and_save_model(df):
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


# 04. PREDICTION FUNCTION
# ==============================================================================
def generate_forecast(city_to_forecast: str, historical_df: pd.DataFrame):
    print(f"--- [PREDICTION MODE] for city: {city_to_forecast} ---")
    model = joblib.load(MODEL_FILE_PATH)
    model_cols = list(pd.read_json(MODEL_COLS_PATH, typ='series'))
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
# 5. VISUALIZATION FUNCTION
# ==============================================================================
def plot_forecast(historical_data, forecast_data, city_name):
    """
    Plots the historical data and the forecasted data on a single graph.
    """
    print("\n-> Generating plot...")
    
    # Filter historical data for the chosen city
    city_historical = historical_data[historical_data['city'] == city_name][TARGET_COL]

    plt.figure(figsize=(15, 7))  # Create a figure with a good size for time series

    # Plot the historical data
    plt.plot(city_historical.index, city_historical.values, label='Historical Data', color='blue', marker='.')

    # Plot the forecast
    plt.plot(forecast_data.index, forecast_data['predicted_value'], label='Forecasted Data', color='red', linestyle='--')

    # Add plot titles and labels for clarity
    plt.title(f'Transaction Value Forecast for {city_name}', fontsize=16)
    plt.xlabel('Date', fontsize=12)
    plt.ylabel('Transaction Value', fontsize=12)
    plt.legend()
    plt.grid(True, which='both', linestyle='--', linewidth=0.5)

    # Display the plot
    plt.show()

# ==============================================================================
# 6. MAIN EXECUTION BLOCK
# ==============================================================================
if __name__ == "__main__":
    print("--- Running Model Training Script ---")
    
    # Step 1: Load the raw data from the source CSV file.
    all_historical_data = load_and_preprocess_data(CSV_FILE_PATH)
    
    # Step 2: Train the model using all the data and save the output files.
    # The train_and_save_model function handles all the logic.
    train_and_save_model(all_historical_data)
    
    print("\n--- Training complete. The model artifacts (forecast_model.joblib and model_columns.json) have been created/updated. ---")
    print("You can now run the API server.")