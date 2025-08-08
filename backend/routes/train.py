import pandas as pd
import xgboost as xgb
import joblib
import os
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_percentage_error, mean_squared_error, mean_absolute_error
from sklearn.model_selection import TimeSeriesSplit
from supabase import create_client, Client
from dotenv import load_dotenv

# --- CONFIGURATION ---
load_dotenv()
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
BACKEND_DIR = os.path.dirname(SCRIPT_DIR)
OUTPUT_DIR = os.path.join(BACKEND_DIR, 'timeseries_models') 
MODEL_FILE_PATH = os.path.join(OUTPUT_DIR, 'forecast_model.joblib')
MODEL_COLS_PATH = os.path.join(OUTPUT_DIR, 'model_columns.json')

# Constants
GROUPING_KEY = 'city'
TARGET_COL = 'transaction_value'
CITIES = ['baabda', 'beirut', 'bekaa', 'kesrouan', 'tripoli']

# --- FUNCTIONS ---
def load_data_from_supabase():
    """Connects to Supabase and fetches all regional transaction data."""
    print("-> Connecting to Supabase to fetch training data...")
    url: str = os.environ.get("SUPABASE_URL")
    key: str = os.environ.get("SUPABASE_KEY")

    if not url or not key:
        raise ValueError("FATAL: Supabase URL or Key not found in .env file.")
        
    supabase: Client = create_client(url, key)
    response = supabase.table('agg_trans').select("*").order('date').execute()
    
     # Convert the list of dictionaries to a pandas DataFrame
    df = pd.DataFrame(response.data)
    print(f"-> Successfully fetched {len(df)} rows from Supabase.")

    # Convert date column to datetime objects and set as index
    parts = df['date'].str.split('-', expand=True)
    df['date'] = pd.to_datetime(('01-' + parts[1] + '-' + parts[0]), format='%d-%b-%y')
    df['date'] = pd.to_datetime(df['date'], format='%Y-%m-%d')
    df.set_index('date', inplace=True)
    df.sort_index(inplace=True)
    print("-> Date column converted to datetime and set as index.")
    
    df[GROUPING_KEY] = df[GROUPING_KEY].str.lower()
    
    return df

def create_features(df):
    """Feature engineering for time-series data."""
    df_features = df.copy()
    df_features['year'] = df_features.index.year
    df_features['month'] = df_features.index.month
    df_features['month_sin'] = np.sin(2 * np.pi * df_features.index.month / 12.0)
    df_features['month_cos'] = np.cos(2 * np.pi * df_features.index.month / 12.0)
    df_features['is_december'] = (df_features.index.month == 12).astype(int)
    df_features['is_summer'] = df_features.index.month.isin([6, 7, 8]).astype(int)
    return df_features

def plot_validation_results(train_df, validation_df, predictions_df, city_name):
    """Plots validation results for a specific city."""
    plt.figure(figsize=(15, 7))
    plt.plot(train_df.index, train_df[TARGET_COL], label='Training Data', color='blue')
    plt.plot(validation_df.index, validation_df[TARGET_COL], label='Actual Values', color='green', marker='o')
    plt.plot(validation_df.index, predictions_df, label='Predicted Values', color='red', linestyle='--')
    plt.title(f'Model Validation for {city_name}: Actual vs. Predicted', fontsize=16)
    plt.xlabel('Date')
    plt.ylabel('Transaction Value')
    plt.legend()
    plt.grid(True)
    plt.show()

# --- MAIN EXECUTION ---
if __name__ == "__main__":
    print("--- Starting Optimized Model Training ---")

    # --- Step 1: Load and Prepare Data ---
    all_data = load_data_from_supabase()
    data_with_features = create_features(all_data)
    data_final = pd.get_dummies(data_with_features, columns=[GROUPING_KEY], prefix=GROUPING_KEY)

    # --- Step 2: Train-Validation Split ---
    train = data_final.loc['2011-01-01':'2014-12-01']  # 4 years training
    val = data_final.loc['2015-01-01':'2016-12-01']    # 2 years validation

    FEATURES = [col for col in data_final.columns if col != TARGET_COL]
    X_train, y_train = train[FEATURES], train[TARGET_COL]
    X_val, y_val = val[FEATURES], val[TARGET_COL]

    print(f"Training: {X_train.index.min()} to {X_train.index.max()} ({len(X_train)} rows)")
    print(f"Validation: {X_val.index.min()} to {X_val.index.max()} ({len(X_val)} rows)")

    # --- Step 3: Train Model with Optimized Params ---
    best_params = {
        'objective': 'reg:squarederror',
        'max_depth': 4,
        'learning_rate': 0.05,
        'n_estimators': 150,
        'subsample': 0.8,
        'colsample_bytree': 0.8,
        'reg_alpha': 0.8,
        'reg_lambda': 0.8,
        'min_child_weight': 4,
        'early_stopping_rounds': 20,
        'eval_metric': 'mae'
    }

    print("\n--- Training Model ---")
    model = xgb.XGBRegressor(**best_params)
    model.fit(
        X_train, y_train,
        eval_set=[(X_train, y_train), (X_val, y_val)],
        verbose=True
    )

    # --- Step 4: Evaluate ---
    predictions = model.predict(X_val)
    mae = mean_absolute_error(y_val, predictions)
    rmse = np.sqrt(mean_squared_error(y_val, predictions))
    mape = mean_absolute_percentage_error(y_val, predictions) * 100

    print("\n--- Validation Metrics ---")
    print(f"MAE: {mae:.2f}")
    print(f"RMSE: {rmse:.2f}")
    print(f"MAPE: {mape:.2f}%")

    # --- Step 5: Plot Results for Beirut ---
    city_to_plot = 'beirut'
    train_city = data_with_features[(data_with_features.index <= '2014-12-01') & 
                                  (data_with_features[GROUPING_KEY] == city_to_plot)]
    val_city = data_with_features[(data_with_features.index >= '2015-01-01') & 
                                 (data_with_features.index <= '2016-12-01') & 
                                 (data_with_features[GROUPING_KEY] == city_to_plot)]
    
    X_val_city = X_val[X_val[f'{GROUPING_KEY}_{city_to_plot}'] == 1]
    if not X_val_city.empty:
        city_preds = model.predict(X_val_city)
        plot_validation_results(train_city, val_city, city_preds, city_to_plot.capitalize())

    # --- Step 6: Retrain on Full Data and Save ---
    print("\n--- Retraining on Full Data ---")
    X_all = pd.concat([X_train, X_val])
    y_all = pd.concat([y_train, y_val])

    final_model = xgb.XGBRegressor(**best_params)
    final_model.fit(X_all, y_all)

    # Save model
    if not os.path.exists(OUTPUT_DIR):
        os.makedirs(OUTPUT_DIR)
    joblib.dump(final_model, MODEL_FILE_PATH)
    pd.Series(FEATURES).to_json(MODEL_COLS_PATH)
    print(f"\nModel saved to {MODEL_FILE_PATH}")

    # Feature Importance
    plt.figure(figsize=(12, 8))
    xgb.plot_importance(final_model, max_num_features=20)
    plt.title("Feature Importance")
    plt.tight_layout()
    plt.show()

    print("\n--- Training Complete ---")