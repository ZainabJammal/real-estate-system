

import pandas as pd
import xgboost as xgb
import joblib
import os
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_percentage_error
from sklearn.model_selection import GridSearchCV, TimeSeriesSplit
from supabase import create_client, Client
from dotenv import load_dotenv

# --- 1. CONFIGURATION ---
# Load environment variables from .env file
load_dotenv()

# Define paths for saving the model artifacts
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_FILE_PATH = os.path.join(SCRIPT_DIR, 'forecast_model.joblib')
MODEL_COLS_PATH = os.path.join(SCRIPT_DIR, 'model_columns.json')

GROUPING_KEY = 'city' 
TARGET_COL = 'transaction_value'

# --- 2. HELPER FUNCTIONS ---

def load_data_from_supabase():
    """Connects to Supabase and fetches all regional transaction data."""
    print("-> Connecting to Supabase to fetch training data...")
    url: str = os.environ.get("SUPABASE_URL")
    key: str = os.environ.get("SUPABASE_KEY")

    if not url or not key:
        raise ValueError("FATAL: Supabase URL or Key not found in .env file.")
        
    supabase: Client = create_client(url, key)

    # Fetch all data from the table, ordered by date
    response = supabase.table('agg_trans').select("*").order('date').execute()
    
    # Convert the list of dictionaries to a pandas DataFrame
    df = pd.DataFrame(response.data)
    print(f"-> Successfully fetched {len(df)} rows from Supabase.")

    # Convert date column to datetime objects and set as index
    df['date'] = pd.to_datetime(df['date'])

     parts = raw_df['date'].str.split('-', expand=True)
    raw_df['date'] = pd.to_datetime('01-' + parts[1] + '-' + parts[0], format='%d-%b-%y')
        city_history_df = raw_df.set_index('date')
    # The GROUPING_KEY column is named 'city' in your database table
    df[GROUPING_KEY] = df[GROUPING_KEY].str.lower()
    
    return df.set_index('date').drop(columns=['id', 'created_at'], errors='ignore')

def create_features(df):
    """
    Creates all time-series features needed for the model.
    This is the "single source of truth" for feature engineering.
    """
    print("-> Creating time-series features...")
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

def plot_validation_results(train_df, validation_df, predictions_df, city_name):
    """Plots the validation results for a specific city."""
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

# --- 3. MAIN TRAINING WORKFLOW ---
if __name__ == "__main__":
    print("--- Starting Production Model Training from Supabase Data ---")

    # Delete old model files to ensure a fresh start
    if os.path.exists(MODEL_FILE_PATH): os.remove(MODEL_FILE_PATH)
    if os.path.exists(MODEL_COLS_PATH): os.remove(MODEL_COLS_PATH)

    # Load data from the database
    all_data = load_data_from_supabase()
    data_with_features = create_features(all_data)

    # Split data for training and validation
    split_date = '2016-01-01'
    train_set_raw = data_with_features[data_with_features.index < split_date]
    validation_set_raw = data_with_features[data_with_features.index >= split_date]
    print(f"-> Data split for validation at: {split_date}")

    # Prepare data for XGBoost (One-Hot Encode region and handle NaNs from lags)
    train_final = pd.get_dummies(train_set_raw, columns=[GROUPING_KEY]).dropna()
    FEATURES = [col for col in train_final.columns if col != TARGET_COL]
    X_train, y_train = train_final[FEATURES], train_final[TARGET_COL]

    # Hyperparameter Tuning with the expanded grid
    param_grid = {
        'n_estimators': [1000, 1500],
        'max_depth': [3, 5, 7, 10],
        'learning_rate': [0.01, 0.05],
        'subsample': [0.7, 0.8, 1.0],
        'colsample_bytree': [0.7, 0.8, 1.0]
    }
    tscv = TimeSeriesSplit(n_splits=5)
    
    print("-> Starting GridSearchCV for hyperparameter tuning...")
    grid_search = GridSearchCV(
        estimator=xgb.XGBRegressor(objective='reg:squarederror', random_state=42),
        param_grid=param_grid, scoring='neg_mean_absolute_percentage_error', cv=tscv, verbose=1
    )
    grid_search.fit(X_train, y_train)
    best_params = grid_search.best_params_
    print(f"-> GridSearchCV complete. Best parameters found: {best_params}")

    # Train a final model on the entire pre-2016 training set with best params
    optimized_model = xgb.XGBRegressor(objective='reg:squarederror', random_state=42, **best_params)
    optimized_model.fit(X_train, y_train)

    # # --- Validate and Plot Results for a key city ---
    # city_to_plot = 'beirut' # Use a lowercase city name
    # print(f"\n--- Validating and Plotting for City: {city_to_plot} ---")

    # history = train_set_raw[train_set_raw[GROUPING_KEY] == city_to_plot].copy()
    # validation_actuals = validation_set_raw[validation_set_raw[GROUPING_KEY] == city_to_plot]

    # if not validation_actuals.empty:
    #     predictions = []
    #     for date in validation_actuals.index:
    #         features_df = create_features(history)
    #         current_features = features_df.tail(1)
    #         current_features_encoded = pd.get_dummies(current_features, columns=[GROUPING_KEY])
    #         current_features_aligned = current_features_encoded.reindex(columns=FEATURES, fill_value=0)
    #         prediction = optimized_model.predict(current_features_aligned)[0]
    #         predictions.append(prediction)
    #         history.loc[date] = {GROUPING_KEY: city_to_plot, TARGET_COL: prediction}

    #     mape = mean_absolute_percentage_error(validation_actuals[TARGET_COL], predictions) * 100
    #     print(f"Validation MAPE for '{city_to_plot}': {mape:.2f}%")
    #     plot_validation_results(
    #         train_set_raw[train_set_raw[GROUPING_KEY] == city_to_plot],
    #         validation_actuals,
    #         predictions,
    #         city_to_plot
    #     )
    # else:
    #     print(f"No validation data found for city '{city_to_plot}'. Skipping plot.")

    # --- NEW, "Honest Recursive Validation" Block ---

    # We will validate on a key region to check authenticity.
    region_to_plot = 'Beirut' # Use a lowercase region name to match your data
    print(f"\n--- Performing HONEST Recursive Validation for Region: '{region_to_plot}' ---")
    print("This mimics the live API's behavior to verify its output.")

    # Isolate the training history and the actual future values for this region
    history = train_set_raw[train_set_raw[GROUPING_KEY].str.lower() == region_to_plot].copy()
    validation_actuals = validation_set_raw[validation_set_raw[GROUPING_KEY].str.lower() == region_to_plot]

    if not validation_actuals.empty:
        # This list will store our recursive predictions
        recursive_predictions = []
        
        # This is the same loop as in your API
        for date in validation_actuals.index:
            # Create features based on the current history
            features_df = create_features(history)
            
            # Get the feature set for the current step
            current_features = features_df.tail(1)
            
            # One-hot encode and align columns
            current_features_encoded = pd.get_dummies(current_features, columns=[GROUPING_KEY])
            current_features_aligned = current_features_encoded.reindex(columns=FEATURES, fill_value=0)
            
            # Make one prediction
            prediction = optimized_model.predict(current_features_aligned)[0]
            recursive_predictions.append(prediction)
            
            # IMPORTANT: Add the *prediction* back to the history for the next loop
            history.loc[date] = {GROUPING_KEY: region_to_plot, TARGET_COL: prediction}

        # Now, evaluate the HONEST forecast
        mape = mean_absolute_percentage_error(validation_actuals[TARGET_COL], recursive_predictions) * 100
        print(f"\nHonest Recursive Validation MAPE for '{region_to_plot}': {mape:.2f}%")
        print("(This MAPE is expected to be higher than a simple one-step forecast)")
        
        # And plot the HONEST forecast
        plot_validation_results(
            train_set_raw[train_set_raw[GROUPING_KEY].str.lower() == region_to_plot], 
            validation_actuals, 
            recursive_predictions, 
            f"Honest Recursive Forecast for {region_to_plot.capitalize()}" # New Title
        )
    else:
        print(f"No validation data found for region '{region_to_plot}'. Skipping honest validation.")

# ... (The rest of the script continues with re-training on all data and saving) ...

    # --- Re-train model on ALL available data for production ---
    print("\n-> Re-training final model on ALL data from Supabase...")
    all_data_final = pd.get_dummies(data_with_features, columns=[GROUPING_KEY]).dropna()
    # Ensure the features match the ones from tuning
    X_all, y_all = all_data_final.reindex(columns=FEATURES, fill_value=0), all_data_final[TARGET_COL]
    
    production_model = xgb.XGBRegressor(objective='reg:squarederror', random_state=42, **best_params)
    production_model.fit(X_all, y_all)
    
    # Save the production-ready model and its columns
    print(f"-> Saving final production model to '{MODEL_FILE_PATH}'")
    joblib.dump(production_model, MODEL_FILE_PATH)
    pd.Series(FEATURES).to_json(MODEL_COLS_PATH)
    
    print("\n--- Production Model Training Complete ---")