# --- 1. IMPORTS ---
import pandas as pd
import numpy as np
import xgboost as xgb
import joblib
import os
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_percentage_error, mean_squared_error, mean_absolute_error
from sklearn.model_selection import GridSearchCV, TimeSeriesSplit
from supabase import create_client, Client
from dotenv import load_dotenv


# --- 2. CONFIGURATION ---
load_dotenv()
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
BACKEND_DIR = os.path.dirname(SCRIPT_DIR)
OUTPUT_DIR = os.path.join(BACKEND_DIR, 'next_models')

MODEL_FILE_PATH = os.path.join(OUTPUT_DIR, 'forecast_model.joblib')
MODEL_COLS_PATH = os.path.join(OUTPUT_DIR, 'model_columns.json')


# --- 3. GLOBAL CONSTANTS ---
GROUPING_KEY = 'city'
TARGET_COL = 'transaction_value'
CITIES = ['baabda', 'beirut', 'london', 'paris', 'tokyo'] # Assuming these are your 5 cities


# --- 4. HELPER FUNCTIONS ---

def load_data_from_supabase():
    """Connects to Supabase and fetches all regional transaction data."""
    print("-> Connecting to Supabase to fetch training data...")
    url: str = os.environ.get("SUPABASE_URL")
    key: str = os.environ.get("SUPABASE_KEY")

    if not url or not key:
        raise ValueError("FATAL: Supabase URL or Key not found in .env file.")

    supabase: Client = create_client(url, key)
    response = supabase.table('merged_trans').select("*").order('date').execute()

    df = pd.DataFrame(response.data)
    print(f"-> Successfully fetched {len(df)} rows from Supabase.")

    df['date'] = pd.to_datetime(df['date'])
    df.set_index('date', inplace=True)
    df.sort_index(inplace=True)
    print("-> Date column converted to datetime and set as index.")

    df[GROUPING_KEY] = df[GROUPING_KEY].str.lower()
    return df

def create_features(df):
    """
    Creates a comprehensive set of time-series features from the datetime index.
    """
    df_features = df.copy()
    
    # Date components
    df_features['year'] = df_features.index.year
    df_features['month'] = df_features.index.month
    df_features['day_of_month'] = df_features.index.day
    df_features['day_of_week'] = df_features.index.dayofweek
    df_features['week_of_year'] = df_features.index.isocalendar().week.astype(int)
    df_features['quarter'] = df_features.index.quarter

    # Binary flags
    df_features['is_weekend'] = (df_features.index.dayofweek >= 5).astype(int)
    df_features['is_month_start'] = df_features.index.is_month_start.astype(int)
    df_features['is_month_end'] = df_features.index.is_month_end.astype(int)
    df_features['is_quarter_start'] = df_features.index.is_quarter_start.astype(int)
    df_features['is_quarter_end'] = df_features.index.is_quarter_end.astype(int)
    df_features['is_year_start'] = df_features.index.is_year_start.astype(int)
    df_features['is_year_end'] = df_features.index.is_year_end.astype(int)
    
    return df_features

def plot_validation_results(train_df, validation_df, predictions_df, city_name):
    """Plots the validation results for a specific city."""
    plt.figure(figsize=(15, 7))
    plt.plot(train_df.index, train_df[TARGET_COL], label='Training Data (2011-2018)', color='blue')
    plt.plot(validation_df.index, validation_df[TARGET_COL], label='Actual Values (2019-2021)', color='green', marker='o')
    plt.plot(validation_df.index, predictions_df, label='Predicted Values (2019-2021)', color='red', linestyle='--')
    plt.title(f'Model Validation for {city_name}: Actual vs. Predicted', fontsize=16)
    plt.xlabel('Date')
    plt.ylabel('Transaction Value')
    plt.legend()
    plt.grid(True)
    plt.show()

def plot_future_forecast(historical_df, forecast_df):
    """Plots the historical data and the future forecast for all cities combined."""
    plt.figure(figsize=(18, 8))
    
    # Plot historical data for each city
    for city in CITIES:
        city_hist_df = historical_df[historical_df[f'{GROUPING_KEY}_{city}'] == 1]
        plt.plot(city_hist_df.index, city_hist_df[TARGET_COL], label=f'Historical - {city.capitalize()}')

    # Plot forecasted data for each city
    for city in CITIES:
        city_forecast_df = forecast_df[forecast_df[f'{GROUPING_KEY}_{city}'] == 1]
        plt.plot(city_forecast_df.index, city_forecast_df['prediction'], label=f'Forecast - {city.capitalize()}', linestyle='--')

    plt.title('Future Forecast (2022-2025) vs. Historical Data', fontsize=16)
    plt.xlabel('Date')
    plt.ylabel('Transaction Value')
    plt.legend()
    plt.grid(True)
    plt.show()


# --- 5. MAIN EXECUTION SCRIPT ---

if __name__ == "__main__":

    print("--- Starting Advanced Model Training and Forecasting ---")

    # --- Step 1: Load and Prepare Data ---
    all_data = load_data_from_supabase()

    data_with_features = create_features(all_data)
    data_final = pd.get_dummies(data_with_features, columns=[GROUPING_KEY], prefix=GROUPING_KEY)

    FEATURES = [col for col in data_final.columns if col not in [TARGET_COL, 'id', 'transaction_number']]

    # --- Step 2: Train-Validation-Test Split ---
    train_start, valid_start, test_start = '2011-01-01', '2019-01-01', '2022-01-01'
    
    train_df = data_final.loc[train_start:valid_start].iloc[:-1] # Exclude the first day of validation
    validation_df = data_final.loc[valid_start:test_start].iloc[:-1] # Exclude the first day of test
    
    X_train, y_train = train_df[FEATURES], train_df[TARGET_COL]
    X_val, y_val = validation_df[FEATURES], validation_df[TARGET_COL]
    
    print(f"-> Training data: {X_train.index.min()} to {X_train.index.max()} ({len(X_train)} rows)")
    print(f"-> Validation data: {X_val.index.min()} to {X_val.index.max()} ({len(X_val)} rows)")

    # --- Step 3: Hyperparameter Tuning with GridSearchCV ---
    print("\n--- Starting GridSearchCV for Hyperparameter Tuning ---")
    tscv = TimeSeriesSplit(n_splits=5)
    
    param_grid = {
        'n_estimators': [100, 200, 300],
        'max_depth': [3, 6, 9],
        'learning_rate': [0.01, 0.05, 0.1],
        'subsample': [0.6, 0.8, 1.0],
        'colsample_bytree': [0.6, 0.8, 1.0],
        'gamma': [0, 0.1, 0.2],
        'min_child_weight': [1, 3, 5],
        'reg_alpha': [0, 0.1, 1],
        'reg_lambda': [0, 0.1, 1]
    }
    
    grid_search = GridSearchCV(
        estimator=xgb.XGBRegressor(objective='reg:squarederror', random_state=42),
        param_grid=param_grid, cv=tscv, scoring='neg_mean_squared_error', verbose=2, n_jobs=-1
    )
    grid_search.fit(X_train, y_train)
    best_params = grid_search.best_params_
    print(f"\nGridSearchCV Complete. Best parameters found: {best_params}")

    # --- Step 4: Train Model with Early Stopping and Evaluate on Validation Set ---
    print("\n--- Training model with best params and early stopping ---")
    
    model = xgb.XGBRegressor(objective='reg:squarederror', random_state=42, **best_params)
    
    eval_set = [(X_train, y_train), (X_val, y_val)]
    model.fit(X_train, y_train, 
              eval_set=eval_set, 
              eval_metric="rmse", 
              early_stopping_rounds=50, 
              verbose=True)

    # --- Step 5: Evaluate Performance on Validation Set ---
    predictions_val = model.predict(X_val)
    mae = mean_absolute_error(y_val, predictions_val)
    rmse = np.sqrt(mean_squared_error(y_val, predictions_val))
    mape = mean_absolute_percentage_error(y_val, predictions_val) * 100
    print("\n--- Model Performance on Held-Out Validation Set (2019-2021) ---")
    print(f"  Root Mean Squared Error (RMSE): {rmse:,.2f}")
    print(f"  Mean Absolute Error (MAE):      {mae:,.2f}")
    print(f"  Validation MAPE: {mape:.2f}%\n")

    # --- Step 6: Visualize Validation Results for a Sample City ---
    city_to_plot = 'beirut'
    print(f"-> Generating validation plot for '{city_to_plot}'")
    
    train_city_df = data_with_features[(data_with_features.index < valid_start) & (data_with_features[GROUPING_KEY] == city_to_plot)]
    validation_city_df = data_with_features[(data_with_features.index >= valid_start) & (data_with_features.index < test_start) & (data_with_features[GROUPING_KEY] == city_to_plot)]
    
    X_val_city = X_val[X_val[f'{GROUPING_KEY}_{city_to_plot}'] == 1]

    if not X_val_city.empty:
        city_plot_predictions = model.predict(X_val_city)
        plot_validation_results(train_city_df, validation_city_df, city_plot_predictions, city_to_plot.capitalize())
    else:
        print(f"No validation data found for '{city_to_plot}'.")


    # --- Step 7: Re-train Final Model on All Data (2011-2021) ---
    print("\n--- Re-training final model on ALL available data (2011-2021) for production... ---")
    
    X_all = pd.concat([X_train, X_val])
    y_all = pd.concat([y_train, y_val])

    production_model = xgb.XGBRegressor(objective='reg:squarederror', random_state=42, **best_params)
    production_model.fit(X_all, y_all)
    print("-> Final production model trained.")

    # --- Step 8: Predict the Next 4 Years (2022-2025) ---
    print("\n--- Predicting the future: 2022-2025 ---")
    
    future_dates = pd.date_range(start='2022-01-01', end='2025-12-31', freq='D')
    future_df_list = []

    for city in CITIES:
        temp_df = pd.DataFrame(index=future_dates)
        temp_df[GROUPING_KEY] = city
        future_df_list.append(temp_df)
    
    future_df = pd.concat(future_df_list)
    
    # Create features for the future dates
    future_features = create_features(future_df)
    future_final = pd.get_dummies(future_features, columns=[GROUPING_KEY], prefix=GROUPING_KEY)

    # Ensure all feature columns from training are present
    for col in FEATURES:
        if col not in future_final.columns:
            future_final[col] = 0
    future_final = future_final[FEATURES] # Keep order and columns consistent

    # Make predictions
    future_predictions = production_model.predict(future_final)
    future_final['prediction'] = future_predictions
    
    print(f"-> Successfully generated {len(future_final)} forecast points for the next 4 years.")
    
    # --- Step 9: Visualize the Future Forecast ---
    plot_future_forecast(data_final, future_final)
    
    # --- Step 10: Save Production Artifacts ---
    print("\n--- Saving PRODUCTION model and feature list ---")
    if not os.path.exists(OUTPUT_DIR):
        os.makedirs(OUTPUT_DIR)

    joblib.dump(production_model, MODEL_FILE_PATH)
    pd.Series(FEATURES).to_json(MODEL_COLS_PATH, indent=4)
    
    print(f"-> Production model saved to: {MODEL_FILE_PATH}")
    print(f"-> Feature list saved to: {MODEL_COLS_PATH}")

    # --- Step 11: Plot Final Feature Importance ---
    print("\n--- Feature Importance of the Final Production Model ---")
    plt.figure(figsize=(12, 9))
    xgb.plot_importance(production_model, height=0.8, max_num_features=20)
    plt.title("Top 20 Feature Importances from Final Model", fontsize=16)
    plt.tight_layout()
    plt.show()

    print("\n--- Model Training and Forecasting Complete ---")