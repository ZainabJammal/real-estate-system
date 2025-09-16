# --- 1. IMPORTS & CONFIGURATION ---
import pandas as pd
import numpy as np
import xgboost as xgb
import joblib
import os
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_percentage_error
from sklearn.model_selection import GridSearchCV, TimeSeriesSplit
from supabase import create_client, Client
from dotenv import load_dotenv

# --- Basic Setup ---
load_dotenv()
OUTPUT_DIR = os.path.join(os.getcwd(), 'series_models')
MODEL_FILE_PATH = os.path.join(OUTPUT_DIR, 'forecast_model.joblib')
MODEL_COLS_PATH = os.path.join(OUTPUT_DIR, 'model_features.json')

# --- Constants ---
GROUPING_KEY = 'city'
TARGET_COL = 'transaction_value'
# *** THIS IS THE MOST IMPORTANT PART OF THE SCRIPT ***
# We explicitly define the columns to exclude to prevent data leakage.
COLUMNS_TO_EXCLUDE = [TARGET_COL, 'id', 'transaction_number']

# --- 2. HELPER FUNCTIONS ---
def load_data_from_supabase():
    print("-> Connecting to Supabase...")
    url, key = os.environ.get("SUPABASE_URL"), os.environ.get("SUPABASE_KEY")
    supabase: Client = create_client(url, key)
    response = supabase.table('merged_trans').select("*").order('date').execute()
    df = pd.DataFrame(response.data)
    df['date'] = pd.to_datetime(df['date'])
    df.set_index('date', inplace=True)
    df.sort_index(inplace=True)
    df[GROUPING_KEY] = df[GROUPING_KEY].str.lower()
    print(f"-> Fetched {len(df)} rows. Columns: {df.columns.tolist()}")
    return df

def create_features(df):
    # This robust function only uses the date index.
    df_features = df.copy()
    df_features['year'] = df_features.index.year
    df_features['is_december'] = (df_features.index.month == 12).astype(int)
    df_features['is_summer_peak'] = (df_features.index.month.isin([7, 8])).astype(int)
    for k in [1, 2, 3, 4]:
        df_features[f'month_sin_{k}'] = np.sin(2 * np.pi * k * df_features.index.month / 12)
        df_features[f'month_cos_{k}'] = np.cos(2 * np.pi * k * df_features.index.month / 12)
    return df_features

def plot_validation_results(train_df, validation_df, predictions_df, city_name):
    # This plotting function is correct.
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

# --- 3. MAIN EXECUTION SCRIPT ---
if __name__ == "__main__":
    print("--- Starting FINAL HONEST Model Training ---")

    # Step 1: Load and Prepare Data
    all_data = load_data_from_supabase()
    data_with_features = create_features(all_data)
    data_final = pd.get_dummies(data_with_features, columns=[GROUPING_KEY])
    
    # Step 2: Define the final, clean feature list
    FEATURES = [col for col in data_final.columns if col not in COLUMNS_TO_EXCLUDE]
    print(f"\n-> Training with {len(FEATURES)} HONEST features.")
    print(f"-> Feature list: {FEATURES}")

    # Step 3: Split Data for Validation
    split_date = '2021-01-01' # Using 1 year test set
    train_df = data_final[data_final.index < split_date]
    test_df = data_final[data_final.index >= split_date]
    print(f"-> Data split for validation at: {split_date}")

    X_train, y_train = train_df[FEATURES], train_df[TARGET_COL]
    X_test, y_test = test_df[FEATURES], test_df[TARGET_COL]

    # Step 4: Hyperparameter Tuning (using your best grid)
    print("\n--- Starting GridSearchCV ---")
    param_grid = {
      'objective': 'reg:squarederror',
      'random_state': 42,
      'max_depth': 3,               # Keep shallow to avoid overfitting
      'learning_rate': 0.01,        # Good for small datasets (slow, precise updates)
      'n_estimators': 2000,         # Increase for better convergence (early stopping will handle actual rounds)
      'subsample': 0.7,             # Lower to reduce overfitting (vs. 0.8)
      'colsample_bytree': 0.7,      # Same as above
      'gamma': 1,                   # Simpler trees (reduced from 5)
      'reg_alpha': 1.0,             # Slightly higher L1 regularization
      'reg_lambda': 2.0,            # Slightly higher L2 regularization
      'min_child_weight': 3,        # New: Controls leaf node splits (higher = more conservative)
      'tree_method': 'hist',        # Faster training for small/medium datasets
      'early_stopping_rounds': 50,  
    }
    grid_search = GridSearchCV(
        estimator=xgb.XGBRegressor(objective='reg:squarederror', random_state=42),
        param_grid=param_grid, cv=TimeSeriesSplit(n_splits=3), scoring='neg_mean_squared_error', n_jobs=-1
    )
    grid_search.fit(X_train, y_train)
    best_params = grid_search.best_params_
    print(f"\nGridSearchCV Complete. Best parameters found: {best_params}")

    # Step 5: Evaluate the HONEST model
    validation_model = xgb.XGBRegressor(objective='reg:squarederror', random_state=42, **best_params)
    validation_model.fit(X_train, y_train)
    predictions = validation_model.predict(X_test)
    mape = mean_absolute_percentage_error(y_test, predictions) * 100
    print(f"\n--- HONEST Model Performance: MAPE = {mape:.2f}% ---")

    # Step 6: Plotting and Feature Importance
    plot_validation_results(train_df, test_df, predictions, 'Beirut')
    plt.figure(figsize=(10, 8))
    xgb.plot_importance(validation_model, height=0.8)
    plt.title("Feature Importance from Final HONEST Model", fontsize=16)
    plt.show()

    # Step 7: Re-train the final model on ALL data for production
    print("\n-> Re-training final model on ALL data for production...")
    X_all, y_all = data_final[FEATURES], data_final[TARGET_COL]
    production_model = xgb.XGBRegressor(objective='reg:squarederror', random_state=42, **best_params)
    production_model.fit(X_all, y_all)
    print("-> Final production model trained.")

    # Step 8: Save the final, honest production model
    print("\n--- Saving PRODUCTION model and feature list ---")
    if not os.path.exists(OUTPUT_DIR):
        os.makedirs(OUTPUT_DIR)
    joblib.dump(production_model, MODEL_FILE_PATH)
    pd.Series(FEATURES).to_json(MODEL_COLS_PATH, indent=4)
    print(f"-> Production model saved to: {MODEL_FILE_PATH}")
    print("\n--- Model Training and Saving Complete ---")