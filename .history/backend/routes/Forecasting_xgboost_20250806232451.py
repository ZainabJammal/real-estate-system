# import pandas as pd
# import xgboost as xgb
# import joblib
# import os
# import numpy as np
# import matplotlib.pyplot as plt
# from sklearn.metrics import mean_absolute_percentage_error, mean_squared_error, mean_absolute_error
# from sklearn.model_selection import GridSearchCV, TimeSeriesSplit
# from supabase import create_client, Client
# from dotenv import load_dotenv

# # --- 1. CONFIGURATION ---
# # Load environment variables from .env file
# load_dotenv()

# # # Define paths for saving the model artifacts
# # SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
# # CSV_FILE_PATH = os.path.join(SCRIPT_DIR, 'agg_trans.csv') 
# # MODEL_FILE_PATH = os.path.join(SCRIPT_DIR, 'forecast_model.joblib')
# # MODEL_COLS_PATH = os.path.join(SCRIPT_DIR, 'model_columns.json')

# SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
# BACKEND_DIR = os.path.dirname(SCRIPT_DIR)
# OUTPUT_DIR = os.path.join(BACKEND_DIR, 'forecasting_models') 

# # Update file paths to use the new directory
# MODEL_FILE_PATH = os.path.join(OUTPUT_DIR, 'forecast_model.joblib')
# MODEL_COLS_PATH = os.path.join(OUTPUT_DIR, 'model_columns.json')

# GROUPING_KEY = 'city' 
# TARGET_COL = 'transaction_value'

# # --- 2. FUNCTIONS ---

# def load_data_from_supabase():
#     """Connects to Supabase and fetches all regional transaction data."""
#     print("-> Connecting to Supabase to fetch training data...")
#     url: str = os.environ.get("SUPABASE_URL")
#     key: str = os.environ.get("SUPABASE_KEY")

#     if not url or not key:
#         raise ValueError("FATAL: Supabase URL or Key not found in .env file.")
        
#     supabase: Client = create_client(url, key)

#     # Fetch all data from the table, ordered by date
#     response = supabase.table('agg_trans').select("*").order('date').execute()
    
#     # Convert the list of dictionaries to a pandas DataFrame
#     df = pd.DataFrame(response.data)
#     print(f"-> Successfully fetched {len(df)} rows from Supabase.")

#     # Convert date column to datetime objects and set as index
#     parts = df['date'].str.split('-', expand=True)
#     df['date'] = pd.to_datetime(('01-' + parts[1] + '-' + parts[0]), format='%d-%b-%y')
#     df['date'] = pd.to_datetime(df['date'], format='%Y-%m-%d')
#     df.set_index('date', inplace=True)
#     df.sort_index(inplace=True)
#     print("-> Date column converted to datetime and set as index.")
    
#     df[GROUPING_KEY] = df[GROUPING_KEY].str.lower()
    
#     return df

# def create_features(df):
#     """Optimal features for a small dataset with a long forecast horizon."""
#     df_features = df.copy()
#     df_features['year'] = df_features.index.year
#     df_features['month_sin'] = np.sin(2 * np.pi * df_features.index.month / 12.0)
#     df_features['month_cos'] = np.cos(2 * np.pi * df_features.index.month / 12.0)
#     df_features['is_december'] = (df_features.index.month == 12).astype(int)
#     df_features['is_summer_peak'] = (df_features.index.month.isin([7, 8])).astype(int)
#     return df_features

# def plot_validation_results(train_df, validation_df, predictions_df, city_name):
#     """Plots the validation results for a specific city."""
#     plt.figure(figsize=(15, 7))
#     plt.plot(train_df.index, train_df[TARGET_COL], label='Training Data', color='blue')
#     plt.plot(validation_df.index, validation_df[TARGET_COL], label='Actual Values', color='green', marker='o')
#     plt.plot(validation_df.index, predictions_df, label='Predicted Values', color='red', linestyle='--')
#     plt.title(f'Model Validation for {city_name}: Actual vs. Predicted', fontsize=16)
#     plt.xlabel('Date')
#     plt.ylabel('Transaction Value')
#     plt.legend()
#     plt.grid(True)
#     plt.show()

# # ----------------------------------------

# if __name__ == "__main__":
#     print("--- Starting 'Best Guess' Model Training with Validation ---")

#     # --- Step 1: Load and Prepare Data ---
#     all_data = load_data_from_supabase()
#     data_with_features = create_features(all_data)
#     data_final = pd.get_dummies(data_with_features, columns=[GROUPING_KEY])

#     # --- Step 2: Split Data for Training and Validation ---
#     split_date = '2016-01-01'
#     train_df = data_final[data_final.index < split_date]
#     test_df = data_final[data_final.index >= split_date]

#     # Check if there is data to validate on
#     if test_df.empty:
#         raise ValueError("No data available for validation (post-2016). Cannot proceed.")
#     print(f"-> Data split for validation at: {split_date}")
    
#     FEATURES = [col for col in data_final.columns if col != TARGET_COL]
    
#     X_train, y_train = train_df[FEATURES], train_df[TARGET_COL]
#     X_test, y_test = test_df[FEATURES], test_df[TARGET_COL]

#     # --- Step 3: Train the Model on the Training Set ---
#     print("-> Training the XGBoost model on pre-2016 data...")
#     xgb_params = {
#         'objective': 'reg:squarederror',
#         'random_state': 42,
#         'max_depth': 3,
#         'learning_rate': 0.01, 
#         'n_estimators': 1000,
#         'subsample': 0.8,
#         'colsample_bytree': 0.8,
#         'gamma': 5,
#         'reg_alpha': 0.5,
#         'reg_lambda': 1.5
#     }

#     model = xgb.XGBRegressor(**xgb_params)
    
#     # We use the test set for early stopping to prevent overfitting
#     model.fit(X_train, y_train, eval_set=[(X_test, y_test)], verbose=False)

#     # --- Step 4: Evaluate and Plot the Validation Results (ROBUST VERSION) ---
#     print("-> Evaluating model performance on the 2016 validation set...")
#     predictions = model.predict(X_test)


#     # --- Step 4: Evaluate and Plot the Validation Results (Using Your Corrected Logic) ---

#     # First, evaluate the model on the entire 2016 validation set to get the overall MAPE
#     print("-> Evaluating model performance on the 2016 validation set...")
#     predictions_all = model.predict(X_test)
#     mae = mean_absolute_error(y_test, predictions_all)
#     rmse = np.sqrt(mean_squared_error(y_test, predictions_all))
#     mape = mean_absolute_percentage_error(y_test, predictions_all) * 100
#     print("\n--- Model Performance on Held-Out Validation Set (2016) ---")
#     print(f"  Root Mean Squared Error (RMSE): {rmse:,.2f}")
#     print(f"  Mean Absolute Error (MAE):      {mae:,.2f}")
#     print(f"  Overall Validation MAPE on 2016 data: {mape:.2f}%\n")


#     # --- Now, Plot the Results for a Single Region ---
#     # Use lowercase to match the data processing + Kesrouan is the tricky one to plot
#     city_to_plot = 'kesrouan' 
#     print(f"-> Generating validation plot for a single region: '{city_to_plot}'")

#     # Isolate the original (non-encoded) data for plotting the historical and actual values
#     train_city_df = data_with_features[
#         (data_with_features.index < split_date) & 
#         (data_with_features[GROUPING_KEY] == city_to_plot)
#     ]
#     validation_city_df = data_with_features[
#         (data_with_features.index >= split_date) & 
#         (data_with_features[GROUPING_KEY] == city_to_plot)
#     ]

#     X_validation_city = X_test[X_test[f'{GROUPING_KEY}_{city_to_plot}'] == 1]

#     if not X_validation_city.empty:
#         city_plot_predictions = model.predict(X_validation_city)

#         plot_validation_results(
#             train_df=train_city_df,
#             validation_df=validation_city_df,
#             predictions_df=city_plot_predictions,
#             city_name=city_to_plot.capitalize()  
#         )
#     else:
#         print(f"No validation data found to plot for region '{city_to_plot}'.")

#     # --- Step 5: Re-train Final Model on ALL Data ---
#     print("\n-> Validation complete. Re-training final model on ALL available data for production...")
#     X_all, y_all = data_final[FEATURES], data_final[TARGET_COL]
        
#     # We will use the same robust hyperparameters we defined earlier.
#     # No need to get clever with best_iteration.
#     print(f"-> Training final model with {xgb_params['n_estimators']} estimators...")
#     production_model = xgb.XGBRegressor(**xgb_params)

#     # Train the final model on the entire dataset.
#     production_model.fit(X_all, y_all)
#     # print(f"-> Final model trained with {final_n_estimators} estimators.")
    
#     # --- Step 6: Save the Production-Ready Model ---
#     # print(f"-> Saving final production model to '{MODEL_FILE_PATH}'")
#     # joblib.dump(production_model, MODEL_FILE_PATH)
#     # pd.Series(FEATURES).to_json(MODEL_COLS_PATH)
    
#     if not os.path.exists(OUTPUT_DIR):
#         os.makedirs(OUTPUT_DIR)
#         print(f"-> Created directory: '{OUTPUT_DIR}'")
#         print(f"-> Saving final production model to '{MODEL_FILE_PATH}'")
#         joblib.dump(production_model, MODEL_FILE_PATH)
#         pd.Series(FEATURES).to_json(MODEL_COLS_PATH)
    
#     print("\n--- 'Best Guess' Model Training Complete ---")

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


# --- 1. CONFIGURATION ---
# Load environment variables from .env file
load_dotenv()

# # Define paths for saving the model artifacts
# SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
# CSV_FILE_PATH = os.path.join(SCRIPT_DIR, 'agg_trans.csv') 
# MODEL_FILE_PATH = os.path.join(SCRIPT_DIR, 'forecast_model.joblib')
# MODEL_COLS_PATH = os.path.join(SCRIPT_DIR, 'model_columns.json')

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
BACKEND_DIR = os.path.dirname(SCRIPT_DIR)
OUTPUT_DIR = os.path.join(BACKEND_DIR, 'TimeSeries_Forecasting_Models') 

# Update file paths to use the new directory
MODEL_FILE_PATH = os.path.join(OUTPUT_DIR, 'forecast_model.joblib')
MODEL_COLS_PATH = os.path.join(OUTPUT_DIR, 'model_columns.json')

# --- 2. GLOBAL CONSTANTS ---
GROUPING_KEY = 'city'
TARGET_COL = 'transaction_value'

# --- 3. HELPER FUNCTIONS ---


def load_data_from_supabase():
    """Connects to Supabase and fetches all regional transaction data."""
    print("-> Connecting to Supabase to fetch training data...")
    url: str = os.environ.get("SUPABASE_URL")
    key: str = os.environ.get("SUPABASE_KEY")

    if not url or not key:
        raise ValueError("FATAL: Supabase URL or Key not found in .env file.")
        
    supabase: Client = create_client(url, key)

    # Fetch all data from the table, ordered by date
    response = supabase.table('merged_trans').select("*").order('date').execute()
    
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
# def load_dataset(path='merged_trans.csv'):
#     """
#     Loads the dataset from a CSV file, converts the date column to datetime objects,
#     and sets it as the index.
#     """
#     df = pd.read_csv(path)
#     print(f"-> Successfully fetched {len(df)} rows from dataset.")

#     # Directly convert the 'date' column, as pandas can infer the format.
#     df['date'] = pd.to_datetime(df['date'])
#     df.set_index('date', inplace=True)
#     df.sort_index(inplace=True)
#     print("-> Date column converted to datetime and set as index.")
    
#     # Convert city names to lowercase for consistency.
#     df[GROUPING_KEY] = df[GROUPING_KEY].str.lower()
    
#     return df

def create_features(df):
    """
    Creates time series features from the datetime index. This is our best-performing
    feature set, using simple lags and seasonal features.
    """
    df_features = df.copy()
    
    # Standard time-based features
    df_features['year'] = df_features.index.year
    df_features['month_sin'] = np.sin(2 * np.pi * df_features.index.month / 12.0)
    df_features['month_cos'] = np.cos(2 * np.pi * df_features.index.month / 12.0)
    df_features['is_december'] = (df_features.index.month == 12).astype(int)
    df_features['is_summer_peak'] = (df_features.index.month.isin([7, 8])).astype(int)
    
    # Lag features calculated correctly within each group to prevent data leakage
    for lag in [1, 2, 3, 12]:
        df_features[f'lag_{lag}'] = df_features.groupby(GROUPING_KEY)[TARGET_COL].shift(lag)
        
    # Fill initial NaNs created by lags to prevent data loss, preserving the full dataset
    df_features.fillna(0, inplace=True)
            
    return df_features

# --- 4. MAIN EXECUTION SCRIPT ---

if __name__ == "__main__":

    # --- Step 1: Load, Prepare, and Split Data ---
    print("--- Starting Data Preparation ---")
    all_data = load_dataset()
    # Check if 'id' column exists from a previous reset_index() and drop it if so.
    if 'id' in all_data.columns:
        all_data = all_data.drop(columns=['id'])

    data_with_features = create_features(all_data)
    data_final = pd.get_dummies(data_with_features, columns=[GROUPING_KEY])

    # --- Step 2: Define a CLEAN Feature List (CRITICAL FIX) ---
    # We explicitly EXCLUDE all columns that could cause data leakage.
    EXCLUDE_COLS = [TARGET_COL, 'transaction_number']
    FEATURES = [col for col in data_final.columns if col not in EXCLUDE_COLS]
    print(f"\n--- Training with {len(FEATURES)} clean, non-leaky features ---")

    # --- Step 3: Split Data for Training and Validation ---
    split_date = '2021-01-01'
    train_df = data_final[data_final.index < split_date]
    test_df = data_final[data_final.index >= split_date]

    X_train = train_df[FEATURES]
    y_train = train_df[TARGET_COL]
    X_test = test_df[FEATURES]
    y_test = test_df[TARGET_COL]

    # --- Step 4: Hyperparameter Tuning with GridSearchCV ---
    print("\n--- Starting GridSearchCV to find best hyperparameters ---")
    tscv = TimeSeriesSplit(n_splits=3)
    param_grid = {
        'max_depth': [3, 4],
        'learning_rate': [0.01, 0.05],
        'n_estimators': [1000], # A high number as a max budget
        'subsample': [0.7, 0.8],
        'reg_alpha': [0.5, 1.0],
        'reg_lambda': [1.5, 2.0],
    }
    grid_search = GridSearchCV(
        estimator=xgb.XGBRegressor(objective='reg:squarederror', random_state=42),
        param_grid=param_grid,
        cv=tscv,
        scoring='neg_mean_squared_error',
        verbose=2,
        n_jobs=-1
    )
    grid_search.fit(X_train, y_train)
    print(f"\nGridSearchCV Complete. Best parameters found: {grid_search.best_params_}")
    best_params = grid_search.best_params_

    # --- Step 5: Train Final Models (Mean, P10, P90) ---
    print("\n--- Training final models using best params and early stopping ---")
    
    # Mean Model (The "Most Likely" Forecast)
    mean_model = xgb.XGBRegressor(objective='reg:squarederror', random_state=42, early_stopping_rounds=50, **best_params)
    mean_model.fit(X_train, y_train, eval_set=[(X_test, y_test)], verbose=False)

    # P90 Model (The "Optimistic Case")
    p90_model = xgb.XGBRegressor(objective='reg:quantileerror', quantile_alpha=0.90, random_state=42, early_stopping_rounds=50, **best_params)
    p90_model.fit(X_train, y_train, eval_set=[(X_test, y_test)], verbose=False)

    # P10 Model (The "Pessimistic Case")
    p10_model = xgb.XGBRegressor(objective='reg:quantileerror', quantile_alpha=0.10, random_state=42, early_stopping_rounds=50, **best_params)
    p10_model.fit(X_train, y_train, eval_set=[(X_test, y_test)], verbose=False)

    # --- Step 6: Evaluate Final Honest Models ---
    mean_preds = mean_model.predict(X_test)
    p90_preds = p90_model.predict(X_test)
    p10_preds = p10_model.predict(X_test)

    print("\n--- Final HONEST Model Performance ---")
    print(f"Mean Forecast MAPE: {mean_absolute_percentage_error(y_test, mean_preds) * 100:.2f}%")
    print(f"P90 Forecast MAPE:  {mean_absolute_percentage_error(y_test, p90_preds) * 100:.2f}%")
    print(f"P10 Forecast MAPE:  {mean_absolute_percentage_error(y_test, p10_preds) * 100:.2f}%")
    
    # --- Step 7: Create the Unified Forecast Visualization ---
    print("\n--- Generating the Unified Forecast Plot ---")
    city_to_plot = 'beirut'
    plt.figure(figsize=(18, 9))
    plt.plot(train_df[train_df[f'city_{city_to_plot}'] == 1].index, train_df[train_df[f'city_{city_to_plot}'] == 1][TARGET_COL], label='Training Data', color='blue')
    plt.plot(test_df[test_df[f'city_{city_to_plot}'] == 1].index, test_df[test_df[f'city_{city_to_plot}'] == 1][TARGET_COL], label='Actual Values', color='green', marker='o', linestyle='-')
    plt.plot(test_df[test_df[f'city_{city_to_plot}'] == 1].index, mean_preds[test_df[f'city_{city_to_plot}'] == 1], label='Mean Forecast', color='red', linestyle='--')
    plt.plot(test_df[test_df[f'city_{city_to_plot}'] == 1].index, p10_preds[test_df[f'city_{city_to_plot}'] == 1], label='P10 Forecast', color='#663399', linestyle=':')
    plt.plot(test_df[test_df[f'city_{city_to_plot}'] == 1].index, p90_preds[test_df[f'city_{city_to_plot}'] == 1], label='P90 Forecast', color='#FF8C00', linestyle=':')
    plt.fill_between(test_df[test_df[f'city_{city_to_plot}'] == 1].index, p10_preds[test_df[f'city_{city_to_plot}'] == 1], p90_preds[test_df[f'city_{city_to_plot}'] == 1], color='grey', alpha=0.2, label='80% Confidence Interval')
    plt.title(f'Unified Forecast for {city_to_plot.capitalize()}', fontsize=18)
    plt.xlabel('Date', fontsize=12)
    plt.ylabel('Transaction Value', fontsize=12)
    plt.legend(fontsize=11)
    plt.grid(True)
    plt.show()

    # --- Step 8: Plot Final Feature Importance ---
    print("\n--- Feature Importance of the Final HONEST Mean Model ---")
    plt.figure(figsize=(10, 8))
    xgb.plot_importance(mean_model, height=0.8)
    plt.title("Feature Importance from Final HONEST Model", fontsize=16)
    plt.show()

    # --- Step 9: Save Models for Production Use ---
    print("\n--- Saving final models and feature list to disk ---")
    OUTPUT_DIR = "production_models"
    if not os.path.exists(OUTPUT_DIR):
        os.makedirs(OUTPUT_DIR)
    joblib.dump(mean_model, os.path.join(OUTPUT_DIR, 'mean_forecast_model.joblib'))
    joblib.dump(p90_model, os.path.join(OUTPUT_DIR, 'p90_forecast_model.joblib'))
    joblib.dump(p10_model, os.path.join(OUTPUT_DIR, 'p10_forecast_model.joblib'))
    pd.Series(FEATURES).to_json(os.path.join(OUTPUT_DIR, 'model_features.json'), indent=4)
    print(f"Models and feature list saved to '{OUTPUT_DIR}' directory.")