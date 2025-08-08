import pandas as pd
import xgboost as xgb
import joblib
import os
import numpy as np
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
OUTPUT_DIR = os.path.join(BACKEND_DIR, 'timeseries_models') 

# Update file paths to use the new directory
MODEL_FILE_PATH = os.path.join(OUTPUT_DIR, 'forecast_model.joblib')
MODEL_COLS_PATH = os.path.join(OUTPUT_DIR, 'model_columns.json')

GROUPING_KEY = 'city' 
TARGET_COL = 'transaction_value'

# --- 2. FUNCTIONS ---

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
    parts = df['date'].str.split('-', expand=True)
    df['date'] = pd.to_datetime(('01-' + parts[1] + '-' + parts[0]), format='%d-%b-%y')
    df['date'] = pd.to_datetime(df['date'], format='%Y-%m-%d')
    df.set_index('date', inplace=True)
    df.sort_index(inplace=True)
    print("-> Date column converted to datetime and set as index.")
    
    df[GROUPING_KEY] = df[GROUPING_KEY].str.lower()
    
    return df

def create_features(df):
    """Optimal features for a small dataset with a long forecast horizon."""
    df_features = df.copy()
    df_features['year'] = df_features.index.year
    df_features['month_sin'] = np.sin(2 * np.pi * df_features.index.month / 12.0)
    df_features['month_cos'] = np.cos(2 * np.pi * df_features.index.month / 12.0)
    df_features['is_december'] = (df_features.index.month == 12).astype(int)
    df_features['is_summer_peak'] = (df_features.index.month.isin([7, 8])).astype(int)
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

# ----------------------------------------

if __name__ == "__main__":
    print("--- Starting 'Best Guess' Model Training with Validation ---")

    # --- Step 1: Load and Prepare Data ---
    all_data = load_data_from_supabase()
    data_with_features = create_features(all_data)
    data_final = pd.get_dummies(data_with_features, columns=[GROUPING_KEY])

    # --- Step 2: Split Data for Training and Validation ---
    split_date = '2016-01-01'
    train_df = data_final[data_final.index < split_date]
    test_df = data_final[data_final.index >= split_date]

    # Check if there is data to validate on
    if test_df.empty:
        raise ValueError("No data available for validation (post-2016). Cannot proceed.")
    print(f"-> Data split for validation at: {split_date}")
    
    FEATURES = [col for col in data_final.columns if col != TARGET_COL]
    
    X_train, y_train = train_df[FEATURES], train_df[TARGET_COL]
    X_test, y_test = test_df[FEATURES], test_df[TARGET_COL]

    # --- Step 3: Train the Model on the Training Set ---
    print("-> Training the XGBoost model on pre-2016 data...")
    xgb_params = {
        'objective': 'reg:squarederror',
        'random_state': 42,
        'max_depth': 3,
        'learning_rate': 0.01, 
        'n_estimators': 1000,
        'subsample': 0.8,
        'colsample_bytree': 0.8,
        'gamma': 5,
        'reg_alpha': 0.5,
        'reg_lambda': 1.5
    }

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

    model = xgb.XGBRegressor(**xgb_params)
    
    # We use the test set for early stopping to prevent overfitting
    model.fit(X_train, y_train, eval_set=[(X_test, y_test)], verbose=False)

    # --- Step 4: Evaluate and Plot the Validation Results (ROBUST VERSION) ---
    print("-> Evaluating model performance on the 2016 validation set...")
    predictions = model.predict(X_test)


    # --- Step 4: Evaluate and Plot the Validation Results (Using Your Corrected Logic) ---

    # First, evaluate the model on the entire 2016 validation set to get the overall MAPE
    print("-> Evaluating model performance on the 2016 validation set...")
    predictions_all = model.predict(X_test)
    mae = mean_absolute_error(y_test, predictions_all)
    rmse = np.sqrt(mean_squared_error(y_test, predictions_all))
    mape = mean_absolute_percentage_error(y_test, predictions_all) * 100
    print("\n--- Model Performance on Held-Out Validation Set (2016) ---")
    print(f"  Root Mean Squared Error (RMSE): {rmse:,.2f}")
    print(f"  Mean Absolute Error (MAE):      {mae:,.2f}")
    print(f"  Overall Validation MAPE on 2016 data: {mape:.2f}%\n")


    # --- Now, Plot the Results for a Single Region ---
    # Use lowercase to match the data processing + Kesrouan is the tricky one to plot
    city_to_plot = 'kesrouan' 
    print(f"-> Generating validation plot for a single region: '{city_to_plot}'")

    # Isolate the original (non-encoded) data for plotting the historical and actual values
    train_city_df = data_with_features[
        (data_with_features.index < split_date) & 
        (data_with_features[GROUPING_KEY] == city_to_plot)
    ]
    validation_city_df = data_with_features[
        (data_with_features.index >= split_date) & 
        (data_with_features[GROUPING_KEY] == city_to_plot)
    ]

    X_validation_city = X_test[X_test[f'{GROUPING_KEY}_{city_to_plot}'] == 1]

    if not X_validation_city.empty:
        city_plot_predictions = model.predict(X_validation_city)

        plot_validation_results(
            train_df=train_city_df,
            validation_df=validation_city_df,
            predictions_df=city_plot_predictions,
            city_name=city_to_plot.capitalize()  
        )
    else:
        print(f"No validation data found to plot for region '{city_to_plot}'.")

    # --- Step 5: Re-train Final Model on ALL Data ---
    print("\n-> Validation complete. Re-training final model on ALL available data for production...")
    X_all, y_all = data_final[FEATURES], data_final[TARGET_COL]
        
    # We will use the same robust hyperparameters we defined earlier.
    # No need to get clever with best_iteration.
    print(f"-> Training final model with {xgb_params['n_estimators']} estimators...")
    production_model = xgb.XGBRegressor(**xgb_params)

    # Train the final model on the entire dataset.
    production_model.fit(X_all, y_all)
    # print(f"-> Final model trained with {final_n_estimators} estimators.")
    
    # --- Step 6: Save the Production-Ready Model ---
    # print(f"-> Saving final production model to '{MODEL_FILE_PATH}'")
    # joblib.dump(production_model, MODEL_FILE_PATH)
    # pd.Series(FEATURES).to_json(MODEL_COLS_PATH)
    
    if not os.path.exists(OUTPUT_DIR):
        os.makedirs(OUTPUT_DIR)
        print(f"-> Created directory: '{OUTPUT_DIR}'")
        print(f"-> Saving final production model to '{MODEL_FILE_PATH}'")
        joblib.dump(production_model, MODEL_FILE_PATH)
        pd.Series(FEATURES).to_json(MODEL_COLS_PATH)
    
    print("\n--- 'Best Guess' Model Training Complete ---")

# # --- 1. IMPORTS ---
# import pandas as pd
# import numpy as np
# import xgboost as xgb
# import joblib
# import os
# import matplotlib.pyplot as plt
# from sklearn.metrics import mean_absolute_percentage_error, mean_squared_error, mean_absolute_error
# from sklearn.model_selection import GridSearchCV, TimeSeriesSplit
# from supabase import create_client, Client
# from dotenv import load_dotenv


# # --- 1. CONFIGURATION ---
# # Load environment variables from .env file
# load_dotenv()
# SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
# BACKEND_DIR = os.path.dirname(SCRIPT_DIR)
# OUTPUT_DIR = os.path.join(BACKEND_DIR, 'timeseries_models') 

# # Update file paths to use the new directory
# MODEL_FILE_PATH = os.path.join(OUTPUT_DIR, 'forecast_model.joblib')
# MODEL_COLS_PATH = os.path.join(OUTPUT_DIR, 'model_columns.json')

# # --- 2. GLOBAL CONSTANTS ---
# GROUPING_KEY = 'city'
# TARGET_COL = 'transaction_value'

# # --- 3. HELPER FUNCTIONS ---


# def load_data_from_supabase():
#     """Connects to Supabase and fetches all regional transaction data."""
#     print("-> Connecting to Supabase to fetch training data...")
#     url: str = os.environ.get("SUPABASE_URL")
#     key: str = os.environ.get("SUPABASE_KEY")

#     if not url or not key:
#         raise ValueError("FATAL: Supabase URL or Key not found in .env file.")
        
#     supabase: Client = create_client(url, key)

#     # Fetch all data from the table, ordered by date
#     response = supabase.table('merged_trans').select("*").order('date').execute()
    
#     # Convert the list of dictionaries to a pandas DataFrame
#     df = pd.DataFrame(response.data)
#     print(f"-> Successfully fetched {len(df)} rows from Supabase.")

#     # Directly convert the 'date' column, as pandas can infer the format.
#     df['date'] = pd.to_datetime(df['date'])
#     df.set_index('date', inplace=True)
#     df.sort_index(inplace=True)
#     print("-> Date column converted to datetime and set as index.")
    
#     # Convert city names to lowercase for consistency.
#     df[GROUPING_KEY] = df[GROUPING_KEY].str.lower()
    
#     return df

# def create_features(df):
#     """
#     Creates time-series features without using sequential lags.
#     This allows for simple "batch" forecasting.
#     """
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


# # --- 4. MAIN EXECUTION SCRIPT ---

# if __name__ == "__main__":

#     print("--- Starting 'Best Guess' Model Training with Validation ---")

#    # --- Step 1: Load and Prepare Data ---
#     all_data = load_data_from_supabase()

#     EXCLUDE_COLS = [TARGET_COL, 'id', 'transaction_number']
#     print(f"-> EXCLUDING LEAKY COLUMNS: {EXCLUDE_COLS}")
#     data_with_features = create_features(all_data)
#     data_final = pd.get_dummies(data_with_features, columns=[GROUPING_KEY])

#     FEATURES = [col for col in data_final.columns if col != TARGET_COL]

#     # --- Step 2: Split for Validation ---
#     split_date = '2021-01-01'
#     train_df = data_final[data_final.index < split_date]
#     test_df = data_final[data_final.index >= split_date]
#     print(f"-> Data split for validation at: {split_date}")

    
#     X_train, y_train = train_df[FEATURES], train_df[TARGET_COL]
#     X_test, y_test = test_df[FEATURES], test_df[TARGET_COL]

#     # --- Step 3: Find Best Hyperparameters ---
#     print("\n--- Starting GridSearchCV ---")
#     tscv = TimeSeriesSplit(n_splits=3)
    
#     # *** FIX 1: Correct parameter grid ***
#     param_grid = {
#         'colsample_bytree': [0.8, 0.9, 1.0],
#         'max_depth': [3, 5, 7],
#         'learning_rate': [0.01, 0.05, 0.1],
#         'n_estimators': [100, 200, 300],
#         'subsample':  [0.8, 0.9, 1.0],
#         'reg_alpha': [0, 0.1, 1, 10],      # L1 regularization
#         'reg_lambda': [0, 0.1, 1, 10],
#         'gamma': [0, 0.1, 0.2],
#         'min_child_weight': [1, 3, 5]
#     }
#     grid_search = GridSearchCV(
#         estimator=xgb.XGBRegressor(objective='reg:squarederror', random_state=42),
#         param_grid=param_grid, cv=tscv, scoring='neg_mean_squared_error', verbose=2, n_jobs=-1
#     )
#     grid_search.fit(X_train, y_train)
#     best_params = grid_search.best_params_
#     print(f"\nGridSearchCV Complete. Best parameters found: {best_params}")

#     # --- Step 4: Evaluate Model on Test Set ---
#     print("\n--- Evaluating model with best params on test set ---")
#     validation_model = xgb.XGBRegressor(objective='reg:squarederror', random_state=42, **best_params)
#     validation_model.fit(X_train, y_train)

  
#     # --- Step 5: Evaluate Final Honest Models ---
#     predictions = validation_model.predict(X_test)

#     print("-> Evaluating model performance on the 2020 validation set...")
#     predictions_all = validation_model.predict(X_test)
#     mae = mean_absolute_error(y_test, predictions_all)
#     rmse = np.sqrt(mean_squared_error(y_test, predictions_all))
#     mape = mean_absolute_percentage_error(y_test, predictions_all) * 100
#     print("\n--- Model Performance on Held-Out Validation Set (2020) ---")
#     print(f"  Root Mean Squared Error (RMSE): {rmse:,.2f}")
#     print(f"  Mean Absolute Error (MAE):      {mae:,.2f}")
#     print(f"  Overall Validation MAPE on 2020 data: {mape:.2f}%\n")

#     # --- Step 6: Create the Unified Forecast Visualization ---
#     print("\n--- Generating the Unified Forecast Plot ---")
#     city_to_plot = 'beirut'

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
#         city_plot_predictions = validation_model.predict(X_validation_city)

#         plot_validation_results(
#             train_df=train_city_df,
#             validation_df=validation_city_df,
#             predictions_df=city_plot_predictions,
#             city_name=city_to_plot.capitalize()  
#         )
#     else:
#         print(f"No validation data found to plot for region '{city_to_plot}'.")

#     print("\n-> Validation complete. Re-training final model on ALL available data for production...")
#     X_all, y_all = data_final[FEATURES], data_final[TARGET_COL]
 
    
#     # *** FIX 2: Use best_params here ***
#     production_model = xgb.XGBRegressor(objective='reg:squarederror', random_state=42, **best_params)
#     production_model.fit(X_all, y_all)
#     print("-> Final production model trained.")

#     # print(f"-> Final model trained with {final_n_estimators} estimators.")




    
#     # --- Step 8: Plot Final Feature Importance ---
#     print("\n--- Feature Importance of the Final HONEST Mean Model ---")
#     plt.figure(figsize=(10, 8))
#     xgb.plot_importance(production_model, height=0.8)
#     plt.title("Feature Importance from Final HONEST Model", fontsize=16)
#     plt.show()
    
#      # --- Step 7: Save Production Artifacts ---
#     print("\n--- Saving PRODUCTION model and feature list to disk ---")
#     if not os.path.exists(OUTPUT_DIR):
#         os.makedirs(OUTPUT_DIR)

#     # *** FIX 3: Save the PRODUCTION model, not the validation one ***
#     joblib.dump(production_model, MODEL_FILE_PATH)
#     pd.Series(FEATURES).to_json(MODEL_COLS_PATH, indent=4)

#     print(f"-> Production model saved to: {MODEL_FILE_PATH}")
#     print(f"-> Feature list saved to: {MODEL_COLS_PATH}")
#     print("\n--- Model Training and Saving Complete ---")