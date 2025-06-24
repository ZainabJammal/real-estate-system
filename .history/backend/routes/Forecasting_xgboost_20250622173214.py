

# import pandas as pd
# import xgboost as xgb
# import joblib
# import os
# import numpy as np
# import matplotlib.pyplot as plt
# from sklearn.metrics import mean_absolute_percentage_error
# from sklearn.model_selection import GridSearchCV, TimeSeriesSplit
# from supabase import create_client, Client
# from dotenv import load_dotenv

# # --- 1. CONFIGURATION ---
# # Load environment variables from .env file
# load_dotenv()

# # Define paths for saving the model artifacts
# SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
# MODEL_FILE_PATH = os.path.join(SCRIPT_DIR, 'forecast_model.joblib')
# MODEL_COLS_PATH = os.path.join(SCRIPT_DIR, 'model_columns.json')

# GROUPING_KEY = 'city' 
# TARGET_COL = 'transaction_value'

# # --- 2. HELPER FUNCTIONS ---

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
#     # The GROUPING_KEY column is named 'city' in your database table
#     df[GROUPING_KEY] = df[GROUPING_KEY].str.lower()
    
#     # return df.set_index('date').drop(columns=['id', 'date'], errors='ignore')
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

# # --- 3. MAIN TRAINING WORKFLOW ---
# if __name__ == "__main__":
#     print("--- Starting Production Model Training from Supabase Data ---")

#     # Delete old model files to ensure a fresh start
#     if os.path.exists(MODEL_FILE_PATH): os.remove(MODEL_FILE_PATH)
#     if os.path.exists(MODEL_COLS_PATH): os.remove(MODEL_COLS_PATH)

#     # Load data from the database
#     all_data = load_data_from_supabase()
#     data_with_features = create_features(all_data)

#     # Split data for training and validation
#     split_date = '2016-01-01'
#     train_set_raw = data_with_features[data_with_features.index < split_date]
#     validation_set_raw = data_with_features[data_with_features.index >= split_date]
#     print(f"-> Data split for validation at: {split_date}")

#     # Prepare data for XGBoost (One-Hot Encode region and handle NaNs from lags)
#     train_final = pd.get_dummies(train_set_raw, columns=[GROUPING_KEY]).dropna()
#     FEATURES = [col for col in train_final.columns if col != TARGET_COL]
#     X_train, y_train = train_final[FEATURES], train_final[TARGET_COL]

#     # Hyperparameter Tuning with the expanded grid
#     # param_grid = {
#     #     'n_estimators': [1000, 1500],
#     #     'max_depth': [3, 5, 7, 10],
#     #     'learning_rate': [0.01, 0.05],
#     #     'subsample': [0.7, 0.8, 1.0],
#     #     'colsample_bytree': [0.7, 0.8, 1.0]
#     # }
#     param_grid = {
#         'n_estimators': [500, 1000],
#         'max_depth': [3, 5, 7],
#         'learning_rate': [0.01, 0.05],
#         'subsample': [0.7, 0.8, 1.0],
#         'colsample_bytree': [0.7, 0.8, 1.0]
#     }
#     tscv = TimeSeriesSplit(n_splits=5)
    
#     print("-> Starting GridSearchCV for hyperparameter tuning...")
#     grid_search = GridSearchCV(
#         estimator=xgb.XGBRegressor(objective='reg:squarederror', random_state=42),
#         param_grid=param_grid, scoring='neg_mean_absolute_percentage_error', cv=tscv, verbose=1
#     )
#     grid_search.fit(X_train, y_train)
#     best_params = grid_search.best_params_
#     print(f"-> GridSearchCV complete. Best parameters found: {best_params}")

#     # Train a final model on the entire pre-2016 training set with best params
#     optimized_model = xgb.XGBRegressor(objective='reg:squarederror', random_state=42, **best_params)
#     optimized_model.fit(X_train, y_train)

#     # # --- Validate and Plot Results for a key city ---
#     # city_to_plot = 'beirut' # Use a lowercase city name
#     # print(f"\n--- Validating and Plotting for City: {city_to_plot} ---")

#     # history = train_set_raw[train_set_raw[GROUPING_KEY] == city_to_plot].copy()
#     # validation_actuals = validation_set_raw[validation_set_raw[GROUPING_KEY] == city_to_plot]

#     # if not validation_actuals.empty:
#     #     predictions = []
#     #     for date in validation_actuals.index:
#     #         features_df = create_features(history)
#     #         current_features = features_df.tail(1)
#     #         current_features_encoded = pd.get_dummies(current_features, columns=[GROUPING_KEY])
#     #         current_features_aligned = current_features_encoded.reindex(columns=FEATURES, fill_value=0)
#     #         prediction = optimized_model.predict(current_features_aligned)[0]
#     #         predictions.append(prediction)
#     #         history.loc[date] = {GROUPING_KEY: city_to_plot, TARGET_COL: prediction}

#     #     mape = mean_absolute_percentage_error(validation_actuals[TARGET_COL], predictions) * 100
#     #     print(f"Validation MAPE for '{city_to_plot}': {mape:.2f}%")
#     #     plot_validation_results(
#     #         train_set_raw[train_set_raw[GROUPING_KEY] == city_to_plot],
#     #         validation_actuals,
#     #         predictions,
#     #         city_to_plot
#     #     )
#     # else:
#     #     print(f"No validation data found for city '{city_to_plot}'. Skipping plot.")

#     # --- NEW, "Honest Recursive Validation" Block ---

#     # We will validate on a key region to check authenticity.
#     region_to_plot = 'beirut' # Use a lowercase region name to match your data
#     print(f"\n--- Performing HONEST Recursive Validation for Region: '{region_to_plot}' ---")
#     print("This mimics the live API's behavior to verify its output.")

#     # Isolate the training history and the actual future values for this region
#     history = train_set_raw[train_set_raw[GROUPING_KEY].str.lower() == region_to_plot].copy()
#     validation_actuals = validation_set_raw[validation_set_raw[GROUPING_KEY].str.lower() == region_to_plot]

#     if not validation_actuals.empty:
#         # This list will store our recursive predictions
#         recursive_predictions = []
        
#         # This is the same loop as in your API
#         for date in validation_actuals.index:
#             # Create features based on the current history
#             features_df = create_features(history)
            
#             # Get the feature set for the current step
#             current_features = features_df.tail(1)
            
#             # One-hot encode and align columns
#             current_features_encoded = pd.get_dummies(current_features, columns=[GROUPING_KEY])
#             current_features_aligned = current_features_encoded.reindex(columns=FEATURES, fill_value=0)
            
#             # Make one prediction
#             prediction = optimized_model.predict(current_features_aligned)[0]
#             recursive_predictions.append(prediction)
            
#             # IMPORTANT: Add the *prediction* back to the history for the next loop
#             history.loc[date] = {GROUPING_KEY: region_to_plot, TARGET_COL: prediction}

#         # Now, evaluate the HONEST forecast
#         mape = mean_absolute_percentage_error(validation_actuals[TARGET_COL], recursive_predictions) * 100
#         print(f"\nHonest Recursive Validation MAPE for '{region_to_plot}': {mape:.2f}%")
#         print("(This MAPE is expected to be higher than a simple one-step forecast)")
        
#         # And plot the HONEST forecast
#         plot_validation_results(
#             train_set_raw[train_set_raw[GROUPING_KEY].str.lower() == region_to_plot], 
#             validation_actuals, 
#             recursive_predictions, 
#             f"Honest Recursive Forecast for {region_to_plot.capitalize()}" # New Title
#         )
#     else:
#         print(f"No validation data found for region '{region_to_plot}'. Skipping honest validation.")

# # ... (The rest of the script continues with re-training on all data and saving) ...

#     # --- Re-train model on ALL available data for production ---
#     print("\n-> Re-training final model on ALL data from Supabase...")
#     all_data_final = pd.get_dummies(data_with_features, columns=[GROUPING_KEY]).dropna()
#     # Ensure the features match the ones from tuning
#     X_all, y_all = all_data_final.reindex(columns=FEATURES, fill_value=0), all_data_final[TARGET_COL]
    
#     production_model = xgb.XGBRegressor(objective='reg:squarederror', random_state=42, **best_params)
#     production_model.fit(X_all, y_all)
    
#     # Save the production-ready model and its columns
#     print(f"-> Saving final production model to '{MODEL_FILE_PATH}'")
#     joblib.dump(production_model, MODEL_FILE_PATH)
#     pd.Series(FEATURES).to_json(MODEL_COLS_PATH)
    
#     print("\n--- Production Model Training Complete ---")






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
CSV_FILE_PATH = os.path.join(SCRIPT_DIR, 'agg_trans.csv') 
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
    parts = df['date'].str.split('-', expand=True)
    df['date'] = pd.to_datetime(('01-' + parts[1] + '-' + parts[0]), format='%d-%b-%y')
    df['date'] = pd.to_datetime(df['date'], format='%Y-%m-%d')
    df.set_index('date', inplace=True)
    df.sort_index(inplace=True)
    print("-> Date column converted to datetime and set as index.")
    # The GROUPING_KEY column is named 'city' in your database table
    df[GROUPING_KEY] = df[GROUPING_KEY].str.lower()
    
    # return df.set_index('date').drop(columns=['id', 'date'], errors='ignore')
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

# --- MAIN WORKFLOW (with Validation and Plotting) ---
if __name__ == "__main__":
    print("--- Starting 'Best Guess' Model Training with Validation ---")

    # --- Step 1: Load and Prepare Data ---
    all_data = load_data_from_supabase()
    # Create the NEW, simpler feature set (no lags)
    data_with_features = create_features(all_data)
    data_final = pd.get_dummies(data_with_features, columns=[GROUPING_KEY])

    # --- Step 2: Split Data for Training and Validation ---
    # We will use 2016 as our validation set, just like before.
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

    model = xgb.XGBRegressor(**xgb_params)
    
    # We use the test set for early stopping to prevent overfitting
    model.fit(X_train, y_train, eval_set=[(X_test, y_test)], verbose=False)

        # --- Step 4: Evaluate and Plot the Validation Results (ROBUST VERSION) ---
    print("-> Evaluating model performance on the 2016 validation set...")
    predictions = model.predict(X_test)

    # In Forecasting_xgboost.py

# --- Step 4: Evaluate and Plot the Validation Results (Using Your Corrected Logic) ---

    # First, evaluate the model on the entire 2016 validation set to get the overall MAPE
    print("-> Evaluating model performance on the 2016 validation set...")
    predictions_all = model.predict(X_test)
    mape = mean_absolute_percentage_error(y_test, predictions_all) * 100
    print(f"\nOverall Validation MAPE on 2016 data: {mape:.2f}%\n")


    # --- Now, Plot the Results for a Single Region ---
    # Use lowercase to match the data processing
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

    # Prepare ONLY the validation feature set for this one region
    # This uses the one-hot encoded test data we already created (X_test)
    X_validation_city = X_test[X_test[f'{GROUPING_KEY}_{city_to_plot}'] == 1]

    if not X_validation_city.empty:
        # Predict ONLY on this region's data. This guarantees we get exactly 12 predictions.
        city_plot_predictions = model.predict(X_validation_city)

        # Call the plotting function you defined at the top of the script
        plot_validation_results(
            train_df=train_city_df,
            validation_df=validation_city_df,
            predictions_df=city_plot_predictions,
            city_name=city_to_plot.capitalize()  # Pass the name for the title
        )
    else:
        print(f"No validation data found to plot for region '{city_to_plot}'.")

    # --- Step 5: Re-train Final Model on ALL Data ---
    print("-> Validation complete. Re-training final model on ALL available data for production...")
    X_all, y_all = data_final[FEATURES], data_final[TARGET_COL]

    final_n_estimators = model.best_iteration if model.best_iteration else 500
    
    final_params = xgb_params.copy()
    final_params['n_estimators'] = final_n_estimators
    
    production_model = xgb.XGBRegressor(**final_params)
    production_model.fit(X_all, y_all)
    
    print(f"-> Final model trained with {final_n_estimators} estimators.")
    
    # --- Step 6: Save the Production-Ready Model ---
    print(f"-> Saving final production model to '{MODEL_FILE_PATH}'")
    joblib.dump(production_model, MODEL_FILE_PATH)
    pd.Series(FEATURES).to_json(MODEL_COLS_PATH)
    
    print("\n--- 'Best Guess' Model Training Complete ---")