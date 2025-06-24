# import pandas as pd
# import xgboost as xgb
# import joblib
# import os
# from sklearn.metrics import mean_squared_error, mean_absolute_error
# from sklearn.model_selection import GridSearchCV, TimeSeriesSplit
# import numpy as np
# import matplotlib.pyplot as plt

# # 1. CONFIGURATION AND PATH SETUP
# SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
# CSV_FILE_PATH = os.path.join(SCRIPT_DIR, 'transactions.csv')
# MODEL_FILE_PATH = os.path.join(SCRIPT_DIR, 'forecast_model.joblib')
# MODEL_COLS_PATH = os.path.join(SCRIPT_DIR, 'model_columns.json')
# TARGET_COL = 'transaction_value'

# # 2. HELPER FUNCTIONS
# def load_and_preprocess_data(filepath):
#     print("-> Loading and preprocessing data...")
#     df = pd.read_csv(filepath)
#     parts = df['date'].str.split('-', expand=True)
#     df['date_str'] = '01-' + parts[1] + '-' + parts[0]
#     df['date'] = pd.to_datetime(df['date_str'], format='%d-%b-%y')
#     df = df.set_index('date').drop(columns=['id', 'date_str'])
#     df.sort_index(inplace=True)
#     return df

# def create_features(df):
#     df['month'] = df.index.month
#     df['year'] = df.index.year
#     df['quarter'] = df.index.quarter
#     df['lag_1'] = df.groupby('city')[TARGET_COL].shift(1)
#     df['lag_3'] = df.groupby('city')[TARGET_COL].shift(3)
#     df['lag_12'] = df.groupby('city')[TARGET_COL].shift(12)
#     df['rolling_mean_3'] = df.groupby('city')[TARGET_COL].shift(1).rolling(window=3).mean()
#     df['rolling_std_3'] = df.groupby('city')[TARGET_COL].shift(1).rolling(window=3).std()
#     return df

# # 3. EVALUATION AND PLOTTING FUNCTIONS

# def evaluate_model(true_values, predicted_values):
#     mae = mean_absolute_error(true_values, predicted_values)
#     rmse = np.sqrt(mean_squared_error(true_values, predicted_values))
#     print("\n--- Model Performance on Held-Out Validation Set (2016) ---")
#     print(f"  Root Mean Squared Error (RMSE): {rmse:,.2f}")
#     print(f"  Mean Absolute Error (MAE):      {mae:,.2f}")
#     print("----------------------------------------------------------")

# def plot_validation_results(train_df, validation_df, predictions_df):
#     plt.figure(figsize=(15, 7))
#     plt.plot(train_df.index, train_df[TARGET_COL], label='Training Data (2011-2015)', color='blue')
#     plt.plot(validation_df.index, validation_df[TARGET_COL], label='Actual Values (2016)', color='green', marker='o', linestyle='-')
#     plt.plot(validation_df.index, predictions_df, label='Predicted Values (2016)', color='red', linestyle='--')
#     plt.title('Model Validation: Actual vs. Predicted for 2016', fontsize=16)
#     plt.xlabel('Date')
#     plt.ylabel('Transaction Value')
#     plt.legend()
#     plt.grid(True)
#     print("\n-> Displaying validation plot. Close the plot window to continue the script.")
#     plt.show()


# # 4. MAIN WORKFLOW: THE FINAL, CORRECTED VERSION
# if __name__ == "__main__":
#     print("--- Starting Hybrid ML Workflow: Tune, Train, Validate, Re-train ---")

#     # --- Step 1: Load, Prepare, and Split Data ---
#     all_data = load_and_preprocess_data(CSV_FILE_PATH)
#     data_with_features = create_features(all_data)

#     split_date = '2016-01-01'
#     train_set_raw = data_with_features[data_with_features.index < split_date]
#     validation_set_raw = data_with_features[data_with_features.index >= split_date]
    
#     print(f"-> Data split: Training set (for tuning) ends on {train_set_raw.index.max().date()}")
    
#     train_final_for_tuning = pd.get_dummies(train_set_raw, columns=['city'], prefix='city').dropna()
#     FEATURES = [col for col in train_final_for_tuning.columns if col != TARGET_COL]
#     X_train_tune, y_train_tune = train_final_for_tuning[FEATURES], train_final_for_tuning[TARGET_COL]

#     # --- Step 2: Hyperparameter Tuning with GridSearchCV ---
#     param_grid = {
#         'n_estimators': [500, 1000],
#         'max_depth': [3, 5],
#         'learning_rate': [0.01, 0.05],
#         'subsample': [0.7, 1.0],
#     }
#     print(f"\n-> Defined hyperparameter grid with {len(param_grid['n_estimators'])*len(param_grid['max_depth'])*len(param_grid['learning_rate'])*len(param_grid['subsample'])} combinations.")
    
#     tscv = TimeSeriesSplit(n_splits=5)
    
#     print("-> Starting GridSearchCV on the training set (2011-2015)...")
#     grid_search = GridSearchCV(
#         estimator=xgb.XGBRegressor(objective='reg:squarederror', random_state=42),
#         param_grid=param_grid, scoring='neg_mean_squared_error',
#         cv=tscv, verbose=1, n_jobs=-1
#     )
#     grid_search.fit(X_train_tune, y_train_tune)
#     best_params = grid_search.best_params_
#     print(f"-> GridSearchCV complete. Best parameters found: {best_params}")

#     # --- Step 3: Train an Optimized Model and Validate on Hold-Out Set ---
#     print("\n-> Training a single, optimized model on 2011-2015 data...")
#     optimized_model = xgb.XGBRegressor(objective='reg:squarederror', random_state=42, **best_params)
#     optimized_model.fit(X_train_tune, y_train_tune)

#     # Prepare the 2016 validation set for ALL cities to get performance metrics
#     validation_final = pd.get_dummies(validation_set_raw, columns=['city'], prefix='city').dropna()
#     X_validation = validation_final.reindex(columns=FEATURES, fill_value=0)
#     y_validation = validation_final[TARGET_COL]

#     print("-> Evaluating the optimized model on unseen 2016 data (all cities)...")
#     all_validation_predictions = optimized_model.predict(X_validation)
#     evaluate_model(y_validation, all_validation_predictions)

#     # --- Step 4: Plot the Validation Results for ONE City (THE FIX) ---
#     city_to_plot = 'Beirut'
#     print(f"\n-> Generating validation plot for a single city: {city_to_plot}")
    
#     # Isolate the raw data for the plot
#     train_city_df = train_set_raw[train_set_raw['city'] == city_to_plot]
#     validation_city_df = validation_set_raw[validation_set_raw['city'] == city_to_plot]
    
#     # Prepare ONLY the validation data for this one city
#     validation_city_features = pd.get_dummies(validation_city_df, columns=['city'], prefix='city')
#     X_validation_city = validation_city_features.reindex(columns=FEATURES, fill_value=0)
    
#     # Predict ONLY on this city's data. This guarantees we get exactly 12 predictions.
#     city_plot_predictions = optimized_model.predict(X_validation_city)
    
#     plot_validation_results(train_city_df, validation_city_df, city_plot_predictions)

#     # --- Step 5: Re-train Final Production Model on ALL Data ---
#     print("-> Validation and plotting complete. Re-training final model on ALL data (2011-2016)...")
#     all_data_final = pd.get_dummies(data_with_features, columns=['city'], prefix='city').dropna()
#     X_all = all_data_final.reindex(columns=FEATURES, fill_value=0)
#     y_all = all_data_final[TARGET_COL]

#     production_model = xgb.XGBRegressor(objective='reg:squarederror', random_state=42, **best_params)
#     production_model.fit(X_all, y_all)
#     print("-> Final production model trained.")

#     # --- Step 6: Save the Production-Ready Model ---
#     print(f"-> Saving final, optimized model to '{MODEL_FILE_PATH}'")
#     joblib.dump(production_model, MODEL_FILE_PATH)
#     pd.Series(FEATURES).to_json(MODEL_COLS_PATH)

#     print("\n--- Workflow Complete ---")

import pandas as pd
import xgboost as xgb
import joblib
import os
import re # Import the regular expression library for sanitizing
from sklearn.metrics import mean_squared_error, mean_absolute_error
from sklearn.model_selection import GridSearchCV, TimeSeriesSplit
import numpy as np
import matplotlib.pyplot as plt

# ==============================================================================
# 1. CONFIGURATION
# ==============================================================================
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
CSV_FILE_PATH = os.path.join(SCRIPT_DIR, 'transactions.csv')
MODEL_FILE_PATH = os.path.join(SCRIPT_DIR, 'forecast_model.joblib')
MODEL_COLS_PATH = os.path.join(SCRIPT_DIR, 'model_columns.json')
TARGET_COL = 'transaction_value'

# ==============================================================================
# 2. DATA PREPARATION FUNCTIONS
# ==============================================================================
def sanitize_city_name(name):
    """Replaces spaces, commas, and slashes with underscores for clean column names."""
    return re.sub(r'[ ,/]', '_', name)

def load_and_preprocess_data(filepath):
    print("-> Loading and preprocessing data...")
    df = pd.read_csv(filepath)
    parts = df['date'].str.split('-', expand=True)
    df['date_str'] = '01-' + parts[1] + '-' + parts[0]
    df['date'] = pd.to_datetime(df['date_str'], format='%d-%b-%y')
    
    # --- NEW: Sanitize the city names ---
    df['city'] = df['city'].apply(sanitize_city_name)
    
    df = df.set_index('date').drop(columns=['id', 'date_str'])
    df.sort_index(inplace=True)
    return df

def create_features(df):
    # ... (no change needed here)
    df['month'] = df.index.month
    df['year'] = df.index.year
    df['quarter'] = df.index.quarter
    df['lag_1'] = df.groupby('city')[TARGET_COL].shift(1)
    df['lag_3'] = df.groupby('city')[TARGET_COL].shift(3)
    df['lag_12'] = df.groupby('city')[TARGET_COL].shift(12)
    df['rolling_mean_3'] = df.groupby('city')[TARGET_COL].shift(1).rolling(window=3).mean()
    df['rolling_std_3'] = df.groupby('city')[TARGET_COL].shift(1).rolling(window=3).std()
    return df

# ... (The rest of the helper functions: tune_hyperparameters, plot_validation_results, etc. can remain the same) ...
# I will include the full script below for completeness.

def tune_hyperparameters(X_train, y_train):
    param_grid = {
        'n_estimators': [500, 1000], 'max_depth': [3, 5],
        'learning_rate': [0.01, 0.05], 'subsample': [0.7, 1.0],
    }
    print(f"\n-> Tuning with {len(param_grid['n_estimators'])*len(param_grid['max_depth'])*len(param_grid['learning_rate'])*len(param_grid['subsample'])} combinations...")
    tscv = TimeSeriesSplit(n_splits=5)
    
    grid_search = GridSearchCV(
        estimator=xgb.XGBRegressor(objective='reg:squarederror', random_state=42),
        param_grid=param_grid, scoring='neg_mean_squared_error',
        cv=tscv, verbose=1, n_jobs=-1
    )
    grid_search.fit(X_train, y_train)
    print(f"-> GridSearchCV complete. Best parameters found: {grid_search.best_params_}")
    return grid_search.best_params_

def train_and_validate_model(train_set_raw, validation_set_raw, best_params):
    print("\n-> Training optimized model for validation...")
    train_final = pd.get_dummies(train_set_raw, columns=['city'], prefix='city').dropna()
    FEATURES = [col for col in train_final.columns if col != TARGET_COL]
    X_train, y_train = train_final[FEATURES], train_final[TARGET_COL]

    optimized_model = xgb.XGBRegressor(objective='reg:squarederror', random_state=42, **best_params)
    optimized_model.fit(X_train, y_train)
    
    X_validation = pd.get_dummies(validation_set_raw, columns=['city'], prefix='city').reindex(columns=FEATURES, fill_value=0)
    y_validation = validation_set_raw[TARGET_COL]
    
    print("-> Evaluating the optimized model on unseen 2016 data...")
    validation_predictions = optimized_model.predict(X_validation)
    mae = mean_absolute_error(y_validation, validation_predictions)
    print(f"  Validation MAE: {mae:,.2f}")
    
    predictions_series = pd.Series(validation_predictions, index=y_validation.index)
    return optimized_model, FEATURES, predictions_series

def plot_validation_results(train_df_raw, validation_df_raw, predictions_series, city_to_plot):
    """Plots the results for a single chosen city."""
    print(f"\n-> Generating validation plot for: {city_to_plot}")
    
    # Sanitize the city name for filtering, just like in the loading step
    sanitized_city_to_plot = sanitize_city_name(city_to_plot)

    train_city = train_df_raw[train_df_raw['city'] == sanitized_city_to_plot]
    validation_city = validation_df_raw[validation_df_raw['city'] == sanitized_city_to_plot]
    
    if validation_city.empty:
        print(f"Warning: No validation data found for city '{city_to_plot}'. Skipping plot.")
        return

    predictions_city = predictions_series.loc[validation_city.index]
    
    train_city = train_city.sort_index()
    validation_city = validation_city.sort_index()
    predictions_city = predictions_city.sort_index()

    plt.figure(figsize=(15, 7))
    plt.plot(train_city.index, train_city[TARGET_COL], label='Training Data', color='blue')
    plt.plot(validation_city.index, validation_city[TARGET_COL], label='Actual Values (2016)', color='green', marker='o')
    plt.plot(predictions_city.index, predictions_city.values, label='Predicted Values (2016)', color='red', linestyle='--')
    
    plt.title(f'Model Validation for {city_to_plot}: Actual vs. Predicted (2016)', fontsize=16)
    plt.xlabel('Date')
    plt.ylabel('Transaction Value')
    plt.legend()
    plt.grid(True)
    print("\n-> Displaying plot. Close the plot window to save the final model.")
    plt.show()

def train_and_save_production_model(full_data_with_features, features_list, best_params):
    print("\n-> Re-training final production model on all available data...")
    all_data_final = pd.get_dummies(full_data_with_features, columns=['city'], prefix='city').dropna()
    X_all = all_data_final.reindex(columns=features_list, fill_value=0)
    y_all = all_data_final[TARGET_COL]

    production_model = xgb.XGBRegressor(objective='reg:squarederror', random_state=42, **best_params)
    production_model.fit(X_all, y_all)
    
    print(f"-> Saving final, optimized model to '{MODEL_FILE_PATH}'")
    joblib.dump(production_model, MODEL_FILE_PATH)
    pd.Series(features_list).to_json(MODEL_COLS_PATH)

# ==============================================================================
# 5. MAIN WORKFLOW
# ==============================================================================
if __name__ == "__main__":
    # 1. DATA PREP
    all_data = load_and_preprocess_data(CSV_FILE_PATH)
    data_with_features = create_features(all_data)
    train_set_raw = data_with_features[data_with_features.index < '2016-01-01'].copy()
    validation_set_raw = data_with_features[data_with_features.index >= '2016-01-01'].copy()
    print(f"-> Data split complete. Training on data up to {train_set_raw.index.max().date()}.")

    # 2. HYPERPARAMETER TUNING
    X_train_for_tuning = pd.get_dummies(train_set_raw, columns=['city'], prefix='city').dropna()
    best_hyperparams = tune_hyperparameters(
        X_train_for_tuning.drop(columns=[TARGET_COL]), 
        X_train_for_tuning[TARGET_COL]
    )

    # 3. VALIDATION
    model, features, all_predictions_series = train_and_validate_model(
        train_set_raw, validation_set_raw, best_hyperparams
    )
    
    # 4. VISUALIZATION
    # Now you can test any city name, even the complex ones
    plot_validation_results(
        train_set_raw, 
        validation_set_raw, 
        all_predictions_series, 
        city_to_plot='Baabda, Aley, Chouf' 
    )
    
    # 5. FINAL BUILD & SAVE
    train_and_save_production_model(data_with_features, features, best_hyperparams)

    print("\n--- Workflow Complete: Production model is ready for the API. ---")