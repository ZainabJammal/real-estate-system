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
import json # Import the json library
from sklearn.metrics import mean_squared_error, mean_absolute_error
from sklearn.model_selection import GridSearchCV, TimeSeriesSplit
import numpy as np
import matplotlib.pyplot as plt

# 1. CONFIGURATION AND PATH SETUP
try:
    SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
except NameError:
    SCRIPT_DIR = os.getcwd() 

CSV_FILE_PATH = os.path.join(SCRIPT_DIR, 'transactions.csv')
# MAPPING_FILE_PATH = os.path.join(SCRIPT_DIR, 'region_mapping.json') # Path to our new mapping file
MODEL_FILE_PATH = os.path.join(SCRIPT_DIR, 'forecast_model.joblib')
MODEL_COLS_PATH = os.path.join(SCRIPT_DIR, 'model_columns.json')

# *** THE KEY CHANGE: We are now forecasting by 'region', not 'city' ***
REGION_COL = 'region'
TARGET_COL = 'transaction_value'


# 2. HELPER FUNCTIONS

# --- THIS FUNCTION IS COMPLETELY REWRITTEN ---
# --- THIS IS THE ONLY FUNCTION YOU NEED TO REPLACE ---
# def load_and_preprocess_data(filepath, mapping_filepath):
#     """
#     Loads data, normalizes cities into regions using a mapping file,
#     aggregates values by region, and performs date preprocessing.
#     """
#     print("-> Loading and preprocessing data with region mapping...")
#     df = pd.read_csv(filepath)

#     # --- Step 1: Normalize the raw city strings ---
#     df['city'] = df['city'].str.split(',')
#     df = df.explode('city')
#     df['city'] = df['city'].str.strip()

#     # --- Step 2: Load the region mapping ---
#     with open(mapping_filepath, 'r') as f:
#         region_map = json.load(f)

#     # --- Step 3: Create a "reverse map" (city -> region) ---
#     print("-> Creating city-to-region mapping...")
#     reverse_map = {}
#     for region_name, cities in region_map.items():
#         if region_name == "_DEFAULT_":
#             # For default cities, the region name is the city name itself
#             for city in cities:
#                 reverse_map[city] = city
#         else:
#             # For grouped cities, they all map to the region name
#             for city in cities:
#                 # --- THIS IS THE CORRECTED LOGIC ---
#                 # The key is the individual city, the value is the region.
#                 reverse_map[city] = region_name
    
#     # --- Step 4: Apply the mapping to create the new 'region' column ---
#     df[REGION_COL] = df['city'].map(reverse_map)
    
#     # Handle any cities that might have been in the data but not the mapping file
#     unmapped_cities = df[df[REGION_COL].isnull()]['city'].unique()
#     if len(unmapped_cities) > 0:
#         print(f"Warning: Found cities not in mapping file. Treating them as their own region: {unmapped_cities}")
#         df[REGION_COL].fillna(df['city'], inplace=True) 

#     # --- Step 5: Process dates and aggregate by the NEW region column ---
#     parts = df['date'].str.split('-', expand=True)
#     df['date_str'] = '01-' + parts[1] + '-' + parts[0]
#     df['date'] = pd.to_datetime(df['date_str'], format='%d-%b-%y')

#     print(f"-> Aggregating transaction values by date and '{REGION_COL}'...")
#     regional_df = df.groupby(['date', REGION_COL])[[TARGET_COL]].sum().reset_index()
    
#     regional_df = regional_df.set_index('date')
#     regional_df.sort_index(inplace=True)
    
#     print("-> Data preprocessing and regional aggregation complete.")
#     return regional_df

def load_and_preprocess_data(filepath):
    """
    Loads data, normalizes cities, and HARD-CODES them into new regions.
    This logic will be mirrored in the API.
    """
    print("-> Loading and preprocessing data with hard-coded regions...")
    df = pd.read_csv(filepath)

    # Step 1: Explode the cities as before
    df['city'] = df['city'].str.split(',').apply(lambda x: [c.strip() for c in x])
    df = df.explode('city')

    # Step 2: Define the hard-coded mapping logic in a function
    def map_city_to_region(city_name):
        # Define your groups here. This is the single source of truth.
        if city_name in ["Tripoli", "Akkar"]:
            return "North"
        if city_name in ["Baabda", "Aley", "Chouf"]:
            return "Mount Lebanon South"
        if city_name in ["Kesrouan", "Jbeil", "Metn"]:
            return "Mount Lebanon North"
        # Any other city becomes its own region (e.g., "Beirut" -> "Beirut")
        return city_name

    # Step 3: Apply this mapping to create the new 'region' column
    df[REGION_COL] = df['city'].apply(map_city_to_region)

    # Step 4: Process dates and aggregate by the NEW region column
    parts = df['date'].str.split('-', expand=True)
    df['date'] = pd.to_datetime('01-' + parts[1] + '-' + parts[0], format='%d-%b-%y')
    
    print(f"-> Aggregating transaction values by date and '{REGION_COL}'...")
    regional_df = df.groupby(['date', REGION_COL])[[TARGET_COL]].sum().reset_index()
    
    return regional_df.set_index('date')


def create_features(df):
    """Creates time-series features based on the dataframe index and target."""
    df_features = df.copy()
    df_features['month'] = df_features.index.month
    df_features['year'] = df_features.index.year
    df_features['quarter'] = df_features.index.quarter
    df_features['is_december'] = (df_features.index.month == 12).astype(int)
    df_features['month_sin'] = np.sin(2 * np.pi * df_features.index.month/12.0)
    df_features['month_cos'] = np.cos(2 * np.pi * df_features.index.month/12.0)
   
    # *** CRITICAL: Group by the new REGION_COL ***
    df_features['lag_1'] = df_features.groupby(REGION_COL)[TARGET_COL].shift(1)
    df_features['lag_3'] = df_features.groupby(REGION_COL)[TARGET_COL].shift(3)
    df_features['lag_12'] = df_features.groupby(REGION_COL)[TARGET_COL].shift(12)
    df_features['rolling_mean_3'] = df_features.groupby(REGION_COL)[TARGET_COL].shift(1).rolling(window=3).mean()
    df_features['rolling_std_3'] = df_features.groupby(REGION_COL)[TARGET_COL].shift(1).rolling(window=3).std()
   
    return df_features

# (Evaluation and plotting functions remain the same, but we will call them with region data)
def evaluate_model(true_values, predicted_values):
    """Calculates and prints key regression metrics, including MAPE."""
    mae = mean_absolute_error(true_values, predicted_values)
    rmse = np.sqrt(mean_squared_error(true_values, predicted_values))
    
    # --- ADD MAPE CALCULATION ---
    # Add a small epsilon to avoid division by zero if true_values has 0s
    epsilon = 1e-10 
    mape = np.mean(np.abs((true_values - predicted_values) / (true_values + epsilon))) * 100
    
    print("\n--- Model Performance on Held-Out Validation Set (2016) ---")
    print(f"  Root Mean Squared Error (RMSE): {rmse:,.2f}")
    print(f"  Mean Absolute Error (MAE):      {mae:,.2f}")
    print(f"  Mean Absolute Percentage Error (MAPE): {mape:.2f}%") # Print the new metric
    print("----------------------------------------------------------")

def plot_validation_results(train_df, validation_df, predictions_df, region_name):
    plt.figure(figsize=(15, 7))
    plt.plot(train_df.index, train_df[TARGET_COL], label='Training Data (2011-2015)', color='blue')
    plt.plot(validation_df.index, validation_df[TARGET_COL], label='Actual Values (2016)', color='green', marker='o', linestyle='-')
    plt.plot(validation_df.index, predictions_df, label='Predicted Values (2016)', color='red', linestyle='--')
    plt.title(f'Model Validation for {region_name}: Actual vs. Predicted for 2016', fontsize=16)
    plt.xlabel('Date')
    plt.ylabel('Transaction Value')
    plt.legend()
    plt.grid(True)
    print(f"\n-> Displaying validation plot for {region_name}. Close the plot window to continue the script.")
    plt.show()

# 4. MAIN WORKFLOW
if __name__ == "__main__":
    print("--- Starting Regional ML Workflow: Tune, Train, Validate, Re-train ---")

    # --- Step 1: Load, Prepare (with region mapping), and Split Data ---
    # all_data = load_and_preprocess_data(CSV_FILE_PATH, MAPPING_FILE_PATH)
    
    all_data = load_and_preprocess_data(CSV_FILE_PATH) 
    data_with_features = create_features(all_data)
    

    split_date = '2016-01-01'
    train_set_raw = data_with_features[data_with_features.index < split_date]
    validation_set_raw = data_with_features[data_with_features.index >= split_date]
    
    # *** CRITICAL: One-hot encode the new REGION_COL ***
    train_final_for_tuning = pd.get_dummies(train_set_raw, columns=[REGION_COL], prefix=REGION_COL).dropna()
    FEATURES = [col for col in train_final_for_tuning.columns if col != TARGET_COL]
    X_train_tune, y_train_tune = train_final_for_tuning[FEATURES], train_final_for_tuning[TARGET_COL]

    # --- Step 2: Hyperparameter Tuning (no changes here) ---
    param_grid = {
        'n_estimators': [500, 1000], 'max_depth':  [3, 5],
        'learning_rate': [0.01, 0.05], 'subsample': [0.7, 1.0],
    }
    tscv = TimeSeriesSplit(n_splits=5)
    grid_search = GridSearchCV(
        estimator=xgb.XGBRegressor(objective='reg:squarederror', random_state=42),
        param_grid=param_grid, scoring='neg_mean_squared_error',
        cv=tscv, verbose=1, n_jobs=-1
    )
    grid_search.fit(X_train_tune, y_train_tune)
    best_params = grid_search.best_params_
    print(f"-> GridSearchCV complete. Best parameters found: {best_params}")

    # --- Step 3: Train an Optimized Model and Validate ---
    optimized_model = xgb.XGBRegressor(objective='reg:squarederror', random_state=42, **best_params)
    optimized_model.fit(X_train_tune, y_train_tune)

    validation_final = pd.get_dummies(validation_set_raw, columns=[REGION_COL], prefix=REGION_COL).dropna()
    X_validation = validation_final.reindex(columns=FEATURES, fill_value=0)
    y_validation = validation_final[TARGET_COL]
    all_validation_predictions = optimized_model.predict(X_validation)
    evaluate_model(y_validation, all_validation_predictions)

    # --- Step 4: Plot the Validation Results for a specific REGION ---
    region_to_plot = 'Mount Lebanon South' # Use the name from your JSON file
    print(f"\n-> Generating validation plot for a single region: {region_to_plot}")
    
    train_region_df = train_set_raw[train_set_raw[REGION_COL] == region_to_plot]
    validation_region_df = validation_set_raw[validation_set_raw[REGION_COL] == region_to_plot]
    
    if validation_region_df.empty:
        print(f"Warning: No validation data found for '{region_to_plot}'. Skipping plot.")
    else:
        validation_region_features = pd.get_dummies(validation_region_df, columns=[REGION_COL], prefix=REGION_COL)
        X_validation_region = validation_region_features.reindex(columns=FEATURES, fill_value=0)
        region_plot_predictions = optimized_model.predict(X_validation_region)
        plot_validation_results(train_region_df, validation_region_df, region_plot_predictions, region_to_plot)

    # --- Step 5 & 6: Re-train and Save Final PRODUCTION Model ---
    print("\n-> Re-training final model on ALL regional data (2011-2016)...")
    all_data_final = pd.get_dummies(data_with_features, columns=[REGION_COL], prefix=REGION_COL).dropna()
    X_all = all_data_final.reindex(columns=FEATURES, fill_value=0)
    y_all = all_data_final[TARGET_COL]
    production_model = xgb.XGBRegressor(objective='reg:squarederror', random_state=42, **best_params)
    production_model.fit(X_all, y_all)
    print("-> Final production model trained.")

    joblib.dump(production_model, MODEL_FILE_PATH)
    pd.Series(X_all.columns.tolist()).to_json(MODEL_COLS_PATH)
    print("\n--- Workflow Complete ---")