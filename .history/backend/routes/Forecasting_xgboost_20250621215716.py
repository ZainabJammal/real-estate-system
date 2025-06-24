import pandas as pd
import xgboost as xgb
import joblib
import os
from sklearn.metrics import mean_squared_error, mean_absolute_error
from sklearn.model_selection import GridSearchCV, TimeSeriesSplit
import numpy as np
import matplotlib.pyplot as plt

# 1. CONFIGURATION AND PATH SETUP
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
CSV_FILE_PATH = os.path.join(SCRIPT_DIR, 'agg_trans.csv')
MODEL_FILE_PATH = os.path.join(SCRIPT_DIR, 'forecast_model.joblib')
MODEL_COLS_PATH = os.path.join(SCRIPT_DIR, 'model_columns.json')
TARGET_COL = 'transaction_value'

# 2. HELPER FUNCTIONS
def load_and_preprocess_data(filepath):
    print("-> Loading and preprocessing data...")
    df = pd.read_csv(filepath)
    parts = df['date'].str.split('-', expand=True)
    df['date_str'] = '01-' + parts[1] + '-' + parts[0]
    df['date'] = pd.to_datetime(df['date_str'], format='%d-%b-%y')
    df = df.set_index('date').drop(columns=['id', 'date_str'])
    df.sort_index(inplace=True)
    return df

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

def create_features(df):
    """
    Creates more powerful time-based features to help the model
    understand the magnitude of seasonal events.
    """
    # Create a copy to avoid SettingWithCopyWarning
    df_features = df.copy()

    # --- NEW FEATURES ---
    # 1. Cyclical Features for Month (more powerful than just the number 1-12)
    df_features['month_sin'] = np.sin(2 * np.pi * df_features.index.month / 12.0)
    df_features['month_cos'] = np.cos(2 * np.pi * df_features.index.month / 12.0)
    
    # 2. Explicit "High-Impact Month" Flags
    # This explicitly tells the model that December and the summer are special.
    df_features['is_december'] = (df_features.index.month == 12).astype(int)
    df_features['is_summer_peak'] = (df_features.index.month.isin([7, 8])).astype(int)

    # --- ORIGINAL FEATURES ---
    df_features['year'] = df_features.index.year
    df_features['quarter'] = df_features.index.quarter # Keep quarter as it's useful
    
    # The 'city' column in your agg_trans.csv is now the aggregated region name.
    # We will use this as the grouping key.
    GROUPING_KEY = 'city' # This corresponds to your aggregated regions like 'tripoli', 'beirut'
    
    df_features['lag_1'] = df_features.groupby(GROUPING_KEY)[TARGET_COL].shift(1)
    df_features['lag_3'] = df_features.groupby(GROUPING_KEY)[TARGET_COL].shift(3)
    df_features['lag_12'] = df_features.groupby(GROUPING_KEY)[TARGET_COL].shift(12)
    df_features['rolling_mean_3'] = df_features.groupby(GROUPING_KEY)[TARGET_COL].shift(1).rolling(window=3).mean()
    df_features['rolling_std_3'] = df_features.groupby(GROUPING_KEY)[TARGET_COL].shift(1).rolling(window=3).std()

    # We can now drop the original 'month' column as sin/cos are better
    if 'month' in df_features.columns:
        df_features = df_features.drop('month', axis=1)

    return df_features

# 3. EVALUATION AND PLOTTING FUNCTIONS

def evaluate_model(true_values, predicted_values):
    mae = mean_absolute_error(true_values, predicted_values)
    rmse = np.sqrt(mean_squared_error(true_values, predicted_values))
    print("\n--- Model Performance on Held-Out Validation Set (2016) ---")
    print(f"  Root Mean Squared Error (RMSE): {rmse:,.2f}")
    print(f"  Mean Absolute Error (MAE):      {mae:,.2f}")
    print("----------------------------------------------------------")

def plot_validation_results(train_df, validation_df, predictions_df):
    plt.figure(figsize=(15, 7))
    plt.plot(train_df.index, train_df[TARGET_COL], label='Training Data (2011-2015)', color='blue')
    plt.plot(validation_df.index, validation_df[TARGET_COL], label='Actual Values (2016)', color='green', marker='o', linestyle='-')
    plt.plot(validation_df.index, predictions_df, label='Predicted Values (2016)', color='red', linestyle='--')
    plt.title('Model Validation: Actual vs. Predicted for 2016', fontsize=16)
    plt.xlabel('Date')
    plt.ylabel('Transaction Value')
    plt.legend()
    plt.grid(True)
    print("\n-> Displaying validation plot. Close the plot window to continue the script.")
    plt.show()


# 4. MAIN WORKFLOW: THE FINAL, CORRECTED VERSION
if __name__ == "__main__":
    print("--- Starting Hybrid ML Workflow: Tune, Train, Validate, Re-train ---")

    # --- Step 1: Load, Prepare, and Split Data ---
    all_data = load_and_preprocess_data(CSV_FILE_PATH)
    data_with_features = create_features(all_data)

    split_date = '2016-01-01'
    train_set_raw = data_with_features[data_with_features.index < split_date]
    validation_set_raw = data_with_features[data_with_features.index >= split_date]
    
    print(f"-> Data split: Training set (for tuning) ends on {train_set_raw.index.max().date()}")
    
    train_final_for_tuning = pd.get_dummies(train_set_raw, columns=['city'], prefix='city').dropna()
    FEATURES = [col for col in train_final_for_tuning.columns if col != TARGET_COL]
    X_train_tune, y_train_tune = train_final_for_tuning[FEATURES], train_final_for_tuning[TARGET_COL]

    # --- Step 2: Hyperparameter Tuning with GridSearchCV ---
    # param_grid = {
    #     'n_estimators': [500, 1000],
    #     'max_depth': [3, 5],
    #     'learning_rate': [0.01, 0.05],
    #     'subsample': [0.7, 1.0],
    # }
    param_grid = {
        'n_estimators': [1000, 1500],       # Allow more trees
        'max_depth': [3, 5, 7, 10],       # *** THIS IS THE KEY CHANGE ***
        'learning_rate': [0.01, 0.05],
        'subsample': [0.7, 0.8, 1.0],       # Add more variety
        'colsample_bytree': [0.7, 0.8, 1.0] # Add feature sampling
    }
    print(f"\n-> Defined hyperparameter grid with {len(param_grid['n_estimators'])*len(param_grid['max_depth'])*len(param_grid['learning_rate'])*len(param_grid['subsample'])} combinations.")
    
    tscv = TimeSeriesSplit(n_splits=5)
    
    print("-> Starting GridSearchCV on the training set (2011-2015)...")
    grid_search = GridSearchCV(
        estimator=xgb.XGBRegressor(objective='reg:squarederror', random_state=42),
        param_grid=param_grid, scoring='neg_mean_squared_error',
        cv=tscv, verbose=1, n_jobs=-1
    )
    grid_search.fit(X_train_tune, y_train_tune)
    best_params = grid_search.best_params_
    print(f"-> GridSearchCV complete. Best parameters found: {best_params}")

    # --- Step 3: Train an Optimized Model and Validate on Hold-Out Set ---
    print("\n-> Training a single, optimized model on 2011-2015 data...")
    optimized_model = xgb.XGBRegressor(objective='reg:squarederror', random_state=42, **best_params)
    optimized_model.fit(X_train_tune, y_train_tune)

    # Prepare the 2016 validation set for ALL cities to get performance metrics
    validation_final = pd.get_dummies(validation_set_raw, columns=['city'], prefix='city').dropna()
    X_validation = validation_final.reindex(columns=FEATURES, fill_value=0)
    y_validation = validation_final[TARGET_COL]

    print("-> Evaluating the optimized model on unseen 2016 data (all cities)...")
    all_validation_predictions = optimized_model.predict(X_validation)
    evaluate_model(y_validation, all_validation_predictions)

    # --- Step 4: Plot the Validation Results for ONE City (THE FIX) ---
    city_to_plot = 'Beirut'
    print(f"\n-> Generating validation plot for a single city: {city_to_plot}")
    
    # Isolate the raw data for the plot
    train_city_df = train_set_raw[train_set_raw['city'] == city_to_plot]
    validation_city_df = validation_set_raw[validation_set_raw['city'] == city_to_plot]
    
    # Prepare ONLY the validation data for this one city
    validation_city_features = pd.get_dummies(validation_city_df, columns=['city'], prefix='city')
    X_validation_city = validation_city_features.reindex(columns=FEATURES, fill_value=0)
    
    # Predict ONLY on this city's data. This guarantees we get exactly 12 predictions.
    city_plot_predictions = optimized_model.predict(X_validation_city)
    
    plot_validation_results(train_city_df, validation_city_df, city_plot_predictions)

    # --- Step 5: Re-train Final Production Model on ALL Data ---
    print("-> Validation and plotting complete. Re-training final model on ALL data (2011-2016)...")
    all_data_final = pd.get_dummies(data_with_features, columns=['city'], prefix='city').dropna()
    X_all = all_data_final.reindex(columns=FEATURES, fill_value=0)
    y_all = all_data_final[TARGET_COL]

    production_model = xgb.XGBRegressor(objective='reg:squarederror', random_state=42, **best_params)
    production_model.fit(X_all, y_all)
    print("-> Final production model trained.")

    # --- Step 6: Save the Production-Ready Model ---
    print(f"-> Saving final, optimized model to '{MODEL_FILE_PATH}'")
    joblib.dump(production_model, MODEL_FILE_PATH)
    pd.Series(FEATURES).to_json(MODEL_COLS_PATH)

    print("\n--- Workflow Complete ---")


