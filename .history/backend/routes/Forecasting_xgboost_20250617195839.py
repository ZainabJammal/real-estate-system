import pandas as pd
import xgboost as xgb
import joblib
import os
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
# 2. HELPER FUNCTIONS
# ==============================================================================
def load_and_preprocess_data(filepath):
    print("-> Loading and preprocessing data...")
    df = pd.read_csv(filepath)
    parts = df['date'].str.split('-', expand=True)
    df['date_str'] = '01-' + parts[1] + '-' + parts[0]
    df['date'] = pd.to_datetime(df['date_str'], format='%d-%b-%y')
    df = df.set_index('date').drop(columns=['id', 'date_str'])
    df.sort_index(inplace=True)
    return df

def create_features(df):
    df['month'] = df.index.month
    df['year'] = df.index.year
    df['quarter'] = df.index.quarter
    df['lag_1'] = df.groupby('city')[TARGET_COL].shift(1)
    df['lag_3'] = df.groupby('city')[TARGET_COL].shift(3)
    df['lag_12'] = df.groupby('city')[TARGET_COL].shift(12)
    df['rolling_mean_3'] = df.groupby('city')[TARGET_COL].shift(1).rolling(window=3).mean()
    df['rolling_std_3'] = df.groupby('city')[TARGET_COL].shift(1).rolling(window=3).std()
    return df

def tune_hyperparameters(X_train, y_train):
    """Performs GridSearchCV to find the best model parameters."""
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

def plot_validation_results(train_df, validation_df, predictions, city_to_plot):
    """Plots the training data, actual validation data, and predicted validation data for a single city."""
    print(f"\n-> Generating validation plot for: {city_to_plot}")
    
    # Isolate the data for the specific city
    train_city = train_df[train_df['city'] == city_to_plot]
    validation_city = validation_df[validation_df['city'] == city_to_plot]
    
    # Match predictions to the validation set for the city
    predictions_city = predictions.loc[validation_city.index]
    
    # --- THE CRITICAL FIX: Ensure all data is sorted by date before plotting ---
    train_city = train_city.sort_index()
    validation_city = validation_city.sort_index()
    predictions_city = predictions_city.sort_index()

    plt.figure(figsize=(15, 7))
    plt.plot(train_city.index, train_city[TARGET_COL], label='Training Data (2011-2015)', color='blue')
    plt.plot(validation_city.index, validation_city[TARGET_COL], label='Actual Values (2016)', color='green', marker='o')
    plt.plot(predictions_city.index, predictions_city.values, label='Predicted Values (2016)', color='red', linestyle='--')
    
    plt.title(f'Model Validation for {city_to_plot}: Actual vs. Predicted (2016)', fontsize=16)
    plt.xlabel('Date')
    plt.ylabel('Transaction Value')
    plt.legend()
    plt.grid(True)
    print("\n-> Displaying plot. Close the plot window to save the final model.")
    plt.show()

# ==============================================================================
# 5. MAIN WORKFLOW
# ==============================================================================
if __name__ == "__main__":
    print("--- Starting Full ML Pipeline: Tune, Validate, and Build ---")

    # 1. DATA PREP: Load, create features, and split the data by time
    all_data = load_and_preprocess_data(CSV_FILE_PATH)
    data_with_features = create_features(all_data)
    train_set_raw = data_with_features[data_with_features.index < '2016-01-01'].copy()
    validation_set_raw = data_with_features[data_with_features.index >= '2016-01-01'].copy()
    print(f"-> Data split complete. Training on data up to {train_set_raw.index.max().date()}.")

    # 2. HYPERPARAMETER TUNING: Use GridSearchCV on the training set
    X_train_for_tuning = pd.get_dummies(train_set_raw, columns=['city'], prefix='city').dropna()
    FEATURES = [col for col in X_train_for_tuning.columns if col != TARGET_COL]
    y_train_for_tuning = X_train_for_tuning[TARGET_COL]
    X_train_for_tuning = X_train_for_tuning[FEATURES]
    
    best_params = tune_hyperparameters(X_train_for_tuning, y_train_for_tuning)

    # 3. VALIDATION: Train a model with the best params and test on the 2016 hold-out set
    print("\n-> Training optimized model for validation...")
    optimized_model = xgb.XGBRegressor(objective='reg:squarederror', random_state=42, **best_params)
    optimized_model.fit(X_train_for_tuning, y_train_for_tuning)
    
    # Prepare validation data
    X_validation = pd.get_dummies(validation_set_raw, columns=['city'], prefix='city').reindex(columns=FEATURES, fill_value=0)
    y_validation = validation_set_raw[TARGET_COL]
    
    # Make predictions and evaluate
    validation_predictions = optimized_model.predict(X_validation)
    mae = mean_absolute_error(y_validation, validation_predictions)
    print(f"-> Evaluation on 2016 data complete. Mean Absolute Error: {mae:,.2f}")

    # 4. VISUALIZATION: Plot the results for one city
    all_predictions_series = pd.Series(validation_predictions, index=y_validation.index)
    plot_validation_results(
        train_set_raw, 
        validation_set_raw, 
        all_predictions_series, 
        city_to_plot='Beirut'
    )

    # 5. FINAL BUILD: Re-train the model on ALL data using the best parameters
    print("\n-> Re-training final production model on all available data (2011-2016)...")
    all_data_final = pd.get_dummies(data_with_features, columns=['city'], prefix='city').dropna()
    X_all = all_data_final.reindex(columns=FEATURES, fill_value=0)
    y_all = all_data_final[TARGET_COL]

    production_model = xgb.XGBRegressor(objective='reg:squarederror', random_state=42, **best_params)
    production_model.fit(X_all, y_all)

    # 6. SAVE: Save the final model and its required columns
    print(f"-> Saving final, optimized model to '{MODEL_FILE_PATH}'")
    joblib.dump(production_model, MODEL_FILE_PATH)
    pd.Series(FEATURES).to_json(MODEL_COLS_PATH)

    print("\n--- Workflow Complete: Production model is ready for the API. ---")