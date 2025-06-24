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
# 2. DATA PREPARATION FUNCTIONS
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

# ==============================================================================
# 3. ORGANIZED MODELING & VISUALIZATION FUNCTIONS
# ==============================================================================
def tune_hyperparameters(X_train, y_train):
    """Performs GridSearchCV to find the best model parameters."""
    param_grid = {
        'n_estimators': [500, 1000], 'max_depth': [3, 5],
        'learning_rate': [0.01, 0.05], 'subsample': [0.7, 1.0],
    }
    print(f"\n-> Defined hyperparameter grid.")
    tscv = TimeSeriesSplit(n_splits=5)
    
    print("-> Starting GridSearchCV on the training set (2011-2015)...")
    grid_search = GridSearchCV(
        estimator=xgb.XGBRegressor(objective='reg:squarederror', random_state=42),
        param_grid=param_grid, scoring='neg_mean_squared_error',
        cv=tscv, verbose=1, n_jobs=-1
    )
    grid_search.fit(X_train, y_train)
    print(f"-> GridSearchCV complete. Best parameters found: {grid_search.best_params_}")
    return grid_search.best_params_

def train_and_validate_model(train_set_raw, validation_set_raw, best_params):
    """Trains an optimized model, validates it, and returns results for plotting."""
    print("\n-> Training a single, optimized model on 2011-2015 data...")
    train_final = pd.get_dummies(train_set_raw, columns=['city'], prefix='city').dropna()
    FEATURES = [col for col in train_final.columns if col != TARGET_COL]
    X_train, y_train = train_final[FEATURES], train_final[TARGET_COL]

    optimized_model = xgb.XGBRegressor(objective='reg:squarederror', random_state=42, **best_params)
    optimized_model.fit(X_train, y_train)

    # Prepare validation set
    validation_final = pd.get_dummies(validation_set_raw, columns=['city'], prefix='city').dropna()
    X_validation = validation_final.reindex(columns=FEATURES, fill_value=0)
    y_validation = validation_final[TARGET_COL]

    print("-> Evaluating the optimized model on unseen 2016 data...")
    validation_predictions = optimized_model.predict(X_validation)
    
    # Calculate and print metrics
    mae = mean_absolute_error(y_validation, validation_predictions)
    print(f"  Validation MAE: {mae:,.2f}")
    
    # Create a pandas Series of predictions with the correct dates as the index
    predictions_series = pd.Series(validation_predictions, index=y_validation.index)
    
    return optimized_model, FEATURES, predictions_series

def plot_validation_results(train_df_raw, validation_df_raw, predictions_series, city_to_plot):
    """Plots the results for a single chosen city."""
    print(f"\n-> Generating validation plot for: {city_to_plot}")
    
    # Isolate the raw data for the plot
    train_city_df = train_df_raw[train_df_raw['city'] == city_to_plot]
    validation_city_df = validation_df_raw[validation_df_raw['city'] == city_to_plot]
    
    # Filter the predictions Series by the index of the city's validation data
    city_plot_predictions = predictions_series.loc[validation_city_df.index]

    plt.figure(figsize=(15, 7))
    plt.plot(train_city_df.index, train_city_df[TARGET_COL], label='Training Data', color='blue')
    plt.plot(validation_city_df.index, validation_city_df[TARGET_COL], label='Actual Values (2016)', color='green', marker='o')
    plt.plot(city_plot_predictions.index, city_plot_predictions.values, label='Predicted Values (2016)', color='red', linestyle='--')
    
    plt.title(f'Model Validation for {city_to_plot}: Actual vs. Predicted (2016)', fontsize=16)
    plt.xlabel('Date')
    plt.ylabel('Transaction Value')
    plt.legend()
    plt.grid(True)
    print("\n-> Displaying plot. Close the plot window to save the final model.")
    plt.show()

def train_and_save_production_model(full_data_with_features, features_list, best_params):
    """Trains the final model on all data and saves it."""
    print("\n-> Re-training final production model on ALL available data...")
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
    # 1. Load, prepare, and split data
    all_data = load_and_preprocess_data(CSV_FILE_PATH)
    data_with_features = create_features(all_data)
    train_set_raw = data_with_features[data_with_features.index < '2016-01-01']
    validation_set_raw = data_with_features[data_with_features.index >= '2016-01-01']

    # 2. Find best hyperparameters using only training data
    X_train_for_tuning = pd.get_dummies(train_set_raw, columns=['city'], prefix='city').dropna()
    best_hyperparams = tune_hyperparameters(
        X_train_for_tuning.drop(columns=[TARGET_COL]), 
        X_train_for_tuning[TARGET_COL]
    )

    # 3. Train an optimized model, validate it, and get predictions
    model, features, all_predictions = train_and_validate_model(
        train_set_raw, validation_set_raw, best_hyperparams
    )
    
    # 4. Plot the results for a specific city
    plot_validation_results(
        train_set_raw, validation_set_raw, all_predictions, city_to_plot='Beirut'
    )
    
    # 5. Train and save the final production model on all data
    train_and_save_production_model(data_with_features, features, best_hyperparams)

    print("\n--- Workflow Complete: Production model is ready. ---")