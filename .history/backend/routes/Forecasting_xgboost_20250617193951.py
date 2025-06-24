import pandas as pd
import xgboost as xgb
import joblib
import os
from sklearn.metrics import mean_squared_error, mean_absolute_error
from sklearn.model_selection import GridSearchCV, TimeSeriesSplit
import numpy as np
import matplotlib.pyplot as plt

# ==============================================================================
# 1. CONFIGURATION (Stays at the top)
# ==============================================================================
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
CSV_FILE_PATH = os.path.join(SCRIPT_DIR, 'transactions.csv')
MODEL_FILE_PATH = os.path.join(SCRIPT_DIR, 'forecast_model.joblib')
MODEL_COLS_PATH = os.path.join(SCRIPT_DIR, 'model_columns.json')
TARGET_COL = 'transaction_value'

# ==============================================================================
# 2. DATA PREPARATION FUNCTIONS (These are already well-organized)
# ==============================================================================
def load_and_preprocess_data(filepath):
    # ... (no change)
    print("-> Loading and preprocessing data...")
    df = pd.read_csv(filepath)
    parts = df['date'].str.split('-', expand=True)
    df['date_str'] = '01-' + parts[1] + '-' + parts[0]
    df['date'] = pd.to_datetime(df['date_str'], format='%d-%b-%y')
    df = df.set_index('date').drop(columns=['id', 'date_str'])
    df.sort_index(inplace=True)
    return df

def create_features(df):
    # ... (no change)
    df['month'] = df.index.month
    df['year'] = df.index.year
    df['quarter'] = df.index.quarter
    # ... etc.
    return df

# ==============================================================================
# 3. NEW, ORGANIZED MODELING FUNCTIONS
# ==============================================================================

def tune_hyperparameters(X_train, y_train):
    """
    Performs GridSearchCV to find the best model parameters.
    Returns the dictionary of best parameters.
    """
    param_grid = {
        'n_estimators': [500, 1000], 'max_depth': [3, 5],
        'learning_rate': [0.01, 0.05], 'subsample': [0.7, 1.0],
    }
    print(f"\n-> Defined hyperparameter grid with {len(param_grid['n_estimators'])*len(param_grid['max_depth'])*len(param_grid['learning_rate'])*len(param_grid['subsample'])} combinations.")
    
    tscv = TimeSeriesSplit(n_splits=5)
    
    print("-> Starting GridSearchCV...")
    grid_search = GridSearchCV(
        estimator=xgb.XGBRegressor(objective='reg:squarederror', random_state=42),
        param_grid=param_grid, scoring='neg_mean_squared_error',
        cv=tscv, verbose=1, n_jobs=-1
    )
    grid_search.fit(X_train, y_train)
    print(f"-> GridSearchCV complete. Best parameters found: {grid_search.best_params_}")
    return grid_search.best_params_

def train_and_validate_model(train_set_raw, validation_set_raw, best_params):
    """
    Trains an optimized model, validates it, and plots the results.
    Returns the list of features used for the final model.
    """
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
    rmse = np.sqrt(mean_squared_error(y_validation, validation_predictions))
    print(f"  Validation MAE: {mae:,.2f} | Validation RMSE: {rmse:,.2f}")

    # Plot results
    city_to_plot = 'Beirut'
    train_city_df = train_set_raw[train_set_raw['city'] == city_to_plot]
    validation_city_df = validation_set_raw[validation_set_raw['city'] == city_to_plot]
    X_validation_city = X_validation[X_validation[f'city_{city_to_plot}'] == 1]
    city_plot_predictions = optimized_model.predict(X_validation_city)
    
    plot_validation_results(train_city_df, validation_city_df, city_plot_predictions)
    
    return FEATURES # Return the feature list for the final model

def train_and_save_production_model(full_data_with_features, features_list, best_params):
    """
    Trains the final model on all data and saves it.
    """
    print("\n-> Re-training final production model on ALL available data...")
    all_data_final = pd.get_dummies(full_data_with_features, columns=['city'], prefix='city').dropna()
    X_all = all_data_final.reindex(columns=features_list, fill_value=0)
    y_all = all_data_final[TARGET_COL]

    production_model = xgb.XGBRegressor(objective='reg:squarederror', random_state=42, **best_params)
    production_model.fit(X_all, y_all)
    
    print(f"-> Saving final, optimized model to '{MODEL_FILE_PATH}'")
    joblib.dump(production_model, MODEL_FILE_PATH)
    pd.Series(features_list).to_json(MODEL_COLS_PATH)

def plot_validation_results(train_df, validation_df, predictions_df):
    # ... (no change)
    plt.figure(figsize=(15, 7))
    # ... etc.
    plt.show()

# ==============================================================================
# 5. THE MAIN "CONDUCTOR" BLOCK (Clean and Simple)
# ==============================================================================
if __name__ == "__main__":
    # 1. Load and prepare data
    all_data = load_and_preprocess_data(CSV_FILE_PATH)
    data_with_features = create_features(all_data)

    # 2. Split data for training/tuning and final validation
    train_set_raw = data_with_features[data_with_features.index < '2016-01-01']
    validation_set_raw = data_with_features[data_with_features.index >= '2016-01-01']

    # 3. Find the best hyperparameters using only the training data
    X_train_for_tuning = pd.get_dummies(train_set_raw, columns=['city'], prefix='city').dropna()
    best_hyperparams = tune_hyperparameters(
        X_train_for_tuning.drop(columns=[TARGET_COL]), 
        X_train_for_tuning[TARGET_COL]
    )

    # 4. Train an optimized model, validate it against the hold-out set, and plot it
    final_features = train_and_validate_model(train_set_raw, validation_set_raw, best_hyperparams)
    
    # 5. Train the final production model on all data using the best params
    train_and_save_production_model(data_with_features, final_features, best_hyperparams)

    print("\n--- Workflow Complete: Production model is ready. ---")