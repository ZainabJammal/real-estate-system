import pandas as pd
import xgboost as xgb
import joblib
import os
from sklearn.metrics import mean_squared_error, mean_absolute_error
from sklearn.model_selection import GridSearchCV, TimeSeriesSplit
import numpy as np
import matplotlib.pyplot as plt

# ==============================================================================
# 1. CONFIGURATION AND PATH SETUP
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

# ==============================================================================
# 3. EVALUATION AND PLOTTING FUNCTIONS
# ==============================================================================
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

# ==============================================================================
# 4. MAIN WORKFLOW
# ==============================================================================
if __name__ == "__main__":
    print("--- Starting Hybrid ML Workflow: Tune, Train, Validate, Re-train ---")

    # --- Step 1: Load, Prepare, and Split Data ---
    all_data = load_and_preprocess_data(CSV_FILE_PATH)
    data_with_features = create_features(all_data)

    split_date = '2016-01-01'
    train_set_raw = data_with_features[data_with_features.index < split_date]
    validation_set_raw = data_with_features[data_with_features.index >= split_date]
    
    print(f"-> Data split: Training set (for tuning) ends on {train_set_raw.index.max().date()}")
    
    train_final = pd.get_dummies(train_set_raw, columns=['city'], prefix='city').dropna()
    FEATURES = [col for col in train_final.columns if col != TARGET_COL]
    X_train_tune, y_train_tune = train_final[FEATURES], train_final[TARGET_COL]

    # --- Step 2: Hyperparameter Tuning with GridSearchCV on the Training Set ---
    param_grid = {
        'n_estimators': [500, 1000],
        'max_depth': [3, 5],
        'learning_rate': [0.01, 0.05],
        'subsample': [0.7, 1.0],
    }
    print(f"\n-> Defined hyperparameter grid with {len(param_grid['n_estimators'])*len(param_grid['max_depth'])*len(param_grid['learning_rate'])*len(param_grid['subsample'])} combinations.")
    
    tscv = TimeSeriesSplit(n_splits=5)
    
    print("-> Starting GridSearchCV on the training set (2011-2015)...")
    grid_search = GridSearchCV(
        estimator=xgb.XGBRegressor(objective='reg:squarederror', random_state=42),
        param_grid=param_grid,
        scoring='neg_mean_squared_error',
        cv=tscv,
        verbose=1,
        n_jobs=-1
    )
    grid_search.fit(X_train_tune, y_train_tune)
    best_params = grid_search.best_params_
    print(f"-> GridSearchCV complete. Best parameters found: {best_params}")

    # --- Step 3: Train a Model with Best Params and Validate on Hold-Out Set ---
    print("\n-> Training a single, optimized model on 2011-2015 data...")
    optimized_model = xgb.XGBRegressor(objective='reg:squarederror', random_state=42, **best_params)
    optimized_model.fit(X_train_tune, y_train_tune)

    # Prepare the 2016 validation set
    validation_final = pd.get_dummies(validation_set_raw, columns=['city'], prefix='city').dropna()
    X_validation = validation_final.reindex(columns=FEATURES, fill_value=0) # Align columns
    y_validation = validation_final[TARGET_COL]

    print("-> Evaluating the optimized model on unseen 2016 data...")
    validation_predictions = optimized_model.predict(X_validation)
    evaluate_model(y_validation, validation_predictions)

    # Plot the results for a single city to visualize performance
    city_to_plot = 'Beirut'
    train_city_df = train_set_raw[train_set_raw['city'] == city_to_plot]
    validation_city_df = validation_set_raw[validation_set_raw['city'] == city_to_plot]
    
    # Get the predictions for just that city from our overall validation predictions
    city_indices = y_validation.index[validation_final['city_Beirut'] == 1]
    city_predictions_series = pd.Series(validation_predictions, index=y_validation.index)
    city_plot_predictions = city_predictions_series.loc[city_indices]

    plot_validation_results(train_city_df, validation_city_df, city_plot_predictions)

    # --- Step 4: Re-train Final Production Model on ALL Data ---
    print("-> Validation and plotting complete. Re-training final model on ALL data (2011-2016)...")
    all_data_final = pd.get_dummies(data_with_features, columns=['city'], prefix='city').dropna()
    X_all, y_all = all_data_final[FEATURES], all_data_final[TARGET_COL]

    production_model = xgb.XGBRegressor(objective='reg:squarederror', random_state=42, **best_params)
    production_model.fit(X_all, y_all)
    print("-> Final production model trained.")

    # --- Step 5: Save the Production-Ready Model ---
    print(f"-> Saving final, optimized model to '{MODEL_FILE_PATH}'")
    joblib.dump(production_model, MODEL_FILE_PATH)
    pd.Series(X_all.columns).to_json(MODEL_COLS_PATH)

    print("\n--- Workflow Complete ---")