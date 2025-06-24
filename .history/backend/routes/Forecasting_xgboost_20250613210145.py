import pandas as pd
import xgboost as xgb
import joblib
import os
from sklearn.metrics import mean_squared_error, mean_absolute_error
import numpy as np
import matplotlib.pyplot as plt # Import the plotting library

# ==============================================================================
# 1. CONFIGURATION AND PATH SETUP
# ==============================================================================
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
CSV_FILE_PATH = os.path.join(SCRIPT_DIR, 'transactions.csv')
MODEL_FILE_PATH = os.path.join(SCRIPT_DIR, 'forecast_model.joblib')
MODEL_COLS_PATH = os.path.join(SCRIPT_DIR, 'model_columns.json')
TARGET_COL = 'transaction_value'

# ==============================================================================
# 2. HELPER FUNCTIONS (load_data, create_features - no changes)
# ==============================================================================
def load_and_preprocess_data(filepath):
    # ... (same as before) ...
    print("-> Loading and preprocessing data...")
    df = pd.read_csv(filepath)
    parts = df['date'].str.split('-', expand=True)
    df['date_str'] = '01-' + parts[1] + '-' + parts[0]
    df['date'] = pd.to_datetime(df['date_str'], format='%d-%b-%y')
    df = df.set_index('date').drop(columns=['id', 'date_str'])
    df.sort_index(inplace=True)
    return df

def create_features(df):
    # ... (same as before) ...
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
    # ... (same as before) ...
    mae = mean_absolute_error(true_values, predicted_values)
    rmse = np.sqrt(mean_squared_error(true_values, predicted_values))
    print("\n--- Model Performance on Validation Set (2016) ---")
    print(f"  Root Mean Squared Error (RMSE): {rmse:,.2f}")
    print(f"  Mean Absolute Error (MAE):      {mae:,.2f}")
    print("--------------------------------------------------")

def plot_validation_results(train_df, validation_df, predictions_df):
    """
    Plots the training data, actual validation data, and predicted validation data.
    """
    plt.figure(figsize=(15, 7))
    
    # Plot training data
    plt.plot(train_df.index, train_df[TARGET_COL], label='Training Data (2011-2015)', color='blue')
    
    # Plot actual validation data
    plt.plot(validation_df.index, validation_df[TARGET_COL], label='Actual Values (2016)', color='green', marker='o')
    
    # Plot predicted validation data
    plt.plot(validation_df.index, predictions_df, label='Predicted Values (2016)', color='red', linestyle='--')
    
    plt.title('Model Validation: Actual vs. Predicted for 2016', fontsize=16)
    plt.xlabel('Date', fontsize=12)
    plt.ylabel('Transaction Value', fontsize=12)
    plt.legend()
    plt.grid(True, which='both', linestyle='--', linewidth=0.5)
    
    print("\n-> Displaying validation plot. Close the plot window to continue the script.")
    plt.show()

# ==============================================================================
# 4. MAIN EXECUTION BLOCK (The New Workflow)
# ==============================================================================
if __name__ == "__main__":
    print("--- Starting Professional ML Workflow: Train, Validate, Re-train ---")

    # --- Step 1: Load and Prepare All Data ---
    all_data = load_and_preprocess_data(CSV_FILE_PATH)
    data_with_features = create_features(all_data)

    # --- Step 2: Split Data into Training and Validation Sets ---
    split_date = '2016-01-01'
    train_set = data_with_features[data_with_features.index < split_date]
    validation_set = data_with_features[data_with_features.index >= split_date]
    print(f"-> Data split: Training set ends on {train_set.index.max().date()}, Validation set starts on {validation_set.index.min().date()}")

    # One-hot encode cities AFTER splitting
    train_final = pd.get_dummies(train_set, columns=['city'], prefix='city').dropna()
    validation_final = pd.get_dummies(validation_set, columns=['city'], prefix='city').dropna()
    
    # Align columns to handle cases where a city might only appear in one set
    common_cols = list(set(train_final.columns) & set(validation_final.columns))
    train_final = train_final[common_cols]
    validation_final = validation_final[common_cols]

    FEATURES = [col for col in train_final.columns if col != TARGET_COL]
    X_train, y_train = train_final[FEATURES], train_final[TARGET_COL]
    X_validation, y_validation = validation_final[FEATURES], validation_final[TARGET_COL]

    # --- Step 3: Train Initial Model ONLY on the Training Set ---
    print("\n-> Training initial model on data from 2011-2015...")
    model_for_validation = xgb.XGBRegressor(n_estimators=1000, learning_rate=0.01, max_depth=3, subsample=0.8, colsample_bytree=0.8, random_state=42, objective='reg:squarederror')
    model_for_validation.fit(X_train, y_train)

    # --- Step 4: Evaluate the Model on the Validation Set (2016) ---
    print("-> Evaluating model performance on unseen 2016 data...")
    validation_predictions = model_for_validation.predict(X_validation)
    evaluate_model(y_validation, validation_predictions)

    # --- Step 4b: PLOT THE VALIDATION RESULTS ---
    # We need to plot a single city's data for a clean graph. Let's pick Beirut.
    city_to_plot = 'Beirut'
    train_city_df = train_set[train_set['city'] == city_to_plot]
    validation_city_df = validation_set[validation_set['city'] == city_to_plot]
    
    # We need to get the predictions for just Beirut from our overall validation predictions
    beirut_indices = y_validation[y_validation.index.isin(validation_city_df.index)].index
    beirut_predictions = pd.Series(validation_predictions, index=y_validation.index).loc[beirut_indices]

    plot_validation_results(train_city_df, validation_city_df, beirut_predictions)

    # --- Step 5: Re-train Final Production Model on ALL Data ---
    print("-> Re-training final model on ALL available data (2011-2016)...")
    all_data_final = pd.get_dummies(data_with_features, columns=['city'], prefix='city').dropna()
    X_all, y_all = all_data_final[FEATURES], all_data_final[TARGET_COL]
    production_model = xgb.XGBRegressor(n_estimators=1000, learning_rate=0.01, max_depth=3, subsample=0.8, colsample_bytree=0.8, random_state=42, objective='reg:squarederror')
    production_model.fit(X_all, y_all)
    print("-> Final production model trained.")

    # --- Step 6: Save the Production-Ready Model ---
    print(f"-> Saving final model to '{MODEL_FILE_PATH}'")
    joblib.dump(production_model, MODEL_FILE_PATH)
    pd.Series(X_all.columns).to_json(MODEL_COLS_PATH)

    print("\n--- Workflow Complete ---")