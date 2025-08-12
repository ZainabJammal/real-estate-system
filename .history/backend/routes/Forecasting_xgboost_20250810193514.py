import pandas as pd
import xgboost as xgb
import joblib
import os
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_percentage_error, mean_squared_error, mean_absolute_error

from supabase import create_client, Client
from dotenv import load_dotenv

# --- 1. CONFIGURATION ---
# Load environment variables from .env file
load_dotenv()


SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
BACKEND_DIR = os.path.dirname(SCRIPT_DIR)
OUTPUT_DIR = os.path.join(BACKEND_DIR, 'forecasting_models') 

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

    if 'id' in df.columns and 'transaction_number' in df.columns:
         print("-> Dropping 'id' and 'transaction_number' columns.")
         df = df.drop(columns=['id', 'transaction_number'], errors='ignore')
    else:
        print("-> 'id' and 'transaction_number' not found in df")

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
    """Enhanced feature engineering for time-series forecasting"""
    df_features = df.copy()
    
    # --- Temporal Features ---
    # Date components
    df_features['year'] = df_features.index.year
    df_features['month'] = df_features.index.month
    df_features['quarter'] = df_features.index.quarter
    
    # Cyclical encoding for months
    df_features['month_sin'] = np.sin(2 * np.pi * df_features.index.month/12)
    df_features['month_cos'] = np.cos(2 * np.pi * df_features.index.month/12)
    
    # --- Seasonal Flags ---
    df_features['is_december'] = (df_features.index.month == 12).astype(int)
    df_features['is_summer'] = df_features.index.month.isin([6,7,8]).astype(int)
    df_features['is_quarter_end'] = df_features.index.is_quarter_end.astype(int)
    
    # # --- Economic Cycle Features ---
    # # Moving averages (no future leakage)
    # for city in df[GROUPING_KEY].unique():
    #     city_mask = df[GROUPING_KEY] == city
    #     df_features.loc[city_mask, 'city_rolling_3m'] = (
    #         df.loc[city_mask, TARGET_COL]
    #         .rolling(3, min_periods=1)
    #         .mean()
    #         .values
    #     )
    #     df_features.loc[city_mask, 'city_rolling_12m'] = (
    #         df.loc[city_mask, TARGET_COL]
    #         .rolling(12, min_periods=1)
    #         .mean()
    #         .values
    #     )
    
    # --- City Interaction Features ---
    # Relative performance metrics
    df_features['monthly_avg'] = df_features.groupby('month')[TARGET_COL].transform('mean')
    df_features['city_avg'] = df_features.groupby(GROUPING_KEY)[TARGET_COL].transform('mean')
    
    # --- Special Event Flags ---
    # Add known event markers (customize for your domain)
    df_features['post_2013'] = (df_features.index.year > 2013).astype(int)
    
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
    'max_depth': 4,            # Slightly deeper than before
    'learning_rate': 0.05,     # Increased from 0.01
    'n_estimators': 500,       # Reduced from 1000 (with higher learning rate)
    'subsample': 0.8,
    'colsample_bytree': 0.8,
    'gamma': 1,                # Reduced from 5
    'reg_alpha': 0.5,
    'reg_lambda': 1.0,         # Reduced from 1.5
    'min_child_weight': 3,     # New parameter to control leaf size
    'early_stopping_rounds': 50
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
    # Use lowercase to match the data processing
    city_to_plot = 'beirut' 
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

    plt.figure(figsize=(12, 8))
    xgb.plot_importance(model, max_num_features=20)
    plt.show()

    # --- Step 5: Re-train Final Model on ALL Data ---
    
    print("\n-> Validation complete. Re-training final model on ALL available data for production...")
    X_all, y_all = data_final[FEATURES], data_final[TARGET_COL]

    # We will use the same robust hyperparameters we defined earlier.
    # Create a copy of the params for the production model
    production_params = xgb_params.copy()

    # *** FIX: Remove the early_stopping_rounds parameter for the final training ***
    if 'early_stopping_rounds' in production_params:
        print("-> Removing 'early_stopping_rounds' for final model training.")
        del production_params['early_stopping_rounds']

    print(f"-> Training final model with {production_params.get('n_estimators', 'N/A')} estimators...")
    production_model = xgb.XGBRegressor(**production_params)

    # Train the final model on the entire dataset without early stopping
    production_model.fit(X_all, y_all)

    print("-> Final model trained successfully.")

    # --- Step 6: Save the Production-Ready Model ---
    print(f"-> Saving final production model to '{MODEL_FILE_PATH}'")
    if not os.path.exists(OUTPUT_DIR):
        os.makedirs(OUTPUT_DIR)
        print(f"-> Created directory: '{OUTPUT_DIR}'")
    joblib.dump(production_model, MODEL_FILE_PATH)
    pd.Series(FEATURES).to_json(MODEL_COLS_PATH)

    print("\n--- 'Best Guess' Model Training Complete ---")