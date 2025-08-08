# --- 1. IMPORTS ---
import pandas as pd
import numpy as np
import xgboost as xgb
import joblib
import os
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import mean_absolute_percentage_error, mean_squared_error, mean_absolute_error
from sklearn.model_selection import GridSearchCV, TimeSeriesSplit
from supabase import create_client, Client
from dotenv import load_dotenv
from sklearn.preprocessing import MinMaxScaler 

# --- 2. CONFIGURATION ---
load_dotenv()
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
BACKEND_DIR = os.path.dirname(SCRIPT_DIR)
OUTPUT_DIR = os.path.join(BACKEND_DIR, 'next_models')

MODEL_FILE_PATH = os.path.join(OUTPUT_DIR, 'forecast_model.joblib')
MODEL_COLS_PATH = os.path.join(OUTPUT_DIR, 'model_columns.json')


# --- 3. GLOBAL CONSTANTS ---
GROUPING_KEY = 'city'
TARGET_COL = 'transaction_value'
CITIES = ['baabda', 'beirut', 'bekaa', 'kesrouan', 'tripoli'] 
SYNTHETIC_START_YEAR = 2017

# --- 4. HELPER FUNCTIONS ---

def load_data_from_supabase():
    """Connects to Supabase and fetches all regional transaction data."""
    print("-> Connecting to Supabase to fetch training data...")
    url: str = os.environ.get("SUPABASE_URL")
    key: str = os.environ.get("SUPABASE_KEY")

    if not url or not key:
        raise ValueError("FATAL: Supabase URL or Key not found in .env file.")

    supabase: Client = create_client(url, key)
    response = supabase.table('merged_trans').select("*").order('date').execute()

    df = pd.DataFrame(response.data)
    print(f"-> Successfully fetched {len(df)} rows from Supabase.")

    df['date'] = pd.to_datetime(df['date'])
    df.set_index('date', inplace=True)
    df.sort_index(inplace=True)
    print("-> Date column converted to datetime and set as index.")

    df[GROUPING_KEY] = df[GROUPING_KEY].str.lower()
    return df

def create_features(df):
    """Enhanced feature engineering with synthetic data flags"""
    df_features = df.copy()
    
    # Date components
    df_features['year'] = df_features.index.year
    df_features['month'] = df_features.index.month
    df_features['month_sin'] = np.sin(2 * np.pi * df_features.index.month / 12.0)
    df_features['month_cos'] = np.cos(2 * np.pi * df_features.index.month / 12.0)
    
    # Seasonal flags
    df_features['is_december'] = (df_features.index.month == 12).astype(int)
    df_features['is_summer'] = df_features.index.month.isin([6,7,8]).astype(int)
    
    # Synthetic data flag
    df_features['is_synthetic'] = (df_features.index.year >= SYNTHETIC_START_YEAR).astype(int)
    
    # City-specific features (added dynamically in main script)
    return df_features

def add_city_specific_features(df):
    """Adds city-specific rolling stats and interaction terms"""
    for city in CITIES:
        # Rolling averages per city
        df[f'{city}_rolling_3m'] = df[df[GROUPING_KEY] == city][TARGET_COL].rolling(3).mean()
        df[f'{city}_rolling_12m'] = df[df[GROUPING_KEY] == city][TARGET_COL].rolling(12).mean()
        
        # Synthetic interaction terms
        df[f'{city}_synthetic'] = ((df[GROUPING_KEY] == city) & (df['is_synthetic'] == 1)).astype(int)
    return df

def scale_target_by_city(df):
    """Normalizes target variable per city to handle magnitude differences"""
    scalers = {}
    df_scaled = df.copy()
    for city in CITIES:
        mask = df[GROUPING_KEY] == city
        scaler = MinMaxScaler()
        df_scaled.loc[mask, TARGET_COL] = scaler.fit_transform(
            df.loc[mask, TARGET_COL].values.reshape(-1, 1)
        scalers[city] = scaler
    return df_scaled, scalers

# --- 5. MAIN EXECUTION SCRIPT ---
if __name__ == "__main__":
    print("--- Starting Enhanced Model Training ---")
    
    # --- Step 1: Load and Prepare Data ---
    all_data = load_data_from_supabase()
    
    # Feature engineering
    data_with_features = create_features(all_data)
    data_with_features = add_city_specific_features(data_with_features)  # New
    
    # Scale target per city
    data_scaled, scalers = scale_target_by_city(data_with_features)  # New
    
    # One-hot encoding
    data_final = pd.get_dummies(data_scaled, columns=[GROUPING_KEY], prefix=GROUPING_KEY)
    FEATURES = [col for col in data_final.columns if col not in [TARGET_COL, 'id', 'transaction_number']]
    
    # --- Step 2: Train-Validation Split with Synthetic Weighting ---
    train_df = data_final.loc[:'2018-12-31']
    validation_df = data_final.loc['2019-01-01':]
    
    # Apply sample weights (higher weight to original data)
    train_df['weight'] = np.where(
        train_df.index.year < SYNTHETIC_START_YEAR, 1.5, 0.8  # Downweight synthetic
    )
    
    X_train, y_train = train_df[FEATURES], train_df[TARGET_COL]
    X_val, y_val = validation_df[FEATURES], validation_df[TARGET_COL]
    sample_weights = train_df['weight']  # New
    
    # --- Step 3: Hyperparameter Tuning ---
    param_grid = {
        'max_depth': [3, 5],  # Reduced to prevent overfitting
        'learning_rate': [0.01, 0.05],
        'n_estimators': [200],
        'reg_alpha': [0.1, 1],  # More regularization
        'reg_lambda': [0.1, 1],
        'subsample': [0.8]  # Stochastic training
    }
    
    # --- Step 4: Train Model with Weighting ---
    model = xgb.XGBRegressor(
        objective='reg:squarederror',
        eval_metric=['rmse', 'mae'],
        early_stopping_rounds=50,
        **best_params
    )
    
    model.fit(
        X_train, y_train,
        eval_set=[(X_train, y_train), (X_val, y_val)],
        sample_weight=sample_weights,  # Critical for synthetic handling
        verbose=True
    )
    
    # --- Step 5: Residual Analysis (Per City) ---
    residuals = y_val - model.predict(X_val)
    validation_df['residuals'] = residuals
    
    plt.figure(figsize=(12,6))
    sns.boxplot(x=GROUPING_KEY, y='residuals', data=validation_df.reset_index())
    plt.title('Residual Distribution by City')
    plt.xticks(rotation=45)
    plt.show()
    
    # --- Step 6: Future Forecast Adjustments ---
    # When generating future predictions:
    future_df['is_synthetic'] = 1  # Mark all future data as synthetic
    # Apply city-specific scaling inversions:
    for city in CITIES:
        mask = future_df[GROUPING_KEY] == city
        future_df.loc[mask, 'prediction'] = scalers[city].inverse_transform(
            future_df.loc[mask, 'prediction'].values.reshape(-1, 1))

 