import pandas as pd
import numpy as np
from catboost import CatBoostRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_absolute_error
import joblib
import os
import warnings

warnings.filterwarnings('ignore', category=UserWarning)

# --- 1. Your Custom Hyperparameters and Feature Definitions ---
# This dictionary holds the tailored parameters for each model
MODEL_CONFIG = {
    'apartment': {
        'params': {'depth': 8, 'learning_rate': 0.03, 'l2_leaf_reg': 5, 'iterations': 2000},
        'features': {'numerical': ['log_size_m2', 'bedrooms', 'bathrooms'], 'categorical': ['province', 'district', 'city']}
    },
    'land': {
        'params': {'depth': 6, 'learning_rate': 0.01, 'l2_leaf_reg': 10, 'iterations': 2000},
        'features': {'numerical': ['log_size_m2'], 'categorical': ['province', 'district', 'city']}
    },
    'house/villa': {
        'params': {'depth': 7, 'learning_rate': 0.02, 'l2_leaf_reg': 3, 'iterations': 2000},
        'features': {'numerical': ['log_size_m2', 'bedrooms', 'bathrooms'], 'categorical': ['province', 'district', 'city']}
    },
    # Add configs for other types if they become viable
    'office': {
        'params': {'depth': 6, 'learning_rate': 0.05, 'l2_leaf_reg': 7, 'iterations': 1500},
        'features': {'numerical': ['log_size_m2', 'bedrooms', 'bathrooms'], 'categorical': ['province', 'district', 'city']}
    },
    'shop': {
        'params': {'depth': 6, 'learning_rate': 0.05, 'l2_leaf_reg': 7, 'iterations': 1500},
        'features': {'numerical': ['log_size_m2'], 'categorical': ['province', 'district', 'city']}
    },
     'chalet': {
        'params': {'depth': 6, 'learning_rate': 0.03, 'l2_leaf_reg': 5, 'iterations': 1500},
        'features': {'numerical': ['log_size_m2', 'bedrooms', 'bathrooms'], 'categorical': ['province', 'district', 'city']}
    }
}

def train_specialist_model(df_segment, property_type, config):
    """
    Trains a specialized CatBoost model using tailored hyperparameters and features.
    """
    print(f"\n--- Training Specialist Model for: {property_type.upper()} ---")
    
    # 1. Feature Engineering
    df = df_segment.copy()
    
    # ** NEW TARGET VARIABLE: log_price_per_sqm **
    df['log_price_per_sqm'] = np.log1p(df['price_$'] / df['size_m2'])
    df['log_size_m2'] = np.log1p(df['size_m2'])
    
    features_to_use = config['features']['numerical'] + config['features']['categorical']
    
    y = df['log_price_per_sqm']
    X = df[features_to_use]
    
    # 2. Data Split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # 3. Model Training
    params = config['params']
    model = CatBoostRegressor(
        loss_function='RMSE', # A standard for regression
        eval_metric='R2',
        random_seed=42,
        verbose=0,
        cat_features=config['features']['categorical'],
        **params
    )
    
    model.fit(
        X_train, y_train,
        eval_set=(X_test, y_test),
        early_stopping_rounds=50,
        verbose=False
    )
    
    # 4. Evaluation
    y_pred_log_per_sqm = model.predict(X_test)
    
    # Convert prediction back to total dollar value
    test_log_size = X_test['log_size_m2']
    test_size = np.expm1(test_log_size)
    
    pred_price_per_sqm = np.expm1(y_pred_log_per_sqm)
    y_pred_dollars = pred_price_per_sqm * test_size
    
    # Get original test prices for comparison
    y_test_dollars = np.expm1(y_test) * test_size

    r2 = r2_score(y_test_dollars, y_pred_dollars)
    mae = mean_absolute_error(y_test_dollars, y_pred_dollars)
    
    print(f"Result -> R²: {r2:.4f}, MAE: ${mae:,.2f}")
    
    # 5. Feature Importance
    feature_importances = pd.Series(model.get_feature_importance(), index=X.columns).sort_values(ascending=False)
    print("Top 5 Features:")
    print(feature_importances.head(5).to_string())
    
    return model, r2

def main():
    print("Step 1: Loading and preparing base data...")
    df = pd.read_csv('properties.csv')
    df.drop_duplicates(subset=['id'], inplace=True)
    df = df[df['price_$'] > 10000].copy()
    df = df[df['size_m2'] > 20].copy()
    
    for col in ['type', 'province', 'district', 'city']:
        df[col] = df[col].astype(str).str.lower().str.strip()
        
    df['bedrooms'].fillna(df.groupby('type')['bedrooms'].transform('median'), inplace=True)
    df['bathrooms'].fillna(df.groupby('type')['bathrooms'].transform('median'), inplace=True)
    df.fillna({'bedrooms': 1, 'bathrooms': 1}, inplace=True)
    df[['bedrooms', 'bathrooms']] = df[['bedrooms', 'bathrooms']].astype(int)
    
    output_dir = "ultimate_specialist_models"
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        
    viable_types = list(MODEL_CONFIG.keys())
    print(f"\nTraining specialist models for: {viable_types}")

    for prop_type in viable_types:
        df_segment = df[df['type'] == prop_type]
        if len(df_segment) < 50: # Check if there is enough data
            print(f"\n--- Skipping {prop_type.upper()}: Not enough data ({len(df_segment)} rows) ---")
            continue

        model, r2 = train_specialist_model(df_segment, prop_type, MODEL_CONFIG[prop_type])
        
        if r2 > 0.50: # A reasonable threshold for saving a specialist model
            filename = f"model_{prop_type.replace('/', '_')}.joblib"
            filepath = os.path.join(output_dir, filename)
            joblib.dump(model, filepath)
            print(f"   -> Model saved to '{filepath}'")
        else:
            print("   -> Model performance is too low, not saving.")
            
    print("\n--- Process Complete ---")

if __name__ == '__main__':
    main()