import pandas as pd
import numpy as np
import lightgbm as lgb
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.metrics import r2_score, mean_absolute_error
import joblib
import warnings

warnings.filterwarnings('ignore', category=UserWarning)

def run_focused_experiment(df, numerical_features, categorical_features, experiment_name):
    """A helper function to run our focused training and evaluation."""
    print(f"\n--- Running Experiment: {experiment_name} ---")
    print(f"Features: {numerical_features + categorical_features}")

    # Define the data for this experiment
    X = df[numerical_features + categorical_features]
    y = df['log_price']
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    preprocessor = ColumnTransformer(
        transformers=[
            ('num', StandardScaler(), numerical_features),
            ('cat', OneHotEncoder(handle_unknown='ignore', sparse_output=False), categorical_features)
        ],
        remainder='drop'
    )

    pipeline = Pipeline(steps=[
        ('preprocessor', preprocessor),
        ('regressor', lgb.LGBMRegressor(objective='regression_l1', random_state=42))
    ])

    pipeline.fit(X_train, y_train)

    y_pred_log = pipeline.predict(X_test)
    y_pred_dollars = np.expm1(y_pred_log)
    y_test_dollars = np.expm1(y_test)

    r2 = r2_score(y_test_dollars, y_pred_dollars)
    mae = mean_absolute_error(y_test_dollars, y_pred_dollars)
    
    print(f"Result -> R²: {r2:.4f}, MAE: ${mae:,.2f}")
    return r2

def main():
    """
    V20: The 'last shot' based on a focused strategy. We model only the most
    consistent property types to see if a signal can be found.
    """
    # --- 1. Load and Filter Data ---
    print("Step 1: Loading and Filtering Data to a Consistent Subset...")
    df = pd.read_csv('properties.csv')
    df.drop_duplicates(subset=['id'], inplace=True)
    
    # ** CORE STRATEGY: Only model these types **
    property_types_to_model = ['apartment', 'house/villa', 'chalet']
    df['type'] = df['type'].str.lower().str.strip()
    df = df[df['type'].isin(property_types_to_model)].copy()
    
    # Basic sanity filtering
    df = df[(df['price_$'] > 10000) & (df['size_m2'] > 20) & (df['price_$'] < 4000000)].copy()
    
    print(f"Filtered down to {len(df)} properties of types: {property_types_to_model}")

    # --- 2. Feature Engineering & Cleaning ---
    print("\nStep 2: Feature Engineering and Cleaning...")
    for col in ['province', 'district']:
        df[col] = df[col].str.lower().str.strip()
        
    df['bedrooms'].fillna(df.groupby('type')['bedrooms'].transform('median'), inplace=True)
    df['bathrooms'].fillna(df.groupby('type')['bathrooms'].transform('median'), inplace=True)
    df.fillna({'bedrooms': 1, 'bathrooms': 1}, inplace=True)
    df[['bedrooms', 'bathrooms']] = df[['bedrooms', 'bathrooms']].astype(int)

    # Create combined location feature
    df['province_district'] = df['province'] + '_' + df['district']

    # Log transforms
    df['log_price'] = np.log1p(df['price_$'])
    df['log_size_m2'] = np.log1p(df['size_m2'])
    
    # --- 3. Run Experiments ---
    
    numerical_features = ['log_size_m2', 'bedrooms', 'bathrooms']
    
    # Experiment A: Broad location feature (province only)
    run_focused_experiment(
        df, 
        numerical_features=numerical_features,
        categorical_features=['province', 'type'],
        experiment_name="Strategy A (Province Only)"
    )
    
    # Experiment B: Combined location feature
    run_focused_experiment(
        df, 
        numerical_features=numerical_features,
        categorical_features=['province_district', 'type'],
        experiment_name="Strategy B (Province + District Combined)"
    )

    # Experiment C: The original best attempt on this filtered data
    r2_final = run_focused_experiment(
        df, 
        numerical_features=numerical_features,
        categorical_features=['province', 'district', 'type'],
        experiment_name="Strategy C (Province and District Separate)"
    )

    # --- 4. Final Conclusion ---
    print("\n--- FINAL CONCLUSION ---")
    if r2_final > 0.75:
        print(f"SUCCESS! R² score of {r2_final:.4f} is strong.")
        print("The issue was data noise from other property types. A focused model works.")
    elif r2_final > 0.60:
        print(f"PARTIAL SUCCESS. R² score of {r2_final:.4f} is moderately useful.")
        print("The model has some predictive power but is limited by data consistency.")
    else:
        print(f"FAILURE. R² score of {r2_final:.4f} is too low.")
        print("The data inconsistencies persist even within residential properties.")
        print("Pivoting to the 'Market Analyser' approach is the recommended final step.")

if __name__ == '__main__':
    main()