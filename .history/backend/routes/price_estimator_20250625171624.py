import pandas as pd
import numpy as np
import lightgbm as lgb
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import r2_score, mean_absolute_error
import joblib
import warnings

warnings.filterwarnings('ignore', category=UserWarning)

def run_experiment(X_train, X_test, y_train, y_test, numerical_features, categorical_features, experiment_name):
    """A helper function to run a training and evaluation experiment."""
    print(f"\n--- Running Experiment: {experiment_name} ---")
    print(f"Features: {numerical_features + categorical_features}")

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
    return r2, mae, pipeline

def train_and_evaluate_model():
    """
    V19: A diagnostic script to add features one by one and validate their impact.
    """
    # --- 1. Load and Clean Data ---
    print("Step 1: Loading and Cleaning Data...")
    df = pd.read_csv('properties.csv')
    df.drop_duplicates(subset=['id'], inplace=True)
    df = df[(df['price_$'] > 10000) & (df['size_m2'] > 15)].copy()

    for col in ['type', 'province', 'district']:
        df[col] = df[col].str.lower().str.strip()

    types_with_rooms = ['apartment', 'house/villa', 'chalet']
    df['bedrooms'] = np.where((df['type'].isin(types_with_rooms)) & (df['bedrooms'] == 0), np.nan, df['bedrooms'])
    df['bathrooms'] = np.where((df['type'].isin(types_with_rooms)) & (df['bathrooms'] == 0), np.nan, df['bathrooms'])
    
    for col in ['bedrooms', 'bathrooms']:
        df[col].fillna(df.groupby('type')[col].transform('median'), inplace=True)
        df[col].fillna(1, inplace=True)
        df[col] = df[col].astype(int)

    # ** Geospatial Clustering (Done once) **
    kmeans = KMeans(n_clusters=20, random_state=42, n_init='auto')
    df['location_cluster'] = kmeans.fit_predict(df[['latitude', 'longitude']]).astype(str)

    df['log_price'] = np.log1p(df['price_$'])
    df['log_size_m2'] = np.log1p(df['size_m2'])
    
    log_price_cap = df['log_price'].quantile(0.99)
    df = df[df['log_price'] < log_price_cap].copy()
    
    print(f"Data shape after cleaning: {df.shape}")

    # --- 2. Define Full Feature Set and Split Data ---
    y = df['log_price']
    X = df.drop(columns=['id', 'city', 'created_at', 'price_$', 'log_price'])
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # --- 3. Run Incremental Experiments ---
    
    # Experiment 1: The absolute baseline (Size only)
    run_experiment(X_train, X_test, y_train, y_test, 
                   numerical_features=['log_size_m2'], 
                   categorical_features=[],
                   experiment_name="Baseline (Size Only)")

    # Experiment 2: Add Room Counts
    run_experiment(X_train, X_test, y_train, y_test, 
                   numerical_features=['log_size_m2', 'bedrooms', 'bathrooms'], 
                   categorical_features=[],
                   experiment_name="Add Rooms")

    # Experiment 3: Add Property Type
    run_experiment(X_train, X_test, y_train, y_test, 
                   numerical_features=['log_size_m2', 'bedrooms', 'bathrooms'], 
                   categorical_features=['type'],
                   experiment_name="Add Property Type")

    # Experiment 4: Add Broad Location
    run_experiment(X_train, X_test, y_train, y_test, 
                   numerical_features=['log_size_m2', 'bedrooms', 'bathrooms'], 
                   categorical_features=['type', 'province', 'district'],
                   experiment_name="Add Full Location (District)")

    # Experiment 5: Test Geospatial Cluster instead of lat/lon
    r2_final, mae_final, final_pipeline = run_experiment(
                   X_train, X_test, y_train, y_test, 
                   numerical_features=['log_size_m2', 'bedrooms', 'bathrooms'], 
                   categorical_features=['type', 'province', 'district', 'location_cluster'],
                   experiment_name="Add Geospatial Cluster")
                   
    # --- 4. Final Conclusion and Saving ---
    print("\n--- DIAGNOSTIC COMPLETE ---")
    if r2_final > 0.8:
        print("\nSUCCESS: The final model with geospatial clustering is strong.")
        model_filename = 'property_price_model_final.joblib'
        print(f"Saving best model to {model_filename}...")
        joblib.dump(final_pipeline, model_filename)
        joblib.dump(kmeans, 'location_cluster_model.joblib')
        print("Models saved successfully.")
    else:
        print("\nCONCLUSION: The model's predictive power is limited even with advanced features.")
        print("This strongly suggests an issue with the underlying data's consistency or that critical predictive features are missing.")


if __name__ == '__main__':
    train_and_evaluate_model()