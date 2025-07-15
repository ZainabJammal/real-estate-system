import pandas as pd
import numpy as np
import lightgbm as lgb
from sklearn.model_selection import train_test_split
from sklearn.cluster import KMeans
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, r2_score
import joblib
import warnings

# New imports for advanced feature engineering
from geopy.distance import geodesic
from category_encoders import LeaveOneOutEncoder

# --- Setup ---
warnings.filterwarnings('ignore', category=UserWarning)
warnings.filterwarnings('ignore', category=FutureWarning)

def train_and_evaluate_model():
    """
    V17: Implements a suite of advanced, high-impact features to finally
    unlock the predictive power of the dataset. This is a full-scale feature
    engineering approach.
    """
    # --- 1. Load and Clean Data ---
    print("Step 1: Loading and Cleaning Data...")
    df = pd.read_csv('properties.csv')
    df.drop_duplicates(subset=['id'], inplace=True)
    df = df[df['price_$'] > 10000].copy()
    df = df[df['size_m2'] > 15].copy()
    
    # Standardize text columns
    for col in ['type', 'province', 'district']:
        df[col] = df[col].str.lower().str.strip()

    # --- 2. Advanced Feature Engineering ---
    print("\nStep 2: Advanced Feature Engineering...")

    # A) Geospatial Features
    print("... creating geospatial features (distance and clusters)...")
    BEIRUT_CENTER = (33.8938, 35.5018)
    df['dist_to_center_km'] = df.apply(
        lambda row: geodesic((row['latitude'], row['longitude']), BEIRUT_CENTER).km,
        axis=1
    )
    kmeans = KMeans(n_clusters=20, random_state=42, n_init='auto')
    df['location_cluster'] = kmeans.fit_predict(df[['latitude', 'longitude']]).astype(str)

    # B) Structural & Room-Based Features
    print("... creating structural and room-based features...")
    types_with_rooms = ['apartment', 'house/villa', 'chalet']
    df['bedrooms'] = np.where((df['type'].isin(types_with_rooms)) & (df['bedrooms'] == 0), np.nan, df['bedrooms'])
    df['bathrooms'] = np.where((df['type'].isin(types_with_rooms)) & (df['bathrooms'] == 0), np.nan, df['bathrooms'])
    for col in ['bedrooms', 'bathrooms']:
        df[col].fillna(df.groupby('type')[col].transform('median'), inplace=True)
        df[col].fillna(1, inplace=True)
        df[col] = df[col].astype(int)
    
    # Avoid division by zero
    df['bathrooms_safe'] = df['bathrooms'].replace(0, 1)
    df['bedrooms_safe'] = df['bedrooms'].replace(0, 1)
    df['total_rooms'] = df['bedrooms'] + df['bathrooms_safe']
    
    df['beds_to_baths_ratio'] = df['bedrooms_safe'] / df['bathrooms_safe']
    df['sqm_per_room'] = df['size_m2'] / df['total_rooms'].replace(0, 1)

    # C) Log Transforms and Outlier Handling
    print("... applying log transforms and handling outliers...")
    df['log_price'] = np.log1p(df['price_$'])
    df['log_size_m2'] = np.log1p(df['size_m2'])
    
    # Use a more robust outlier detection on log price
    log_price_cap = df['log_price'].quantile(0.99)
    log_price_floor = df['log_price'].quantile(0.01)
    df = df[(df['log_price'] < log_price_cap) & (df['log_price'] > log_price_floor)].copy()
    
    print(f"Final data shape for modeling: {df.shape}")

    # --- 3. Define Features and Split Data ---
    print("\nStep 3: Defining Final Features and Splitting Data...")
    
    # We will encode these with LeaveOneOutEncoder
    high_cardinality_categorical = ['district', 'location_cluster']
    
    # We will encode these with OneHotEncoder
    low_cardinality_categorical = ['province', 'type']

    numerical_features = [
        'log_size_m2', 'bedrooms', 'bathrooms', 'latitude', 'longitude',
        'dist_to_center_km', 'beds_to_baths_ratio', 'sqm_per_room'
    ]
    
    y = df['log_price']
    X = df[high_cardinality_categorical + low_cardinality_categorical + numerical_features]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # --- 4. Create Final Preprocessing and Model Pipeline ---
    print("\nStep 4: Building Final Model Pipeline...")
    
    # Create a preprocessor that handles different encoding types
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', StandardScaler(), numerical_features),
            # Use LeaveOneOutEncoder for features with many categories
            ('loo_encoder', LeaveOneOutEncoder(), high_cardinality_categorical),
            ('ohe_encoder', OneHotEncoder(handle_unknown='ignore'), low_cardinality_categorical)
        ],
        remainder='passthrough'
    )
    
    # Use a solid set of LightGBM parameters
    model = lgb.LGBMRegressor(
        objective='regression_l1', n_estimators=1000, learning_rate=0.03,
        num_leaves=50, max_depth=8, subsample=0.8, colsample_bytree=0.8,
        random_state=42, n_jobs=-1
    )

    pipeline = Pipeline(steps=[
        ('preprocessor', preprocessor),
        ('regressor', model)
    ])
    
    # --- 5. Train the Model ---
    print("\nStep 5: Training Final Model...")
    pipeline.fit(X_train, y_train)
    print("Model training complete.")

    # --- 6. Final Evaluation ---
    print("\n--- Step 6: Final Model Evaluation on Test Set ---")
    y_pred_log = pipeline.predict(X_test)
    y_pred_dollars = np.expm1(y_pred_log)
    y_test_dollars = np.expm1(y_test)

    r2 = r2_score(y_test_dollars, y_pred_dollars)
    mae = mean_absolute_error(y_test_dollars, y_pred_dollars)

    print(f"\nFinal R-squared (R²): {r2:.4f}")
    print(f"Final Mean Absolute Error (MAE): ${mae:,.2f}")

    if r2 > 0.8:
        print("\nSUCCESS: Model performance is strong!")
        # --- 7. Save the Final Model ---
        model_filename = 'property_price_model_final.joblib'
        print(f"\nStep 7: Saving final model...")
        joblib.dump(pipeline, model_filename)
        joblib.dump(kmeans, 'location_cluster_model.joblib') # Save the fitted k-means model
        print(f"Model and k-means clusterer saved successfully.")
    else:
        print("\nWARNING: Model performance is still low. The issue may be fundamental to the data's consistency.")

if __name__ == '__main__':
    train_and_evaluate_model()