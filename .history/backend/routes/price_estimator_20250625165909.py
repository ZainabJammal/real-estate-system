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

def train_and_evaluate_model():
    """
    V16: A methodical approach to validate feature impact. We start with a
    simple baseline and add features to see their effect on performance.
    """
    # --- 1. Load and Clean Data ---
    print("Step 1: Loading and Cleaning Data...")
    df = pd.read_csv('properties.csv')
    df.drop_duplicates(subset=['id'], inplace=True)
    df = df[(df['price_$'] > 10000) & (df['size_m2'] > 15) & (df['price_$'] < 5000000)].copy() # Basic sanity filtering

    # --- 2. Feature Engineering & Cleaning ---
    print("\nStep 2: Feature Engineering and Data Cleaning...")
    for col in ['type', 'province', 'district']:
        df[col] = df[col].str.lower().str.strip()

    types_with_rooms = ['apartment', 'house/villa', 'chalet']
    df['bedrooms'] = np.where((df['type'].isin(types_with_rooms)) & (df['bedrooms'] == 0), np.nan, df['bedrooms'])
    df['bathrooms'] = np.where((df['type'].isin(types_with_rooms)) & (df['bathrooms'] == 0), np.nan, df['bathrooms'])
    
    for col in ['bedrooms', 'bathrooms']:
        df[col].fillna(df.groupby('type')[col].transform('median'), inplace=True)
        df[col].fillna(1, inplace=True)
        df[col] = df[col].astype(int)

    # ** NEW: Create the geospatial cluster feature **
    print("Creating geospatial features using K-Means clustering...")
    kmeans = KMeans(n_clusters=20, random_state=42, n_init='auto') # Increase clusters for more granularity
    df['location_cluster'] = kmeans.fit_predict(df[['latitude', 'longitude']]).astype(str)

    # --- 3. Log Transforms and Outlier Handling ---
    print("\nStep 3: Log Transforms and Outlier Handling...")
    df['log_price'] = np.log1p(df['price_$'])
    df['log_size_m2'] = np.log1p(df['size_m2'])
    
    log_price_cap = df['log_price'].quantile(0.995) # Use a slightly more lenient cap
    df = df[df['log_price'] < log_price_cap].copy()
    print(f"Data shape after cleaning and outlier removal: {df.shape}")

    # --- 4. Define Features and Split Data ---
    print("\nStep 4: Defining Features and Splitting Data...")
    
    # We will use all our powerful features now, as the previous simplified model failed.
    # The new 'location_cluster' is the key addition we are testing.
    categorical_features = ['province', 'district', 'type', 'location_cluster']
    numerical_features = ['log_size_m2', 'bedrooms', 'bathrooms', 'latitude', 'longitude']
    
    X = df[categorical_features + numerical_features]
    y = df['log_price']

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # --- 5. Create Preprocessing and Model Pipeline ---
    print("\nStep 5: Building Model Pipeline...")
    
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', StandardScaler(), numerical_features),
            ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_features)
        ],
        remainder='passthrough'
    )
    
    # Use a robust set of LightGBM parameters known to perform well
    model = lgb.LGBMRegressor(
        objective='regression_l1', # MAE is robust to outliers
        n_estimators=1000,
        learning_rate=0.03,
        num_leaves=40,
        max_depth=8,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42,
        n_jobs=-1
    )

    pipeline = Pipeline(steps=[
        ('preprocessor', preprocessor),
        ('regressor', model)
    ])
    
    # --- 6. Train the Model with Early Stopping (Corrected) ---
    print("\nStep 6: Training Model...")
    
    # To use early stopping with a pipeline, we need to create a temporary validation set
    # and manually transform it *after* the pipeline's preprocessor has been fitted.
    
    # A cleaner approach is to fit on the full training set and rely on good parameters.
    # For simplicity and robustness, let's fit on all of X_train.
    pipeline.fit(X_train, y_train)
    print("Model training complete.")


    # --- 7. Final Evaluation ---
    print("\n--- Step 7: Final Model Evaluation on Test Set ---")
    y_pred_log = pipeline.predict(X_test)
    y_pred_dollars = np.expm1(y_pred_log)
    y_test_dollars = np.expm1(y_test)

    r2 = r2_score(y_test_dollars, y_pred_dollars)
    mae = mean_absolute_error(y_test_dollars, y_pred_dollars)

    print(f"\nFinal R-squared (R²): {r2:.4f}")
    print(f"Final Mean Absolute Error (MAE): ${mae:,.2f}")

    if r2 > 0.8:
        print("\nSUCCESS: Model performance is strong!")
        # --- 8. Save the Final Model ---
        model_filename = 'property_price_model_final.joblib'
        print(f"\nStep 8: Saving final model...")
        joblib.dump(pipeline, model_filename)
        joblib.dump(kmeans, 'location_cluster_model.joblib') # Save the fitted k-means model
        print(f"Model and k-means clusterer saved successfully.")
    else:
        print("\nWARNING: Model performance is still low. Further feature engineering or data review is needed.")

if __name__ == '__main__':
    train_and_evaluate_model()