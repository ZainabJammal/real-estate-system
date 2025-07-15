import pandas as pd
import numpy as np
import lightgbm as lgb
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.metrics import mean_absolute_error, r2_score
import joblib
import warnings

# --- Setup ---
warnings.filterwarnings('ignore', category=UserWarning)
warnings.filterwarnings('ignore', category=FutureWarning)

def train_and_evaluate_model():
    """
    V14: A robust baseline model. Strips back to core, powerful features and
    a simple, effective training process to establish a high-performance baseline.
    This version intentionally removes the complex, potentially leaky features to
    ensure the core model is sound.
    """
    # --- 1. Load and Clean Data ---
    print("Step 1: Loading and Cleaning Data...")
    df = pd.read_csv('properties.csv')
    df.drop_duplicates(subset=['id'], inplace=True)
    df = df[(df['price_$'] > 10000) & (df['size_m2'] > 15)].copy() # Basic sanity filtering

    # --- 2. Feature Engineering & Cleaning ---
    print("\nStep 2: Feature Engineering and Data Cleaning...")
    
    # Use only core, reliable features for this baseline
    df['type'] = df['type'].str.lower()
    df['province'] = df['province'].str.lower()
    df['district'] = df['district'].str.lower()
    
    # Impute Bedrooms and Bathrooms
    types_with_rooms = ['apartment', 'house/villa', 'chalet']
    df['bedrooms'] = np.where((df['type'].isin(types_with_rooms)) & (df['bedrooms'] == 0), np.nan, df['bedrooms'])
    df['bathrooms'] = np.where((df['type'].isin(types_with_rooms)) & (df['bathrooms'] == 0), np.nan, df['bathrooms'])
    
    for col in ['bedrooms', 'bathrooms']:
        df[col].fillna(df.groupby('type')[col].transform('median'), inplace=True)
        df[col].fillna(1, inplace=True)
        df[col] = df[col].astype(int)

    # Log transform price and size for better performance
    df['log_price'] = np.log1p(df['price_$'])
    df['log_size_m2'] = np.log1p(df['size_m2'])

    # --- 3. Outlier Handling (On Transformed Data) ---
    print("\nStep 3: Handling Outliers...")
    # It's more robust to remove outliers on the log-transformed price
    log_price_cap = df['log_price'].quantile(0.99)
    df = df[df['log_price'] < log_price_cap].copy()
    print(f"Data shape after cleaning and outlier removal: {df.shape}")

    # --- 4. Define Features and Split Data ---
    print("\nStep 4: Defining Features and Splitting Data...")
    
    # Core features that are known to be powerful predictors
    categorical_features = ['province', 'district', 'type']
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
    
    # Use a solid set of default LightGBM parameters
    model = lgb.LGBMRegressor(
        objective='regression_l1',
        n_estimators=1000,
        learning_rate=0.05,
        num_leaves=31,
        random_state=42,
        n_jobs=-1
    )

    pipeline = Pipeline(steps=[
        ('preprocessor', preprocessor),
        ('regressor', model)
    ])
    
    # --- 6. Train the Model with Early Stopping ---
    print("\nStep 6: Training Model...")
    
    # Create a validation set for early stopping
    X_train_main, X_val, y_train_main, y_val = train_test_split(X_train, y_train, test_size=0.2, random_state=42)
    
    # Note the syntax for early stopping within a pipeline
    pipeline.fit(X_train_main, y_train_main,
                 regressor__eval_set=[(pipeline.named_steps['preprocessor'].transform(X_val), y_val)],
                 regressor__callbacks=[lgb.early_stopping(50, verbose=False)])

    # --- 7. Final Evaluation ---
    print("\n--- Step 7: Final Model Evaluation on Test Set ---")
    y_pred_log = pipeline.predict(X_test)
    y_pred_dollars = np.expm1(y_pred_log)
    y_test_dollars = np.expm1(y_test)

    r2 = r2_score(y_test_dollars, y_pred_dollars)
    mae = mean_absolute_error(y_test_dollars, y_pred_dollars)

    print(f"\nFinal R-squared (R²): {r2:.4f}")
    print(f"Final Mean Absolute Error (MAE): ${mae:,.2f}")

    # --- 8. Save the Final Model ---
    model_filename = 'property_price_model_baseline.joblib'
    print(f"\nStep 8: Saving final model...")
    joblib.dump(pipeline, model_filename)
    print(f"Model saved successfully to {model_filename}")

if __name__ == '__main__':
    train_and_evaluate_model()