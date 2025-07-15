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
    V15: The definitive, direct approach. Avoids all pipeline errors by handling
    preprocessing manually before training. Assembles the final pipeline for
    deployment convenience only. This is a robust and guaranteed workflow.
    """
    # --- 1. Load and Clean Data ---
    print("Step 1: Loading and Cleaning Data...")
    df = pd.read_csv('properties.csv')
    df.drop_duplicates(subset=['id'], inplace=True)
    df = df[(df['price_$'] > 10000) & (df['size_m2'] > 15)].copy()
    
    # --- 2. Feature Engineering & Cleaning ---
    print("\nStep 2: Feature Engineering and Data Cleaning...")
    df['type'] = df['type'].str.lower()
    df['province'] = df['province'].str.lower()
    df['district'] = df['district'].str.lower()
    
    types_with_rooms = ['apartment', 'house/villa', 'chalet']
    df['bedrooms'] = np.where((df['type'].isin(types_with_rooms)) & (df['bedrooms'] == 0), np.nan, df['bedrooms'])
    df['bathrooms'] = np.where((df['type'].isin(types_with_rooms)) & (df['bathrooms'] == 0), np.nan, df['bathrooms'])
    
    for col in ['bedrooms', 'bathrooms']:
        df[col].fillna(df.groupby('type')[col].transform('median'), inplace=True)
        df[col].fillna(1, inplace=True)
        df[col] = df[col].astype(int)

    # Log transform for better performance
    df['log_price'] = np.log1p(df['price_$'])
    df['log_size_m2'] = np.log1p(df['size_m2'])

    # --- 3. Outlier Handling ---
    print("\nStep 3: Handling Outliers...")
    log_price_cap = df['log_price'].quantile(0.99)
    df = df[df['log_price'] < log_price_cap].copy()
    print(f"Data shape after cleaning and outlier removal: {df.shape}")

    # --- 4. Define Features and Split Data ---
    print("\nStep 4: Defining Features and Splitting Data...")
    categorical_features = ['province', 'district', 'type']
    numerical_features = ['log_size_m2', 'bedrooms', 'bathrooms', 'latitude', 'longitude']
    
    X = df[categorical_features + numerical_features]
    y = df['log_price']

    # Create the main train/test split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    # Create a validation set from the training data for early stopping
    X_train_main, X_val, y_train_main, y_val = train_test_split(X_train, y_train, test_size=0.2, random_state=42)

    # --- 5. Manual Preprocessing (The Core Fix) ---
    print("\nStep 5: Manually Preprocessing Data...")
    
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', StandardScaler(), numerical_features),
            ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_features)
        ],
        remainder='passthrough'
    )
    
    # Fit on the main training data, then transform all sets
    X_train_main_processed = preprocessor.fit_transform(X_train_main)
    X_val_processed = preprocessor.transform(X_val)
    X_test_processed = preprocessor.transform(X_test)
    
    # --- 6. Train the Model Directly ---
    print("\nStep 6: Training Model Directly on Processed Data...")
    
    model = lgb.LGBMRegressor(
        objective='regression_l1',
        n_estimators=2000, # Set high and let early stopping find the best
        learning_rate=0.03,
        num_leaves=40,
        random_state=42,
        n_jobs=-1
    )

    # Fit the model with early stopping. This will now work without error.
    model.fit(X_train_main_processed, y_train_main,
              eval_set=[(X_val_processed, y_val)],
              eval_metric='mae',
              callbacks=[lgb.early_stopping(100, verbose=False)])

    # --- 7. Final Evaluation ---
    print("\n--- Step 7: Final Model Evaluation on Test Set ---")
    y_pred_log = model.predict(X_test_processed)
    y_pred_dollars = np.expm1(y_pred_log)
    y_test_dollars = np.expm1(y_test)

    r2 = r2_score(y_test_dollars, y_pred_dollars)
    mae = mean_absolute_error(y_test_dollars, y_pred_dollars)

    print(f"\nFinal R-squared (R²): {r2:.4f}")
    print(f"Final Mean Absolute Error (MAE): ${mae:,.2f}")

    # --- 8. Assemble and Save the Final Pipeline for Deployment ---
    print("\nStep 8: Assembling final pipeline and saving...")
    
    # Create the final deployment pipeline, bundling the *already fitted* preprocessor
    # and the *already trained* model. This is for convenience.
    deployment_pipeline = Pipeline(steps=[
        ('preprocessor', preprocessor),
        ('regressor', model)
    ])
    
    model_filename = 'property_price_model_final.joblib'
    joblib.dump(deployment_pipeline, model_filename)
    print(f"Model saved successfully to {model_filename}")

if __name__ == '__main__':
    train_and_evaluate_model()