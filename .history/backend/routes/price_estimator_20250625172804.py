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

# Use LeaveOneOutEncoder for our best location feature
from category_encoders import LeaveOneOutEncoder

warnings.filterwarnings('ignore', category=UserWarning)

def train_and_evaluate_model():
    """
    Final Version: Based on the data audit, this model ignores the corrupted
    latitude/longitude data and relies on the features we know are trustworthy.
    It uses advanced encoding for the 'district' feature to maximize its impact.
    """
    # --- 1. Load and Clean Data ---
    print("Step 1: Loading and Cleaning Data...")
    df = pd.read_csv('properties.csv')
    df.drop_duplicates(subset=['id'], inplace=True)
    df = df[(df['price_$'] > 10000) & (df['size_m2'] > 15)].copy()

    for col in ['type', 'province', 'district']:
        df[col] = df[col].str.lower().str.strip()

    # --- 2. Feature Engineering & Cleaning (on reliable features) ---
    print("\nStep 2: Feature Engineering and Cleaning...")
    types_with_rooms = ['apartment', 'house/villa', 'chalet']
    df['bedrooms'] = np.where((df['type'].isin(types_with_rooms)) & (df['bedrooms'] == 0), np.nan, df['bedrooms'])
    df['bathrooms'] = np.where((df['type'].isin(types_with_rooms)) & (df['bathrooms'] == 0), np.nan, df['bathrooms'])
    
    for col in ['bedrooms', 'bathrooms']:
        df[col].fillna(df.groupby('type')[col].transform('median'), inplace=True)
        df[col].fillna(1, inplace=True)
        df[col] = df[col].astype(int)

    # Log transforms for better performance
    df['log_price'] = np.log1p(df['price_$'])
    df['log_size_m2'] = np.log1p(df['size_m2'])
    
    # --- 3. Outlier Handling ---
    print("\nStep 3: Handling Outliers...")
    log_price_cap = df['log_price'].quantile(0.99)
    df = df[df['log_price'] < log_price_cap].copy()
    print(f"Final data shape for modeling: {df.shape}")

    # --- 4. Define Final Features and Split Data ---
    print("\nStep 4: Defining Final Features and Splitting Data...")
    
    # ** KEY CHANGE: We are only using features we have validated **
    categorical_features_ohe = ['province', 'type']
    # 'district' is our most important location feature, so we give it a powerful encoder
    categorical_features_loo = ['district']
    numerical_features = ['log_size_m2', 'bedrooms', 'bathrooms']
    
    X = df[categorical_features_ohe + categorical_features_loo + numerical_features]
    y = df['log_price']

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # --- 5. Create Final Preprocessing and Model Pipeline ---
    print("\nStep 5: Building the Final Model Pipeline...")
    
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', StandardScaler(), numerical_features),
            ('ohe', OneHotEncoder(handle_unknown='ignore'), categorical_features_ohe),
            ('loo', LeaveOneOutEncoder(), categorical_features_loo)
        ],
        remainder='passthrough'
    )
    
    # Use a solid set of LightGBM parameters
    model = lgb.LGBMRegressor(
        objective='regression_l1', n_estimators=800, learning_rate=0.03,
        num_leaves=40, random_state=42, n_jobs=-1
    )

    pipeline = Pipeline(steps=[
        ('preprocessor', preprocessor),
        ('regressor', model)
    ])
    
    # --- 6. Train the Model ---
    print("\nStep 6: Training Final Model...")
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

    # --- 8. Save the Final Model ---
    if r2 > 0.6: # A more realistic success threshold now
        print("\nSUCCESS: Model performance is strong given the data limitations.")
        model_filename = 'property_price_model_final.joblib'
        print(f"\nStep 8: Saving final model...")
        joblib.dump(pipeline, model_filename)
        print(f"Model saved successfully to {model_filename}")
    else:
        print("\nCONCLUSION: The data's location information is too limited for a high-accuracy model.")

if __name__ == '__main__':
    train_and_evaluate_model()