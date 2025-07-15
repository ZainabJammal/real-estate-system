import pandas as pd
import numpy as np
import lightgbm as lgb
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_absolute_error, r2_score
import joblib
import warnings

warnings.filterwarnings('ignore', category=UserWarning)

def train_and_save_model():
    """
    Loads, cleans, trains, and saves the property price prediction model from a local CSV.
    """
    # --- Load Data ---
    print("Loading properties.csv...")
    try:
        df = pd.read_csv('properties.csv')
    except FileNotFoundError:
        print("Error: 'properties.csv' not found. Make sure it's in the same directory.")
        return

    # --- Data Cleaning: Fix Mistaken Zeros ---
    types_with_rooms = ['Apartment', 'House/Villa', 'Chalet', 'Office', 'Residential Building']
    bedrooms_mistake_condition = (
        (df['type'].isin(types_with_rooms)) &
        (df['bedrooms'] == 0) &
        (df['size_m2'] > 35)
    )
    df.loc[bedrooms_mistake_condition, 'bedrooms'] = np.nan
    df['bedrooms'].fillna(df.groupby('type')['bedrooms'].transform('median'), inplace=True)
    df['bedrooms'].fillna(1, inplace=True)
    df['bedrooms'] = df['bedrooms'].round().astype(int)
    
    # --- Outlier Handling ---
    price_cap = df['price_$'].quantile(0.99)
    size_cap = df['size_m2'].quantile(0.99)
    df_trimmed = df[(df['price_$'] < price_cap) & (df['size_m2'] < size_cap)].copy()

    # --- Feature Engineering and Splitting ---
    X = df_trimmed.drop(columns=['id', 'city', 'created_at', 'price_$', 'latitude', 'longitude'])
    y = df_trimmed['price_$']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # --- Preprocessing and Modeling Pipeline ---
    categorical_features = ['district', 'province', 'type']
    preprocessor = ColumnTransformer(
        transformers=[('cat', OneHotEncoder(handle_unknown='ignore'), categorical_features)],
        remainder='passthrough'
    )
    lgbm_regressor = lgb.LGBMRegressor(objective='regression_l1', random_state=42, n_estimators=1000)
    model_pipeline = Pipeline(steps=[('preprocessor', preprocessor), ('regressor', lgbm_regressor)])

    # --- Train and Evaluate ---
    print("Training the model...")
    model_pipeline.fit(X_train, y_train)
    y_pred = model_pipeline.predict(X_test)
    r2 = r2_score(y_test, y_pred)
    mae = mean_absolute_error(y_test, y_pred)
    print(f"Model Training Complete. R²: {r2:.4f}, MAE: ${mae:,.2f}")

    # --- Save the Model ---
    model_filename = 'property_price_model.joblib'
    print(f"Saving model to '{model_filename}'...")
    joblib.dump(model_pipeline, model_filename)
    print("Model saved successfully.")

if __name__ == '__main__':
    train_and_save_model()