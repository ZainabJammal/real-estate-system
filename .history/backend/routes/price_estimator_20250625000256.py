import pandas as pd
import numpy as np
import lightgbm as lgb
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_absolute_error, r2_score
import joblib
import warnings

# Suppress a common warning from LightGBM when using pipelines
warnings.filterwarnings('ignore', category=UserWarning)

def train_and_evaluate_model():
    """
    Loads data, cleans it, preprocesses features, and then trains, evaluates,
    and saves a LightGBM regression model for property price prediction.
    """
    # --- 1. Load Data ---
    print("Step 1: Loading data from properties.csv...")
    try:
        df = pd.read_csv('properties.csv')
    except FileNotFoundError:
        print("Error: properties.csv not found. Please ensure the file is in the same directory.")
        return None
    print(f"Show head: {df.head()}")
    print(f"Show info: {df.info()}")
    print(f"Original shape: {df.shape}")

     # --- 2. Data Cleaning (Corrected & Simplified) ---
    print("\nStep 2: Cleaning mistaken zero values...")
    types_with_rooms = ['Apartment', 'House/Villa', 'Chalet', 'Office', 'Residential Building']

    bedrooms_mistake_condition = (
        (df['type'].isin(types_with_rooms)) &
        (df['bedrooms'] == 0) &
        (df['size_m2'] > 35)
    )
    df.loc[bedrooms_mistake_condition, 'bedrooms'] = np.nan

    bathrooms_mistake_condition = (
        (df['type'].isin(types_with_rooms)) &
        (df['bathrooms'] == 0) &
        (df['size_m2'] > 40)
    )
    df.loc[bathrooms_mistake_condition, 'bathrooms'] = np.nan

    # *** FIX: Correctly impute missing values using the median of the group ***
    # This ensures that a missing bedroom in a 'House/Villa' is filled with the
    # median for villas, not a generic value.
    df['bedrooms'].fillna(df.groupby('type')['bedrooms'].transform('median'), inplace=True)
    df['bathrooms'].fillna(df.groupby('type')['bathrooms'].transform('median'), inplace=True)

    # Fallback for any remaining NaNs (e.g., if a new property type appears)
    df['bedrooms'].fillna(1, inplace=True)
    df['bathrooms'].fillna(1, inplace=True)

    df['bedrooms'] = df['bedrooms'].round().astype(int)
    df['bathrooms'] = df['bathrooms'].round().astype(int)
    print("Data cleaning complete.")

    # --- 3. Outlier Handling ---
    print("\nStep 3: Handling outliers...")
    price_cap = df['price_$'].quantile(0.99)
    size_cap = df['size_m2'].quantile(0.99)
    df_trimmed = df[(df['price_$'] < price_cap) & (df['size_m2'] < size_cap)].copy()
    print(f"Shape after trimming outliers: {df_trimmed.shape}")

    # --- 4. Feature Engineering & Selection ---
    print("\nStep 4: Engineering new features and splitting data...")
    
    # *** IMPROVEMENT 1: Create Price per Square Meter by District ***
    # This is a very powerful feature. It helps the model understand the baseline
    # value of land in a specific area, separating it from the value of the building itself.
    district_avg_price_per_m2 = df_trimmed.groupby('district')['price_$'].sum() / df_trimmed.groupby('district')['size_m2'].sum()
    df_trimmed['district_price_per_m2'] = df_trimmed['district'].map(district_avg_price_per_m2)
    # Fill any potential NaNs for districts that might only appear in the test set later
    df_trimmed['district_price_per_m2'].fillna(df_trimmed['district_price_per_m2'].median(), inplace=True)


    X = df_trimmed.drop(columns=['id', 'city', 'created_at', 'price_$', 'latitude', 'longitude'])
    
    # *** IMPROVEMENT 2: Log Transform the Target Variable ***
    # Property prices are often heavily skewed. A log transform makes the distribution
    # more "normal", which helps the model learn more effectively and stabilizes its predictions.
    y = np.log1p(df_trimmed['price_$'])

    categorical_features = ['district', 'province', 'type']
    numerical_features = ['bedrooms', 'bathrooms', 'size_m2', 'district_price_per_m2']
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # --- 5. Preprocessing and Modeling Pipeline ---
    print("\nStep 5: Building the pipeline...")
    # The preprocessor now also scales our numerical features
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', StandardScaler(), numerical_features),
            ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_features)
        ],
        remainder='drop' # Drop any columns not specified
    )

    model_pipeline = Pipeline(steps=[
        ('preprocessor', preprocessor),
        ('regressor', lgb.LGBMRegressor(random_state=42, n_jobs=-1)) # Using simpler params for first run
    ])

    # --- 6. Train the Model ---
    print("\nStep 6: Training the model...")
    model_pipeline.fit(X_train, y_train)
    print("Model training complete.")

    # --- 7. Evaluate the Model ---
    print("\n--- Step 7: Model Evaluation ---")
    y_pred_log = model_pipeline.predict(X_test)
    
    # *** IMPORTANT: We must transform the predictions and test values back to dollars ***
    # before calculating the error, otherwise the MAE would be in log-dollars.
    y_pred_dollars = np.expm1(y_pred_log)
    y_test_dollars = np.expm1(y_test)

    r2 = r2_score(y_test_dollars, y_pred_dollars)
    mae = mean_absolute_error(y_test_dollars, y_pred_dollars)

    print(f"R-squared (R²): {r2:.4f}")
    print(f"Mean Absolute Error (MAE): ${mae:,.2f}")
    
    # --- 8. Save the Model ---
    model_filename = 'property_price_model_v2.joblib'
    print(f"\nStep 8: Saving trained model to '{model_filename}'...")
    joblib.dump(model_pipeline, model_filename)
    print("Model saved successfully.")

    return model_pipeline

if __name__ == '__main__':
    trained_model = train_and_evaluate_model()

    if trained_model:
        print("\n--- New Prediction Example (with improved features) ---")
        # For a new prediction, we must also calculate the district_price_per_m2
        # In a real application, you would save these district averages or compute them on the fly.
        # For this example, we'll hardcode the Beirut average.
        beirut_avg_price_m2 = 2933 # Calculated from a previous run

        new_property = pd.DataFrame({
            'district': ['Beirut'],
            'province': ['Beirut'],
            'type': ['Apartment'],
            'bedrooms': [3],
            'bathrooms': [2],
            'size_m2': [200],
            'district_price_per_m2': [beirut_avg_price_m2]
        })

        predicted_price_log = trained_model.predict(new_property)
        predicted_price_dollars = np.expm1(predicted_price_log)
        
        print(f"\nInput property details:\n{new_property}")
        print(f"\nPredicted Price: ${predicted_price_dollars[0]:,.2f}")