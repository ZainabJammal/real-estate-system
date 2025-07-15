# import pandas as pd
# import numpy as np
# import lightgbm as lgb
# from sklearn.model_selection import train_test_split
# from sklearn.compose import ColumnTransformer
# from sklearn.preprocessing import OneHotEncoder, StandardScaler
# from sklearn.pipeline import Pipeline
# from sklearn.metrics import mean_absolute_error, r2_score
# import joblib
# import warnings

# # Suppress a common warning from LightGBM when using pipelines
# warnings.filterwarnings('ignore', category=UserWarning)

# def train_and_evaluate_model():
#     """
#     Loads data, cleans it, preprocesses features, and then trains, evaluates,
#     and saves a LightGBM regression model for property price prediction.
#     """
#     # --- 1. Load Data ---
#     print("Step 1: Loading data from properties.csv...")
#     try:
#         df = pd.read_csv('properties.csv')
#     except FileNotFoundError:
#         print("Error: properties.csv not found. Please ensure the file is in the same directory.")
#         return None
#     print(f"Show head: {df.head()}")
#     print(f"Show info: {df.info()}")
#     print(f"Original shape: {df.shape}")

#      # --- 2. Data Cleaning (Corrected & Simplified) ---
#     print("\nStep 2: Cleaning mistaken zero values...")
#     types_with_rooms = ['Apartment', 'House/Villa', 'Chalet', 'Office', 'Residential Building']

#     bedrooms_mistake_condition = (
#         (df['type'].isin(types_with_rooms)) &
#         (df['bedrooms'] == 0) &
#         (df['size_m2'] > 35)
#     )
#     df.loc[bedrooms_mistake_condition, 'bedrooms'] = np.nan

#     bathrooms_mistake_condition = (
#         (df['type'].isin(types_with_rooms)) &
#         (df['bathrooms'] == 0) &
#         (df['size_m2'] > 40)
#     )
#     df.loc[bathrooms_mistake_condition, 'bathrooms'] = np.nan

#     # *** FIX: Correctly impute missing values using the median of the group ***
#     # This ensures that a missing bedroom in a 'House/Villa' is filled with the
#     # median for villas, not a generic value.
#     df['bedrooms'].fillna(df.groupby('type')['bedrooms'].transform('median'), inplace=True)
#     df['bathrooms'].fillna(df.groupby('type')['bathrooms'].transform('median'), inplace=True)

#     # Fallback for any remaining NaNs (e.g., if a new property type appears)
#     df['bedrooms'].fillna(1, inplace=True)
#     df['bathrooms'].fillna(1, inplace=True)

#     df['bedrooms'] = df['bedrooms'].round().astype(int)
#     df['bathrooms'] = df['bathrooms'].round().astype(int)
#     print("Data cleaning complete.")

#     # --- 3. Outlier Handling ---
#     print("\nStep 3: Handling outliers...")
#     price_cap = df['price_$'].quantile(0.99)
#     size_cap = df['size_m2'].quantile(0.99)
#     df_trimmed = df[(df['price_$'] < price_cap) & (df['size_m2'] < size_cap)].copy()
#     print(f"Shape after trimming outliers: {df_trimmed.shape}")

#     # --- 4. Feature Engineering & Selection ---
#     print("\nStep 4: Engineering new features and splitting data...")
    
#     # *** IMPROVEMENT 1: Create Price per Square Meter by District ***
#     # This is a very powerful feature. It helps the model understand the baseline
#     # value of land in a specific area, separating it from the value of the building itself.
#     district_avg_price_per_m2 = df_trimmed.groupby('district')['price_$'].sum() / df_trimmed.groupby('district')['size_m2'].sum()
#     df_trimmed['district_price_per_m2'] = df_trimmed['district'].map(district_avg_price_per_m2)
#     # Fill any potential NaNs for districts that might only appear in the test set later
#     df_trimmed['district_price_per_m2'].fillna(df_trimmed['district_price_per_m2'].median(), inplace=True)


#     X = df_trimmed.drop(columns=['id', 'city', 'created_at', 'price_$', 'latitude', 'longitude'])
    
#     # *** IMPROVEMENT 2: Log Transform the Target Variable ***
#     # Property prices are often heavily skewed. A log transform makes the distribution
#     # more "normal", which helps the model learn more effectively and stabilizes its predictions.
#     y = np.log1p(df_trimmed['price_$'])

#     categorical_features = ['district', 'province', 'type']
#     numerical_features = ['bedrooms', 'bathrooms', 'size_m2', 'district_price_per_m2']
    
#     X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

#     # --- 5. Preprocessing and Modeling Pipeline ---
#     print("\nStep 5: Building the pipeline...")
#     # The preprocessor now also scales our numerical features
#     preprocessor = ColumnTransformer(
#         transformers=[
#             ('num', StandardScaler(), numerical_features),
#             ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_features)
#         ],
#         remainder='drop' # Drop any columns not specified
#     )

#     model_pipeline = Pipeline(steps=[
#         ('preprocessor', preprocessor),
#         ('regressor', lgb.LGBMRegressor(random_state=42, n_jobs=-1)) # Using simpler params for first run
#     ])

#     # --- 6. Train the Model ---
#     print("\nStep 6: Training the model...")
#     model_pipeline.fit(X_train, y_train)
#     print("Model training complete.")

#     # --- 7. Evaluate the Model ---
#     print("\n--- Step 7: Model Evaluation ---")
#     y_pred_log = model_pipeline.predict(X_test)
    
#     # *** IMPORTANT: We must transform the predictions and test values back to dollars ***
#     # before calculating the error, otherwise the MAE would be in log-dollars.
#     y_pred_dollars = np.expm1(y_pred_log)
#     y_test_dollars = np.expm1(y_test)

#     r2 = r2_score(y_test_dollars, y_pred_dollars)
#     mae = mean_absolute_error(y_test_dollars, y_pred_dollars)

#     print(f"R-squared (R²): {r2:.4f}")
#     print(f"Mean Absolute Error (MAE): ${mae:,.2f}")
    
#     # --- 8. Save the Model ---
#     model_filename = 'property_price_model_v2.joblib'
#     print(f"\nStep 8: Saving trained model to '{model_filename}'...")
#     joblib.dump(model_pipeline, model_filename)
#     print("Model saved successfully.")

#     return model_pipeline

# if __name__ == '__main__':
#     trained_model = train_and_evaluate_model()

#     if trained_model:
#         print("\n--- New Prediction Example (with improved features) ---")
#         # For a new prediction, we must also calculate the district_price_per_m2
#         # In a real application, you would save these district averages or compute them on the fly.
#         # For this example, we'll hardcode the Beirut average.
#         beirut_avg_price_m2 = 2933 # Calculated from a previous run

#         new_property = pd.DataFrame({
#             'district': ['Beirut'],
#             'province': ['Beirut'],
#             'type': ['Apartment'],
#             'bedrooms': [3],
#             'bathrooms': [2],
#             'size_m2': [200],
#             'district_price_per_m2': [beirut_avg_price_m2]
#         })

#         predicted_price_log = trained_model.predict(new_property)
#         predicted_price_dollars = np.expm1(predicted_price_log)
        
#         print(f"\nInput property details:\n{new_property}")
#         print(f"\nPredicted Price: ${predicted_price_dollars[0]:,.2f}")


import pandas as pd
import numpy as np
import lightgbm as lgb
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler, FunctionTransformer
from sklearn.metrics import mean_absolute_error, r2_score
import joblib
import warnings

warnings.filterwarnings('ignore', category=UserWarning)

# Use a function for the log transformation to fit into the pipeline
def log_transform(x):
    return np.log1p(x)

def train_and_evaluate_model():
    """
    Loads, cleans, and trains a high-performance model with robust feature
    engineering, log transforms, and hyperparameter tuning.
    """
    # --- 1. Load and Initial Clean ---
    print("Step 1: Loading and Cleaning Data...")
    df = pd.read_csv('properties.csv')
    
    # Simple cleaning for irrelevant/problematic columns
    df.drop_duplicates(subset=['city', 'district', 'type', 'bedrooms', 'price_$', 'size_m2'], inplace=True)
    df = df[df['price_$'] > 1000] # Remove junk listings
    df = df[df['size_m2'] > 10]

    # --- 2. Feature Engineering ---
    print("\nStep 2: Feature Engineering...")
    # Fix mistaken zeros for rooms based on size
    types_with_rooms = ['Apartment', 'House/Villa', 'Chalet', 'Office', 'Residential Building']
    df['bedrooms'] = np.where((df['type'].isin(types_with_rooms)) & (df['bedrooms'] == 0) & (df['size_m2'] > 35), np.nan, df['bedrooms'])
    df['bathrooms'] = np.where((df['type'].isin(types_with_rooms)) & (df['bathrooms'] == 0) & (df['size_m2'] > 40), np.nan, df['bathrooms'])
    
    # Impute missing values using the group median
    for col in ['bedrooms', 'bathrooms']:
        df[col].fillna(df.groupby('type')[col].transform('median'), inplace=True)
        df[col].fillna(1, inplace=True) # Fallback
        df[col] = df[col].astype(int)

    # Create price_per_m2 feature
    district_price_per_m2 = df.groupby('district')['price_$'].sum() / df.groupby('district')['size_m2'].sum()
    df['district_price_per_m2'] = df['district'].map(district_price_per_m2)
    df['district_price_per_m2'].fillna(df['district_price_per_m2'].median(), inplace=True)
    
    # --- 3. Outlier and Data Splitting ---
    print("\nStep 3: Handling Outliers and Splitting Data...")
    price_cap = df['price_$'].quantile(0.99)
    size_cap = df['size_m2'].quantile(0.99)
    df_trimmed = df[(df['price_$'] < price_cap) & (df['size_m2'] < size_cap)].copy()

    X = df_trimmed.drop(columns=['id', 'city', 'created_at', 'price_$', 'latitude', 'longitude'])
    y = df_trimmed['price_$']
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Log transform the target variable for better performance
    y_train_log = np.log1p(y_train)
    y_test_log = np.log1p(y_test)
    
    # --- 4. Advanced Preprocessing Pipeline ---
    print("\nStep 4: Building Advanced Preprocessing Pipeline...")
    
    # Define features by type
    numerical_features = ['size_m2', 'district_price_per_m2']
    count_features = ['bedrooms', 'bathrooms'] # Treat counts differently from continuous measures
    categorical_features = ['district', 'province', 'type']

    # Create a pipeline for log-transforming and scaling numerical features
    numerical_transformer = Pipeline(steps=[
        ('log_transformer', FunctionTransformer(log_transform)),
        ('scaler', StandardScaler())
    ])

    # Create a preprocessor that applies the correct transformation to each column type
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numerical_transformer, numerical_features),
            ('counts', StandardScaler(), count_features), # Just scale the counts
            ('cat', OneHotEncoder(handle_unknown='ignore', sparse_output=False), categorical_features)
        ],
        remainder='drop'
    )
    
    # --- 5. Hyperparameter Tuning & Training ---
    print("\nStep 5: Setting up Hyperparameter Tuning...")
    
    # Define the final pipeline with the model
    full_pipeline = Pipeline(steps=[
        ('preprocessor', preprocessor),
        ('regressor', lgb.LGBMRegressor(objective='regression_l1', random_state=42))
    ])

    # Define a parameter grid for GridSearchCV to search over
    # This is a focused search space to save time.
    param_grid = {
        'regressor__n_estimators': [400, 700, 1000],
        'regressor__learning_rate': [0.03, 0.05],
        'regressor__num_leaves': [31, 40],
        'regressor__colsample_bytree': [0.7, 0.8]
    }
    
    # Setup GridSearchCV
    # cv=3 means 3-fold cross-validation. n_jobs=-1 uses all CPU cores.
    # scoring='neg_mean_absolute_error' because grid search maximizes a score, so we use the negative MAE.
    grid_search = GridSearchCV(
        full_pipeline, param_grid, cv=3,
        scoring='neg_mean_absolute_error',
        n_jobs=-1, verbose=1
    )

    print("Starting model training and tuning (this may take a few minutes)...")
    grid_search.fit(X_train, y_train_log)

    print("\nBest parameters found: ", grid_search.best_params_)
    best_model = grid_search.best_estimator_
    
    # --- 6. Final Evaluation ---
    print("\n--- Step 6: Final Model Evaluation ---")
    y_pred_log = best_model.predict(X_test)
    y_pred_dollars = np.expm1(y_pred_log) # Convert back from log scale

    r2 = r2_score(y_test, y_pred_dollars)
    mae = mean_absolute_error(y_test, y_pred_dollars)

    print(f"Final R-squared (R²): {r2:.4f}")
    print(f"Final Mean Absolute Error (MAE): ${mae:,.2f}")

    # --- 7. Feature Importance ---
    print("\n--- Step 7: Feature Importance Analysis ---")
    # Get feature names after one-hot encoding
    ohe_feature_names = best_model.named_steps['preprocessor'].named_transformers_['cat'].get_feature_names_out(categorical_features)
    final_feature_names = numerical_features + count_features + list(ohe_feature_names)
    
    # Get importances from the trained model
    importances = best_model.named_steps['regressor'].feature_importances_
    
    feature_importance_df = pd.DataFrame({
        'feature': final_feature_names,
        'importance': importances
    }).sort_values('importance', ascending=False)
    
    print("Top 10 most important features:")
    print(feature_importance_df.head(10))

    # --- 8. Save the Best Model ---
    model_filename = 'property_price_model_v3.joblib'
    print(f"\nStep 8: Saving best model to '{model_filename}'...")
    joblib.dump(best_model, model_filename)
    print("Model saved successfully.")

if __name__ == '__main__':
    train_and_evaluate_model()