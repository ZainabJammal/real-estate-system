import pandas as pd
import numpy as np
import lightgbm as lgb
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_absolute_error, r2_score, mean_squared_error
import joblib
import warnings
import traceback

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
    print(f"Show head: {np.info()}")
    print(f"Original shape: {df.shape}")

    # --- 2. Data Cleaning: Fix Mistaken Zeros ---
    # This is a critical domain-specific step to ensure data integrity.
    print("\nStep 2: Cleaning mistaken zero values in 'bedrooms' and 'bathrooms'...")
    types_with_rooms = ['Apartment', 'House/Villa', 'Chalet', 'Office', 'Residential Building']

    # Condition for likely mistaken zero bedrooms
    bedrooms_mistake_condition = (
        (df['type'].isin(types_with_rooms)) &
        (df['bedrooms'] == 0) &
        (df['size_m2'] > 35) # A property >35m² is unlikely to be a studio/have 0 rooms
    )
    print(f"Found {bedrooms_mistake_condition.sum()} rows with likely mistaken zero bedrooms.")
    df.loc[bedrooms_mistake_condition, 'bedrooms'] = np.nan

    # Condition for likely mistaken zero bathrooms
    bathrooms_mistake_condition = (
        (df['type'].isin(types_with_rooms)) &
        (df['bathrooms'] == 0) &
        (df['size_m2'] > 40) # A property >40m² is unlikely to have zero bathrooms
    )
    print(f"Found {bathrooms_mistake_condition.sum()} rows with likely mistaken zero bathrooms.")
    df.loc[bathrooms_mistake_condition, 'bathrooms'] = np.nan

    # Impute the created NaN values using the median for that property type.
#     # Using transform preserves the original index, making it safe for .fillna()
#     df['bedrooms'].fillna(df.groupby('type')['bedrooms'].transform('median'), inplace=True)
#     df['bathrooms'].fillna(df.groupby('type')['bathrooms'].transform('median'), inplace=True)
    df['bedrooms'] = df['bedrooms'].fillna(df.groupby('type')['bedrooms'].transform('median'))
    df['bathrooms'] = df['bathrooms'].fillna(df.groupby('type')['bathrooms'].transform('median'))
    df['bedrooms'] = df['bedrooms'].fillna(1)
    df['bathrooms'] = df['bathrooms'].fillna(1)


    # Fill any remaining NaNs (e.g., for a type that only had NaN values) with 1
    df['bedrooms'].fillna(1, inplace=True)
    df['bathrooms'].fillna(1, inplace=True)

    # Convert to integer as they are discrete counts
    df['bedrooms'] = df['bedrooms'].round().astype(int)
    df['bathrooms'] = df['bathrooms'].round().astype(int)
    print("Data cleaning complete.")

    # --- 3. Outlier Handling ---
    print("\nStep 3: Handling outliers...")
    # Removing the top 1% of prices and sizes creates a more stable model
    # that is less skewed by multi-million dollar mansions or huge plots of land.
    price_cap = df['price_$'].quantile(0.99)
    size_cap = df['size_m2'].quantile(0.99)

    original_rows = len(df)
    df_trimmed = df[(df['price_$'] < price_cap) & (df['size_m2'] < size_cap)].copy()
    rows_removed = original_rows - len(df_trimmed)
    print(f"Removed {rows_removed} rows as outliers (top 1% of price/size).")
    print(f"Shape after trimming: {df_trimmed.shape}")

    # --- 4. Feature Selection and Data Splitting ---
    print("\nStep 4: Defining features and splitting data...")
    # 'city' is often redundant with 'district' and has higher cardinality.
    # 'latitude'/'longitude' can be powerful but add complexity; we omit them for this model.
    X = df_trimmed.drop(columns=['id', 'city', 'created_at', 'price_$', 'latitude', 'longitude'])
    y = df_trimmed['price_$']

    categorical_features = ['district', 'province', 'type']
    
    # Split data into training and testing sets for model evaluation
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    print(f"Training set size: {X_train.shape[0]} samples")
    print(f"Testing set size: {X_test.shape[0]} samples")

    # --- 5. Preprocessing and Modeling Pipeline ---
    print("\nStep 5: Building the preprocessing and modeling pipeline...")
    # The preprocessor handles categorical features by converting them to a numerical format (one-hot encoding)
    preprocessor = ColumnTransformer(
        transformers=[
            ('cat', OneHotEncoder(handle_unknown='ignore', sparse=False), categorical_features)
        ],
        remainder='passthrough' # Keep numerical features (bedrooms, bathrooms, size_m2) as they are
    )

    # Define the LightGBM model with chosen hyperparameters
    # These are solid starting parameters, selected for a balance of speed and accuracy.
    # 'objective':'regression_l1' (Mean Absolute Error) is robust to remaining price outliers.
    lgbm_regressor = lgb.LGBMRegressor(
        objective='regression_l1',
        n_estimators=1500,
        learning_rate=0.03,
        num_leaves=31,
        max_depth=-1,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42,
        n_jobs=-1
    )

    # The pipeline chains the preprocessing and the model into a single object
    model_pipeline = Pipeline(steps=[
        ('preprocessor', preprocessor),
        ('regressor', lgbm_regressor)
    ])

    # --- 6. Train the Model ---
    print("\nStep 6: Training the LightGBM model...")
    model_pipeline.fit(X_train, y_train)
    print("Model training complete.")

    # --- 7. Evaluate the Model ---
    print("\n--- Step 7: Model Evaluation ---")
    y_pred = model_pipeline.predict(X_test)

    r2 = r2_score(y_test, y_pred)
    mae = mean_absolute_error(y_test, y_pred)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))

    print(f"R-squared (R²): {r2:.4f}")
    print(f"Mean Absolute Error (MAE): ${mae:,.2f}")
    print(f"Root Mean Squared Error (RMSE): ${rmse:,.2f}")
    print("-" * 30)
    print("Metric Explanations:")
    print(f"  - R²: The model explains {r2:.1%} of the variance in property prices in the test set.")
    print(f"  - MAE: On average, the model's price prediction is off by ${mae:,.2f}.")
    print("  - RMSE: Similar to MAE but gives a higher penalty to large prediction errors.")

    # --- 8. Save the Trained Model Pipeline ---
    model_filename = 'property_price_model.joblib'
    print(f"\nStep 8: Saving trained model pipeline to '{model_filename}'...")
    joblib.dump(model_pipeline, model_filename)
    print("Model saved successfully.")

    return model_pipeline

if __name__ == '__main__':
    # Train the model when the script is run directly
    trained_model = train_and_evaluate_model()

    if trained_model:
        # --- Example of making a new prediction ---
        print("\n--- New Prediction Example ---")
        new_property = pd.DataFrame({
            'district': ['Beirut'],
            'province': ['Beirut'],
            'type': ['Apartment'],
            'bedrooms': [3],
            'bathrooms': [2],
            'size_m2': [200]
        })

        predicted_price = trained_model.predict(new_property)
        print(f"\nInput property details:\n{new_property}")
        print(f"\nPredicted Price: ${predicted_price[0]:,.2f}")
# import pandas as pd
# import numpy as np
# import lightgbm as lgb
# from sklearn.model_selection import train_test_split, GridSearchCV
# from sklearn.compose import ColumnTransformer
# from sklearn.preprocessing import OneHotEncoder, StandardScaler
# from sklearn.pipeline import Pipeline
# from sklearn.impute import KNNImputer
# from sklearn.metrics import mean_absolute_error, r2_score, mean_squared_error
# import joblib
# import warnings

# warnings.filterwarnings('ignore', category=UserWarning)

# def train_and_evaluate_model():
#     print("Step 1: Loading data from properties.csv...")
#     try:
#         df = pd.read_csv('properties.csv')
#     except FileNotFoundError:
#         print("Error: properties.csv not found.")
#         return None

#     # --- Fix obviously wrong zeros ---
#     types_with_rooms = ['Apartment', 'House/Villa', 'Chalet', 'Office', 'Residential Building']

#     df.loc[(df['type'].isin(types_with_rooms)) & (df['bedrooms'] == 0) & (df['size_m2'] > 35), 'bedrooms'] = np.nan
#     df.loc[(df['type'].isin(types_with_rooms)) & (df['bathrooms'] == 0) & (df['size_m2'] > 40), 'bathrooms'] = np.nan

#     # --- Outlier Removal ---
#     price_cap = df['price_$'].quantile(0.99)
#     size_cap = df['size_m2'].quantile(0.99)
#     df = df[(df['price_$'] < price_cap) & (df['size_m2'] < size_cap)].copy()

#     # --- Define features and label ---
#     features = ['district', 'province', 'type', 'bedrooms', 'bathrooms', 'size_m2', 'latitude', 'longitude']
#     target = 'price_$'
#     X = df[features]
#     y = df[target]

#     categorical_features = ['district', 'province', 'type']
#     numerical_features = ['bedrooms', 'bathrooms', 'size_m2', 'latitude', 'longitude']

#     X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

#     # --- Preprocessing ---
#     preprocessor = ColumnTransformer(
#         transformers=[
#             ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_features),
#             ('num', Pipeline(steps=[
#                 ('imputer', KNNImputer(n_neighbors=5)),
#                 ('scaler', StandardScaler())
#             ]), numerical_features)
#         ]
#     )

#     # --- Model + Grid Search ---
#     lgbm = lgb.LGBMRegressor(objective='regression_l1', random_state=42)
#     param_grid = {
#         'regressor__n_estimators': [500, 1000],
#         'regressor__learning_rate': [0.01, 0.03],
#         'regressor__num_leaves': [31, 63],
#     }

#     pipe = Pipeline([
#         ('preprocessor', preprocessor),
#         ('regressor', lgbm)
#     ])

#     grid = GridSearchCV(pipe, param_grid, cv=3, scoring='neg_mean_absolute_error', n_jobs=-1, verbose=1)
#     grid.fit(X_train, y_train)

#     best_model = grid.best_estimator_

#     # --- Evaluation ---
#     y_pred = best_model.predict(X_test)
#     print("R²:", r2_score(y_test, y_pred))
#     print("MAE:", mean_absolute_error(y_test, y_pred))
#     print("RMSE:", np.sqrt(mean_squared_error(y_test, y_pred)))

#     # --- Save ---
#     joblib.dump(best_model, 'enhanced_property_price_model.joblib')
#     print("Model saved as 'enhanced_property_price_model.joblib'")

#     return best_model

# if __name__ == '__main__':
#     model = train_and_evaluate_model()
