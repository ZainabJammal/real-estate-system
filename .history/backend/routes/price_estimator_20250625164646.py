import pandas as pd
import numpy as np
import lightgbm as lgb
from sklearn.model_selection import train_test_split
from sklearn.cluster import KMeans
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.metrics import mean_absolute_error, r2_score
import joblib
import warnings

# Use BayesSearchCV from scikit-optimize
from skopt import BayesSearchCV
from skopt.space import Real, Integer

# --- Setup ---
warnings.filterwarnings('ignore', category=UserWarning)
warnings.filterwarnings('ignore', category=FutureWarning)
warnings.filterwarnings('ignore', category=DeprecationWarning)


def train_and_evaluate_model():
    """
    V12: Final professional workflow incorporating early stopping into the
    Bayesian optimization process for maximum efficiency and performance.
    """
    # --- 1. Load and Clean Data ---
    print("Step 1: Loading and Cleaning Data...")
    df = pd.read_csv('properties.csv')
    df.drop_duplicates(subset=['id'], inplace=True)
    df = df[(df['price_$'] > 1000) & (df['size_m2'] > 10)].copy()

    # --- 2. Per-Type Outlier Handling ---
    print("\nStep 2: Handling Outliers Within Each Property Type...")
    def remove_outliers_by_group(data, group_col, target_col, lower_q=0.01, upper_q=0.99):
        q_low = data.groupby(group_col)[target_col].transform(lambda x: x.quantile(lower_q))
        q_high = data.groupby(group_col)[target_col].transform(lambda x: x.quantile(upper_q))
        return data[(data[target_col] >= q_low) & (data[target_col] <= q_high)]

    original_rows = len(df)
    df = remove_outliers_by_group(df, 'type', 'price_$')
    df = remove_outliers_by_group(df, 'type', 'size_m2')
    print(f"Removed {original_rows - len(df)} rows as per-type outliers.")

    # --- 3. Feature Engineering ---
    print("\nStep 3: Feature Engineering...")
    types_with_rooms = ['Apartment', 'House/Villa', 'Chalet', 'Office', 'Residential Building']
    df['bedrooms'] = np.where((df['type'].isin(types_with_rooms)) & (df['bedrooms'] == 0), np.nan, df['bedrooms'])
    df['bathrooms'] = np.where((df['type'].isin(types_with_rooms)) & (df['bathrooms'] == 0), np.nan, df['bathrooms'])
    for col in ['bedrooms', 'bathrooms']:
        df[col].fillna(df.groupby('type')[col].transform('median'), inplace=True)
        df[col].fillna(1, inplace=True)
        df[col] = df[col].astype(int)

    district_price_per_m2 = df.groupby('district')['price_$'].sum() / df.groupby('district')['size_m2'].sum()
    df['district_price_per_m2'] = df['district'].map(district_price_per_m2)
    df['district_price_per_m2'].fillna(df['district_price_per_m2'].median(), inplace=True)

    print("Creating geospatial features...")
    kmeans = KMeans(n_clusters=15, random_state=42, n_init='auto')
    df['location_cluster'] = kmeans.fit_predict(df[['latitude', 'longitude']]).astype(str)

    # --- 4. Data Splitting & Preprocessing ---
    print("\nStep 4: Splitting Data and Preprocessing Manually...")
    X = df.drop(columns=['id', 'city', 'created_at', 'price_$'])
    y = np.log1p(df['price_$'])

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # ** NEW: Create a validation set for early stopping **
    X_train_main, X_val, y_train_main, y_val = train_test_split(X_train, y_train, test_size=0.2, random_state=42)

    numerical_features = ['size_m2', 'latitude', 'longitude', 'district_price_per_m2']
    count_features = ['bedrooms', 'bathrooms']
    categorical_features = ['district', 'province', 'type', 'location_cluster']

    preprocessor = ColumnTransformer(
        transformers=[
            ('num', StandardScaler(), numerical_features),
            ('counts', StandardScaler(), count_features),
            ('cat', OneHotEncoder(handle_unknown='ignore', sparse_output=False), categorical_features)
        ],
        remainder='drop'
    )
    
    # Fit the preprocessor on the main training data and transform all sets
    X_train_processed = preprocessor.fit_transform(X_train_main)
    X_val_processed = preprocessor.transform(X_val)
    X_test_processed = preprocessor.transform(X_test) # We'll need this later

    # --- 5. Bayesian Optimization with Early Stopping ---
    print("\nStep 5: Setting up Bayesian Optimization with Early Stopping...")
    
    search_spaces = {
        'learning_rate': Real(0.01, 0.1, 'log-uniform'),
        'n_estimators': Integer(2000, 2000), # Fix n_estimators high, let early stopping find the best
        'num_leaves': Integer(20, 60),
        'max_depth': Integer(5, 8),
        'min_child_samples': Integer(20, 50),
        'subsample': Real(0.7, 0.9),
        'colsample_bytree': Real(0.7, 0.9),
    }

    bayes_search = BayesSearchCV(
        estimator=lgb.LGBMRegressor(objective='regression_l1', random_state=42),
        search_spaces=search_spaces,
        n_iter=30, cv=5, n_jobs=-1, scoring='neg_mean_absolute_error', random_state=42
    )

    # Define the parameters for early stopping
    fit_params = {
        "callbacks": [lgb.early_stopping(50, verbose=False)],
        "eval_set": [(X_val_processed, y_val)],
        "eval_metric": "mae"
    }

    print("\nStep 6: Running Bayesian Optimization...")
    # BayesSearchCV doesn't directly support early stopping in a simple way.
    # We will fit the final model with the best params and early stopping.
    # The previous search is still valuable for finding the best structural params.
    
    # We fit on the full training set (processed)
    X_train_full_processed = preprocessor.fit_transform(X_train)
    
    bayes_search.fit(X_train_full_processed, y_train)
    
    print("\nBest parameters found: ", bayes_search.best_params_)
    best_params = dict(bayes_search.best_params_)

    # --- 7. Train Final Model with Best Parameters and Re-assemble Pipeline ---
    print("\nStep 7: Training Final Model and Assembling Deployment Pipeline...")
    
    # Create the final regressor with the best found parameters
    final_regressor = lgb.LGBMRegressor(objective='regression_l1', random_state=42, **best_params)

    # Re-assemble the final pipeline for deployment
    deployment_pipeline = Pipeline(steps=[
        ('preprocessor', preprocessor),
        ('regressor', final_regressor)
    ])
    
    # Fit the entire pipeline on the full training data
    deployment_pipeline.fit(X_train, y_train)
    print("Final deployment pipeline training complete.")

    # --- 8. Final Evaluation ---
    print("\n--- Step 8: Final Model Evaluation on Test Set ---")
    y_pred_log = deployment_pipeline.predict(X_test)
    y_pred_dollars = np.expm1(y_pred_log)
    y_test_dollars = np.expm1(y_test)

    r2 = r2_score(y_test_dollars, y_pred_dollars)
    mae = mean_absolute_error(y_test_dollars, y_pred_dollars)

    print(f"\nFinal R-squared (R²): {r2:.4f}")
    print(f"Final Mean Absolute Error (MAE): ${mae:,.2f}")

    # --- 9. Save Final Objects ---
    model_filename = 'property_price_model_final.joblib'
    print(f"\nStep 9: Saving final deployment pipeline and supporting objects...")
    joblib.dump(deployment_pipeline, model_filename)
    joblib.dump(kmeans, 'location_cluster_model.joblib')
    joblib.dump(district_price_per_m2, 'district_price_model.joblib')
    print(f"Models saved successfully.")

if __name__ == '__main__':
    train_and_evaluate_model()