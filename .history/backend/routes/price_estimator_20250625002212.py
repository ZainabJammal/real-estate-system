import pandas as pd
import numpy as np
import lightgbm as lgb
import optuna Zrom sklearn.model_selection import train_test_split, KFold, cross_val_score
from sklearn.cluster import KMeans
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler, FunctionTransformer
from sklearn.metrics import mean_absolute_error, r2_score
import joblib
import warnings

# --- Setup ---
warnings.filterwarnings('ignore', category=UserWarning)
warnings.filterwarnings('ignore', category=FutureWarning)
optuna.logging.set_verbosity(optuna.logging.WARNING) # Keep Optuna's output clean

# Helper for the pipeline
def log_transform(x):
    return np.log1p(x)

# --- Main Training Function ---
def train_and_evaluate_model():
    """
    V5: Professional workflow with per-type outliers, geospatial features,
    and advanced hyperparameter tuning with Optuna.
    """
    # --- 1. Load and Initial Clean ---
    print("Step 1: Loading and Initial Cleaning...")
    df = pd.read_csv('properties.csv')
    df.drop_duplicates(subset=['id'], inplace=True)
    df = df[df['price_$'] > 1000].copy()
    df = df[df['size_m2'] > 10].copy()

    # --- 2. Per-Type Outlier Handling ---
    print("\nStep 2: Handling Outliers Within Each Property Type...")
    
    def remove_outliers_by_group(data, group_col, target_col, lower_q=0.01, upper_q=0.99):
        # This function calculates quantile thresholds for the target_col within each group
        # defined by group_col and removes rows outside these thresholds.
        q_low = data.groupby(group_col)[target_col].transform(lambda x: x.quantile(lower_q))
        q_high = data.groupby(group_col)[target_col].transform(lambda x: x.quantile(upper_q))
        return data[(data[target_col] >= q_low) & (data[target_col] <= q_high)]

    original_rows = len(df)
    df = remove_outliers_by_group(df, 'type', 'price_$')
    df = remove_outliers_by_group(df, 'type', 'size_m2')
    print(f"Removed {original_rows - len(df)} rows as per-type outliers.")
    
    # --- 3. Feature Engineering ---
    print("\nStep 3: Feature Engineering...")
    # Room/Bath Imputation (as before)
    types_with_rooms = ['Apartment', 'House/Villa', 'Chalet', 'Office', 'Residential Building']
    df['bedrooms'] = np.where((df['type'].isin(types_with_rooms)) & (df['bedrooms'] == 0), np.nan, df['bedrooms'])
    df['bathrooms'] = np.where((df['type'].isin(types_with_rooms)) & (df['bathrooms'] == 0), np.nan, df['bathrooms'])
    for col in ['bedrooms', 'bathrooms']:
        df[col].fillna(df.groupby('type')[col].transform('median'), inplace=True)
        df[col].fillna(1, inplace=True)
        df[col] = df[col].astype(int)

    # Price Per M2 (as before)
    district_price_per_m2 = df.groupby('district')['price_$'].sum() / df.groupby('district')['size_m2'].sum()
    df['district_price_per_m2'] = df['district'].map(district_price_per_m2)
    df['district_price_per_m2'].fillna(df['district_price_per_m2'].median(), inplace=True)

    # ** NEW: Geospatial Clustering **
    print("Creating geospatial features from latitude and longitude...")
    kmeans = KMeans(n_clusters=12, random_state=42, n_init=10) # 12 clusters is a good start for Lebanon
    df['location_cluster'] = kmeans.fit_predict(df[['latitude', 'longitude']])
    df['location_cluster'] = df['location_cluster'].astype('category') # Treat it as a category
    
    # --- 4. Data Splitting ---
    print("\nStep 4: Splitting Data...")
    X = df.drop(columns=['id', 'city', 'created_at', 'price_$'])
    y = df['price_$']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    y_train_log = np.log1p(y_train)

    # --- 5. Optuna Objective Function ---
    print("\nStep 5: Defining Optuna Objective for Hyperparameter Search...")
    def objective(trial):
        # Define the hyperparameter search space using your suggestions
        params = {
            'objective': 'regression_l1',
            'metric': 'mae',
            'n_estimators': 1000, # We use a high number and rely on early stopping
            'random_state': 42,
            'n_jobs': -1,
            'learning_rate': trial.suggest_float('learning_rate', 0.02, 0.07),
            'num_leaves': trial.suggest_int('num_leaves', 12, 524),
            'max_depth': trial.suggest_int('max_depth', 4, 6),
            'min_child_samples': trial.suggest_int('min_child_samples', 40, 80),
            'subsample': trial.suggest_float('subsample', 0.7, 0.85),
            'colsample_bytree': trial.suggest_float('colsample_bytree', 0.7, 0.85),
            'reg_lambda': trial.suggest_float('reg_lambda', 0.01, 0.05),
        }
        
        # Define the full pipeline
        numerical_features = ['size_m2', 'latitude', 'longitude', 'district_price_per_m2']
        count_features = ['bedrooms', 'bathrooms']
        categorical_features = ['district', 'province', 'type', 'location_cluster']

        preprocessor = ColumnTransformer(
            transformers=[
                ('num', StandardScaler(), numerical_features),
                ('counts', StandardScaler(), count_features),
                ('cat', OneHotEncoder(handle_unknown='ignore', sparse_output=False), categorical_features)
            ])
        
        model = Pipeline(steps=[
            ('preprocessor', preprocessor),
            ('regressor', lgb.LGBMRegressor(**params))
        ])
        
        # Use cross-validation for robust error estimation
        kf = KFold(n_splits=5, shuffle=True, random_state=42)
        score = cross_val_score(model, X_train, y_train_log, cv=kf, scoring='neg_mean_absolute_error').mean()
        return -score # Optuna minimizes, so we return negative MAE

    # --- 6. Run Optuna Study ---
    print("\nStep 6: Running Optuna Study (this will take a few minutes)...")
    study = optuna.create_study(direction='minimize')
    study.optimize(objective, n_trials=70) # Run 70 trials to find the best params

    print("Study finished!")
    print("Best MAE (on log scale):", study.best_value)
    print("Best parameters found: ", study.best_params)

    # --- 7. Train Final Model with Best Parameters ---
    print("\nStep 7: Training Final Model with Best Parameters...")
    best_params = study.best_params
    best_params['objective'] = 'regression_l1'
    best_params['metric'] = 'mae'
    best_params['random_state'] = 42
    
    # Define the final pipeline with all features
    final_numerical_features = ['size_m2', 'latitude', 'longitude', 'district_price_per_m2']
    final_count_features = ['bedrooms', 'bathrooms']
    final_categorical_features = ['district', 'province', 'type', 'location_cluster']

    final_preprocessor = ColumnTransformer(
        transformers=[
            ('num', StandardScaler(), final_numerical_features),
            ('counts', StandardScaler(), final_count_features),
            ('cat', OneHotEncoder(handle_unknown='ignore', sparse_output=False), final_categorical_features)
        ])
    
    # Use early stopping for the final fit to find optimal n_estimators
    X_train_processed = final_preprocessor.fit_transform(X_train_main)
    X_val_processed = final_preprocessor.transform(X_val)
    
    final_model = lgb.LGBMRegressor(n_estimators=2000, **best_params)
    final_model.fit(X_train_processed, y_train_main_log,
                    eval_set=[(X_val_processed, y_val_log)],
                    eval_metric='mae',
                    callbacks=[lgb.early_stopping(100, verbose=False)])

    # Re-wrap in a full pipeline for easy deployment
    final_pipeline = Pipeline(steps=[
        ('preprocessor', final_preprocessor),
        ('regressor', final_model)
    ])
    
    # --- 8. Final Evaluation ---
    print("\n--- Step 8: Final Model Evaluation on Test Set ---")
    y_pred_log = final_pipeline.predict(X_test)
    y_pred_dollars = np.expm1(y_pred_log)

    r2 = r2_score(y_test, y_pred_dollars)
    mae = mean_absolute_error(y_test, y_pred_dollars)

    print(f"Final R-squared (R²): {r2:.4f}")
    print(f"Final Mean Absolute Error (MAE): ${mae:,.2f}")

    # --- 9. Save the Best Model ---
    model_filename = 'property_price_model_v5_optuna.joblib'
    print(f"\nStep 9: Saving best model to '{model_filename}'...")
    joblib.dump(final_pipeline, model_filename)
    print("Model saved successfully.")

if __name__ == '__main__':
    train_and_evaluate_model()