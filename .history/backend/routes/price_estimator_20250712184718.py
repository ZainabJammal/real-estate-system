
# # # MODEL_CONFIG = {
# # #     'apartment': {
# # #         'params': {'depth': 8, 'learning_rate': 0.03, 'l2_leaf_reg': 5, 'iterations': 2000},
# # #         'features': {'numerical': ['log_size_m2', 'bedrooms', 'bathrooms'], 'categorical': ['province', 'district', 'city']}
# # #     },
# # #     'land': {
# # #         'params': {'depth': 6, 'learning_rate': 0.01, 'l2_leaf_reg': 10, 'iterations': 2000},
# # #         'features': {'numerical': ['log_size_m2'], 'categorical': ['province', 'district', 'city']}
# # #     },
# # #     'house/villa': {
# # #         'params': {'depth': 7, 'learning_rate': 0.02, 'l2_leaf_reg': 3, 'iterations': 2000},
# # #         'features': {'numerical': ['log_size_m2', 'bedrooms', 'bathrooms'], 'categorical': ['province', 'district', 'city']}
# # #     },
# # #  
# # # }
    
# import pandas as pd
# import numpy as np
# import lightgbm as lgb
# import matplotlib.pyplot as plt
# import joblib
# import warnings
# import os
# from sklearn.model_selection import KFold, cross_val_score
# from sklearn.pipeline import Pipeline
# from sklearn.compose import ColumnTransformer
# from sklearn.preprocessing import StandardScaler, OneHotEncoder
# from sklearn.impute import SimpleImputer

# # --- 1. SETUP & CONFIGURATION (Unchanged) ---
# def setup_environment():
#     """Sets up directories and configurations for the script."""
#     warnings.filterwarnings('ignore', category=UserWarning)
#     pd.options.mode.chained_assignment = None
    
#     # Get the absolute path of the currently running script (e.g., .../backend/routes/your_script.py)
#     script_path = os.path.abspath(__file__)
    
#     # Get the directory the script is in (e.g., .../backend/routes/)
#     routes_dir = os.path.dirname(script_path)
    
#     # Go UP ONE LEVEL to get the backend directory (e.g., .../backend/)
#     backend_dir = os.path.dirname(routes_dir)

#     # Now, build the paths relative to the `backend_dir`
#     output_dir = os.path.join(backend_dir, 'final_model_output')
#     data_path = os.path.join(backend_dir, 'properties.csv')
    
#     os.makedirs(output_dir, exist_ok=True)
    
#     print("="*60)
#     print("Environment Setup Complete")
#     print(f"-> Model artifacts will be saved to: '{os.path.abspath(output_dir)}'")
#     print("="*60)
    
#     return data_path, output_dir

# MODEL_CONFIG = { 'apartment': { 'params': { 'objective': 'regression_l1', 'metric': 'rmse', 'random_state': 42, 'n_estimators': 1500, 'learning_rate': 0.02, 'num_leaves': 40, 'max_depth': 8, 'subsample': 0.8, 'colsample_bytree': 0.8 } } }

# # --- 2. DATA PREPARATION & FEATURE ENGINEERING WORKFLOW (REVISED) ---
# def prepare_apartment_data(df):
#     """Loads, cleans, and engineers ALL features for the apartment dataset."""
#      # --- Basic Cleaning ---
#     df_apartments = df[df['property_type'] == 'apartment'].copy()
#     df_apartments['bedrooms'].replace(0, np.nan, inplace=True)
#     df_apartments['bathrooms'].replace(0, np.nan, inplace=True)
#     df_apartments.dropna(subset=['price_$', 'size_m2', 'bedrooms', 'bathrooms'], inplace=True)
#     df_apartments = df_apartments[df_apartments['size_m2'] > 0]
    
#     # --- Basic Feature Engineering ---

#     df_apartments['price_per_m2'] = df_apartments['price_$'] / df_apartments['size_m2']
#     df_apartments['size_per_room'] = df_apartments['size_m2'] / (df_apartments['bedrooms'] + df_apartments['bathrooms'] + 1)
    
#     # --- Outlier Removal ---
#     q_low = df_apartments['price_per_m2'].quantile(0.01)
#     q_high = df_apartments['price_per_m2'].quantile(0.99)
#     df_apartments_clean = df_apartments[(df_apartments['price_per_m2'] >= q_low) & (df_apartments['price_per_m2'] <= q_high)].copy()
#     print(f"   -> {len(df_apartments_clean)} rows after cleaning and outlier removal.")

#     # --- Advanced Feature Engineering: Handling Rare Categories ---
#     print("\n--- Advanced Feature Engineering ---")
#     district_counts = df_apartments_clean['district'].value_counts()
#     THRESHOLD = 30 
#     rare_districts = district_counts[district_counts < THRESHOLD].index.tolist()
#     df_apartments_clean.loc[df_apartments_clean['district'].isin(rare_districts), 'district'] = 'Other'
#     print(f"   -> Grouped {len(rare_districts)} rare districts into 'Other' category.")

#     # --- Advanced Feature Engineering: Interaction Features ---
#     district_dummies = pd.get_dummies(df_apartments_clean['district'], prefix='dist', dtype=int)
#     for district_col in district_dummies.columns:
#         interaction_col_name = f'size_x_{district_col}'
#         df_apartments_clean[interaction_col_name] = df_apartments_clean['size_m2'] * district_dummies[district_col]
#     print(f"   -> Created {len(district_dummies.columns)} size-district interaction features.")

#     # --- Final Transformation ---
#     df_apartments_clean['log_price'] = np.log1p(df_apartments_clean['price_$'])
#     print("   -> Created 'log_price' as the final target variable.")
    
#     return df_apartments_clean

# # --- 3. MODELING WORKFLOW (REVISED) ---
# def build_and_evaluate_model(df_featured, config, output_dir):
#     """Builds the preprocessing pipeline, validates, trains, and saves the final model."""
#     print("\n--- Step 3: Building and Validating the Model ---")
    
#     # --- Define Features and Target from the prepared DataFrame ---
#     TARGET = 'log_price'
    
#     # Original features
#     NUMERICAL_FEATURES = ['size_m2', 'bedrooms', 'bathrooms', 'latitude', 'longitude', 'size_per_room']
#     CATEGORICAL_FEATURES = ['province', 'district']
    
#     # Identify the new interaction features that were created
#     INTERACTION_FEATURES = [col for col in df_featured.columns if col.startswith('size_x_dist_')]
    
#     # The final list of numerical features for the model
#     UPDATED_NUMERICAL_FEATURES = NUMERICAL_FEATURES + INTERACTION_FEATURES
    
#     # The final list of all predictor columns
#     FEATURES = UPDATED_NUMERICAL_FEATURES + CATEGORICAL_FEATURES
    
#     X = df_featured[FEATURES]
#     y = df_featured[TARGET]

#     # --- Define the preprocessing pipeline ---
#     preprocessor = ColumnTransformer(
#         transformers=[
#             # IMPORTANT: Use the UPDATED list of numerical features here
#             ('num', Pipeline(steps=[
#                 ('imputer', SimpleImputer(strategy='median')),
#                 ('scaler', StandardScaler())]), UPDATED_NUMERICAL_FEATURES),
#             ('cat', Pipeline(steps=[
#                 ('imputer', SimpleImputer(strategy='most_frequent')),
#                 ('onehot', OneHotEncoder(handle_unknown='ignore', sparse_output=False))]), CATEGORICAL_FEATURES)
#         ],
#         remainder='drop' # This is safe as all our features are included
#     )

#     # Create the full model pipeline
#     pipeline = Pipeline(steps=[
#         ('preprocessor', preprocessor),
#         ('regressor', lgb.LGBMRegressor(**config['params']))
#     ])

#     # --- Cross-Validation ---
#     kf = KFold(n_splits=5, shuffle=True, random_state=42)
#     cv_scores_r2 = cross_val_score(pipeline, X, y, cv=kf, scoring='r2')
    
#     print("\n--- Cross-Validation Performance ---")
#     print(f"   -> 5-Fold R² Scores: {[f'{s:.4f}' for s in cv_scores_r2]}")
#     print(f"   -> Average R² Score: {np.mean(cv_scores_r2):.4f} (± {np.std(cv_scores_r2):.4f})")

#     # --- Final Model Training ---
#     print("\n--- Step 4: Training Final Model on All Data ---")
#     pipeline.fit(X, y)
#     print("   -> Final model training complete.")
    
#     return pipeline, np.mean(cv_scores_r2)

# # --- 4. FEATURE IMPORTANCE & 5. MAIN EXECUTION (Largely Unchanged) ---
# # --- 4. FEATURE IMPORTANCE (CORRECTED VERSION) ---
# def plot_feature_importance(pipeline, output_dir):
#     """Extracts and plots feature importances with correct names from the trained model."""
#     print("\n--- Step 5: Generating Feature Importance Plot ---")

#     # Extract the two main components from the pipeline
#     preprocessor = pipeline.named_steps['preprocessor']
#     regressor = pipeline.named_steps['regressor']
    
#     # --- THIS IS THE KEY PART ---
#     # Get feature names from the 'num' transformer (numerical features)
#     # The names are passed through unchanged in order
#     num_feature_names = preprocessor.transformers_[0][2] 
    
#     # Get feature names from the 'cat' transformer (categorical features) after one-hot encoding
#     cat_feature_names = preprocessor.named_transformers_['cat'].named_steps['onehot'].get_feature_names_out()
    
#     # Concatenate all feature names in the correct order
#     all_feature_names = np.concatenate([num_feature_names, cat_feature_names])
    
#     # Create a DataFrame for feature importances
#     importances = pd.DataFrame({
#         'feature': all_feature_names,
#         'importance': regressor.feature_importances_
#     }).sort_values('importance', ascending=False).head(20) # Get top 20

#     # Plot
#     plt.figure(figsize=(10, 8))
#     plt.barh(importances['feature'], importances['importance'])
#     plt.xlabel("LightGBM Feature Importance")
#     plt.ylabel("Feature")
#     plt.title("Top 20 Feature Importances")
#     plt.gca().invert_yaxis()
#     plt.tight_layout()
    
#     plot_path = os.path.join(output_dir, 'feature_importance_advanced.png')
#     plt.savefig(plot_path)
#     print(f"   -> Feature importance plot saved to '{plot_path}'")
#     plt.close()

# def main():
#     # ... (code is correct, no changes needed, just call the revised functions) ...
#     data_path, output_dir = setup_environment()
#     df = pd.read_csv(data_path)
#     df.drop_duplicates(subset=['id'], inplace=True)
#     df.rename(columns={'type': 'property_type'}, inplace=True)
#     df['property_type'] = df['property_type'].str.lower().str.strip()
#     df_featured = prepare_apartment_data(df)
#     final_pipeline, avg_r2_score = build_and_evaluate_model(df_featured, MODEL_CONFIG['apartment'], output_dir)
#     plot_feature_importance(final_pipeline, output_dir)
#     model_path = os.path.join(output_dir, 'model_apartment_advanced.joblib') 
#     joblib.dump(final_pipeline, model_path)
#     print(f"\n--- Step 6: Final model saved to '{model_path}' ---")
#     print("\n" + "="*60); 
#     print("--- MODEL PERFORMANCE SUMMARY ---")

#     summary = {'Model': 'Apartment Price Estimator (Advanced Features)', 'Average R2 (5-Fold CV)': f"{avg_r2_score:.4f}", 'Number of Training Samples': len(df_featured), 'Model Saved At': model_path, 'Importance Plot Saved At': os.path.join(output_dir, 'feature_importance_advanced.png')}
#     for key, value in summary.items(): print(f"   -> {key}: {value}")
#     print("="*60)
#     print("--- WORKFLOW COMPLETE ---")

# if __name__ == '__main__':
#     main()


import pandas as pd
import numpy as np
import lightgbm as lgb
import matplotlib.pyplot as plt
import joblib
import warnings
import os
from sklearn.model_selection import KFold, cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer

# --- 1. SETUP & CONFIGURATION (Unchanged) ---
def setup_environment():
    """Sets up directories and configurations for the script."""
    warnings.filterwarnings('ignore', category=UserWarning)
    pd.options.mode.chained_assignment = None
    
    # Get the absolute path of the currently running script (e.g., .../backend/routes/your_script.py)
    script_path = os.path.abspath(__file__)
    
    # Get the directory the script is in (e.g., .../backend/routes/)
    routes_dir = os.path.dirname(script_path)
    
    # Go UP ONE LEVEL to get the backend directory (e.g., .../backend/)
    backend_dir = os.path.dirname(routes_dir)

    # Now, build the paths relative to the `backend_dir`
    output_dir = os.path.join(backend_dir, 'final_model_output_v2')
    data_path = os.path.join(backend_dir, 'properties.csv')
    
    os.makedirs(output_dir, exist_ok=True)
    
    print("="*60)
    print("Environment Setup Complete")
    print(f"-> Model artifacts will be saved to: '{os.path.abspath(output_dir)}'")
    print("="*60)
    
    return data_path, output_dir

MODEL_CONFIG = { 'apartment': { 'params': { 'objective': 'regression_l1', 'metric': 'rmse', 'random_state': 42, 'n_estimators': 1500, 'learning_rate': 0.02, 'num_leaves': 40, 'max_depth': 8, 'subsample': 0.8, 'colsample_bytree': 0.8 } } }

# --- 2. DATA PREPARATION & FEATURE ENGINEERING WORKFLOW (REVISED) ---
def prepare_apartment_data(df):
    """Loads, cleans, and engineers features for the apartment dataset using size categories."""
    print("\n--- Step 2: Preparing Apartment Data with Size Categories ---")
        
    # --- Basic Cleaning ---
    df_apartments = df[df['property_type'] == 'apartment'].copy()
    df_apartments['bedrooms'].replace(0, np.nan, inplace=True)
    df_apartments['bathrooms'].replace(0, np.nan, inplace=True)
    df_apartments.dropna(subset=['price_$', 'size_m2', 'bedrooms', 'bathrooms'], inplace=True)
    df_apartments = df_apartments[df_apartments['size_m2'] > 0]
    
    # --- CHANGE 1: Feature Engineering for Size Category ---
    print("\n--- Feature Engineering ---")
    bins = [0, 80, 120, 180, 250, 400, float('inf')]
    labels = ['Studio/Small (0-80m²)',
               'Standard (81-120m²)',
                 'Comfortable (121-180m²)', 
                 'Large (181-250m²)',
                   'Very Large (251-400m²)', 'Penthouse (401m²+)']
    df_apartments['size_category'] = pd.cut(df_apartments['size_m2'], bins=bins, labels=labels, right=True)
    print(f"   -> Created 'size_category' feature from 'size_m2'.")

    # --- Basic Feature Engineering (still useful) ---
    df_apartments['price_per_m2'] = df_apartments['price_$'] / df_apartments['size_m2']
    
    # --- Outlier Removal (using price_per_m2 is still a good idea) ---
    q_low = df_apartments['price_per_m2'].quantile(0.01)
    q_high = df_apartments['price_per_m2'].quantile(0.99)
    df_apartments_clean = df_apartments[(df_apartments['price_per_m2'] >= q_low) & (df_apartments['price_per_m2'] <= q_high)].copy()
    print(f"   -> {len(df_apartments_clean)} rows after cleaning and outlier removal.")

    # --- Handling Rare Categories (still a good practice) ---
    district_counts = df_apartments_clean['district'].value_counts()
    THRESHOLD = 30 
    rare_districts = district_counts[district_counts < THRESHOLD].index.tolist()
    df_apartments_clean.loc[df_apartments_clean['district'].isin(rare_districts), 'district'] = 'Other'
    print(f"   -> Grouped {len(rare_districts)} rare districts into 'Other' category.")

    # --- Final Transformation ---
    df_apartments_clean['log_price'] = np.log1p(df_apartments_clean['price_$'])
    print("   -> Created 'log_price' as the final target variable.")
    
    return df_apartments_clean

# --- 3. MODELING WORKFLOW (REVISED) ---
def build_and_evaluate_model(df_featured, config, output_dir):
    """Builds the preprocessing pipeline, validates, trains, and saves the final model."""
    print("\n--- Step 3: Building and Validating the Model ---")
    
    # --- CHANGE 2: Define New Features and Target ---
    TARGET = 'log_price'
    
    # Define features based on the new approach
    NUMERICAL_FEATURES = ['bedrooms', 'bathrooms', 'latitude', 'longitude']
    CATEGORICAL_FEATURES = ['province', 'district', 'size_category'] # Add size_category here
    
    # The interaction features are no longer needed as the model will learn the interaction
    # between the 'district' category and the new 'size_category' automatically.
    
    FEATURES = NUMERICAL_FEATURES + CATEGORICAL_FEATURES
    
    X = df_featured[FEATURES]
    y = df_featured[TARGET]

    # --- CHANGE 3: Define the new preprocessing pipeline ---
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', Pipeline(steps=[
                ('imputer', SimpleImputer(strategy='median')),
                ('scaler', StandardScaler())]), NUMERICAL_FEATURES),
            ('cat', Pipeline(steps=[
                ('imputer', SimpleImputer(strategy='most_frequent')),
                ('onehot', OneHotEncoder(handle_unknown='ignore', sparse_output=False))]), CATEGORICAL_FEATURES)
        ],
        remainder='drop'
    )

    # Create the full model pipeline (this part is unchanged)
    pipeline = Pipeline(steps=[
        ('preprocessor', preprocessor),
        ('regressor', lgb.LGBMRegressor(**config['params']))
    ])

    # --- Cross-Validation (Unchanged) ---
    kf = KFold(n_splits=5, shuffle=True, random_state=42)
    cv_scores_r2 = cross_val_score(pipeline, X, y, cv=kf, scoring='r2')
    
    print("\n--- Cross-Validation Performance ---")
    print(f"   -> 5-Fold R² Scores: {[f'{s:.4f}' for s in cv_scores_r2]}")
    print(f"   -> Average R² Score: {np.mean(cv_scores_r2):.4f} (± {np.std(cv_scores_r2):.4f})")

    # --- Final Model Training (Unchanged) ---
    print("\n--- Step 4: Training Final Model on All Data ---")
    pipeline.fit(X, y)
    print("   -> Final model training complete.")
    
    return pipeline, np.mean(cv_scores_r2)

# --- 4. FEATURE IMPORTANCE & 5. MAIN EXECUTION (Largely Unchanged but will use new features) ---
def plot_feature_importance(pipeline, output_dir):
    """Extracts and plots feature importances from the trained model."""
    print("\n--- Step 5: Generating Feature Importance Plot ---")

    preprocessor = pipeline.named_steps['preprocessor']
    regressor = pipeline.named_steps['regressor']
    
    # Get names from numerical transformer
    num_feature_names = preprocessor.transformers_[0][2]
    # Get names from categorical transformer after one-hot encoding
    cat_feature_names = preprocessor.named_transformers_['cat'].named_steps['onehot'].get_feature_names_out()
    
    all_feature_names = np.concatenate([num_feature_names, cat_feature_names])
    
    importances = pd.DataFrame({
        'feature': all_feature_names,
        'importance': regressor.feature_importances_
    }).sort_values('importance', ascending=False).head(20)

    plt.figure(figsize=(10, 8))
    plt.barh(importances['feature'], importances['importance'])
    plt.xlabel("LightGBM Feature Importance")
    plt.ylabel("Feature")
    plt.title("Top 20 Feature Importances (with Size Category)")
    plt.gca().invert_yaxis()
    plt.tight_layout()
    
    plot_path = os.path.join(output_dir, 'feature_importance_size_category.png')
    plt.savefig(plot_path)
    print(f"   -> Feature importance plot saved to '{plot_path}'")
    plt.close()

def main():
    data_path, output_dir = setup_environment()
    df = pd.read_csv(data_path)
    df.drop_duplicates(subset=['id'], inplace=True)
    df.rename(columns={'type': 'property_type'}, inplace=True)
    df['property_type'] = df['property_type'].str.lower().str.strip()
    
    df_featured = prepare_apartment_data(df)
    
    final_pipeline, avg_r2_score = build_and_evaluate_model(df_featured, MODEL_CONFIG['apartment'], output_dir)
    
    plot_feature_importance(final_pipeline, output_dir)
    
    model_path = os.path.join(output_dir, 'model_apartment_size_category.joblib') 
    joblib.dump(final_pipeline, model_path)
    
    print(f"\n--- Step 6: Final model saved to '{model_path}' ---")
    print("\n" + "="*60); 
    print("--- MODEL PERFORMANCE SUMMARY ---")
    summary = {
        'Model': 'Apartment Price Estimator (Size Category)',
        'Average R2 (5-Fold CV)': f"{avg_r2_score:.4f}",
        'Number of Training Samples': len(df_featured),
        'Model Saved At': model_path,
        'Importance Plot Saved At': os.path.join(output_dir, 'feature_importance_size_category.png')
    }
    for key, value in summary.items(): 
        print(f"   -> {key}: {value}")
    print("="*60)
    print("--- WORKFLOW COMPLETE ---")

if __name__ == '__main__':
    main()