
# # MODEL_CONFIG = {
# #     'apartment': {
# #         'params': {'depth': 8, 'learning_rate': 0.03, 'l2_leaf_reg': 5, 'iterations': 2000},
# #         'features': {'numerical': ['log_size_m2', 'bedrooms', 'bathrooms'], 'categorical': ['province', 'district', 'city']}
# #     },
# #     'land': {
# #         'params': {'depth': 6, 'learning_rate': 0.01, 'l2_leaf_reg': 10, 'iterations': 2000},
# #         'features': {'numerical': ['log_size_m2'], 'categorical': ['province', 'district', 'city']}
# #     },
# #     'house/villa': {
# #         'params': {'depth': 7, 'learning_rate': 0.02, 'l2_leaf_reg': 3, 'iterations': 2000},
# #         'features': {'numerical': ['log_size_m2', 'bedrooms', 'bathrooms'], 'categorical': ['province', 'district', 'city']}
# #     },
# #  
# # }


# import pandas as pd
# import numpy as np
# import lightgbm as lgb
# import matplotlib.pyplot as plt
# from sklearn.model_selection import train_test_split
# from sklearn.cluster import KMeans
# from sklearn.pipeline import Pipeline
# from sklearn.compose import ColumnTransformer
# from sklearn.preprocessing import OneHotEncoder, StandardScaler
# from sklearn.metrics import r2_score, mean_absolute_error
# import joblib
# import warnings
# import os

# # --- Setup ---
# warnings.filterwarnings('ignore', category=UserWarning)
# warnings.filterwarnings('ignore', category=FutureWarning)
# pd.options.mode.chained_assignment = None

# # --- Helper Functions ---

# def plot_feature_importance(pipeline, prop_type, output_dir):
#     """Saves a readable feature importance plot to the specified directory."""
#     regressor = pipeline.named_steps['regressor']
#     if not hasattr(regressor, 'feature_importances_') or not hasattr(regressor, 'feature_name_'):
#         return
        
#     feature_importances = pd.Series(regressor.feature_importances_, index=regressor.feature_name_).sort_values(ascending=False)
#     plt.figure(figsize=(10, 8))
#     top_features = feature_importances.head(15)
#     plt.barh(top_features.index, top_features.values)
#     plt.gca().invert_yaxis()
#     plt.title(f'Top 15 Feature Importance - {prop_type.title()}', fontsize=16)
#     plt.xlabel('Feature Importance (Gain)', fontsize=12)
#     plt.tight_layout()
    
#     plot_path = os.path.join(output_dir, f'feature_importance_{prop_type}.png')
#     plt.savefig(plot_path)
#     print(f"   -> Feature importance plot saved to '{plot_path}'")
#     plt.close()

# def create_and_train_pipeline(X_train, y_train, X_test, y_test, params, feature_config):
#     """Creates, trains, evaluates, and returns a model pipeline."""
#     preprocessor = ColumnTransformer(
#         transformers=[
#             ('num', StandardScaler(), feature_config['numerical']),
#             ('cat', OneHotEncoder(handle_unknown='ignore', sparse_output=False), feature_config['categorical'])
#         ],
#         remainder='drop',
#         verbose_feature_names_out=False
#     )
#     pipeline = Pipeline(steps=[
#         ('preprocessor', preprocessor),
#         ('regressor', lgb.LGBMRegressor(**params))
#     ])
    
#     preprocessor.fit(X_train)
#     final_feature_names = list(preprocessor.get_feature_names_out())
    
#     pipeline.fit(X_train, y_train, regressor__feature_name=final_feature_names)
    
#     y_pred_log = pipeline.predict(X_test)
#     y_pred_dollars = np.expm1(y_pred_log)
#     y_test_dollars = np.expm1(y_test)
    
#     r2 = r2_score(y_test_dollars, y_pred_dollars)
#     mae = mean_absolute_error(y_test_dollars, y_pred_dollars)
    
#     print(f"Result -> R²: {r2:.4f}, MAE: ${mae:,.2f}")
    
#     return pipeline, r2

# def main():
#     """Main function to run the final hybrid model training workflow."""
#     print("Starting the Final Hybrid Model Training Process...")
    
#     # --- 1. DEFINE PATHS CORRECTLY ---
#     # Assumes this script is run from within the 'backend' directory or a subdirectory like 'routes'.
#     try:
#         # This handles running from both 'backend/' and 'backend/routes/'
#         script_path = os.path.dirname(os.path.abspath(__file__))
#         if os.path.basename(script_path) == 'routes':
#             BACKEND_DIR = os.path.dirname(script_path)
#         else:
#             BACKEND_DIR = script_path
#     except NameError:
#          # Fallback for interactive environments
#         BACKEND_DIR = os.getcwd()

#     OUTPUT_DIR = os.path.join(BACKEND_DIR, 'final_models')
#     os.makedirs(OUTPUT_DIR, exist_ok=True)
#     print(f"-> Models will be saved to: '{os.path.abspath(OUTPUT_DIR)}'")
    
#     # --- 2. MODEL CONFIGURATION ---
#     MODEL_CONFIG = {
#         'apartment': {
#             'params': {'objective': 'regression_l1', 'random_state': 42, 'n_estimators': 1500, 'learning_rate': 0.02, 'num_leaves': 40, 'max_depth': 8},
#             'features': {'numerical': ['size_m2', 'latitude', 'longitude', 'bedrooms', 'bathrooms'], 'categorical': ['province', 'district', 'location_cluster']}
#         },
#         'office': {
#             'params': {'objective': 'regression_l1', 'random_state': 42, 'n_estimators': 1000, 'learning_rate': 0.03, 'num_leaves': 30, 'max_depth': 6},
#             'features': {'numerical': ['size_m2', 'latitude', 'longitude', 'bedrooms', 'bathrooms'], 'categorical': ['province', 'district', 'location_cluster']}
#         },
#         'shop': {
#             'params': {'objective': 'regression_l1', 'random_state': 42, 'n_estimators': 1200, 'learning_rate': 0.03},
#             'features': {'numerical': ['size_m2', 'latitude', 'longitude', 'bathrooms'], 'categorical': ['province', 'district', 'location_cluster']}
#         }
#     }
    
#     # --- 3. DATA PREPARATION ---
#     df = pd.read_csv('properties.csv')
#     df.drop_duplicates(subset=['id'], inplace=True)
#     df = df[(df['price_$'] > 1000) & (df['size_m2'] > 10)].copy()
#     df['type'] = df['type'].str.lower().str.strip()
#     df = df[df['type'] != 'land'].copy()
#     print(f"Removed 'Land' properties. Working with {len(df)} properties.")
#     df.drop(columns=['created_at', 'city'], inplace=True, errors='ignore')

#     # --- 4. GLOBAL FEATURE ENGINEERING ---
#     print("\nCreating global K-Means model for location clustering...")
#     kmeans = KMeans(n_clusters=20, random_state=42, n_init='auto')
#     df['location_cluster'] = kmeans.fit_predict(df[['latitude', 'longitude']]).astype(str)
    
#     # --- 5. TRAINING SPECIALIST MODELS ---
#     specialist_types = ['apartment', 'office', 'shop']
#     for prop_type in specialist_types:
#         print(f"\n--- Training Specialist Model for: {prop_type.upper()} ---")
        
#         df_slice = df[df['type'] == prop_type].copy()
#         print(f"Found {len(df_slice)} samples for type '{prop_type}'.")
        
#         if len(df_slice) < 50:
#             print("   -> Insufficient data. This type will be handled by the generalist model.")
#             continue
        
#         config = MODEL_CONFIG[prop_type]
#         for col in config['features']['numerical'] + config['features']['categorical']:
#             if col not in df_slice.columns:
#                 df_slice[col] = 0

#         X = df_slice.drop(columns=['price_$', 'type'], errors='ignore')
#         y = np.log1p(df_slice['price_$'])
#         X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

#         pipeline, r2 = create_and_train_pipeline(X_train, y_train, X_test, y_test, config['params'], config['features'])

#         if r2 > 0.65:
#             model_path = os.path.join(OUTPUT_DIR, f'model_{prop_type}.joblib')
#             joblib.dump(pipeline, model_path)
#             print(f"   -> High-quality model saved to '{model_path}'")
#             plot_feature_importance(pipeline, prop_type, OUTPUT_DIR)
#         else:
#             print(f"   -> Model performance below threshold. This type will use the Generalist Fallback.")

#     # --- 6. TRAINING GENERALIST FALLBACK MODEL (ON ALL DATA EXCEPT LAND) ---
#     print("\n--- Training Generalist Fallback Model ---")
#     df_general = df.copy() 

#     general_config = {
#         'params': {'objective': 'regression_l1', 'random_state': 42, 'n_estimators': 1500, 'learning_rate': 0.02, 'num_leaves': 40, 'max_depth': 7},
#         'features': {'numerical': ['size_m2', 'latitude', 'longitude', 'bedrooms', 'bathrooms'], 'categorical': ['province', 'district', 'location_cluster', 'type']}
#     }
    
#     X_general = df_general.drop(columns=['price_$'])
#     y_general = np.log1p(df_general['price_$'])
#     X_train_gen, X_test_gen, y_train_gen, y_test_gen = train_test_split(X_general, y_general, test_size=0.2, random_state=42)
    
#     general_pipeline, _ = create_and_train_pipeline(
#         X_train_gen, y_train_gen, X_test_gen, y_test_gen, general_config['params'], general_config['features']
#     )
    
#     model_path = os.path.join(OUTPUT_DIR, 'model_general_fallback.joblib')
#     joblib.dump(general_pipeline, model_path)
#     print(f"   -> Generalist Fallback Model saved to '{model_path}'")
#     plot_feature_importance(general_pipeline, "general_fallback", OUTPUT_DIR)

#     # --- 7. SAVE SUPPORTING ARTIFACTS ---
#     joblib.dump(kmeans, os.path.join(OUTPUT_DIR, 'kmeans_model.joblib'))
    
#     print(f"\n\n--- Workflow Complete ---")
#     print(f"All necessary models and plots have been generated and saved to the '{os.path.abspath(OUTPUT_DIR)}' directory.")

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
from supabase import create_client, Client
from dotenv import load_dotenv

load_dotenv()

# --- 1. SETUP & CONFIGURATION (Unchanged) ---
def setup_environment():
    """Sets up directories and configurations for the script."""
    warnings.filterwarnings('ignore', category=UserWarning)
    pd.options.mode.chained_assignment = None
    
   
    # script_path = os.path.abspath(__file__) 
    # routes_dir = os.path.dirname(script_path)
    # backend_dir = os.path.dirname(routes_dir)
    # output_dir = os.path.join(backend_dir, 'final_model_output')
    # data_path = os.path.join(backend_dir, 'properties.csv')
    # os.makedirs(output_dir, exist_ok=True)
        
    SCRIPT_DIR = os.path.abspath(__file__) 
    BACKEND_DIR = os.path.dirname(SCRIPT_DIR)
    OUTPUT_DIR = os.path.join(BACKEND_DIR, 'final_model_output') 
    DATA_PATH = os.path.join(BACKEND_DIR, 'properties.csv')    
    print("="*60)
    print("Environment Setup Complete")
    print(f"-> Model artifacts will be saved to: '{os.path.abspath(output_dir)}'")
    print("="*60)
    
    return data_path, output_dir

MODEL_CONFIG = { 
    'apartment': { 
        'params': { 
            'objective': 'regression_l1', 
            'metric': 'rmse', 
            'random_state': 42, 
            'n_estimators': 1500, 
            'learning_rate': 0.02, 
            'num_leaves': 40, 
            'max_depth': 8, 
            'subsample': 0.8, 
            'colsample_bytree': 0.8 
        } 
    } 
}

def load_data_from_supabase():
    """Connects to Supabase and fetches all regional transaction data."""
    print("-> Connecting to Supabase to fetch training data...")
    url: str = os.environ.get("SUPABASE_URL")
    key: str = os.environ.get("SUPABASE_KEY")

    if not url or not key:
        raise ValueError("FATAL: Supabase URL or Key not found in .env file.")
        
    supabase: Client = create_client(url, key)

    # Fetch all data from the table, ordered by id
    response = supabase.table('properties').select("*").order('id').execute()
    
    # Convert the list of dictionaries to a pandas DataFrame
    df = pd.DataFrame(response.data)
    print(f"-> Successfully fetched {len(df)} rows from Supabase.")
    return df


# --- 2. DATA PREPARATION & FEATURE ENGINEERING WORKFLOW (REVISED) ---
def prepare_apartment_data(df):
    """Loads, cleans, and engineers ALL features for the apartment dataset."""
    print("\n--- Step 2: Preparing Apartment Data ---")

    # --- Basic Cleaning ---
    df_apartments = df[df['property_type'] == 'apartment'].copy()
    df_apartments['bedrooms'].replace(0, np.nan, inplace=True)
    df_apartments['bathrooms'].replace(0, np.nan, inplace=True)
    df_apartments.dropna(subset=['price_$', 'size_m2', 'bedrooms', 'bathrooms'], inplace=True)
    df_apartments = df_apartments[df_apartments['size_m2'] > 0]
    
    # --- Basic Feature Engineering ---
    df_apartments['price_per_m2'] = df_apartments['price_$'] / df_apartments['size_m2']
    df_apartments['size_per_room'] = df_apartments['size_m2'] / (df_apartments['bedrooms'] + df_apartments['bathrooms'] + 1)
    
    # --- Outlier Removal ---
    q_low = df_apartments['price_per_m2'].quantile(0.01)
    q_high = df_apartments['price_per_m2'].quantile(0.99)
    df_apartments_clean = df_apartments[(df_apartments['price_per_m2'] >= q_low) & (df_apartments['price_per_m2'] <= q_high)].copy()
    print(f"   -> {len(df_apartments_clean)} rows after cleaning and outlier removal.")

    # --- Advanced Feature Engineering: Handling Rare Categories ---
    print("\n--- Advanced Feature Engineering ---")
    district_counts = df_apartments_clean['district'].value_counts()
    THRESHOLD = 30 
    rare_districts = district_counts[district_counts < THRESHOLD].index.tolist()
    df_apartments_clean.loc[df_apartments_clean['district'].isin(rare_districts), 'district'] = 'Other'
    print(f"   -> Grouped {len(rare_districts)} rare districts into 'Other' category.")

    # --- Advanced Feature Engineering: Interaction Features ---
    district_dummies = pd.get_dummies(df_apartments_clean['district'], prefix='dist', dtype=int)
    for district_col in district_dummies.columns:
        interaction_col_name = f'size_x_{district_col}'
        df_apartments_clean[interaction_col_name] = df_apartments_clean['size_m2'] * district_dummies[district_col]
    print(f"   -> Created {len(district_dummies.columns)} size-district interaction features.")

    # --- Final Transformation ---
    df_apartments_clean['log_price'] = np.log1p(df_apartments_clean['price_$'])
    print("   -> Created 'log_price' as the final target variable.")
    
    return df_apartments_clean

# --- 3. MODELING WORKFLOW (REVISED) ---
def build_and_evaluate_model(df_featured, config, output_dir):
    """Builds the preprocessing pipeline, validates, trains, and saves the final model."""
    print("\n--- Step 3: Building and Validating the Model ---")
    
    # --- Define Features and Target from the prepared DataFrame ---
    TARGET = 'log_price'
    
    # Original features
    NUMERICAL_FEATURES = ['size_m2', 'bedrooms', 'bathrooms', 'latitude', 'longitude', 'size_per_room']
    CATEGORICAL_FEATURES = ['province', 'district']
    
    # Identify the new interaction features that were created
    INTERACTION_FEATURES = [col for col in df_featured.columns if col.startswith('size_x_dist_')]
    
    # The final list of numerical features for the model
    UPDATED_NUMERICAL_FEATURES = NUMERICAL_FEATURES + INTERACTION_FEATURES
    
    # The final list of all predictor columns
    FEATURES = UPDATED_NUMERICAL_FEATURES + CATEGORICAL_FEATURES
    
    X = df_featured[FEATURES]
    y = df_featured[TARGET]

    # --- Define the preprocessing pipeline ---
    preprocessor = ColumnTransformer(
        transformers=[
            # IMPORTANT: Use the UPDATED list of numerical features here
            ('num', Pipeline(steps=[
                ('imputer', SimpleImputer(strategy='median')),
                ('scaler', StandardScaler())]), UPDATED_NUMERICAL_FEATURES),
            ('cat', Pipeline(steps=[
                ('imputer', SimpleImputer(strategy='most_frequent')),
                ('onehot', OneHotEncoder(handle_unknown='ignore', sparse_output=False))]), CATEGORICAL_FEATURES)
        ],
        remainder='drop' # This is safe as all our features are included
    )

    # Create the full model pipeline
    pipeline = Pipeline(steps=[
        ('preprocessor', preprocessor),
        ('regressor', lgb.LGBMRegressor(**config['params']))
    ])

    # --- Cross-Validation ---
    kf = KFold(n_splits=5, shuffle=True, random_state=42)
    cv_scores_r2 = cross_val_score(pipeline, X, y, cv=kf, scoring='r2')
    
    print("\n--- Cross-Validation Performance ---")
    print(f"   -> 5-Fold R² Scores: {[f'{s:.4f}' for s in cv_scores_r2]}")
    print(f"   -> Average R² Score: {np.mean(cv_scores_r2):.4f} (± {np.std(cv_scores_r2):.4f})")

    # --- Final Model Training ---
    print("\n--- Step 4: Training Final Model on All Data ---")
    pipeline.fit(X, y)
    print("   -> Final model training complete.")
    
    return pipeline, np.mean(cv_scores_r2)

# --- 4. FEATURE IMPORTANCE & 5. MAIN EXECUTION (Largely Unchanged) ---
# --- 4. FEATURE IMPORTANCE (CORRECTED VERSION) ---
def plot_feature_importance(pipeline, output_dir):
    """Extracts and plots feature importances with correct names from the trained model."""
    print("\n--- Step 5: Generating Feature Importance Plot ---")

    # Extract the two main components from the pipeline
    preprocessor = pipeline.named_steps['preprocessor']
    regressor = pipeline.named_steps['regressor']
    
    # --- THIS IS THE KEY PART ---
    # Get feature names from the 'num' transformer (numerical features)
    # The names are passed through unchanged in order
    num_feature_names = preprocessor.transformers_[0][2] 
    
    # Get feature names from the 'cat' transformer (categorical features) after one-hot encoding
    cat_feature_names = preprocessor.named_transformers_['cat'].named_steps['onehot'].get_feature_names_out()
    
    # Concatenate all feature names in the correct order
    all_feature_names = np.concatenate([num_feature_names, cat_feature_names])
    
    # Create a DataFrame for feature importances
    importances = pd.DataFrame({
        'feature': all_feature_names,
        'importance': regressor.feature_importances_
    }).sort_values('importance', ascending=False).head(20) # Get top 20

    # Plot
    plt.figure(figsize=(10, 8))
    plt.barh(importances['feature'], importances['importance'])
    plt.xlabel("LightGBM Feature Importance")
    plt.ylabel("Feature")
    plt.title("Top 20 Feature Importances")
    plt.gca().invert_yaxis()
    plt.tight_layout()
    
    plot_path = os.path.join(output_dir, 'feature_importance_advanced.png')
    plt.savefig(plot_path)
    print(f"   -> Feature importance plot saved to '{plot_path}'")
    plt.close()

def main():
    # ... (code is correct, no changes needed, just call the revised functions) ...
    data_path, output_dir = setup_environment()
    df = pd.read_csv(data_path)
    df.drop_duplicates(subset=['id'], inplace=True)
    df.rename(columns={'type': 'property_type'}, inplace=True)
    df['property_type'] = df['property_type'].str.lower().str.strip()
    df_featured = prepare_apartment_data(df)
    final_pipeline, avg_r2_score = build_and_evaluate_model(df_featured, MODEL_CONFIG['apartment'], output_dir)
    plot_feature_importance(final_pipeline, output_dir)
    model_path = os.path.join(output_dir, 'model_apartment_advanced.joblib') 
    joblib.dump(final_pipeline, model_path)
    print(f"\n--- Step 6: Final model saved to '{model_path}' ---")
    print("\n" + "="*60); 
    print("--- MODEL PERFORMANCE SUMMARY ---")

    summary = {'Model': 'Apartment Price Estimator (Advanced Features)', 'Average R2 (5-Fold CV)': f"{avg_r2_score:.4f}", 'Number of Training Samples': len(df_featured), 'Model Saved At': model_path, 'Importance Plot Saved At': os.path.join(output_dir, 'feature_importance_advanced.png')}
    for key, value in summary.items(): print(f"   -> {key}: {value}")
    print("="*60)
    print("--- WORKFLOW COMPLETE ---")

if __name__ == '__main__':
    main()