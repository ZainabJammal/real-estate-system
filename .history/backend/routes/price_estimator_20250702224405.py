
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




# import pandas as pd
# import numpy as np
# import lightgbm as lgb
# import matplotlib.pyplot as plt
# from sklearn.model_selection import train_test_split, cross_val_score, KFold
# from sklearn.cluster import KMeans
# from sklearn.pipeline import Pipeline
# from sklearn.compose import ColumnTransformer
# from sklearn.preprocessing import StandardScaler, OneHotEncoder
# from sklearn.impute import SimpleImputer
# from sklearn.metrics import r2_score, mean_squared_error
# import joblib
# import warnings
# import os

# # --- 1. SETUP & CONFIGURATION ---
# warnings.filterwarnings('ignore', category=UserWarning)
# warnings.filterwarnings('ignore', category=FutureWarning)
# pd.options.mode.chained_assignment = None

# # --- Configuration for paths ---
# BACKEND_DIR = os.path.dirname(os.path.abspath(__file__))
# OUTPUT_DIR = os.path.join(BACKEND_DIR, 'final_models') # Saving to a single, clear folder
# DATA_PATH = os.path.join(BACKEND_DIR, 'properties.csv')
# os.makedirs(OUTPUT_DIR, exist_ok=True)

# # --- 2. HELPER FUNCTIONS ---

# def plot_feature_importance(pipeline, model_group_name, output_dir):
#     """Saves a readable feature importance plot."""
#     try:
#         regressor = pipeline.named_steps['regressor']
#         feature_names = pipeline.named_steps['preprocessor'].get_feature_names_out()
        
#         feature_importances = pd.Series(regressor.feature_importances_, index=feature_names).sort_values(ascending=False)
        
#         plt.figure(figsize=(10, 8))
#         top_features = feature_importances.head(20)
#         plt.barh(top_features.index, top_features.values, color='cornflowerblue')
#         plt.gca().invert_yaxis()
#         plt.title(f'Top 20 Feature Importance - {model_group_name.replace("_", " ").title()}', fontsize=16)
#         plt.xlabel('Feature Importance (Gain)', fontsize=12)
#         plt.tight_layout()
        
#         plot_path = os.path.join(output_dir, f'feature_importance_{model_group_name}.png')
#         plt.savefig(plot_path)
#         print(f"   -> Feature importance plot saved to '{plot_path}'")
#         plt.close()
#     except Exception as e:
#         print(f"   -> Could not generate feature importance plot for {model_group_name}: {e}")

# # --- 3. MAIN WORKFLOW ---

# def main():
#     """Main function to run the final grouped model training workflow."""
#     print("="*60)
#     print("Starting Final Grouped Model Training Process (v4)...")
#     print(f"-> Models will be saved to: '{os.path.abspath(OUTPUT_DIR)}'")
#     print("="*60)

#     # --- Step 1 & 2: Loading, Cleaning, and Grouping (WITH FINAL FIX) ---
#     print("\n--- Step 1 & 2: Loading, Cleaning, and Final Grouping ---")
#     df = pd.read_csv(DATA_PATH)
#     df.drop_duplicates(subset=['id'], inplace=True)
#     df.rename(columns={'type': 'property_type'}, inplace=True)
#     df['property_type'] = df['property_type'].str.lower().str.strip()
#     df = df[(df['price_$'] > 1000) & (df['size_m2'] > 10)].copy()

#     # *** THE FINAL GROUPING FIX ***
#     def assign_model_group(ptype):
#         # High-quality, distinct groups
#         if ptype in ['apartment', 'house/villa', 'land', 'chalet', 'residential building']:
#             return ptype
#         # A combined group for similar commercial types
#         if ptype in ['office', 'shop', 'warehouse', 'commercial building']:
#             return 'commercial_other'
#         # Discard everything else
#         return 'discard'

#     df['model_group'] = df['property_type'].apply(assign_model_group)
#     df = df[df['model_group'] != 'discard'].copy()
#     print("Final Group counts:\n", df['model_group'].value_counts())

#     # --- Step 3: Feature Engineering (Same as before) ---
#     print("\n--- Step 3: Feature Engineering ---")
#     kmeans = KMeans(n_clusters=20, random_state=42, n_init='auto')
#     df['location_cluster'] = kmeans.fit_predict(df[['latitude', 'longitude']]).astype(str)
#     df['bedrooms'].replace(0, np.nan, inplace=True)
#     df['bathrooms'].replace(0, np.nan, inplace=True)
#     denominator = df['bedrooms'].fillna(0) + df['bathrooms'].fillna(0)
#     df['size_per_room'] = np.where(denominator > 0, df['size_m2'] / denominator, np.nan)
#     df['price_per_m2'] = df['price_$'] / df['size_m2']
#     df_clean = df.groupby('model_group', group_keys=False).apply(
#         lambda x: x[(x['price_per_m2'] >= x['price_per_m2'].quantile(0.01)) & (x['price_per_m2'] <= x['price_per_m2'].quantile(0.99))]
#     )
#     df_clean['log_price_per_m2'] = np.log1p(df_clean['price_per_m2'])

#     # --- Step 4: Model Training Loop (with Summary Report) ---
#     print("\n--- Step 4: Training One Model Per Group ---")
    
#     # *** YOUR IDEA: A DICTIONARY TO STORE RESULTS ***
#     results_summary = {}

#     for group_name in df_clean['model_group'].unique():
#         print(f"\n--- Processing Model for Group: {group_name.upper()} ---")
#         df_group = df_clean[df_clean['model_group'] == group_name]

#         if len(df_group) < 30: # Add a threshold to skip tiny groups
#             print(f"   -> Skipping group '{group_name}' due to insufficient data ({len(df_group)} samples).")
#             continue

#         # Define features dynamically
#         numerical_features = ['size_m2', 'latitude', 'longitude']
#         if group_name in ['apartment', 'house/villa', 'chalet']:
#             numerical_features.extend(['bedrooms', 'bathrooms', 'size_per_room'])
#         # 'residential building' gets no room features as they are ambiguous
#         elif group_name == 'commercial_other':
#             numerical_features.append('bathrooms')
        
#         categorical_features = ['province', 'district', 'location_cluster']
        
#         preprocessor = ColumnTransformer(
#             transformers=[
#                 ('num', Pipeline(steps=[('imputer', SimpleImputer(strategy='median')), ('scaler', StandardScaler())]), [f for f in numerical_features if f in df_group.columns]),
#                 ('cat', OneHotEncoder(handle_unknown='ignore', sparse_output=False), [f for f in categorical_features if f in df_group.columns])
#             ], remainder='drop', verbose_feature_names_out=False
#         )
#         pipeline = Pipeline(steps=[('preprocessor', preprocessor), ('regressor', lgb.LGBMRegressor(objective='regression_l1', random_state=42))])
        
#         X = df_group
#         y = df_group['log_price_per_m2']
#         X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        
#         pipeline.fit(X_train, y_train)
        
#         y_pred_log = pipeline.predict(X_test)
#         actual_dollars = df_group.loc[y_test.index]['price_$']
#         predicted_dollars = np.expm1(y_pred_log) * X_test['size_m2']

#         final_r2 = r2_score(y_test, y_pred_log)
#         final_rmse = np.sqrt(mean_squared_error(actual_dollars, predicted_dollars))
        
#         # Store results for the summary table
#         results_summary[group_name] = {
#             'R2_Score': final_r2,
#             'RMSE_Dollars': final_rmse,
#             'Num_Samples': len(df_group)
#         }
        
#         safe_group_name = group_name.replace('/', '_')
#         model_path = os.path.join(OUTPUT_DIR, f'model_{safe_group_name}.joblib')
#         joblib.dump(pipeline, model_path)
#         # plot_feature_importance(pipeline, safe_group_name, OUTPUT_DIR) # Optional: uncomment to save plots

#     # --- Step 5: Final Report & Artifacts ---
#     print("\n" + "="*60)
#     print("--- FINAL MODEL PERFORMANCE SUMMARY ---")
    
#     summary_df = pd.DataFrame.from_dict(results_summary, orient='index')
#     summary_df['R2_Score'] = summary_df['R2_Score'].map('{:.4f}'.format)
#     summary_df['RMSE_Dollars'] = summary_df['RMSE_Dollars'].map('${:,.2f}'.format)
#     print(summary_df)
    
#     print("\n--- Saving Supporting Artifacts ---")
#     joblib.dump(kmeans, os.path.join(OUTPUT_DIR, 'kmeans_model.joblib'))
#     print(f"   -> K-Means model saved.")
    
#     print("\n" + "="*60)
#     print("--- WORKFLOW COMPLETE ---")
#     print("="*60)

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
    # ... (code is perfect, no changes needed) ...
    warnings.filterwarnings('ignore', category=UserWarning)
    pd.options.mode.chained_assignment = None
    
    backend_dir = os.path.dirname(os.path.abspath(__file__))
    output_dir = os.path.join(backend_dir, 'final_model_output')
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
def plot_feature_importance(pipeline, output_dir):
    # ... (code is correct, no changes needed) ...
    print("\n--- Step 5: Generating Feature Importance Plot ---")
    preprocessor = pipeline.named_steps['preprocessor']
    regressor = pipeline.named_steps['regressor']
    num_features_processed = preprocessor.named_transformers_['num'].named_steps['scaler'].get_feature_names_out()
    cat_features_encoded = preprocessor.named_transformers_['cat'].named_steps['onehot'].get_feature_names_out()
    all_feature_names = np.concatenate([num_features_processed, cat_features_encoded])
    importances = pd.DataFrame({'feature': all_feature_names, 'importance': regressor.feature_importances_}).sort_values('importance', ascending=False).head(20)
    plt.figure(figsize=(10, 8)); plt.barh(importances['feature'], importances['importance']); 
    plt.xlabel("LightGBM Feature Importance"); 
    plt.ylabel("Feature"); 
    plt.title("Top 20 Feature Importances"); 
    plt.gca().invert_yaxis(); 
    plt.tight_layout()
    plot_path = os.path.join(output_dir, 'feature_importance_advanced.png'); 
    plt.savefig(plot_path); print(f"   -> Feature importance plot saved to '{plot_path}'"); 
    plt.close()

def main():
    # ... (code is correct, no changes needed, just call the revised functions) ...
    data_path, output_dir = setup_environment()
    df = pd.read_csv(data_path); 
    df.drop_duplicates(subset=['id'], inplace=True); 
    df.rename(columns={'type': 'property_type'}, inplace=True); 
    df['property_type'] = df['property_type'].str.lower().str.strip()
    df_featured = prepare_apartment_data(df)
    final_pipeline, avg_r2_score = build_and_evaluate_model(df_featured, MODEL_CONFIG['apartment'], output_dir)
    plot_feature_importance(final_pipeline, output_dir)
    model_path = os.path.join(output_dir, 'model_apartment_advanced.joblib'); 
    joblib.dump(final_pipeline, model_path); 
    print(f"\n--- Step 6: Final model saved to '{model_path}' ---")
    print("\n" + "="*60); 
    print("--- MODEL PERFORMANCE SUMMARY ---")
    summary = {'Model': 'Apartment Price Estimator (Advanced Features)', 'Average R2 (5-Fold CV)': f"{avg_r2_score:.4f}", 'Number of Training Samples': len(df_featured), 'Model Saved At': model_path, 'Importance Plot Saved At': os.path.join(output_dir, 'feature_importance_advanced.png')}
    for key, value in summary.items(): print(f"   -> {key}: {value}")
    print("="*60); print("--- WORKFLOW COMPLETE ---")

if __name__ == '__main__':
    main()