
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
# #     # Add configs for other types if they become viable
# #     'office': {
# #         'params': {'depth': 6, 'learning_rate': 0.05, 'l2_leaf_reg': 7, 'iterations': 1500},
# #         'features': {'numerical': ['log_size_m2', 'bedrooms', 'bathrooms'], 'categorical': ['province', 'district', 'city']}
# #     },
# #     'shop': {
# #         'params': {'depth': 6, 'learning_rate': 0.05, 'l2_leaf_reg': 7, 'iterations': 1500},
# #         'features': {'numerical': ['log_size_m2'], 'categorical': ['province', 'district', 'city']}
# #     },
# #      'chalet': {
# #         'params': {'depth': 6, 'learning_rate': 0.03, 'l2_leaf_reg': 5, 'iterations': 1500},
# #         'features': {'numerical': ['log_size_m2', 'bedrooms', 'bathrooms'], 'categorical': ['province', 'district', 'city']}
# #     }
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
from sklearn.model_selection import train_test_split, cross_val_score, KFold # <-- IMPORT NEW MODULES
from sklearn.cluster import KMeans
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.metrics import r2_score, mean_absolute_error
import joblib
import warnings
import os

# --- 1. SETUP & CONFIGURATION (No changes here) ---
# ... (Keep this section exactly the same) ...
warnings.filterwarnings('ignore', category=UserWarning)
warnings.filterwarnings('ignore', category=FutureWarning)
pd.options.mode.chained_assignment = None
BACKEND_DIR = os.path.dirname(os.path.abspath(__file__))
OUTPUT_DIR = os.path.join(BACKEND_DIR, 'final_models')
DATA_PATH = os.path.join(BACKEND_DIR, 'properties.csv')
os.makedirs(OUTPUT_DIR, exist_ok=True)
MODEL_CONFIG = {
    'apartment': {
        'params': {'objective': 'regression_l1', 'random_state': 42, 'n_estimators': 1500, 'learning_rate': 0.02, 'num_leaves': 40, 'max_depth': 8},
        'features': {'numerical': ['size_m2', 'latitude', 'longitude'], 'room': ['bedrooms', 'bathrooms'], 'categorical': ['province', 'district', 'location_cluster']}
    },
    'office': {
        'params': {'objective': 'regression_l1', 'random_state': 42, 'n_estimators': 1000, 'learning_rate': 0.03, 'num_leaves': 30, 'max_depth': 6},
        'features': {'numerical': ['size_m2', 'latitude', 'longitude'], 'room': ['bedrooms', 'bathrooms'], 'categorical': ['province', 'district', 'location_cluster']}
    },
    'shop': {
        'params': {'objective': 'regression_l1', 'random_state': 42, 'n_estimators': 1200, 'learning_rate': 0.03},
        'features': {'numerical': ['size_m2', 'latitude', 'longitude'], 'room': ['bathrooms'], 'categorical': ['province', 'district', 'location_cluster']}
    },
    'general_fallback': {
        'params': {'objective': 'regression_l1', 'random_state': 42, 'n_estimators': 1500, 'learning_rate': 0.02, 'num_leaves': 40, 'max_depth': 7},
        'features': {'numerical': ['size_m2', 'latitude', 'longitude'], 'room': ['bedrooms', 'bathrooms'], 'categorical': ['province', 'district', 'location_cluster', 'property_type']}
    }
}
# --- 2. HELPER FUNCTIONS (No changes here) ---
# ... (Keep plot_feature_importance and create_and_train_pipeline the same as my last correct version) ...
def plot_feature_importance(pipeline, prop_type, output_dir):
    try:
        regressor = pipeline.named_steps['regressor']
        feature_names = pipeline.named_steps['preprocessor'].get_feature_names_out()
        if not hasattr(regressor, 'feature_importances_'):
            print("   -> Regressor does not have feature importances. Skipping plot.")
            return
        feature_importances = pd.Series(regressor.feature_importances_, index=feature_names).sort_values(ascending=False)
        plt.figure(figsize=(10, 8))
        top_features = feature_importances.head(20)
        plt.barh(top_features.index, top_features.values, color='skyblue')
        plt.gca().invert_yaxis()
        plt.title(f'Top 20 Feature Importance - {prop_type.replace("_", " ").title()}', fontsize=16)
        plt.xlabel('Feature Importance (Gain)', fontsize=12)
        plt.tight_layout()
        plot_path = os.path.join(output_dir, f'feature_importance_{prop_type}.png')
        plt.savefig(plot_path)
        print(f"   -> Feature importance plot saved to '{plot_path}'")
        plt.close()
    except Exception as e:
        print(f"   -> Could not generate feature importance plot for {prop_type}: {e}")

def create_and_train_pipeline(X, y, params, feature_config): # Simplified signature
    numerical_features_to_use = [col for col in feature_config.get('numerical', []) if col in X.columns]
    room_features_to_use = [col for col in feature_config.get('room', []) if col in X.columns]
    categorical_features_to_use = [col for col in feature_config.get('categorical', []) if col in X.columns]
    transformers = []
    if numerical_features_to_use:
        transformers.append(('num', StandardScaler(), numerical_features_to_use))
    if room_features_to_use:
        transformers.append(('room', Pipeline(steps=[('imputer', SimpleImputer(strategy='median')), ('scaler', StandardScaler())]), room_features_to_use))
    if categorical_features_to_use:
        transformers.append(('cat', OneHotEncoder(handle_unknown='ignore', sparse_output=False), categorical_features_to_use))
    preprocessor = ColumnTransformer(transformers=transformers, remainder='drop', verbose_feature_names_out=False)
    pipeline = Pipeline(steps=[('preprocessor', preprocessor), ('regressor', lgb.LGBMRegressor(**params))])
    return pipeline


# --- 3. MAIN WORKFLOW (This is where all the changes are) ---

def main():
    """Main function to run the final hybrid model training workflow."""
    print("="*60)
    print("Starting the Final Hybrid Model Training Process...")
    print(f"-> Models will be saved to: '{os.path.abspath(OUTPUT_DIR)}'")
    print(f"-> Loading data from: '{os.path.abspath(DATA_PATH)}'")
    print("="*60)

    # --- DATA LOADING AND CLEANING (Same as before) ---
    try:
        df = pd.read_csv(DATA_PATH)
    except FileNotFoundError:
        print(f"ERROR: Data file not found at '{DATA_PATH}'.")
        return
    df.drop_duplicates(subset=['id'], inplace=True)
    df.rename(columns={'type': 'property_type'}, inplace=True)
    df['property_type'] = df['property_type'].str.lower().str.strip()
    df = df[(df['price_$'] > 1000) & (df['size_m2'] > 10)].copy()
    df = df[df['property_type'] != 'land'].copy()
    df['bedrooms'].replace(0, np.nan, inplace=True)
    df['bathrooms'].replace(0, np.nan, inplace=True)
    kmeans = KMeans(n_clusters=20, random_state=42, n_init='auto')
    df['location_cluster'] = kmeans.fit_predict(df[['latitude', 'longitude']]).astype(str)
    df_clean = df.groupby('property_type', group_keys=False).apply(
        lambda x: x[(x['price_$'] >= x['price_$'].quantile(0.01)) & (x['price_$'] <= x['price_$'].quantile(0.99))]
    )
    df_clean['log_price'] = np.log1p(df_clean['price_$'])

    # --- MODEL TRAINING ---
    print("\nStep 2: Training Specialist & Generalist Models...")
    
    model_groups = {**{pt: 'specialist' for pt in ['apartment', 'office', 'shop']}, 'general_fallback': 'generalist'}
    all_trained_types = []

    for prop_type, model_class in model_groups.items():
        print(f"\n--- Processing Model for: {prop_type.upper()} ---")
        
        if model_class == 'specialist':
            df_slice = df_clean[df_clean['property_type'] == prop_type].copy()
            print(f"Found {len(df_slice)} cleaned samples.")
            if len(df_slice) < 50:
                print("   -> Insufficient data. Skipping specialist model, will be handled by generalist.")
                continue
        else: # Generalist model
            df_slice = df_clean[~df_clean['property_type'].isin(all_trained_types)].copy()
            print(f"Training generalist model on remaining {len(df_slice)} samples.")

        config = MODEL_CONFIG[prop_type]
        
        # ** THE ERROR FIX IS HERE **
        all_potential_features = config['features'].get('numerical', []) + config['features'].get('room', []) + config['features'].get('categorical', [])
        features_to_use = [col for col in all_potential_features if col in df_slice.columns]
        print(f"   -> Using {len(features_to_use)} available features.")

        if not features_to_use:
            print("   -> No usable features found. Skipping.")
            continue
            
        X = df_slice[features_to_use]
        y = df_slice['log_price']
        
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        # ** PHASE 1: CROSS-VALIDATION TO GET A RELIABLE SCORE **
        print("   Phase 1: Running 5-Fold Cross-Validation for robust evaluation...")
        pipeline_for_cv = create_and_train_pipeline(X_train, y_train, config['params'], config['features'])
        cv_splitter = KFold(n_splits=5, shuffle=True, random_state=42)
        cv_scores = cross_val_score(pipeline_for_cv, X_train, y_train, cv=cv_splitter, scoring='r2')
        print(f"   -> CV R² Scores: {[round(s, 4) for s in cv_scores]}")
        print(f"   -> Average CV R²: {np.mean(cv_scores):.4f} (± {np.std(cv_scores):.4f})")

        # ** PHASE 2: FINAL MODEL TRAINING & EVALUATION **
        print("   Phase 2: Training final model on full training data...")
        final_pipeline = create_and_train_pipeline(X_train, y_train, config['params'], config['features'])
        final_pipeline.fit(X_train, y_train)
        
        # Evaluate on the held-out test set
        y_pred_log = final_pipeline.predict(X_test)
        y_pred_dollars = np.expm1(y_pred_log)
        y_test_dollars = np.expm1(y_test)
        
        final_r2 = r2_score(y_test, y_pred_log) 
        final_mae = mean_absolute_error(y_test_dollars, y_pred_dollars)
        print(f"   -> Final Test Set Performance -> R²: {final_r2:.4f}, MAE: ${final_mae:,.2f}")

        # Save model if it's good enough (or if it's the generalist)
        if model_class == 'generalist' or (model_class == 'specialist' and np.mean(cv_scores) > 0.65):
            model_path = os.path.join(OUTPUT_DIR, f'model_{prop_type}.joblib')
            joblib.dump(final_pipeline, model_path)
            if model_class == 'specialist': all_trained_types.append(prop_type)
            print(f"   -> Model saved to '{model_path}'")
            plot_feature_importance(final_pipeline, prop_type, OUTPUT_DIR)
        else:
            print(f"   -> Model performance below threshold. Not saving specialist model.")

    # --- SAVE SUPPORTING ARTIFACTS ---
    print("\nStep 3: Saving Supporting Artifacts...")
    joblib.dump(kmeans, os.path.join(OUTPUT_DIR, 'kmeans_model.joblib'))
    print(f"   -> K-Means model saved.")
    
    print("\n" + "="*60)
    print("--- WORKFLOW COMPLETE ---")
    print(f"All models and artifacts saved to: '{os.path.abspath(OUTPUT_DIR)}'")
    print("="*60)

if __name__ == '__main__':
    main()