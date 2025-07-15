
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
from sklearn.model_selection import train_test_split, cross_val_score, KFold
from sklearn.cluster import KMeans
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.metrics import r2_score, mean_squared_error
import joblib
import warnings
import os

# --- 1. SETUP & CONFIGURATION ---
warnings.filterwarnings('ignore', category=UserWarning)
warnings.filterwarnings('ignore', category=FutureWarning)
pd.options.mode.chained_assignment = None

# --- Configuration for paths ---
BACKEND_DIR = os.path.dirname(os.path.abspath(__file__))
OUTPUT_DIR = os.path.join(BACKEND_DIR, 'final_grouped_models')
DATA_PATH = os.path.join(BACKEND_DIR, 'properties.csv')
os.makedirs(OUTPUT_DIR, exist_ok=True)

# --- 2. HELPER FUNCTIONS ---

def plot_feature_importance(pipeline, model_group, output_dir):
    """Saves a readable feature importance plot."""
    try:
        regressor = pipeline.named_steps['regressor']
        feature_names = pipeline.named_steps['preprocessor'].get_feature_names_out()
        
        feature_importances = pd.Series(regressor.feature_importances_, index=feature_names).sort_values(ascending=False)
        
        plt.figure(figsize=(10, 8))
        top_features = feature_importances.head(20)
        plt.barh(top_features.index, top_features.values, color='teal')
        plt.gca().invert_yaxis()
        plt.title(f'Top 20 Feature Importance - {model_group.replace("_", " ").title()}', fontsize=16)
        plt.xlabel('Feature Importance (Gain)', fontsize=12)
        plt.tight_layout()
        
        plot_path = os.path.join(output_dir, f'feature_importance_{model_group}.png')
        plt.savefig(plot_path)
        print(f"   -> Feature importance plot saved to '{plot_path}'")
        plt.close()
    except Exception as e:
        print(f"   -> Could not generate feature importance plot for {model_group}: {e}")

# --- 3. MAIN WORKFLOW ---

def main():
    """Main function to run the final grouped model training workflow."""
    print("="*60)
    print("Starting the NEW Grouped Model Training Process...")
    print(f"-> Models will be saved to: '{os.path.abspath(OUTPUT_DIR)}'")
    print("="*60)

    # --- Step 1: Data Loading & Initial Cleaning ---
    print("\n--- Step 1: Loading and Initial Cleaning ---")
    try:
        df = pd.read_csv(DATA_PATH)
    except FileNotFoundError:
        print(f"ERROR: Data file not found at '{DATA_PATH}'.")
        return
        
    df.drop_duplicates(subset=['id'], inplace=True)
    df.rename(columns={'type': 'property_type'}, inplace=True)
    df['property_type'] = df['property_type'].str.lower().str.strip()
    df = df[(df['price_$'] > 1000) & (df['size_m2'] > 10)].copy()
    print(f"Loaded {len(df)} properties after basic filtering.")
    
    # --- Step 2: Triage & Grouping ---
    print("\n--- Step 2: Triaging Data into Model Groups ---")
    
    def assign_model_group(ptype):
        if ptype in ['apartment', 'house/villa', 'land']:
            return ptype
        if ptype in ['office', 'shop', 'warehouse', 'commercial building']:
            return 'commercial_other'
        if ptype in ['chalet', 'residential building']:
            return 'residential_other'
        return 'discard' # For restaurants, gas stations, etc.

    df['model_group'] = df['property_type'].apply(assign_model_group)
    
    # Discard with no regrets
    initial_rows = len(df)
    df = df[df['model_group'] != 'discard'].copy()
    print(f"Discarded {initial_rows - len(df)} rows from rare types.")
    print("Group counts:\n", df['model_group'].value_counts())

    # --- Step 3: Global & Group-Specific Feature Engineering ---
    print("\n--- Step 3: Feature Engineering ---")

    # Global: K-Means Clustering
    kmeans = KMeans(n_clusters=20, random_state=42, n_init='auto')
    df['location_cluster'] = kmeans.fit_predict(df[['latitude', 'longitude']]).astype(str)
    print("   -> Created 'location_cluster' feature.")

    # Global: Intelligent Cleaning for rooms
    df['bedrooms'].replace(0, np.nan, inplace=True)
    df['bathrooms'].replace(0, np.nan, inplace=True)
    
    # Group-Specific Feature Engineering
    df['price_per_m2'] = df['price_$'] / df['size_m2']
    df['size_per_room'] = df['size_m2'] / (df['bedrooms'].fillna(0) + df['bathrooms'].fillna(0) + 1)
    
    # Outlier handling on the new target variable
    df_clean = df.groupby('model_group', group_keys=False).apply(
        lambda x: x[(x['price_per_m2'] >= x['price_per_m2'].quantile(0.01)) & (x['price_per_m2'] <= x['price_per_m2'].quantile(0.99))]
    )
    print(f"   -> Created 'price_per_m2' and 'size_per_room' features.")
    print(f"   -> Removed price_per_m2 outliers. {len(df) - len(df_clean)} rows removed.")
    
    # Log-transform the new target
    df_clean['log_price_per_m2'] = np.log1p(df_clean['price_per_m2'])

    # --- Step 4: Model Training Loop ---
    print("\n--- Step 4: Training One Model Per Group ---")
    
    for group_name in df_clean['model_group'].unique():
        print(f"\n--- Processing Model for Group: {group_name.upper()} ---")
        
        df_group = df_clean[df_clean['model_group'] == group_name]
        
        # Define features for this group
        numerical_features = ['size_m2', 'latitude', 'longitude']
        categorical_features = ['province', 'district', 'location_cluster']
        
        # Add special features only for relevant groups
        if group_name in ['apartment', 'house/villa', 'residential_other']:
            numerical_features.extend(['bedrooms', 'bathrooms', 'size_per_room'])
        elif group_name == 'commercial_other':
            # Commercial properties might have bathrooms but not bedrooms
            numerical_features.append('bathrooms')
        # 'Land' model gets no room features
            
        # Define the pipeline dynamically
        numeric_transformer = Pipeline(steps=[
            ('imputer', SimpleImputer(strategy='median')),
            ('scaler', StandardScaler())
        ])
        
        preprocessor = ColumnTransformer(
            transformers=[
                ('num', numeric_transformer, [f for f in numerical_features if f in df_group.columns]),
                ('cat', OneHotEncoder(handle_unknown='ignore', sparse_output=False), [f for f in categorical_features if f in df_group.columns])
            ],
            remainder='drop',
            verbose_feature_names_out=False
        )
        
        pipeline = Pipeline(steps=[
            ('preprocessor', preprocessor),
            ('regressor', lgb.LGBMRegressor(objective='regression_l1', random_state=42, n_estimators=1000))
        ])
        
        X = df_group.drop(columns=['price_$', 'price_per_m2', 'log_price_per_m2'])
        y = df_group['log_price_per_m2']

        # Split data for this group
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        
        # Phase 1: Cross-Validation for robust evaluation
        print("   Phase 1: Running 5-Fold CV...")
        cv_splitter = KFold(n_splits=5, shuffle=True, random_state=42)
        # Set error_score to handle potential issues gracefully in any small fold
        cv_scores = cross_val_score(pipeline, X_train, y_train, cv=cv_splitter, scoring='r2', error_score='raise')
        print(f"   -> Average CV R²: {np.mean(cv_scores):.4f} (± {np.std(cv_scores):.4f})")
        
        # Phase 2: Final model training
        print("   Phase 2: Training final model...")
        pipeline.fit(X_train, y_train)
        
        # Evaluate on the held-out test set
        y_pred_log = pipeline.predict(X_test)
        
        # Convert predictions back to dollar price
        # Predicted price = predicted_price_per_m2 * size_m2
        size_m2_test = X_test['size_m2']
        predicted_price_per_m2 = np.expm1(y_pred_log)
        predicted_dollars = predicted_price_per_m2 * size_m2_test
        
        actual_dollars = df_group.loc[y_test.index]['price_$']

        final_r2 = r2_score(y_test, y_pred_log)
        final_rmse = np.sqrt(mean_squared_error(actual_dollars, predicted_dollars))
        print(f"   -> Final Test Performance -> R² (log-scale): {final_r2:.4f}, RMSE (dollars): ${final_rmse:,.2f}")
        
        # Save the model
        # Sanitize the group_name for use in a filename
        safe_group_name = group_name.replace('/', '_') 
        model_path = os.path.join(OUTPUT_DIR, f'model_{safe_group_name}.joblib')
        joblib.dump(pipeline, model_path)
        print(f"   -> Model saved to '{model_path}'")
        plot_feature_importance(pipeline, group_name, OUTPUT_DIR)

    # --- SAVE SUPPORTING ARTIFACTS ---
    print("\n--- Step 5: Saving Supporting Artifacts ---")
    joblib.dump(kmeans, os.path.join(OUTPUT_DIR, 'kmeans_model.joblib'))
    print(f"   -> K-Means model saved.")
    
    print("\n" + "="*60)
    print("--- WORKFLOW COMPLETE ---")
    print(f"All models saved to: '{os.path.abspath(OUTPUT_DIR)}'")
    print("="*60)

if __name__ == '__main__':
    main()