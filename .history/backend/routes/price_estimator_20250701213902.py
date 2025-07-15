
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


import pandas as pd
import numpy as np
import lightgbm as lgb
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.cluster import KMeans
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.metrics import r2_score, mean_absolute_error
import joblib
import warnings
import os

# --- Setup ---
warnings.filterwarnings('ignore', category=UserWarning)
warnings.filterwarnings('ignore', category=FutureWarning)
pd.options.mode.chained_assignment = None

# --- Helper Functions ---

def plot_feature_importance(pipeline, prop_type, output_dir):
    """Saves a readable feature importance plot to the specified directory."""
    regressor = pipeline.named_steps['regressor']
    if not hasattr(regressor, 'feature_importances_') or not hasattr(regressor, 'feature_name_'):
        return
        
    feature_importances = pd.Series(regressor.feature_importances_, index=regressor.feature_name_).sort_values(ascending=False)
    plt.figure(figsize=(10, 8))
    top_features = feature_importances.head(15)
    plt.barh(top_features.index, top_features.values)
    plt.gca().invert_yaxis()
    plt.title(f'Top 15 Feature Importance - {prop_type.title()}', fontsize=16)
    plt.xlabel('Feature Importance (Gain)', fontsize=12)
    plt.tight_layout()
    
    plot_path = os.path.join(output_dir, f'feature_importance_{prop_type}.png')
    plt.savefig(plot_path)
    print(f"   -> Feature importance plot saved to '{plot_path}'")
    plt.close()

def create_and_train_pipeline(X_train, y_train, X_test, y_test, params, feature_config):
    """Creates, trains, evaluates, and returns a model pipeline."""
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', StandardScaler(), feature_config['numerical']),
            ('cat', OneHotEncoder(handle_unknown='ignore', sparse_output=False), feature_config['categorical'])
        ],
        remainder='drop',
        verbose_feature_names_out=False
    )
    pipeline = Pipeline(steps=[
        ('preprocessor', preprocessor),
        ('regressor', lgb.LGBMRegressor(**params))
    ])
    
    preprocessor.fit(X_train)
    final_feature_names = list(preprocessor.get_feature_names_out())
    
    pipeline.fit(X_train, y_train, regressor__feature_name=final_feature_names)
    
    y_pred_log = pipeline.predict(X_test)
    y_pred_dollars = np.expm1(y_pred_log)
    y_test_dollars = np.expm1(y_test)
    
    r2 = r2_score(y_test_dollars, y_pred_dollars)
    mae = mean_absolute_error(y_test_dollars, y_pred_dollars)
    
    print(f"Result -> R²: {r2:.4f}, MAE: ${mae:,.2f}")
    
    return pipeline, r2

def main():
    """Main function to run the final hybrid model training workflow."""
    print("Starting the Final Hybrid Model Training Process...")
    
    # --- 1. DEFINE PATHS CORRECTLY ---
    # Assumes this script is run from within the 'backend' directory or a subdirectory like 'routes'.
    try:
        # This handles running from both 'backend/' and 'backend/routes/'
        script_path = os.path.dirname(os.path.abspath(__file__))
        if os.path.basename(script_path) == 'routes':
            BACKEND_DIR = os.path.dirname(script_path)
        else:
            BACKEND_DIR = script_path
    except NameError:
         # Fallback for interactive environments
        BACKEND_DIR = os.getcwd()

    OUTPUT_DIR = os.path.join(BACKEND_DIR, 'final_models')
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    print(f"-> Models will be saved to: '{os.path.abspath(OUTPUT_DIR)}'")
    
    # --- 2. MODEL CONFIGURATION ---
    MODEL_CONFIG = {
        'apartment': {
            'params': {'objective': 'regression_l1', 'random_state': 42, 'n_estimators': 1500, 'learning_rate': 0.02, 'num_leaves': 40, 'max_depth': 8},
            'features': {'numerical': ['size_m2', 'latitude', 'longitude', 'bedrooms', 'bathrooms'], 'categorical': ['province', 'district', 'location_cluster']}
        },
        'office': {
            'params': {'objective': 'regression_l1', 'random_state': 42, 'n_estimators': 1000, 'learning_rate': 0.03, 'num_leaves': 30, 'max_depth': 6},
            'features': {'numerical': ['size_m2', 'latitude', 'longitude', 'bedrooms', 'bathrooms'], 'categorical': ['province', 'district', 'location_cluster']}
        },
        'shop': {
            'params': {'objective': 'regression_l1', 'random_state': 42, 'n_estimators': 1200, 'learning_rate': 0.03},
            'features': {'numerical': ['size_m2', 'latitude', 'longitude', 'bathrooms'], 'categorical': ['province', 'district', 'location_cluster']}
        }
    }
    
    # --- 3. DATA PREPARATION ---
    df = pd.read_csv('properties.csv')
    df.drop_duplicates(subset=['id'], inplace=True)
    df = df[(df['price_$'] > 1000) & (df['size_m2'] > 10)].copy()
    df['type'] = df['type'].str.lower().str.strip()
    df = df[df['type'] != 'land'].copy()
    print(f"Removed 'Land' properties. Working with {len(df)} properties.")
    df.drop(columns=['created_at', 'city'], inplace=True, errors='ignore')

    # --- 4. GLOBAL FEATURE ENGINEERING ---
    print("\nCreating global K-Means model for location clustering...")
    kmeans = KMeans(n_clusters=20, random_state=42, n_init='auto')
    df['location_cluster'] = kmeans.fit_predict(df[['latitude', 'longitude']]).astype(str)
    
    # --- 5. TRAINING SPECIALIST MODELS ---
    specialist_types = ['apartment', 'office', 'shop']
    for prop_type in specialist_types:
        print(f"\n--- Training Specialist Model for: {prop_type.upper()} ---")
        
        df_slice = df[df['type'] == prop_type].copy()
        print(f"Found {len(df_slice)} samples for type '{prop_type}'.")
        
        if len(df_slice) < 50:
            print("   -> Insufficient data. This type will be handled by the generalist model.")
            continue
        
        config = MODEL_CONFIG[prop_type]
        for col in config['features']['numerical'] + config['features']['categorical']:
            if col not in df_slice.columns:
                df_slice[col] = 0

        X = df_slice.drop(columns=['price_$', 'type'], errors='ignore')
        y = np.log1p(df_slice['price_$'])
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        pipeline, r2 = create_and_train_pipeline(X_train, y_train, X_test, y_test, config['params'], config['features'])

        if r2 > 0.65:
            model_path = os.path.join(OUTPUT_DIR, f'model_{prop_type}.joblib')
            joblib.dump(pipeline, model_path)
            print(f"   -> High-quality model saved to '{model_path}'")
            plot_feature_importance(pipeline, prop_type, OUTPUT_DIR)
        else:
            print(f"   -> Model performance below threshold. This type will use the Generalist Fallback.")

    # --- 6. TRAINING GENERALIST FALLBACK MODEL (ON ALL DATA EXCEPT LAND) ---
    print("\n--- Training Generalist Fallback Model ---")
    df_general = df.copy() 

    general_config = {
        'params': {'objective': 'regression_l1', 'random_state': 42, 'n_estimators': 1500, 'learning_rate': 0.02, 'num_leaves': 40, 'max_depth': 7},
        'features': {'numerical': ['size_m2', 'latitude', 'longitude', 'bedrooms', 'bathrooms'], 'categorical': ['province', 'district', 'location_cluster', 'type']}
    }
    
    X_general = df_general.drop(columns=['price_$'])
    y_general = np.log1p(df_general['price_$'])
    X_train_gen, X_test_gen, y_train_gen, y_test_gen = train_test_split(X_general, y_general, test_size=0.2, random_state=42)
    
    general_pipeline, _ = create_and_train_pipeline(
        X_train_gen, y_train_gen, X_test_gen, y_test_gen, general_config['params'], general_config['features']
    )
    
    model_path = os.path.join(OUTPUT_DIR, 'model_general_fallback.joblib')
    joblib.dump(general_pipeline, model_path)
    print(f"   -> Generalist Fallback Model saved to '{model_path}'")
    plot_feature_importance(general_pipeline, "general_fallback", OUTPUT_DIR)

    # --- 7. SAVE SUPPORTING ARTIFACTS ---
    joblib.dump(kmeans, os.path.join(OUTPUT_DIR, 'kmeans_model.joblib'))
    
    print(f"\n\n--- Workflow Complete ---")
    print(f"All necessary models and plots have been generated and saved to the '{os.path.abspath(OUTPUT_DIR)}' directory.")

if __name__ == '__main__':
    main()


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

# def plot_feature_importance(pipeline, prop_type, output_dir):
#     """Saves a readable feature importance plot for a trained model."""
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

# def process_and_clean_data(df):
#     """Performs all initial cleaning and feature engineering."""
#     df.drop_duplicates(subset=['id'], inplace=True)
#     df = df[(df['price_$'] > 1000) & (df['size_m2'] > 10)].copy()
#     df['type'] = df['type'].str.lower().str.strip()
#     df = df[df['type'] != 'land'].copy()
#     df.drop(columns=['created_at', 'city'], inplace=True, errors='ignore')
    
#     # Impute missing rooms
#     df['bedrooms'].fillna(df.groupby('type')['bedrooms'].transform('median'), inplace=True)
#     df['bathrooms'].fillna(df.groupby('type')['bathrooms'].transform('median'), inplace=True)
#     df.fillna({'bedrooms': 1, 'bathrooms': 1}, inplace=True)
#     df[['bedrooms', 'bathrooms']] = df[['bedrooms', 'bathrooms']].astype(int)

#     return df

# def main():
#     """Main function to run the final hybrid model training workflow."""
#     print("Starting the Final Hybrid Model Training Process...")
    
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
    
#     output_dir = 'final_models'
#     os.makedirs(output_dir, exist_ok=True)
    
#     df = pd.read_csv('properties.csv')
#     df_clean = process_and_clean_data(df)

#     # --- Train K-Means on ALL data to create a universal location map ---
#     print("\nCreating global K-Means model for location clustering...")
#     kmeans = KMeans(n_clusters=20, random_state=42, n_init='auto')
#     df_clean['location_cluster'] = kmeans.fit_predict(df_clean[['latitude', 'longitude']]).astype(str)
    
#     # --- ISOLATED TRAINING FOR SPECIALIST MODELS ---
#     specialist_types = ['apartment', 'office', 'shop']
#     generalist_data_frames = []

#     for prop_type in df_clean['type'].unique():
#         if prop_type in specialist_types:
#             print(f"\n--- Training Specialist Model for: {prop_type.upper()} ---")
#             df_slice = df_clean[df_clean['type'] == prop_type].copy()
            
#             print(f"Found {len(df_slice)} samples for type '{prop_type}'.")
#             if len(df_slice) < 50:
#                 print("   -> Insufficient data, adding to generalist pool.")
#                 generalist_data_frames.append(df_slice)
#                 continue

#             config = MODEL_CONFIG[prop_type]
#             X = df_slice.drop(columns=['price_$', 'type'])
#             y = np.log1p(df_slice['price_$'])
#             X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

#             pipeline, r2 = create_and_train_pipeline(X_train, y_train, X_test, y_test, config['params'], config['features'])

#             if r2 > 0.65:
#                 model_path = os.path.join(output_dir, f'model_{prop_type}.joblib')
#                 joblib.dump(pipeline, model_path)
#                 print(f"   -> High-quality model saved to '{model_path}'")
#                 plot_feature_importance(pipeline, prop_type, output_dir)
#             else:
#                 print("   -> Model performance below threshold, adding to generalist pool.")
#                 generalist_data_frames.append(df_slice)
#         else:
#             # If it's not a specialist type, add it to the generalist pool
#             generalist_data_frames.append(df_clean[df_clean['type'] == prop_type])

#     # --- TRAINING GENERALIST FALLBACK MODEL ---
#     print("\n--- Training Generalist Fallback Model ---")
#     if not generalist_data_frames:
#         print("No data left for the generalist model. Skipping.")
#     else:
#         df_general = pd.concat(generalist_data_frames, ignore_index=True)
#         print(f"The generalist model will be trained on {len(df_general)} samples from types: {list(df_general['type'].unique())}")
        
#         general_config = {
#             'params': {'objective': 'regression_l1', 'random_state': 42, 'n_estimators': 1500, 'learning_rate': 0.02, 'num_leaves': 40, 'max_depth': 7},
#             'features': {'numerical': ['size_m2', 'latitude', 'longitude', 'bedrooms', 'bathrooms'], 'categorical': ['province', 'district', 'location_cluster', 'type']}
#         }
        
#         X_general = df_general.drop(columns=['price_$'])
#         y_general = np.log1p(df_general['price_$'])
#         X_train_gen, X_test_gen, y_train_gen, y_test_gen = train_test_split(X_general, y_general, test_size=0.2, random_state=42)
        
#         general_pipeline, _ = create_and_train_pipeline(
#             X_train_gen, y_train_gen, X_test_gen, y_test_gen, general_config['params'], general_config['features']
#         )
        
#         model_path = os.path.join(output_dir, 'model_general_fallback.joblib')
#         joblib.dump(general_pipeline, model_path)
#         print(f"   -> Generalist Fallback Model saved to '{model_path}'")
#         plot_feature_importance(general_pipeline, "general_fallback", output_dir)

#     joblib.dump(kmeans, os.path.join(output_dir, 'kmeans_model.joblib'))
    
#     print(f"\n\n--- Workflow Complete ---")
#     print(f"All necessary models and plots have been generated and saved to the '{output_dir}' directory.")

# if __name__ == '__main__':
#     main()