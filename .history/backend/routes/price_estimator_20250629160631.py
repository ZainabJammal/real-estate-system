# import pandas as pd
# import numpy as np
# from catboost import CatBoostRegressor
# from sklearn.model_selection import train_test_split
# from sklearn.metrics import r2_score, mean_absolute_error
# import joblib
# import os
# import warnings

# warnings.filterwarnings('ignore', category=UserWarning)

# # --- 1. Your Custom Hyperparameters and Feature Definitions ---
# # # This dictionary holds the tailored parameters for each model
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
# MODEL_CONFIG = {
#     'apartment': {
#         'params': {
#             'depth': 8,  # Good for complex relationships
#             'learning_rate': 0.02,  # Slightly reduced for better convergence
#             'l2_leaf_reg': 3,  # Reduced regularization (your model is performing well)
#             'iterations': 2500,  # Increased slightly
#             'grow_policy': 'SymmetricTree',  # Better for medium-sized datasets
#             'bootstrap_type': 'Bayesian',  # Better uncertainty estimation
#             'random_strength': 1,  # Helps with generalization
#             'border_count': 128,  # Good default for medium datasets
#             'early_stopping_rounds': 50  # Prevent overfitting
#         },
#         'features': {
#             'numerical': ['log_size_m2', 'bedrooms', 'bathrooms'],
#             'categorical': ['province', 'district', 'city']
#         }
#     },
#     'land': {
#    'params': {'depth': 6, 'learning_rate': 0.01, 'l2_leaf_reg': 10, 'iterations': 2000},
#          'features': {'numerical': ['log_size_m2'], 'categorical': ['province', 'district', 'city']}
#     },
#     'house/villa': {
#    'params': {
#             'depth': 5,  # Reduced from 7
#             'learning_rate': 0.01,  # Slower learning
#             'l2_leaf_reg': 10,  # Increased regularization  
#             'iterations': 2500,
#             'grow_policy': 'Lossguide',
#             'max_leaves': 32,  # Limit complexity
#             'min_data_in_leaf': 5  # ~3% of data
#         },
#         'features': {
#             'numerical': ['log_size_m2', 'bathrooms'],  # Removed bedrooms
#             'categorical': ['district', 'city']  # Removed province
#         }
#     },
#     'office': {
#      'params': {
#             'depth': 6,
#             'learning_rate': 0.03,
#             'l2_leaf_reg': 5,
#             'iterations': 2000,
#             'one_hot_max_size': 15,  # Increased from 10
#             'border_count': 64  # Reduced for stability
#         },
#         'features': {
#             'numerical': ['log_size_m2', 'bedrooms', 'bathrooms'],
#             'categorical': ['province', 'district', 'city']
#         }
#     },
# 'shop': {
#     'params': {
#         'learning_rate': 0.03,  # Between previous and current
#         'l2_leaf_reg': 3,  # Reduced regularization
#         'iterations': 2000
#     },
#     'features': {
#         'numerical': ['log_size_m2'],
#         'categorical': ['province', 'district', 'city']
#     }
# },
# 'chalet': {
#     'params': {
#             'depth': 5,  # Reduced from 6
#             'learning_rate': 0.02,  # Adjusted
#             'l2_leaf_reg': 7,
#             'iterations': 2000,
#             'min_data_in_leaf': 3  # Added
#         },
#  'features': {'numerical': ['log_size_m2', 'bedrooms', 'bathrooms'], 'categorical': ['province', 'district', 'city']}
# #    
#     }

# }
# def train_specialist_model(df_segment, property_type, config):
#     """
#     Trains a specialized CatBoost model using tailored hyperparameters and features.
#     """
#     print(f"\n--- Training Specialist Model for: {property_type.upper()} ---")
    
#     # 1. Feature Engineering
#     df = df_segment.copy()
    
#     # ** NEW TARGET VARIABLE: log_price_per_sqm **
#     df['log_price_per_sqm'] = np.log1p(df['price_$'] / df['size_m2'])
#     df['log_size_m2'] = np.log1p(df['size_m2'])
    
#     features_to_use = config['features']['numerical'] + config['features']['categorical']
    
#     y = df['log_price_per_sqm']
#     X = df[features_to_use]
    
#     # 2. Data Split
#     X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
#     # 3. Model Training
#     params = config['params']
#     model = CatBoostRegressor(
#         loss_function='RMSE', # A standard for regression
#         eval_metric='R2',
#         random_seed=42,
#         verbose=0,
#         cat_features=config['features']['categorical'],
#         **params
#     )
    
#     model.fit(
#         X_train, y_train,
#         eval_set=(X_test, y_test),
#         early_stopping_rounds=50,
#         verbose=False
#     )
    
#     # 4. Evaluation
#     y_pred_log_per_sqm = model.predict(X_test)
    
#     # Convert prediction back to total dollar value
#     test_log_size = X_test['log_size_m2']
#     test_size = np.expm1(test_log_size)
    
#     pred_price_per_sqm = np.expm1(y_pred_log_per_sqm)
#     y_pred_dollars = pred_price_per_sqm * test_size
    
#     # Get original test prices for comparison
#     y_test_dollars = np.expm1(y_test) * test_size

#     r2 = r2_score(y_test_dollars, y_pred_dollars)
#     mae = mean_absolute_error(y_test_dollars, y_pred_dollars)
    
#     print(f"Result -> R²: {r2:.4f}, MAE: ${mae:,.2f}")
    
#     # 5. Feature Importance
#     feature_importances = pd.Series(model.get_feature_importance(), index=X.columns).sort_values(ascending=False)
#     print("Top 5 Features:")
#     print(feature_importances.head(5).to_string())
    
#     return model, r2

# def main():
#     print("Step 1: Loading and preparing base data...")
#     df = pd.read_csv('properties.csv')
#     df.drop_duplicates(subset=['id'], inplace=True)
#     df = df[df['price_$'] > 10000].copy()
#     df = df[df['size_m2'] > 20].copy()
    
#     for col in ['type', 'province', 'district', 'city']:
#         df[col] = df[col].astype(str).str.lower().str.strip()
        
#     df['bedrooms'].fillna(df.groupby('type')['bedrooms'].transform('median'), inplace=True)
#     df['bathrooms'].fillna(df.groupby('type')['bathrooms'].transform('median'), inplace=True)
#     df.fillna({'bedrooms': 1, 'bathrooms': 1}, inplace=True)
#     df[['bedrooms', 'bathrooms']] = df[['bedrooms', 'bathrooms']].astype(int)
    
#     output_dir = "ultimate_specialist_models"
#     if not os.path.exists(output_dir):
#         os.makedirs(output_dir)
        
#     viable_types = list(MODEL_CONFIG.keys())
#     print(f"\nTraining specialist models for: {viable_types}")

#     for prop_type in viable_types:
#         df_segment = df[df['type'] == prop_type]
#         if len(df_segment) < 50: # Check if there is enough data
#             print(f"\n--- Skipping {prop_type.upper()}: Not enough data ({len(df_segment)} rows) ---")
#             continue

#         model, r2 = train_specialist_model(df_segment, prop_type, MODEL_CONFIG[prop_type])
        
#         if r2 > 0.50: # A reasonable threshold for saving a specialist model
#             filename = f"model_{prop_type.replace('/', '_')}.joblib"
#             filepath = os.path.join(output_dir, filename)
#             joblib.dump(model, filepath)
#             print(f"   -> Model saved to '{filepath}'")
#         else:
#             print("   -> Model performance is too low, not saving.")
            
#     print("\n--- Process Complete ---")

# if __name__ == '__main__':
#     main()







# import pandas as pd
# import numpy as np
# import lightgbm as lgb
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

# def create_and_train_pipeline(X_train, y_train, X_test, y_test, params, feature_config):
#     """
#     A generic function to create, train, and evaluate a model pipeline.
#     This centralizes the logic to avoid repetition and errors.
#     """
#     preprocessor = ColumnTransformer(
#         transformers=[
#             ('num', StandardScaler(), feature_config['numerical']),
#             ('cat', OneHotEncoder(handle_unknown='ignore', sparse_output=False), feature_config['categorical'])
#         ],
#         remainder='drop'
#     )
    
#     pipeline = Pipeline(steps=[
#         ('preprocessor', preprocessor),
#         ('regressor', lgb.LGBMRegressor(**params))
#     ])
    
#     pipeline.fit(X_train, y_train)
    
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
    
#     # --- MODEL CONFIGURATION ---
#     # Finalized parameters and features for our high-confidence specialist models.
#     MODEL_CONFIG = {
#         'apartment': {
#             'params': {'objective': 'regression_l1', 'random_state': 42, 'n_estimators': 1500, 'learning_rate': 0.02, 'num_leaves': 40, 'max_depth': 8},
#             'features': {'numerical': ['size_m2', 'latitude', 'longitude', 'bedrooms', 'bathrooms'], 'categorical': ['province', 'district', 'location_cluster']}
#         },
#         'office': {
#             'params': {'objective': 'regression_l1', 'random_state': 42, 'n_estimators': 1000, 'learning_rate': 0.03, 'num_leaves': 30, 'max_depth': 6},
#             'features': {'numerical': ['size_m2', 'latitude', 'longitude', 'bedrooms'], 'categorical': ['province', 'district', 'location_cluster']}
#         },
#         'shop': {
#             'params': {'objective': 'regression_l1', 'random_state': 42, 'n_estimators': 1200, 'learning_rate': 0.03},
#             'features': {'numerical': ['size_m2', 'latitude', 'longitude'], 'categorical': ['province', 'district', 'location_cluster']}
#         }
#     }
    
#     # --- SETUP & DATA PREPARATION ---
#     os.makedirs('final_models', exist_ok=True)
#     df = pd.read_csv('properties.csv')
#     df.drop_duplicates(subset=['id'], inplace=True)
#     df = df[(df['price_$'] > 1000) & (df['size_m2'] > 10)].copy()
    
#     # *** STRATEGIC DECISION: Exclude 'Land' completely from the project ***
#     df = df[df['type'] != 'Land'].copy()
#     print(f"Removed 'Land' properties. Working with {len(df)} properties.")
    
#     df.drop(columns=['created_at', 'city'], inplace=True) # Drop unused columns

#     # --- GLOBAL FEATURE ENGINEERING ---
#     print("\nCreating global K-Means model for location clustering...")
#     kmeans = KMeans(n_clusters=20, random_state=42, n_init='auto')
#     df['location_cluster'] = kmeans.fit_predict(df[['latitude', 'longitude']]).astype(str)
    
#     # --- TRAINING SPECIALIST MODELS ---
#     specialist_types = ['apartment', 'office', 'shop']
#     for prop_type in specialist_types:
#         print(f"\n--- Training Specialist Model for: {prop_type.upper()} ---")
        
#         config = MODEL_CONFIG[prop_type]
#         # Normalize type names like 'House/Villa' for file saving
#         df_slice = df[df['type'].str.lower().replace({'/': '_'}) == prop_type].copy()

#         if len(df_slice) < 50:
#             print(f"   -> Insufficient data for {prop_type}, skipping specialist model.")
#             continue
        
#         X = df_slice.drop(columns=['price_$', 'type'])
#         y = np.log1p(df_slice['price_$'])
#         X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

#         pipeline, r2 = create_and_train_pipeline(X_train, y_train, X_test, y_test, config['params'], config['features'])

#         if r2 > 0.65: # Set a quality bar for saving specialist models
#             model_path = os.path.join('final_models', f'model_{prop_type}.joblib')
#             joblib.dump(pipeline, model_path)
#             print(f"   -> High-quality model saved to '{model_path}'")
#         else:
#             print(f"   -> Model performance is below threshold. This type will use the Generalist Fallback.")

#     # --- TRAINING GENERALIST FALLBACK MODEL ---
#     print("\n--- Training Generalist Fallback Model ---")
#     # The generalist is trained on ALL building types
#     df_general = df.copy() 

#     general_config = {
#         'params': {
#             'objective': 'regression_l1', 'random_state': 42, 'n_estimators': 1500,
#             'learning_rate': 0.02, 'num_leaves': 40, 'max_depth': 7
#         },
#         'features': {
#             'numerical': ['size_m2', 'latitude', 'longitude', 'bedrooms', 'bathrooms'],
#             'categorical': ['province', 'district', 'location_cluster', 'type'] # 'type' is now a feature
#         }
#     }
    
#     X_general = df_general.drop(columns=['price_$'])
#     y_general = np.log1p(df_general['price_$'])
#     X_train_gen, X_test_gen, y_train_gen, y_test_gen = train_test_split(X_general, y_general, test_size=0.2, random_state=42)
    
#     general_pipeline, _ = create_and_train_pipeline(
#         X_train_gen, y_train_gen, X_test_gen, y_test_gen,
#         general_config['params'], general_config['features']
#     )
    
#     model_path = os.path.join('final_models', 'model_general_fallback.joblib')
#     joblib.dump(general_pipeline, model_path)
#     print(f"   -> Generalist Fallback Model saved to '{model_path}'")

#     # Save the essential K-Means model
#     joblib.dump(kmeans, os.path.join('final_models', 'kmeans_model.joblib'))
    
#     print("\n\n--- Workflow Complete ---")
#     print("All necessary models for the hybrid system have been trained and saved to the 'final_models' directory.")

# def plot_feature_importance(pipeline, prop_type):
#     """Save feature importance plot for a trained model"""
#     import matplotlib.pyplot as plt
#     regressor = pipeline.named_steps['regressor']
#     preprocessor = pipeline.named_steps['preprocessor']
    
#     # Get feature names after preprocessing
#     num_features = preprocessor.named_transformers_['num'].get_feature_names_out()
#     cat_features = preprocessor.named_transformers_['cat'].get_feature_names_out()
#     all_features = np.concatenate([num_features, cat_features])
    
#     # Plot feature importance
#     plt.figure(figsize=(10, 6))
#     lgb.plot_importance(regressor, max_num_features=20, 
#                        title=f'Feature Importance - {prop_type.title()}',
#                        height=0.8)
#     plt.tight_layout()
#     plt.savefig(f'final_models/feature_importance_{prop_type}.png')
#     plt.close()

# if __name__ == '__main__':
#     main()


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

def plot_feature_importance(pipeline, prop_type):
    """Saves a readable feature importance plot for a trained model."""
    regressor = pipeline.named_steps['regressor']
    
    if not hasattr(regressor, 'feature_importances_') or not hasattr(regressor, 'feature_name_'):
        print(f"   -> Could not generate plot for {prop_type}, model lacks feature importance attributes.")
        return
        
    feature_importances = pd.Series(regressor.feature_importances_, index=regressor.feature_name_).sort_values(ascending=False)

    plt.figure(figsize=(10, 8))
    top_features = feature_importances.head(15)
    plt.barh(top_features.index, top_features.values)
    plt.gca().invert_yaxis()
    plt.title(f'Top 15 Feature Importance - {prop_type.title()}', fontsize=16)
    plt.xlabel('Feature Importance (Gain)', fontsize=12)
    plt.tight_layout()
    
    plot_path = os.path.join('final_models', f'feature_importance_{prop_type}.png')
    plt.savefig(plot_path)
    print(f"   -> Feature importance plot saved to '{plot_path}'")
    plt.close()

def create_and_train_pipeline(X_train, y_train, X_test, y_test, params, feature_config):
    """Creates, trains, evaluates, and returns a model pipeline, ensuring feature names are handled."""
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
    
    # --- FINAL, DATA-DRIVEN MODEL CONFIGURATION ---
    MODEL_CONFIG = {
        'apartment': {
            'params': {'objective': 'regression_l1', 'random_state': 42, 'n_estimators': 1500, 'learning_rate': 0.02, 'num_leaves': 40, 'max_depth': 8},
            'features': {'numerical': ['size_m2', 'latitude', 'longitude', 'bedrooms', 'bathrooms'], 'categorical': ['province', 'district', 'location_cluster']}
        },
        'office': {
            'params': {'objective': 'regression_l1', 'random_state': 42, 'n_estimators': 1000, 'learning_rate': 0.03, 'num_leaves': 30, 'max_depth': 6},
            # Offices can have bathrooms, so we include it.
            'features': {'numerical': ['size_m2', 'latitude', 'longitude', 'bedrooms', 'bathrooms'], 'categorical': ['province', 'district', 'location_cluster']}
        },
        'shop': {
            'params': {'objective': 'regression_l1', 'random_state': 42, 'n_estimators': 1200, 'learning_rate': 0.03},
            # Shops typically don't have bedrooms, but can have bathrooms.
            'features': {'numerical': ['size_m2', 'latitude', 'longitude', 'bathrooms'], 'categorical': ['province', 'district', 'location_cluster']}
        }
    }
    
    # --- SETUP & DATA PREPARATION ---
    os.makedirs('final_models', exist_ok=True)
    df = pd.read_csv('properties.csv')
    df.drop_duplicates(subset=['id'], inplace=True)
    df = df[(df['price_$'] > 1000) & (df['size_m2'] > 10)].copy()
    
    df = df[df['type'] != 'Land'].copy()
    print(f"Removed 'Land' properties. Working with {len(df)} properties.")
    
    df.drop(columns=['created_at', 'city'], inplace=True)

    # --- GLOBAL FEATURE ENGINEERING ---
    print("\nCreating global K-Means model for location clustering...")
    kmeans = KMeans(n_clusters=20, random_state=42, n_init='auto')
    df['location_cluster'] = kmeans.fit_predict(df[['latitude', 'longitude']]).astype(str)
    
    # --- TRAINING SPECIALIST MODELS ---
    specialist_types = ['apartment', 'office', 'shop']
    for prop_type in specialist_types:
        print(f"\n--- Training Specialist Model for: {prop_type.upper()} ---")
        
        config = MODEL_CONFIG[prop_type]
        df_slice = df[df['type'].str.lower().replace({'/': '_'}) == prop_type].copy()

        if len(df_slice) < 50:
            print(f"   -> Insufficient data for {prop_type}, skipping.")
            continue
        
        # Ensure all columns exist, fill with 0 if they don't (e.g., 'bathrooms' in shops)
        for col in config['features']['numerical'] + config['features']['categorical']:
            if col not in df_slice.columns:
                df_slice[col] = 0

        X = df_slice.drop(columns=['price_$', 'type'])
        y = np.log1p(df_slice['price_$'])
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        pipeline, r2 = create_and_train_pipeline(X_train, y_train, X_test, y_test, config['params'], config['features'])

        if r2 > 0.65:
            model_path = os.path.join('final_models', f'model_{prop_type}.joblib')
            joblib.dump(pipeline, model_path)
            print(f"   -> High-quality model saved to '{model_path}'")
            plot_feature_importance(pipeline, prop_type)
        else:
            print(f"   -> Model performance below threshold. This type will use the Generalist Fallback.")

    # --- TRAINING GENERALIST FALLBACK MODEL ---
    print("\n--- Training Generalist Fallback Model ---")
    df_general = df.copy() 

    general_config = {
        'params': {
            'objective': 'regression_l1', 'random_state': 42, 'n_estimators': 1500,
            'learning_rate': 0.02, 'num_leaves': 40, 'max_depth': 7
        },
        'features': {
            'numerical': ['size_m2', 'latitude', 'longitude', 'bedrooms', 'bathrooms'],
            'categorical': ['province', 'district', 'location_cluster', 'type']
        }
    }
    
    X_general = df_general.drop(columns=['price_$'])
    y_general = np.log1p(df_general['price_$'])
    X_train_gen, X_test_gen, y_train_gen, y_test_gen = train_test_split(X_general, y_general, test_size=0.2, random_state=42)
    
    general_pipeline, _ = create_and_train_pipeline(
        X_train_gen, y_train_gen, X_test_gen, y_test_gen,
        general_config['params'], general_config['features']
    )
    # -------------------------------------------------------------
    model_path = os.path.join('final_models', 'model_general_fallback.joblib')
    joblib.dump(general_pipeline, model_path)
    print(f"   -> Generalist Fallback Model saved to '{model_path}'")
    plot_feature_importance(general_pipeline, "general_fallback")

    # Save the essential K-Means model
    joblib.dump(kmeans, os.path.join('final_models', 'kmeans_model.joblib'))
    # -------------------------------------------------------------

    SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
    BACKEND_DIR = os.path.dirname(SCRIPT_DIR)
    output_dir = os.path.join(BACKEND_DIR, 'final_models', 'model_general_fallback.joblib')
    
    os.makedirs(output_dir, exist_ok=True)
# ------------------------------

    print("\n\n--- Workflow Complete ---")
    print("All necessary models and plots have been generated and saved to the 'final_models' directory.")

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