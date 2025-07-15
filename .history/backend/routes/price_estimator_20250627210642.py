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



import pandas as pd
import numpy as np
import lightgbm as lgb
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

def get_preprocessor(feature_config):
    """Returns a ColumnTransformer based on a feature configuration."""
    # Add our powerful geo features to every model's configuration
    numerical_features = feature_config['numerical'] + ['latitude', 'longitude']
    categorical_features = feature_config['categorical'] + ['location_cluster']
    
    # Use StandardScaler for all numerical features including counts
    all_numerical = list(set(numerical_features)) # Use set to avoid duplicates
    
    return ColumnTransformer(
        transformers=[
            ('num', StandardScaler(), all_numerical),
            ('cat', OneHotEncoder(handle_unknown='ignore', sparse_output=False), categorical_features)
        ],
        remainder='drop' # Explicitly drop any columns not listed
    )

def train_specialist_model(df_type, property_type, kmeans_model, config):
    """Trains a single model for a specific property type using the provided config."""
    print(f"\n--- Training Specialist Model for: {property_type.upper()} ---")
    
    if len(df_type) < 50: # Increased threshold for robustness
        print(f"   -> Insufficient data ({len(df_type)} rows), skipping.")
        return

    # Add geo features
    df_type['location_cluster'] = kmeans_model.predict(df_type[['latitude', 'longitude']]).astype(str)
    
    # Log transform size_m2 if it's in the feature list
    if 'log_size_m2' in config['features']['numerical']:
        df_type['log_size_m2'] = np.log1p(df_type['size_m2'])
    
    X = df_type.drop(columns=['price_$', 'type'])
    y = np.log1p(df_type['price_$'])
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    pipeline = Pipeline(steps=[
        ('preprocessor', get_preprocessor(config['features'])),
        ('regressor', lgb.LGBMRegressor(**config['params']))
    ])

    pipeline.fit(X_train, y_train)
    
    y_pred_log = pipeline.predict(X_test)
    y_pred_dollars = np.expm1(y_pred_log)
    y_test_dollars = np.expm1(y_test)
    
    r2 = r2_score(y_test_dollars, y_pred_dollars)
    mae = mean_absolute_error(y_test_dollars, y_pred_dollars)
    
    print(f"Result -> R²: {r2:.4f}, MAE: ${mae:,.2f}")

    # Save the model only if it's reasonably good
    if r2 > 0.68:
        model_path = os.path.join('specialist_models', f'model_{property_type.lower().replace("/", "_")}.joblib')
        joblib.dump(pipeline, model_path)
        print(f"   -> Model saved to '{model_path}'")
    else:
        print(f"   -> Model performance is too low. This type will use the Generalist Fallback.")

def train_generalist_model(df_general, kmeans_model):
    """Trains the robust generalist fallback model."""
    print("\n--- Training Generalist Fallback Model ---")
    
    df_general['location_cluster'] = kmeans_model.predict(df_general[['latitude', 'longitude']]).astype(str)
    
    # The generalist model needs 'type' as a feature
    features = {
        'numerical': ['size_m2', 'latitude', 'longitude', 'bedrooms', 'bathrooms'],
        'categorical': ['district', 'province', 'location_cluster', 'type']
    }
    
    preprocessor = get_preprocessor(features)
    
    X = df_general.drop(columns=['price_$'])
    y = np.log1p(df_general['price_$'])
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Strong, robust parameters for the generalist model
    general_params = {
        'objective': 'regression_l1', 'random_state': 42, 'n_estimators': 1500,
        'learning_rate': 0.02, 'num_leaves': 40, 'max_depth': 7
    }

    pipeline = Pipeline(steps=[
        ('preprocessor', preprocessor),
        ('regressor', lgb.LGBMRegressor(**general_params))
    ])

    pipeline.fit(X_train, y_train)

    y_pred_log = pipeline.predict(X_test)
    y_pred_dollars = np.expm1(y_pred_log)
    y_test_dollars = np.expm1(y_test)
    
    r2 = r2_score(y_test_dollars, y_pred_dollars)
    mae = mean_absolute_error(y_test_dollars, y_pred_dollars)
    
    print(f"Result -> R²: {r2:.4f}, MAE: ${mae:,.2f}")

    model_path = os.path.join('specialist_models', 'model_general_fallback.joblib')
    joblib.dump(pipeline, model_path)
    print(f"   -> Generalist Fallback Model saved to '{model_path}'")


def main():
    """Main function to run the entire hybrid model training workflow."""
    print("Starting the Hybrid Model Training Process...")
    
    # --- MODEL CONFIGURATION (Your expert tuning + our geo features) ---
    MODEL_CONFIG = {
        'apartment': {
            'params': {'objective': 'regression_l1', 'random_state': 42, 'n_estimators': 1500, 'depth': 8, 'learning_rate': 0.02, 'l2_leaf_reg': 3, 'bootstrap_type': 'Bayesian'},
            'features': {'numerical': ['log_size_m2', 'bedrooms', 'bathrooms'], 'categorical': ['province', 'district']}
        },
        'office': {
            'params': {'objective': 'regression_l1', 'random_state': 42, 'n_estimators': 1000, 'depth': 6, 'learning_rate': 0.03, 'l2_leaf_reg': 5, 'one_hot_max_size': 15},
            'features': {'numerical': ['log_size_m2', 'bedrooms', 'bathrooms'], 'categorical': ['province', 'district']}
        },
        'shop': {
            'params': {'objective': 'regression_l1', 'random_state': 42, 'n_estimators': 1200, 'learning_rate': 0.03, 'l2_leaf_reg': 3},
            'features': {'numerical': ['log_size_m2'], 'categorical': ['province', 'district']}
        }
    }
    
    os.makedirs('specialist_models', exist_ok=True)
    
    # --- Data Loading and Cleaning ---
    df = pd.read_csv('properties.csv')
    df.drop_duplicates(subset=['id'], inplace=True)
    df = df[(df['price_$'] > 1000) & (df['size_m2'] > 10)].copy()
    df.drop(columns=['created_at'], inplace=True) 

    # --- Global Feature Engineering (before splitting) ---
    print("\nCreating global K-Means model for location clustering...")
    kmeans = KMeans(n_clusters=20, random_state=42, n_init='auto')
    kmeans.fit(df[['latitude', 'longitude']])
    joblib.dump(kmeans, os.path.join('specialist_models', 'kmeans_model.joblib'))
    
    specialist_types = ['apartment', 'office', 'shop']
    
    # --- Train Specialist Models ---
    for prop_type in specialist_types:
        df_slice = df[df['type'].str.lower().replace('/', '_') == prop_type].copy()
        train_specialist_model(df_slice, prop_type, kmeans, MODEL_CONFIG[prop_type])

    # --- Train Generalist Fallback Model ---
    df_general = df[~df['type'].str.lower().replace('/', '_').isin(specialist_types + ['land'])].copy()
    train_generalist_model(df_general, kmeans)

    print("\n\n--- Workflow Complete ---")

if __name__ == '__main__':
    main()