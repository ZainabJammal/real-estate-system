import pandas as pd
import numpy as np
from catboost import CatBoostRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_absolute_error
import joblib
import os
import warnings

warnings.filterwarnings('ignore', category=UserWarning)

# --- 1. Your Custom Hyperparameters and Feature Definitions ---
# # This dictionary holds the tailored parameters for each model
# MODEL_CONFIG = {
#     'apartment': {
#         'params': {'depth': 8, 'learning_rate': 0.03, 'l2_leaf_reg': 5, 'iterations': 2000},
#         'features': {'numerical': ['log_size_m2', 'bedrooms', 'bathrooms'], 'categorical': ['province', 'district', 'city']}
#     },
#     'land': {
#         'params': {'depth': 6, 'learning_rate': 0.01, 'l2_leaf_reg': 10, 'iterations': 2000},
#         'features': {'numerical': ['log_size_m2'], 'categorical': ['province', 'district', 'city']}
#     },
#     'house/villa': {
#         'params': {'depth': 7, 'learning_rate': 0.02, 'l2_leaf_reg': 3, 'iterations': 2000},
#         'features': {'numerical': ['log_size_m2', 'bedrooms', 'bathrooms'], 'categorical': ['province', 'district', 'city']}
#     },
#     # Add configs for other types if they become viable
#     'office': {
#         'params': {'depth': 6, 'learning_rate': 0.05, 'l2_leaf_reg': 7, 'iterations': 1500},
#         'features': {'numerical': ['log_size_m2', 'bedrooms', 'bathrooms'], 'categorical': ['province', 'district', 'city']}
#     },
#     'shop': {
#         'params': {'depth': 6, 'learning_rate': 0.05, 'l2_leaf_reg': 7, 'iterations': 1500},
#         'features': {'numerical': ['log_size_m2'], 'categorical': ['province', 'district', 'city']}
#     },
#      'chalet': {
#         'params': {'depth': 6, 'learning_rate': 0.03, 'l2_leaf_reg': 5, 'iterations': 1500},
#         'features': {'numerical': ['log_size_m2', 'bedrooms', 'bathrooms'], 'categorical': ['province', 'district', 'city']}
#     }
# }
MODEL_CONFIG = {
    'apartment': {
        'params': {
            'depth': 8,  # Good for complex relationships
            'learning_rate': 0.02,  # Slightly reduced for better convergence
            'l2_leaf_reg': 3,  # Reduced regularization (your model is performing well)
            'iterations': 2500,  # Increased slightly
            'grow_policy': 'SymmetricTree',  # Better for medium-sized datasets
            'bootstrap_type': 'Bayesian',  # Better uncertainty estimation
            'random_strength': 1,  # Helps with generalization
            'border_count': 128,  # Good default for medium datasets
            'early_stopping_rounds': 50  # Prevent overfitting
        },
        'features': {
            'numerical': ['log_size_m2', 'bedrooms', 'bathrooms'],
            'categorical': ['province', 'district', 'city']
        }
    },
    'land': {
    'params': {
        'depth': 12,  # Increased further for complex land patterns
        'learning_rate': 0.001,  # Much slower learning
        'l2_leaf_reg': 20,  # Stronger regularization
        'iterations': 5000,  # More iterations
        'grow_policy': 'Depthwise',
        'bootstrap_type': 'Bernoulli',
        'random_strength': 3,
        'border_count': 32,  # Reduced for small dataset
        'early_stopping_rounds': 200,
        'min_data_in_leaf': 20,  # Very conservative
        'has_time': True  # Use time-based split if available
    },
    'features': {
        'numerical': ['log_size_m2'],
        'categorical': ['province', 'district', 'city'],
        # Consider adding if available:
        'potential_new': ['zoning_type', 'distance_to_city_center', 'land_terrain']
    }
},
    'house/villa': {
    'params': {
        'depth': 9,  # Increased from 7
        'learning_rate': 0.01,  # Reduced from 0.015
        'l2_leaf_reg': 7,  # Increased regularization
        'iterations': 3000,
        'grow_policy': 'Lossguide',
        'max_leaves': 128,  # Increased
        'min_data_in_leaf': 7  # Added
    }
    # Keep same features
},
    'office': {
        'params': {
            'depth': 6,
            'learning_rate': 0.03,  # Increased slightly for faster convergence
            'l2_leaf_reg': 5,  # Reduced regularization (model is performing well)
            'iterations': 2000,
            'grow_policy': 'SymmetricTree',
            'bootstrap_type': 'Bayesian',
            'random_strength': 1,
            'border_count': 128,
            'early_stopping_rounds': 50,
            'one_hot_max_size': 10  # Important for categoricals
        },
        'features': {
            'numerical': ['log_size_m2', 'bedrooms', 'bathrooms'],
            'categorical': ['province', 'district', 'city']
        }
    },
'shop': {
    'params': {
        'learning_rate': 0.03,  # Between previous and current
        'l2_leaf_reg': 3,  # Reduced regularization
        'iterations': 2000
    }
        'features': {
            'numerical': ['log_size_m2'],
            'categorical': ['province', 'district', 'city']
        }
    },
    'chalet': {
    
    'params': {
        'depth': 4,  # Reduced from 5 (simpler model)
        'learning_rate': 0.005,  # Slower learning
        'l2_leaf_reg': 15,
        'iterations': 3000,
        'min_data_in_leaf': 10,  # Half your dataset in each leaf
        'one_hot_max_size': 5  # Reduce categorical complexity
    }
,
        'features': {
            'numerical': ['log_size_m2', 'bedrooms', 'bathrooms'],
            'categorical': ['province', 'district', 'city']
        }
    }
}
def train_specialist_model(df_segment, property_type, config):
    """
    Trains a specialized CatBoost model using tailored hyperparameters and features.
    """
    print(f"\n--- Training Specialist Model for: {property_type.upper()} ---")
    
    # 1. Feature Engineering
    df = df_segment.copy()
    
    # ** NEW TARGET VARIABLE: log_price_per_sqm **
    df['log_price_per_sqm'] = np.log1p(df['price_$'] / df['size_m2'])
    df['log_size_m2'] = np.log1p(df['size_m2'])
    
    features_to_use = config['features']['numerical'] + config['features']['categorical']
    
    y = df['log_price_per_sqm']
    X = df[features_to_use]
    
    # 2. Data Split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # 3. Model Training
    params = config['params']
    model = CatBoostRegressor(
        loss_function='RMSE', # A standard for regression
        eval_metric='R2',
        random_seed=42,
        verbose=0,
        cat_features=config['features']['categorical'],
        **params
    )
    
    model.fit(
        X_train, y_train,
        eval_set=(X_test, y_test),
        early_stopping_rounds=50,
        verbose=False
    )
    
    # 4. Evaluation
    y_pred_log_per_sqm = model.predict(X_test)
    
    # Convert prediction back to total dollar value
    test_log_size = X_test['log_size_m2']
    test_size = np.expm1(test_log_size)
    
    pred_price_per_sqm = np.expm1(y_pred_log_per_sqm)
    y_pred_dollars = pred_price_per_sqm * test_size
    
    # Get original test prices for comparison
    y_test_dollars = np.expm1(y_test) * test_size

    r2 = r2_score(y_test_dollars, y_pred_dollars)
    mae = mean_absolute_error(y_test_dollars, y_pred_dollars)
    
    print(f"Result -> R²: {r2:.4f}, MAE: ${mae:,.2f}")
    
    # 5. Feature Importance
    feature_importances = pd.Series(model.get_feature_importance(), index=X.columns).sort_values(ascending=False)
    print("Top 5 Features:")
    print(feature_importances.head(5).to_string())
    
    return model, r2

def main():
    print("Step 1: Loading and preparing base data...")
    df = pd.read_csv('properties.csv')
    df.drop_duplicates(subset=['id'], inplace=True)
    df = df[df['price_$'] > 10000].copy()
    df = df[df['size_m2'] > 20].copy()
    
    for col in ['type', 'province', 'district', 'city']:
        df[col] = df[col].astype(str).str.lower().str.strip()
        
    df['bedrooms'].fillna(df.groupby('type')['bedrooms'].transform('median'), inplace=True)
    df['bathrooms'].fillna(df.groupby('type')['bathrooms'].transform('median'), inplace=True)
    df.fillna({'bedrooms': 1, 'bathrooms': 1}, inplace=True)
    df[['bedrooms', 'bathrooms']] = df[['bedrooms', 'bathrooms']].astype(int)
    
    output_dir = "ultimate_specialist_models"
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        
    viable_types = list(MODEL_CONFIG.keys())
    print(f"\nTraining specialist models for: {viable_types}")

    for prop_type in viable_types:
        df_segment = df[df['type'] == prop_type]
        if len(df_segment) < 50: # Check if there is enough data
            print(f"\n--- Skipping {prop_type.upper()}: Not enough data ({len(df_segment)} rows) ---")
            continue

        model, r2 = train_specialist_model(df_segment, prop_type, MODEL_CONFIG[prop_type])
        
        if r2 > 0.50: # A reasonable threshold for saving a specialist model
            filename = f"model_{prop_type.replace('/', '_')}.joblib"
            filepath = os.path.join(output_dir, filename)
            joblib.dump(model, filepath)
            print(f"   -> Model saved to '{filepath}'")
        else:
            print("   -> Model performance is too low, not saving.")
            
    print("\n--- Process Complete ---")

if __name__ == '__main__':
    main()



# import pandas as pd
# import numpy as np
# from catboost import CatBoostRegressor
# from sklearn.model_selection import train_test_split
# from sklearn.metrics import r2_score, mean_absolute_error
# import joblib
# import os
# import warnings

# # Use BayesSearchCV from scikit-optimize
# from skopt import BayesSearchCV
# from skopt.space import Real, Integer

# # --- Setup ---
# warnings.filterwarnings('ignore', category=UserWarning)
# warnings.filterwarnings('ignore', category=FutureWarning)
# warnings.filterwarnings('ignore', category=DeprecationWarning)

# def tune_and_save_apartment_model():
#     """
#     Final Stage: Takes our best segment (apartments) and uses Bayesian
#     Optimization to find the absolute best hyperparameters for the highest
#     possible performance.
#     """
#     # --- 1. Load and hyper-focus the data to ONLY apartments ---
#     print("Step 1: Loading and Filtering Data to ONLY Apartments...")
#     df = pd.read_csv('properties.csv')
#     df.drop_duplicates(subset=['id'], inplace=True)
#     df['type'] = df['type'].str.lower().str.strip()
#     df = df[df['type'] == 'apartment'].copy()
    
#     # Apply robust filtering
#     df = df[(df['price_$'] > 20000) & (df['size_m2'] > 20) & (df['price_$'] < 3000000)].copy()
    
#     print(f"Focused on {len(df)} apartment listings for tuning.")

#     # --- 2. Feature Engineering & Cleaning for Apartments ---
#     print("\nStep 2: Feature Engineering and Cleaning...")
#     for col in ['province', 'district', 'city']:
#         df[col] = df[col].astype(str).str.lower().str.strip()
        
#     df['bedrooms'].fillna(df['bedrooms'].median(), inplace=True)
#     df['bathrooms'].fillna(df['bathrooms'].median(), inplace=True)
#     df[['bedrooms', 'bathrooms']] = df[['bedrooms', 'bathrooms']].astype(int)
    
#     df['log_price'] = np.log1p(df['price_$'])
#     df['log_size_m2'] = np.log1p(df['size_m2'])
    
#     # --- 3. Define Features and Split Data ---
#     print("\nStep 3: Defining Features and Splitting Data...")
    
#     numerical_features = ['log_size_m2', 'bedrooms', 'bathrooms']
#     categorical_features = ['province', 'district', 'city']
    
#     y = df['log_price']
#     X = df[numerical_features + categorical_features]

#     X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

#     # --- 4. Bayesian Optimization with BayesSearchCV ---
#     print("\nStep 4: Setting up and Running Bayesian Optimization...")
    
#     # Define the search space for the hyperparameters
#     search_spaces = {
#         'learning_rate': Real(0.01, 0.1, 'log-uniform'),
#         'n_estimators': Integer(500, 2000),
#         'depth': Integer(5, 10),
#         'l2_leaf_reg': Real(1, 10, 'uniform'),
#         'random_strength': Real(1e-9, 10, 'log-uniform'),
#         'min_child_samples': Integer(5, 30)
#     }

#     # Setup BayesSearchCV
#     # n_iter=40 will run 40 smart trials to find the best combination
#     bayes_search = BayesSearchCV(
#         estimator=CatBoostRegressor(
#             loss_function='MAE',
#             random_seed=42,
#             verbose=0,
#             cat_features=categorical_features,
#             early_stopping_rounds=50 # Use early stopping within each CV fold
#         ),
#         search_spaces=search_spaces,
#         n_iter=40,
#         cv=5, # 5-fold cross-validation
#         n_jobs=-1,
#         scoring='neg_mean_absolute_error',
#         random_state=42,
#         verbose=0
#     )
    
#     # BayesSearchCV requires a callback for early stopping
#     def on_step(optim_result):
#         # This function is called after each trial
#         score = bayes_search.best_score_
#         print(f"Best score so far: {-score:.4f}")

#     print("Starting hyperparameter search (this may take several minutes)...")
#     bayes_search.fit(X_train, y_train, callback=on_step)
    
#     print("\n--- Search Complete ---")
#     print("Best parameters found: ", bayes_search.best_params_)
#     best_model = bayes_search.best_estimator_

#     # --- 5. Final Evaluation ---
#     print("\n--- Step 5: Final Model Evaluation on Test Set ---")
#     y_pred_log = best_model.predict(X_test)
#     y_pred_dollars = np.expm1(y_pred_log)
#     y_test_dollars = np.expm1(y_test)

#     r2 = r2_score(y_test_dollars, y_pred_dollars)
#     mae = mean_absolute_error(y_test_dollars, y_pred_dollars)

#     print(f"\nFinal Tuned R-squared (R²): {r2:.4f}")
#     print(f"Final Tuned Mean Absolute Error (MAE): ${mae:,.2f}")

#     # --- 6. Save the Champion Model ---
#     output_dir = "champion_model"
#     if not os.path.exists(output_dir):
#         os.makedirs(output_dir)
        
#     model_filename = os.path.join(output_dir, 'apartment_champion_model.joblib')
#     print(f"\nStep 6: Saving champion model to '{model_filename}'...")
#     joblib.dump(best_model, model_filename)
#     print("Model saved successfully.")

# if __name__ == '__main__':
#     tune_and_save_apartment_model()