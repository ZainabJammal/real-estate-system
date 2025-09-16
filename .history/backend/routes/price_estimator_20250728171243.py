
# # # # MODEL_CONFIG = {
# # # #     'apartment': {
# # # #         'params': {'depth': 8, 'learning_rate': 0.03, 'l2_leaf_reg': 5, 'iterations': 2000},
# # # #         'features': {'numerical': ['log_size_m2', 'bedrooms', 'bathrooms'], 'categorical': ['province', 'district', 'city']}
# # # #     },
# # # #     'land': {
# # # #         'params': {'depth': 6, 'learning_rate': 0.01, 'l2_leaf_reg': 10, 'iterations': 2000},
# # # #         'features': {'numerical': ['log_size_m2'], 'categorical': ['province', 'district', 'city']}
# # # #     },
# # # #     'house/villa': {
# # # #         'params': {'depth': 7, 'learning_rate': 0.02, 'l2_leaf_reg': 3, 'iterations': 2000},
# # # #         'features': {'numerical': ['log_size_m2', 'bedrooms', 'bathrooms'], 'categorical': ['province', 'district', 'city']}
# # # #     },
# # # #  
# # # # }

# # import pandas as pd
# # import numpy as np
# # import lightgbm as lgb
# # import matplotlib.pyplot as plt
# # import joblib
# # import warnings
# # import os
# # from sklearn.model_selection import KFold, cross_val_score
# # from sklearn.pipeline import Pipeline
# # from sklearn.compose import ColumnTransformer
# # from sklearn.preprocessing import StandardScaler, OneHotEncoder
# # from sklearn.impute import SimpleImputer

# # # --- 1. SETUP & CONFIGURATION (Unchanged) ---
# # def setup_environment():
# #     """Sets up directories and configurations for the script."""
# #     warnings.filterwarnings('ignore', category=UserWarning)
# #     pd.options.mode.chained_assignment = None
    
# #     # Get the absolute path of the currently running script (e.g., .../backend/routes/your_script.py)
# #     script_path = os.path.abspath(__file__)
    
# #     # Get the directory the script is in (e.g., .../backend/routes/)
# #     routes_dir = os.path.dirname(script_path)
    
# #     # Go UP ONE LEVEL to get the backend directory (e.g., .../backend/)
# #     backend_dir = os.path.dirname(routes_dir)

# #     # Now, build the paths relative to the `backend_dir`
# #     output_dir = os.path.join(backend_dir, 'final_model_output_v2')
# #     data_path = os.path.join(backend_dir, 'properties.csv')
    
# #     os.makedirs(output_dir, exist_ok=True)
    
# #     print("="*60)
# #     print("Environment Setup Complete")
# #     print(f"-> Model artifacts will be saved to: '{os.path.abspath(output_dir)}'")
# #     print("="*60)
    
# #     return data_path, output_dir

# # MODEL_CONFIG = { 'apartment': { 'params': { 'objective': 'regression_l1', 'metric': 'rmse', 'random_state': 42, 'n_estimators': 1500, 'learning_rate': 0.02, 'num_leaves': 40, 'max_depth': 8, 'subsample': 0.8, 'colsample_bytree': 0.8 } } }

# # # --- 2. DATA PREPARATION & FEATURE ENGINEERING WORKFLOW (REVISED) ---
# # def prepare_apartment_data(df):
# #     """Loads, cleans, and engineers features for the apartment dataset using size categories."""
# #     print("\n--- Step 2: Preparing Apartment Data with Size Categories ---")
        
# #     # --- Basic Cleaning ---
# #     df_apartments = df[df['property_type'] == 'apartment'].copy()
# #     df_apartments['bedrooms'].replace(0, np.nan, inplace=True)
# #     df_apartments['bathrooms'].replace(0, np.nan, inplace=True)
# #     df_apartments.dropna(subset=['price_$', 'size_m2', 'bedrooms', 'bathrooms'], inplace=True)
# #     df_apartments = df_apartments[df_apartments['size_m2'] > 0]
    
# #     # --- CHANGE 1: Feature Engineering for Size Category ---
# #     print("\n--- Feature Engineering ---")
# #     bins = [0, 80, 120, 180, 250, 400, float('inf')]
# #     labels = ['Studio/Small (0-80m²)',
# #             'Standard (81-120m²)',
# #             'Comfortable (121-180m²)', 
# #             'Large (181-250m²)',
# #             'Very Large (251-400m²)',
# #             'Penthouse (401m²+)']
    
# #     df_apartments['size_category'] = pd.cut(df_apartments['size_m2'], bins=bins, labels=labels, right=True)
# #     print(f"   -> Created 'size_category' feature from 'size_m2'.")

# #     # --- Basic Feature Engineering (still useful) ---
# #     df_apartments['price_per_m2'] = df_apartments['price_$'] / df_apartments['size_m2']
    
# #     # --- Outlier Removal (using price_per_m2 is still a good idea) ---
# #     q_low = df_apartments['price_per_m2'].quantile(0.01)
# #     q_high = df_apartments['price_per_m2'].quantile(0.99)
# #     df_apartments_clean = df_apartments[(df_apartments['price_per_m2'] >= q_low) & (df_apartments['price_per_m2'] <= q_high)].copy()
# #     print(f"   -> {len(df_apartments_clean)} rows after cleaning and outlier removal.")

# #     # --- Handling Rare Categories (still a good practice) ---
# #     district_counts = df_apartments_clean['district'].value_counts()
# #     THRESHOLD = 30 
# #     rare_districts = district_counts[district_counts < THRESHOLD].index.tolist()
# #     df_apartments_clean.loc[df_apartments_clean['district'].isin(rare_districts), 'district'] = 'Other'
# #     print(f"   -> Grouped {len(rare_districts)} rare districts into 'Other' category.")

# #     # --- Final Transformation ---
# #     df_apartments_clean['log_price'] = np.log1p(df_apartments_clean['price_$'])
# #     print("   -> Created 'log_price' as the final target variable.")
    
# #     return df_apartments_clean

# # # --- 3. MODELING WORKFLOW (REVISED) ---
# # def build_and_evaluate_model(df_featured, config, output_dir):
# #     """Builds the preprocessing pipeline, validates, trains, and saves the final model."""
# #     print("\n--- Step 3: Building and Validating the Model ---")
    
# #     # --- CHANGE 2: Define New Features and Target ---
# #     TARGET = 'log_price'
    
# #     # Define features based on the new approach
# #     NUMERICAL_FEATURES = ['bedrooms', 'bathrooms', 'latitude', 'longitude']
# #     CATEGORICAL_FEATURES = ['province', 'district', 'size_category'] # Add size_category here
    
# #     # The interaction features are no longer needed as the model will learn the interaction
# #     # between the 'district' category and the new 'size_category' automatically.
    
# #     FEATURES = NUMERICAL_FEATURES + CATEGORICAL_FEATURES
    
# #     X = df_featured[FEATURES]
# #     y = df_featured[TARGET]

# #     # --- CHANGE 3: Define the new preprocessing pipeline ---
# #     preprocessor = ColumnTransformer(
# #         transformers=[
# #             ('num', Pipeline(steps=[
# #                 ('imputer', SimpleImputer(strategy='median')),
# #                 ('scaler', StandardScaler())]), NUMERICAL_FEATURES),
# #             ('cat', Pipeline(steps=[
# #                 ('imputer', SimpleImputer(strategy='most_frequent')),
# #                 ('onehot', OneHotEncoder(handle_unknown='ignore', sparse_output=False))]), CATEGORICAL_FEATURES)
# #         ],
# #         remainder='drop'
# #     )

# #     # Create the full model pipeline (this part is unchanged)
# #     pipeline = Pipeline(steps=[
# #         ('preprocessor', preprocessor),
# #         ('regressor', lgb.LGBMRegressor(**config['params']))
# #     ])

# #     # --- Cross-Validation (Unchanged) ---
# #     kf = KFold(n_splits=5, shuffle=True, random_state=42)
# #     cv_scores_r2 = cross_val_score(pipeline, X, y, cv=kf, scoring='r2')
    
# #     print("\n--- Cross-Validation Performance ---")
# #     print(f"   -> 5-Fold R² Scores: {[f'{s:.4f}' for s in cv_scores_r2]}")
# #     print(f"   -> Average R² Score: {np.mean(cv_scores_r2):.4f} (± {np.std(cv_scores_r2):.4f})")

# #     # --- Final Model Training (Unchanged) ---
# #     print("\n--- Step 4: Training Final Model on All Data ---")
# #     pipeline.fit(X, y)
# #     print("   -> Final model training complete.")
    
# #     return pipeline, np.mean(cv_scores_r2)

# # # --- 4. FEATURE IMPORTANCE & 5. MAIN EXECUTION (Largely Unchanged but will use new features) ---
# # def plot_feature_importance(pipeline, output_dir):
# #     """Extracts and plots feature importances from the trained model."""
# #     print("\n--- Step 5: Generating Feature Importance Plot ---")

# #     preprocessor = pipeline.named_steps['preprocessor']
# #     regressor = pipeline.named_steps['regressor']
    
# #     # Get names from numerical transformer
# #     num_feature_names = preprocessor.transformers_[0][2]
# #     # Get names from categorical transformer after one-hot encoding
# #     cat_feature_names = preprocessor.named_transformers_['cat'].named_steps['onehot'].get_feature_names_out()
    
# #     all_feature_names = np.concatenate([num_feature_names, cat_feature_names])
    
# #     importances = pd.DataFrame({
# #         'feature': all_feature_names,
# #         'importance': regressor.feature_importances_
# #     }).sort_values('importance', ascending=False).head(20)

# #     plt.figure(figsize=(10, 8))
# #     plt.barh(importances['feature'], importances['importance'])
# #     plt.xlabel("LightGBM Feature Importance")
# #     plt.ylabel("Feature")
# #     plt.title("Top 20 Feature Importances (with Size Category)")
# #     plt.gca().invert_yaxis()
# #     plt.tight_layout()
    
# #     plot_path = os.path.join(output_dir, 'feature_importance_size_category.png')
# #     plt.savefig(plot_path)
# #     print(f"   -> Feature importance plot saved to '{plot_path}'")
# #     plt.close()

# # def main():
# #     data_path, output_dir = setup_environment()
# #     df = pd.read_csv(data_path)
# #     df.drop_duplicates(subset=['id'], inplace=True)
# #     df.rename(columns={'type': 'property_type'}, inplace=True)
# #     df['property_type'] = df['property_type'].str.lower().str.strip()
    
# #     df_featured = prepare_apartment_data(df)
    
# #     final_pipeline, avg_r2_score = build_and_evaluate_model(df_featured, MODEL_CONFIG['apartment'], output_dir)
    
# #     plot_feature_importance(final_pipeline, output_dir)
    
# #     model_path = os.path.join(output_dir, 'model_apartment_size_category.joblib') 
# #     joblib.dump(final_pipeline, model_path)
    
# #     print(f"\n--- Step 6: Final model saved to '{model_path}' ---")
# #     print("\n" + "="*60); 
# #     print("--- MODEL PERFORMANCE SUMMARY ---")
# #     summary = {
# #         'Model': 'Apartment Price Estimator (Size Category)',
# #         'Average R2 (5-Fold CV)': f"{avg_r2_score:.4f}",
# #         'Number of Training Samples': len(df_featured),
# #         'Model Saved At': model_path,
# #         'Importance Plot Saved At': os.path.join(output_dir, 'feature_importance_size_category.png')
# #     }
# #     for key, value in summary.items(): 
# #         print(f"   -> {key}: {value}")
# #     print("="*60)
# #     print("--- WORKFLOW COMPLETE ---")

# # if __name__ == '__main__':
# #     main()


# import pandas as pd
# import numpy as np
# import lightgbm as lgb
# import matplotlib.pyplot as plt
# import seaborn as sns 
# import joblib
# import warnings
# import os
# from sklearn.model_selection import train_test_split, GridSearchCV
# from sklearn.metrics import r2_score, mean_absolute_error
# from sklearn.pipeline import Pipeline
# from sklearn.compose import ColumnTransformer
# from sklearn.preprocessing import StandardScaler, OneHotEncoder
# from sklearn.impute import SimpleImputer

# # --- 1. SETUP & CONFIGURATION ---
# def setup_environment():
#     """Sets up directories and configurations for the script."""
#     warnings.filterwarnings('ignore', category=UserWarning)
#     pd.options.mode.chained_assignment = None

#     output_dir = os.path.join(os.getcwd(), 'final_model_output_v7_with_plots')
#     data_path = os.path.join(os.getcwd(), 'properties.csv')
    
#     os.makedirs(output_dir, exist_ok=True)
    
#     print("="*60)
#     print("Environment Setup Complete")
#     print(f"-> Model artifacts will be saved to: '{os.path.abspath(output_dir)}'")
#     print("="*60)
#     return data_path, output_dir


# # --- 2. DATA PREPARATION ---
# def prepare_apartment_data(df):
#     """Loads, cleans, and engineers features for the apartment dataset."""
#     print("\n--- Step 2: Preparing Apartment Data ---")
    
#     df_apartments = df[df['property_type'] == 'apartment'].copy()
    
#     # Basic data cleaning
#     df_apartments['bedrooms'].replace(0, np.nan, inplace=True)
#     df_apartments['bathrooms'].replace(0, np.nan, inplace=True)
#     df_apartments.dropna(subset=['price_$', 'size_m2', 'bedrooms', 'bathrooms'], inplace=True)
#     df_apartments = df_apartments[df_apartments['size_m2'] > 0]
    
#     # Outlier removal using a sound statistical method
#     df_apartments['price_per_m2'] = df_apartments['price_$'] / df_apartments['size_m2']
#     q_low = df_apartments['price_per_m2'].quantile(0.01)
#     q_high = df_apartments['price_per_m2'].quantile(0.99)
#     df_apartments_clean = df_apartments[(df_apartments['price_per_m2'] >= q_low) & (df_apartments['price_per_m2'] <= q_high)].copy()
    
#     # Consolidate rare categorical features
#     district_counts = df_apartments_clean['district'].value_counts()
#     THRESHOLD = 30 
#     rare_districts = district_counts[district_counts < THRESHOLD].index.tolist()
#     df_apartments_clean.loc[df_apartments_clean['district'].isin(rare_districts), 'district'] = 'Other'
    
#     # Create the target variable
#     df_apartments_clean['log_price'] = np.log1p(df_apartments_clean['price_$'])
#     print(f"   -> Prepared data with {len(df_apartments_clean)} rows.")
    
#     return df_apartments_clean

# # ### --- NEW PLOT 1: Correlation Heatmap (Your function) --- ###
# def plot_correlation_heatmap(df, output_dir):
#     """Generates and saves a correlation heatmap for numerical features."""
#     print("\n--- Generating Correlation Heatmap ---")
#     plt.figure(figsize=(10, 8))
    
#     corr_cols = ['price_$', 'size_m2', 'bedrooms', 'bathrooms', 'latitude', 'longitude']
#     correlation_matrix = df[corr_cols].corr()
    
#     sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt=".2f", linewidths=.5)
#     plt.title('Correlation Matrix of Numerical Features')
#     plt.tight_layout()
    
#     plot_path = os.path.join(output_dir, 'correlation_heatmap.png')
#     plt.savefig(plot_path)
#     plt.close()
#     print(f"   -> Correlation plot saved to '{plot_path}'")

# # ### --- NEW PLOT 2: Regression Plot (Your function) --- ###
# def plot_regression_results(y_true, y_pred, output_dir):
#     """Generates and saves a scatter plot of actual vs. predicted values."""
#     print("\n--- Generating Regression Plot (Actual vs. Predicted) ---")
#     plt.figure(figsize=(10, 8))
    
#     plt.scatter(y_true, y_pred, alpha=0.3, label='Predictions')
#     perfect_line = np.linspace(min(y_true.min(), y_pred.min()), max(y_true.max(), y_pred.max()), 100)
#     plt.plot(perfect_line, perfect_line, color='red', linestyle='--', linewidth=2, label='Perfect Prediction')
    
#     plt.xlabel("Actual Price ($)")
#     plt.ylabel("Predicted Price ($)")
#     plt.title("Actual vs. Predicted Apartment Prices")
#     plt.legend()
#     plt.gca().ticklabel_format(style='plain', axis='both')
#     plt.grid(True)
#     plt.tight_layout()

#     plot_path = os.path.join(output_dir, 'regression_plot_actual_vs_predicted.png')
#     plt.savefig(plot_path)
#     plt.close()
#     print(f"   -> Regression plot saved to '{plot_path}'")

# # --- MODEL EVALUATION (Modified to include the regression plot call) ---
# def evaluate_model(pipeline, X_test, y_test, output_dir):
#     """Evaluates the model and calls the regression plotting function."""
#     print("\n--- Evaluating Model on Unseen Test Data ---")
    
#     predictions_log = pipeline.predict(X_test)
    
#     # Convert back to actual dollar values for MAE and plotting
#     predictions_actual = np.expm1(predictions_log)
#     y_test_actual = np.expm1(y_test)

#     r2 = r2_score(y_test, predictions_log) # R² is on log scale
#     mae = mean_absolute_error(y_test_actual, predictions_actual) # MAE is on dollar scale

#     print(f"   -> Test Set R² Score: {r2:.4f}")
#     print(f"   -> Test Set MAE (in $): ${mae:,.2f}")
    
#     # ### --- CHANGE 2: Call the plotting function from here --- ###
#     plot_regression_results(y_test_actual, predictions_actual, output_dir)
    
#     return r2, mae

# # --- FEATURE IMPORTANCE PLOTTING (Unchanged) ---
# def plot_feature_importance(pipeline, output_dir):
#     """Extracts and plots feature importances from the trained model."""
#     print("\n--- Generating Feature Importance Plot ---")
#     preprocessor = pipeline.named_steps['preprocessor']
#     regressor = pipeline.named_steps['regressor']
    
#     num_feature_names = preprocessor.transformers_[0][2]
#     cat_feature_names = preprocessor.named_transformers_['cat'].named_steps['onehot'].get_feature_names_out()
#     all_feature_names = np.concatenate([num_feature_names, cat_feature_names])
    
#     importances = pd.DataFrame({
#         'feature': all_feature_names,
#         'importance': regressor.feature_importances_
#     }).sort_values('importance', ascending=False).head(20)

#     plt.figure(figsize=(10, 8))
#     plt.barh(importances['feature'], importances['importance'])
#     plt.xlabel("LightGBM Feature Importance")
#     plt.ylabel("Feature")
#     plt.title("Top 20 Feature Importances (Tuned Champion Model)")
#     plt.gca().invert_yaxis()
#     plt.tight_layout()
    
#     plot_path = os.path.join(output_dir, 'feature_importance.png')
#     plt.savefig(plot_path)
#     plt.close()
#     print(f"   -> Feature importance plot saved to '{plot_path}'")

# def main():
#     try:
#         data_path, output_dir = setup_environment()
#         df = pd.read_csv(data_path)
#     except FileNotFoundError:
#         print(f"\nERROR: 'properties.csv' not found. Please place it in the directory: {os.getcwd()}")
#         return
    


#     df = pd.read_csv(data_path)
#     df.drop_duplicates(subset=['id'], inplace=True)
#     df.rename(columns={'type': 'property_type'}, inplace=True)
#     df['property_type'] = df['property_type'].str.lower().str.strip()
    
#     df_featured = prepare_apartment_data(df)
    
#     # ### --- CHANGE 3: Call the correlation heatmap plot after data prep --- ###
#     plot_correlation_heatmap(df_featured, output_dir)
    
#     # --- Define Features and Target ---
#     TARGET = 'log_price'
#     NUMERICAL_FEATURES = ['size_m2', 'bedrooms', 'bathrooms', 'latitude', 'longitude']
#     CATEGORICAL_FEATURES = ['province', 'district']
#     FEATURES = NUMERICAL_FEATURES + CATEGORICAL_FEATURES
    
#     X = df_featured[FEATURES]
#     y = df_featured[TARGET]
    
#     # ### --- CHANGE 4: Split data 70/30 as requested --- ###
#     print("\n--- Step 3: Splitting Data into Training (70%) and Testing (30%) Sets ---")
#     X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
#     print(f"   -> Training set size: {len(X_train)} rows")
#     print(f"   -> Testing set size:  {len(X_test)} rows")
    
#     # --- Build the base pipeline ---
#     preprocessor = ColumnTransformer(
#         transformers=[
#             ('num', Pipeline(steps=[('imputer', SimpleImputer(strategy='median')), ('scaler', StandardScaler())]), NUMERICAL_FEATURES),
#             ('cat', Pipeline(steps=[('imputer', SimpleImputer(strategy='most_frequent')), ('onehot', OneHotEncoder(handle_unknown='ignore', sparse_output=False))]), CATEGORICAL_FEATURES)
#         ])
    
#     pipeline = Pipeline(steps=[
#         ('preprocessor', preprocessor),
#         ('regressor', lgb.LGBMRegressor(objective='regression_l1', metric='rmse', random_state=42))
#     ])

#     # --- Define Parameter Grid ---
#     param_grid = {
#         'regressor__n_estimators': [800, 1200, 1500],
#         'regressor__learning_rate': [0.01, 0.02, 0.05],
#         'regressor__num_leaves': [31, 40],
#         'regressor__max_depth': [7, 9]
#     }
    
#     # --- Run GridSearchCV ---
#     print("\n--- Step 4: Running GridSearchCV to Find Best Hyperparameters ---")
#     grid_search = GridSearchCV(pipeline, param_grid, cv=5, scoring='r2', n_jobs=-1, verbose=1)
#     grid_search.fit(X_train, y_train)

#     print("\n--- GridSearchCV Results ---")
#     print(f"   -> Best Parameters Found: {grid_search.best_params_}")
#     print(f"   -> Best Cross-Validation R² Score: {grid_search.best_score_:.4f}")

#     best_model_pipeline = grid_search.best_estimator_
    
#     # --- Evaluate, Plot, and Save ---
#     test_r2, test_mae = evaluate_model(best_model_pipeline, X_test, y_test, output_dir)
#     plot_feature_importance(best_model_pipeline, output_dir)
    
#     model_path = os.path.join(output_dir, 'model_apartment_tuned.joblib')
#     joblib.dump(best_model_pipeline, model_path)
#     print(f"\n--- Final Tuned Model saved to '{model_path}' ---")
    
#     # --- Final Summary ---
#     print("\n" + "="*60)
#     print("--- MODEL PERFORMANCE SUMMARY ---")
#     summary = {
#         'Model': 'Apartment Price Estimator (Tuned Champion)',
#         'Final Test Set R2 Score': f"{test_r2:.4f}",
#         'Final Test Set MAE ($)': f"${test_mae:,.2f}",
#         'Number of Training Samples': len(X_train),
#         'Number of Test Samples': len(X_test),
#         'Features Used': FEATURES,
#         'Model Saved At': model_path,
#         'Plots Generated': 'correlation_heatmap.png, regression_plot_actual_vs_predicted.png, feature_importance.png'
#     }
#     for key, value in summary.items():
#         print(f"   -> {key}: {value}")
#     print("="*60)
#     print("--- WORKFLOW COMPLETE ---")

# if __name__ == '__main__':
#     main()
#
# Blueprint-Driven Real Estate Price Prediction
# This script implements the best practices from the consolidated research report
# to train, evaluate, and compare LightGBM and XGBoost models for apartment price prediction.
#

import pandas as pd
import numpy as np
import lightgbm as lgb
import xgboost as xgb  # Import XGBoost
import matplotlib.pyplot as plt
import seaborn as sns 
import joblib
import warnings
import os

from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import r2_score, mean_absolute_error
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer

# --- 1. SETUP & CONFIGURATION ---
def setup_environment():
    """Sets up directories and configurations for the script."""
    warnings.filterwarnings('ignore', category=UserWarning)
    pd.options.mode.chained_assignment = None

    # As requested, create a v8 output folder
    output_dir = os.path.join(os.getcwd(), 'final_model_output_v8')
    data_path = os.path.join(os.getcwd(), 'properties.csv')
    
    os.makedirs(output_dir, exist_ok=True)
    
    print("="*60)
    print("Environment Setup Complete")
    print(f"-> Model artifacts will be saved to: '{os.path.abspath(output_dir)}'")
    print("="*60)
    return data_path, output_dir


# --- 2. DATA PREPARATION & EXPLORATORY ANALYSIS ---
def prepare_apartment_data(df):
    """Loads, cleans, and engineers features for the apartment dataset."""
    print("\n--- Step 2: Preparing Apartment Data ---")
    
    # Filter for apartments only, as requested
    df_apartments = df[df['type'] == 'Apartment'].copy()
    print(f"   -> Found {len(df_apartments)} apartment listings.")
    
    # --- Imputation Strategy Part 1: Basic Cleaning ---
    # Convert 0s to NaN so they can be imputed properly later
    df_apartments['bedrooms'].replace(0, np.nan, inplace=True)
    df_apartments['bathrooms'].replace(0, np.nan, inplace=True)
    
    # Drop rows where key values are missing, as they cannot be used for training or reliable imputation
    df_apartments.dropna(subset=['price_$', 'size_m2'], inplace=True)
    df_apartments = df_apartments[df_apartments['size_m2'] > 0]
    
    # --- Feature Engineering based on Blueprint ---
    df_apartments['price_per_m2'] = df_apartments['price_$'] / df_apartments['size_m2']
    # Log transform size_m2 to handle skewness, a best practice for linear models and beneficial for tree models
    df_apartments['log_size_m2'] = np.log1p(df_apartments['size_m2'])
    
    # Outlier removal using a sound statistical method (interquartile range is also good)
    q_low = df_apartments['price_per_m2'].quantile(0.01)
    q_high = df_apartments['price_per_m2'].quantile(0.99)
    df_apartments_clean = df_apartments[(df_apartments['price_per_m2'] >= q_low) & (df_apartments['price_per_m2'] <= q_high)].copy()
    
    # Consolidate rare categorical features to prevent model from overfitting on noise
    district_counts = df_apartments_clean['district'].value_counts()
    THRESHOLD = 10 # Using a threshold to group infrequent districts
    rare_districts = district_counts[district_counts < THRESHOLD].index.tolist()
    if rare_districts:
        df_apartments_clean.loc[df_apartments_clean['district'].isin(rare_districts), 'district'] = 'Other'
        print(f"   -> Grouped {len(rare_districts)} rare districts into 'Other' for model stability.")

    # Create the target variable: log-transforming price helps model learn better
    df_apartments_clean['log_price'] = np.log1p(df_apartments_clean['price_$'])
    print(f"   -> Prepared final dataset with {len(df_apartments_clean)} rows.")
    
    return df_apartments_clean

def plot_correlation_heatmap(df, output_dir):
    """Generates and saves a correlation heatmap for numerical features."""
    print("\n--- Generating Correlation Heatmap ---")
    plt.figure(figsize=(10, 8))
    
    # Select key numerical features for the heatmap
    corr_cols = ['price_$', 'size_m2', 'bedrooms', 'bathrooms', 'latitude', 'longitude']
    correlation_matrix = df[corr_cols].corr()
    
    sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt=".2f", linewidths=.5)
    plt.title('Correlation Matrix of Key Numerical Features')
    plt.tight_layout()
    
    plot_path = os.path.join(output_dir, 'correlation_heatmap.png')
    plt.savefig(plot_path)
    plt.close()
    print(f"   -> Correlation plot saved to '{plot_path}'")

def plot_regression_results(y_true, y_pred, model_name, output_dir):
    """Generates a scatter plot of actual vs. predicted values."""
    print(f"\n--- Generating Regression Plot for {model_name} ---")
    plt.figure(figsize=(10, 8))
    
    plt.scatter(y_true, y_pred, alpha=0.4, label='Predictions')
    # Add a perfect prediction line (y=x)
    perfect_line = np.linspace(min(y_true.min(), y_pred.min()), max(y_true.max(), y_pred.max()), 100)
    plt.plot(perfect_line, perfect_line, color='red', linestyle='--', linewidth=2, label='Perfect Prediction')
    
    plt.xlabel("Actual Price ($)")
    plt.ylabel("Predicted Price ($)")
    plt.title(f"Actual vs. Predicted Prices ({model_name})")
    plt.legend()
    # Format axes to show full numbers instead of scientific notation
    plt.gca().ticklabel_format(style='plain', axis='both')
    plt.grid(True)
    plt.tight_layout()

    plot_path = os.path.join(output_dir, f'regression_plot_{model_name.lower()}.png')
    plt.savefig(plot_path)
    plt.close()
    print(f"   -> Regression plot saved to '{plot_path}'")

def plot_feature_importance(pipeline, model_name, output_dir):
    """Extracts and plots feature importances from the trained model."""
    print(f"\n--- Generating Feature Importance Plot for {model_name} ---")
    preprocessor = pipeline.named_steps['preprocessor']
    regressor = pipeline.named_steps['regressor']
    
    # Get feature names after preprocessing
    num_feature_names = preprocessor.transformers_[0][2]
    cat_feature_names = preprocessor.named_transformers_['cat'].named_steps['onehot'].get_feature_names_out()
    all_feature_names = np.concatenate([num_feature_names, cat_feature_names])
    
    importances = pd.DataFrame({
        'feature': all_feature_names,
        'importance': regressor.feature_importances_
    }).sort_values('importance', ascending=False).head(20)

    plt.figure(figsize=(12, 8))
    sns.barplot(x='importance', y='feature', data=importances, palette='viridis')
    plt.xlabel(f"{model_name} Feature Importance")
    plt.ylabel("Feature")
    plt.title(f"Top 20 Feature Importances ({model_name})")
    plt.tight_layout()
    
    plot_path = os.path.join(output_dir, f'feature_importance_{model_name.lower()}.png')
    plt.savefig(plot_path)
    plt.close()
    print(f"   -> Feature importance plot saved to '{plot_path}'")
    
# --- 3. MODEL TRAINING & EVALUATION WORKFLOW ---
def main():
    try:
        data_path, output_dir = setup_environment()
        df_raw = pd.read_csv(data_path)
    except FileNotFoundError:
        print(f"\nERROR: 'properties.csv' not found. Please place it in the directory: {os.getcwd()}")
        return

    df_raw.drop_duplicates(subset=['id'], inplace=True)
    df_raw.rename(columns={'type': 'type'}, inplace=True) # Standardize column name
    
    df_featured = prepare_apartment_data(df_raw)
    
    # Generate the correlation heatmap after data prep
    plot_correlation_heatmap(df_featured, output_dir)
    
    # --- Define Features and Target ---
    TARGET = 'log_price'
    NUMERICAL_FEATURES = ['log_size_m2', 'bedrooms', 'bathrooms', 'latitude', 'longitude']
    CATEGORICAL_FEATURES = ['province', 'district', 'city']
    FEATURES = NUMERICAL_FEATURES + CATEGORICAL_FEATURES
    
    X = df_featured[FEATURES]
    y = df_featured[TARGET]
    
    # --- Training/Validation/Testing Split ---
    # The data is split into a training set for model tuning and a final test set for evaluation.
    print("\n--- Step 3: Splitting Data into Training (70%) and Testing (30%) Sets ---")
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
    print(f"   -> Training set size: {len(X_train)} rows")
    print(f"   -> Testing set size:  {len(X_test)} rows")
    
    # --- Preprocessing Pipeline with Imputation ---
    # This pipeline handles missing values and scales/encodes data consistently.
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', Pipeline(steps=[
                ('imputer', SimpleImputer(strategy='median')), # Median imputation for robustness to outliers
                ('scaler', StandardScaler())
            ]), NUMERICAL_FEATURES),
            ('cat', Pipeline(steps=[
                ('imputer', SimpleImputer(strategy='most_frequent')), # Fills with most common value
                ('onehot', OneHotEncoder(handle_unknown='ignore', sparse_output=False))
            ]), CATEGORICAL_FEATURES)
        ], remainder='drop')

    # ==============================================================================
    # --- Part 1: Training and Evaluating the LightGBM Model ---
    # ==============================================================================
    print("\n" + "="*20 + " LightGBM Model Workflow " + "="*20)
    pipeline_lgbm = Pipeline(steps=[
        ('preprocessor', preprocessor),
        ('regressor', lgb.LGBMRegressor(objective='regression_l1', metric='mae', random_state=42))
    ])

    # Define Parameter Grid for 5-Fold Cross-Validation
    param_grid_lgbm = {
        'regressor__n_estimators': [1000, 1500],
        'regressor__learning_rate': [0.01, 0.02],
        'regressor__num_leaves': [31, 40],
        'regressor__max_depth': [7, 9],
        'regressor__l2_leaf_reg': [3, 5]
    }
    
    print("\n--- Running GridSearchCV for LightGBM (with 5-Fold CV) ---")
    grid_search_lgbm = GridSearchCV(pipeline_lgbm, param_grid_lgbm, cv=5, scoring='r2', n_jobs=-1, verbose=1)
    grid_search_lgbm.fit(X_train, y_train)
    
    best_lgbm_model = grid_search_lgbm.best_estimator_
    
    # ==============================================================================
    # --- Part 2: Training and Evaluating the XGBoost Model ---
    # ==============================================================================
    print("\n" + "="*20 + " XGBoost Model Workflow " + "="*20)
    pipeline_xgb = Pipeline(steps=[
        ('preprocessor', preprocessor),
        ('regressor', xgb.XGBRegressor(objective='reg:squarederror', eval_metric='mae', random_state=42, n_jobs=-1))
    ])

    param_grid_xgb = {
        'regressor__n_estimators': [1000, 1500],
        'regressor__learning_rate': [0.01, 0.02],
        'regressor__max_depth': [6, 8],
        'regressor__subsample': [0.7, 0.8],
        'regressor__colsample_bytree': [0.7, 0.8]
    }

    print("\n--- Running GridSearchCV for XGBoost (with 5-Fold CV) ---")
    grid_search_xgb = GridSearchCV(pipeline_xgb, param_grid_xgb, cv=5, scoring='r2', n_jobs=-1, verbose=1)
    grid_search_xgb.fit(X_train, y_train)

    best_xgb_model = grid_search_xgb.best_estimator_

    # --- 4. FINAL EVALUATION, PLOTTING, and SAVING ---
    final_results = {}

    # Evaluate LightGBM
    y_pred_log_lgbm = best_lgbm_model.predict(X_test)
    y_pred_actual_lgbm = np.expm1(y_pred_log_lgbm)
    y_test_actual = np.expm1(y_test)
    
    final_results['LightGBM'] = {
        'Best CV R²': grid_search_lgbm.best_score_,
        'Test R²': r2_score(y_test, y_pred_log_lgbm),
        'Test MAE ($)': mean_absolute_error(y_test_actual, y_pred_actual_lgbm),
        'Best Params': grid_search_lgbm.best_params_
    }
    plot_regression_results(y_test_actual, y_pred_actual_lgbm, "LightGBM", output_dir)
    plot_feature_importance(best_lgbm_model, "LightGBM", output_dir)
    joblib.dump(best_lgbm_model, os.path.join(output_dir, 'model_apartment_lgbm_tuned.joblib'))
    
    # Evaluate XGBoost
    y_pred_log_xgb = best_xgb_model.predict(X_test)
    y_pred_actual_xgb = np.expm1(y_pred_log_xgb)
    
    final_results['XGBoost'] = {
        'Best CV R²': grid_search_xgb.best_score_,
        'Test R²': r2_score(y_test, y_pred_log_xgb),
        'Test MAE ($)': mean_absolute_error(y_test_actual, y_pred_actual_xgb),
        'Best Params': grid_search_xgb.best_params_
    }
    plot_regression_results(y_test_actual, y_pred_actual_xgb, "XGBoost", output_dir)
    plot_feature_importance(best_xgb_model, "XGBoost", output_dir)
    joblib.dump(best_xgb_model, os.path.join(output_dir, 'model_apartment_xgb_tuned.joblib'))

    # --- 5. Final Summary ---
    print("\n" + "="*60)
    print("--- FINAL MODEL COMPARISON SUMMARY ---")
    
    for model_name, metrics in final_results.items():
        print(f"\n--- {model_name} ---")
        print(f"   -> Best Cross-Validation R²: {metrics['Best CV R²']:.4f}")
        print(f"   -> Final Test Set R²:          {metrics['Test R²']:.4f}")
        print(f"   -> Final Test Set MAE ($):     ${metrics['Test MAE ($)']:,.2f}")
        print(f"   -> Best Hyperparameters:     {metrics['Best Params']}")
        
    print("\n" + "="*60)
    print("--- WORKFLOW COMPLETE ---")

if __name__ == '__main__':
    main()