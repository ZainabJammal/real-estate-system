
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
#     output_dir = os.path.join(backend_dir, 'final_model_output_v2')
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
#     """Loads, cleans, and engineers features for the apartment dataset using size categories."""
#     print("\n--- Step 2: Preparing Apartment Data with Size Categories ---")
        
#     # --- Basic Cleaning ---
#     df_apartments = df[df['property_type'] == 'apartment'].copy()
#     df_apartments['bedrooms'].replace(0, np.nan, inplace=True)
#     df_apartments['bathrooms'].replace(0, np.nan, inplace=True)
#     df_apartments.dropna(subset=['price_$', 'size_m2', 'bedrooms', 'bathrooms'], inplace=True)
#     df_apartments = df_apartments[df_apartments['size_m2'] > 0]
    
#     # --- CHANGE 1: Feature Engineering for Size Category ---
#     print("\n--- Feature Engineering ---")
#     bins = [0, 80, 120, 180, 250, 400, float('inf')]
#     labels = ['Studio/Small (0-80m²)',
#             'Standard (81-120m²)',
#             'Comfortable (121-180m²)', 
#             'Large (181-250m²)',
#             'Very Large (251-400m²)',
#             'Penthouse (401m²+)']
    
#     df_apartments['size_category'] = pd.cut(df_apartments['size_m2'], bins=bins, labels=labels, right=True)
#     print(f"   -> Created 'size_category' feature from 'size_m2'.")

#     # --- Basic Feature Engineering (still useful) ---
#     df_apartments['price_per_m2'] = df_apartments['price_$'] / df_apartments['size_m2']
    
#     # --- Outlier Removal (using price_per_m2 is still a good idea) ---
#     q_low = df_apartments['price_per_m2'].quantile(0.01)
#     q_high = df_apartments['price_per_m2'].quantile(0.99)
#     df_apartments_clean = df_apartments[(df_apartments['price_per_m2'] >= q_low) & (df_apartments['price_per_m2'] <= q_high)].copy()
#     print(f"   -> {len(df_apartments_clean)} rows after cleaning and outlier removal.")

#     # --- Handling Rare Categories (still a good practice) ---
#     district_counts = df_apartments_clean['district'].value_counts()
#     THRESHOLD = 30 
#     rare_districts = district_counts[district_counts < THRESHOLD].index.tolist()
#     df_apartments_clean.loc[df_apartments_clean['district'].isin(rare_districts), 'district'] = 'Other'
#     print(f"   -> Grouped {len(rare_districts)} rare districts into 'Other' category.")

#     # --- Final Transformation ---
#     df_apartments_clean['log_price'] = np.log1p(df_apartments_clean['price_$'])
#     print("   -> Created 'log_price' as the final target variable.")
    
#     return df_apartments_clean

# # --- 3. MODELING WORKFLOW (REVISED) ---
# def build_and_evaluate_model(df_featured, config, output_dir):
#     """Builds the preprocessing pipeline, validates, trains, and saves the final model."""
#     print("\n--- Step 3: Building and Validating the Model ---")
    
#     # --- CHANGE 2: Define New Features and Target ---
#     TARGET = 'log_price'
    
#     # Define features based on the new approach
#     NUMERICAL_FEATURES = ['bedrooms', 'bathrooms', 'latitude', 'longitude']
#     CATEGORICAL_FEATURES = ['province', 'district', 'size_category'] # Add size_category here
    
#     # The interaction features are no longer needed as the model will learn the interaction
#     # between the 'district' category and the new 'size_category' automatically.
    
#     FEATURES = NUMERICAL_FEATURES + CATEGORICAL_FEATURES
    
#     X = df_featured[FEATURES]
#     y = df_featured[TARGET]

#     # --- CHANGE 3: Define the new preprocessing pipeline ---
#     preprocessor = ColumnTransformer(
#         transformers=[
#             ('num', Pipeline(steps=[
#                 ('imputer', SimpleImputer(strategy='median')),
#                 ('scaler', StandardScaler())]), NUMERICAL_FEATURES),
#             ('cat', Pipeline(steps=[
#                 ('imputer', SimpleImputer(strategy='most_frequent')),
#                 ('onehot', OneHotEncoder(handle_unknown='ignore', sparse_output=False))]), CATEGORICAL_FEATURES)
#         ],
#         remainder='drop'
#     )

#     # Create the full model pipeline (this part is unchanged)
#     pipeline = Pipeline(steps=[
#         ('preprocessor', preprocessor),
#         ('regressor', lgb.LGBMRegressor(**config['params']))
#     ])

#     # --- Cross-Validation (Unchanged) ---
#     kf = KFold(n_splits=5, shuffle=True, random_state=42)
#     cv_scores_r2 = cross_val_score(pipeline, X, y, cv=kf, scoring='r2')
    
#     print("\n--- Cross-Validation Performance ---")
#     print(f"   -> 5-Fold R² Scores: {[f'{s:.4f}' for s in cv_scores_r2]}")
#     print(f"   -> Average R² Score: {np.mean(cv_scores_r2):.4f} (± {np.std(cv_scores_r2):.4f})")

#     # --- Final Model Training (Unchanged) ---
#     print("\n--- Step 4: Training Final Model on All Data ---")
#     pipeline.fit(X, y)
#     print("   -> Final model training complete.")
    
#     return pipeline, np.mean(cv_scores_r2)

# # --- 4. FEATURE IMPORTANCE & 5. MAIN EXECUTION (Largely Unchanged but will use new features) ---
# def plot_feature_importance(pipeline, output_dir):
#     """Extracts and plots feature importances from the trained model."""
#     print("\n--- Step 5: Generating Feature Importance Plot ---")

#     preprocessor = pipeline.named_steps['preprocessor']
#     regressor = pipeline.named_steps['regressor']
    
#     # Get names from numerical transformer
#     num_feature_names = preprocessor.transformers_[0][2]
#     # Get names from categorical transformer after one-hot encoding
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
#     plt.title("Top 20 Feature Importances (with Size Category)")
#     plt.gca().invert_yaxis()
#     plt.tight_layout()
    
#     plot_path = os.path.join(output_dir, 'feature_importance_size_category.png')
#     plt.savefig(plot_path)
#     print(f"   -> Feature importance plot saved to '{plot_path}'")
#     plt.close()

# def main():
#     data_path, output_dir = setup_environment()
#     df = pd.read_csv(data_path)
#     df.drop_duplicates(subset=['id'], inplace=True)
#     df.rename(columns={'type': 'property_type'}, inplace=True)
#     df['property_type'] = df['property_type'].str.lower().str.strip()
    
#     df_featured = prepare_apartment_data(df)
    
#     final_pipeline, avg_r2_score = build_and_evaluate_model(df_featured, MODEL_CONFIG['apartment'], output_dir)
    
#     plot_feature_importance(final_pipeline, output_dir)
    
#     model_path = os.path.join(output_dir, 'model_apartment_size_category.joblib') 
#     joblib.dump(final_pipeline, model_path)
    
#     print(f"\n--- Step 6: Final model saved to '{model_path}' ---")
#     print("\n" + "="*60); 
#     print("--- MODEL PERFORMANCE SUMMARY ---")
#     summary = {
#         'Model': 'Apartment Price Estimator (Size Category)',
#         'Average R2 (5-Fold CV)': f"{avg_r2_score:.4f}",
#         'Number of Training Samples': len(df_featured),
#         'Model Saved At': model_path,
#         'Importance Plot Saved At': os.path.join(output_dir, 'feature_importance_size_category.png')
#     }
#     for key, value in summary.items(): 
#         print(f"   -> {key}: {value}")
#     print("="*60)
#     print("--- WORKFLOW COMPLETE ---")

# if __name__ == '__main__':
#     main()


import pandas as pd
import numpy as np
import lightgbm as lgb
import matplotlib.pyplot as plt
import seaborn as sns # ### --- CHANGE 1: Added seaborn for the heatmap --- ###
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
    
    script_path = os.path.abspath(__file__)
    # Navigate up to the project's root or a suitable base directory
    backend_dir = os.path.dirname(os.path.dirname(script_path)) 
    
    output_dir = os.path.join(backend_dir, 'final_model_output_v_with_plots')
    data_path = os.path.join(backend_dir, 'properties.csv')
    
    os.makedirs(output_dir, exist_ok=True)
    
    print("="*60)
    print("Environment Setup Complete")
    print(f"-> Model artifacts will be saved to: '{os.path.abspath(output_dir)}'")
    print("="*60)
    
    return data_path, output_dir

# --- 2. DATA PREPARATION ---
def prepare_apartment_data(df):
    """Loads, cleans, and engineers features for the apartment dataset."""
    print("\n--- Step 2: Preparing Apartment Data ---")
    
    df_apartments = df[df['property_type'] == 'apartment'].copy()
    
    # Basic data cleaning
    df_apartments['bedrooms'].replace(0, np.nan, inplace=True)
    df_apartments['bathrooms'].replace(0, np.nan, inplace=True)
    df_apartments.dropna(subset=['price_$', 'size_m2', 'bedrooms', 'bathrooms'], inplace=True)
    df_apartments = df_apartments[df_apartments['size_m2'] > 0]
    
    # Outlier removal using a sound statistical method
    df_apartments['price_per_m2'] = df_apartments['price_$'] / df_apartments['size_m2']
    q_low = df_apartments['price_per_m2'].quantile(0.01)
    q_high = df_apartments['price_per_m2'].quantile(0.99)
    df_apartments_clean = df_apartments[(df_apartments['price_per_m2'] >= q_low) & (df_apartments['price_per_m2'] <= q_high)].copy()
    
    # Consolidate rare categorical features
    district_counts = df_apartments_clean['district'].value_counts()
    THRESHOLD = 30 
    rare_districts = district_counts[district_counts < THRESHOLD].index.tolist()
    df_apartments_clean.loc[df_apartments_clean['district'].isin(rare_districts), 'district'] = 'Other'
    
    # Create the target variable
    df_apartments_clean['log_price'] = np.log1p(df_apartments_clean['price_$'])
    print(f"   -> Prepared data with {len(df_apartments_clean)} rows.")
    
    return df_apartments_clean

# ### --- NEW PLOT 1: Correlation Heatmap (Your function) --- ###
def plot_correlation_heatmap(df, output_dir):
    """Generates and saves a correlation heatmap for numerical features."""
    print("\n--- Generating Correlation Heatmap ---")
    plt.figure(figsize=(10, 8))
    
    corr_cols = ['price_$', 'size_m2', 'bedrooms', 'bathrooms', 'latitude', 'longitude']
    correlation_matrix = df[corr_cols].corr()
    
    sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt=".2f", linewidths=.5)
    plt.title('Correlation Matrix of Numerical Features')
    plt.tight_layout()
    
    plot_path = os.path.join(output_dir, 'correlation_heatmap.png')
    plt.savefig(plot_path)
    plt.close()
    print(f"   -> Correlation plot saved to '{plot_path}'")

# ### --- NEW PLOT 2: Regression Plot (Your function) --- ###
def plot_regression_results(y_true, y_pred, output_dir):
    """Generates and saves a scatter plot of actual vs. predicted values."""
    print("\n--- Generating Regression Plot (Actual vs. Predicted) ---")
    plt.figure(figsize=(10, 8))
    
    plt.scatter(y_true, y_pred, alpha=0.3, label='Predictions')
    perfect_line = np.linspace(min(y_true.min(), y_pred.min()), max(y_true.max(), y_pred.max()), 100)
    plt.plot(perfect_line, perfect_line, color='red', linestyle='--', linewidth=2, label='Perfect Prediction')
    
    plt.xlabel("Actual Price ($)")
    plt.ylabel("Predicted Price ($)")
    plt.title("Actual vs. Predicted Apartment Prices")
    plt.legend()
    plt.gca().ticklabel_format(style='plain', axis='both')
    plt.grid(True)
    plt.tight_layout()

    plot_path = os.path.join(output_dir, 'regression_plot_actual_vs_predicted.png')
    plt.savefig(plot_path)
    plt.close()
    print(f"   -> Regression plot saved to '{plot_path}'")

# --- MODEL EVALUATION (Modified to include the regression plot call) ---
def evaluate_model(pipeline, X_test, y_test, output_dir):
    """Evaluates the model and calls the regression plotting function."""
    print("\n--- Evaluating Model on Unseen Test Data ---")
    
    predictions_log = pipeline.predict(X_test)
    
    # Convert back to actual dollar values for MAE and plotting
    predictions_actual = np.expm1(predictions_log)
    y_test_actual = np.expm1(y_test)

    r2 = r2_score(y_test, predictions_log) # R² is on log scale
    mae = mean_absolute_error(y_test_actual, predictions_actual) # MAE is on dollar scale

    print(f"   -> Test Set R² Score: {r2:.4f}")
    print(f"   -> Test Set MAE (in $): ${mae:,.2f}")
    
    # ### --- CHANGE 2: Call the plotting function from here --- ###
    plot_regression_results(y_test_actual, predictions_actual, output_dir)
    
    return r2, mae

# --- FEATURE IMPORTANCE PLOTTING (Unchanged) ---
def plot_feature_importance(pipeline, output_dir):
    """Extracts and plots feature importances from the trained model."""
    print("\n--- Generating Feature Importance Plot ---")
    preprocessor = pipeline.named_steps['preprocessor']
    regressor = pipeline.named_steps['regressor']
    
    num_feature_names = preprocessor.transformers_[0][2]
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
    plt.title("Top 20 Feature Importances (Tuned Champion Model)")
    plt.gca().invert_yaxis()
    plt.tight_layout()
    
    plot_path = os.path.join(output_dir, 'feature_importance.png')
    plt.savefig(plot_path)
    plt.close()
    print(f"   -> Feature importance plot saved to '{plot_path}'")

def main():
    data_path, output_dir = setup_environment()
    df = pd.read_csv(data_path)
    df.drop_duplicates(subset=['id'], inplace=True)
    df.rename(columns={'type': 'property_type'}, inplace=True)
    df['property_type'] = df['property_type'].str.lower().str.strip()
    
    df_featured = prepare_apartment_data(df)
    
    # ### --- CHANGE 3: Call the correlation heatmap plot after data prep --- ###
    plot_correlation_heatmap(df_featured, output_dir)
    
    # --- Define Features and Target ---
    TARGET = 'log_price'
    NUMERICAL_FEATURES = ['size_m2', 'bedrooms', 'bathrooms', 'latitude', 'longitude']
    CATEGORICAL_FEATURES = ['province', 'district']
    FEATURES = NUMERICAL_FEATURES + CATEGORICAL_FEATURES
    
    X = df_featured[FEATURES]
    y = df_featured[TARGET]
    
    # ### --- CHANGE 4: Split data 70/30 as requested --- ###
    print("\n--- Step 3: Splitting Data into Training (70%) and Testing (30%) Sets ---")
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
    print(f"   -> Training set size: {len(X_train)} rows")
    print(f"   -> Testing set size:  {len(X_test)} rows")
    
    # --- Build the base pipeline ---
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', Pipeline(steps=[('imputer', SimpleImputer(strategy='median')), ('scaler', StandardScaler())]), NUMERICAL_FEATURES),
            ('cat', Pipeline(steps=[('imputer', SimpleImputer(strategy='most_frequent')), ('onehot', OneHotEncoder(handle_unknown='ignore', sparse_output=False))]), CATEGORICAL_FEATURES)
        ])
    
    pipeline = Pipeline(steps=[
        ('preprocessor', preprocessor),
        ('regressor', lgb.LGBMRegressor(objective='regression_l1', metric='rmse', random_state=42))
    ])

    # --- Define Parameter Grid ---
    param_grid = {
        'regressor__n_estimators': [800, 1200, 1500],
        'regressor__learning_rate': [0.01, 0.02, 0.05],
        'regressor__num_leaves': [31, 40],
        'regressor__max_depth': [7, 9]
    }
    
    # --- Run GridSearchCV ---
    print("\n--- Step 4: Running GridSearchCV to Find Best Hyperparameters ---")
    grid_search = GridSearchCV(pipeline, param_grid, cv=5, scoring='r2', n_jobs=-1, verbose=1)
    grid_search.fit(X_train, y_train)

    print("\n--- GridSearchCV Results ---")
    print(f"   -> Best Parameters Found: {grid_search.best_params_}")
    print(f"   -> Best Cross-Validation R² Score: {grid_search.best_score_:.4f}")

    best_model_pipeline = grid_search.best_estimator_
    
    # --- Evaluate, Plot, and Save ---
    test_r2, test_mae = evaluate_model(best_model_pipeline, X_test, y_test, output_dir)
    plot_feature_importance(best_model_pipeline, output_dir)
    
    model_path = os.path.join(output_dir, 'model_apartment_tuned.joblib')
    joblib.dump(best_model_pipeline, model_path)
    print(f"\n--- Final Tuned Model saved to '{model_path}' ---")
    
    # --- Final Summary ---
    print("\n" + "="*60)
    print("--- MODEL PERFORMANCE SUMMARY ---")
    summary = {
        'Model': 'Apartment Price Estimator (Tuned Champion)',
        'Final Test Set R2 Score': f"{test_r2:.4f}",
        'Final Test Set MAE ($)': f"${test_mae:,.2f}",
        'Number of Training Samples': len(X_train),
        'Number of Test Samples': len(X_test),
        'Features Used': FEATURES,
        'Model Saved At': model_path,
        'Plots Generated': 'correlation_heatmap.png, regression_plot_actual_vs_predicted.png, feature_importance.png'
    }
    for key, value in summary.items():
        print(f"   -> {key}: {value}")
    print("="*60)
    print("--- WORKFLOW COMPLETE ---")

if __name__ == '__main__':
    main()