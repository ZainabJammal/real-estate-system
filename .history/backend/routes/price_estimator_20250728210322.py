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
from supabase import create_client, Client
from dotenv import load_dotenv

# --- 1. CONFIGURATION ---
# Load environment variables from .env file
load_dotenv()

# Initialize Supabase client
supabase_url = os.getenv("SUPABASE_URL")
supabase_key = os.getenv("SUPABASE_KEY")
supabase: Client = create_client(supabase_url, supabase_key)
def load_data_from_supabase():
    """Connects to Supabase and fetches all regional transaction data."""
    print("-> Connecting to Supabase to fetch training data...")
    url: str = os.environ.get("SUPABASE_URL")
    key: str = os.environ.get("SUPABASE_KEY")

    if not url or not key:
        raise ValueError("FATAL: Supabase URL or Key not found in .env file.")
        
    supabase: Client = create_client(url, key)

    # Fetch all data from the table, ordered by date
    response = supabase.table('properties').select("*").order('date').execute()
    
    # Convert the list of dictionaries to a pandas DataFrame
    df = pd.DataFrame(response.data)
    print(f"-> Successfully fetched {len(df)} rows from Supabase.")

    
    
    return df

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