import pandas as pd
import numpy as np
import xgboost as xgb
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

# --- 1. SETUP & CONFIGURATION ---
def setup_environment():
    warnings.filterwarnings('ignore', category=UserWarning)
    pd.options.mode.chained_assignment = None

    try:
        # This works when running as a .py file
        SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
        BACKEND_DIR = os.path.dirname(SCRIPT_DIR)
    except NameError:
        # Fallback for interactive environments (like Jupyter)
        SCRIPT_DIR = os.getcwd()
        BACKEND_DIR = SCRIPT_DIR

    output_dir = os.path.join(BACKEND_DIR, 'price_models')
    os.makedirs(output_dir, exist_ok=True)

    # output_dir = os.path.join(os.getcwd(), 'price_estimation_model') 
    # os.makedirs(output_dir, exist_ok=True)
    print("="*60)
    print("Final Model Training Environment Setup")
    print(f"-> Model artifacts will be saved to: '{os.path.abspath(output_dir)}'")
    print("="*60)
    return output_dir

def fetch_supabase_data():
    """Connects to Supabase and fetches all properties data."""
    print("-> Connecting to Supabase to fetch training data...")
    load_dotenv()
    url: str = os.environ.get("SUPABASE_URL")
    key: str = os.environ.get("SUPABASE_KEY")
    if not url or not key:
        raise ValueError("FATAL: Supabase URL or Key not found in .env file.")
    supabase: Client = create_client(url, key)
    response = supabase.table('properties').select("*").execute()
    df = pd.DataFrame(response.data)
    print(f"-> Successfully fetched {len(df)} rows from Supabase.")
    return df

# --- 2. DATA PREPARATION (WITH CORRECTIONS) ---
def prepare_apartment_data(df):
    """Loads, cleans, and engineers features for the apartment dataset."""
    print("\n--- Step 2: Preparing Apartment Data with Advanced Features ---")
    df_apartments = df[df['type'] == 'Apartment'].copy()

    for col in ['bedrooms', 'bathrooms', 'price_$', 'size_m2']:
        df_apartments[col] = pd.to_numeric(df_apartments[col], errors='coerce')

    df_apartments.dropna(subset=['price_$', 'size_m2'], inplace=True)
    df_apartments['bedrooms'].replace(0, np.nan, inplace=True)
    df_apartments['bathrooms'].replace(0, np.nan, inplace=True)
    df_apartments = df_apartments[df_apartments['size_m2'] > 0]

    # --- Outlier Removal (Done before creating aggregate features to prevent data leakage) ---
    df_apartments['price_per_m2'] = df_apartments['price_$'] / df_apartments['size_m2']
    q_low = df_apartments['price_per_m2'].quantile(0.01)
    q_high = df_apartments['price_per_m2'].quantile(0.99)
    df_apartments_clean = df_apartments[(df_apartments['price_per_m2'] >= q_low) & (df_apartments['price_per_m2'] <= q_high)].copy()

    # ### CORRECTED: Feature Engineering now happens on the CLEANED dataframe ###
    print("   -> Engineering new features...")
    # 1. Ratio Features (Handle division by zero by replacing results with NaN)
    df_apartments_clean['bed_bath_ratio'] = df_apartments_clean['bedrooms'] / df_apartments_clean['bathrooms']
    df_apartments_clean['size_per_bedroom'] = df_apartments_clean['size_m2'] / df_apartments_clean['bedrooms']
    df_apartments_clean.replace([np.inf, -np.inf], np.nan, inplace=True) # Important step!

    # 2. Location-based Aggregate Features (Calculated on the clean data)
    df_apartments_clean['avg_price_in_district'] = df_apartments_clean.groupby('district')['price_per_m2'].transform('mean')
    df_apartments_clean['avg_size_in_district'] = df_apartments_clean.groupby('district')['size_m2'].transform('mean')
    print("      - Created ratio and location-based aggregate features.")

    # Consolidate rare districts
    district_counts = df_apartments_clean['district'].value_counts()
    THRESHOLD = 10
    rare_districts = district_counts[district_counts < THRESHOLD].index.tolist()
    if rare_districts:
        df_apartments_clean.loc[df_apartments_clean['district'].isin(rare_districts), 'district'] = 'Other'

    # Target and feature transformation
    df_apartments_clean['log_size_m2'] = np.log1p(df_apartments_clean['size_m2'])
    df_apartments_clean['log_price'] = np.log1p(df_apartments_clean['price_$'])

    print(f"   -> Prepared final dataset with {len(df_apartments_clean)} rows.")
    return df_apartments_clean

# --- (Plotting functions remain unchanged) ---
def plot_correlation_heatmap(df, output_dir):
    print("\n--- Generating Correlation Heatmap (with new features) ---")
    plt.figure(figsize=(12, 10))
    corr_cols = ['price_$', 'size_m2', 'bedrooms', 'bathrooms', 'latitude', 'longitude',
                 'avg_size_in_district',  'price_per_m2', 'dist_to_cbd_km', 'size_per_bedroom', 'bed_bath_ratio']
    correlation_matrix = df[corr_cols].corr()
    sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt=".2f", linewidths=.5)
    plt.title('Correlation Matrix with New Engineered Features')
    plt.tight_layout()
    plot_path = os.path.join(output_dir, 'correlation_heatmap_optimized.png')
    plt.savefig(plot_path)
    plt.close()
    print(f"   -> Correlation plot saved to '{plot_path}'")

def plot_regression_results(y_true, y_pred, output_dir):
    print(f"\n--- Generating Regression Plot for Optimized XGBoost ---")
    plt.figure(figsize=(10, 8))
    plt.scatter(y_true, y_pred, alpha=0.4)
    perfect_line = np.linspace(min(y_true.min(), y_pred.min()), max(y_true.max(), y_pred.max()), 100)
    plt.plot(perfect_line, perfect_line, color='red', linestyle='--', linewidth=2, label='Perfect Prediction')
    plt.xlabel("Actual Price ($)"); plt.ylabel("Predicted Price ($)")
    plt.title(f"Actual vs. Predicted Prices (Optimized XGBoost)")
    plt.legend(); plt.gca().ticklabel_format(style='plain', axis='both'); plt.grid(True); plt.tight_layout()
    plot_path = os.path.join(output_dir, f'regression_plot_optimized_xgb.png')
    plt.savefig(plot_path)
    plt.close()
    print(f"   -> Regression plot saved to '{plot_path}'")

def plot_feature_importance(pipeline, output_dir):
    print(f"\n--- Generating Feature Importance Plot for Optimized XGBoost ---")
    preprocessor = pipeline.named_steps['preprocessor']
    regressor = pipeline.named_steps['regressor']
    num_feature_names = preprocessor.transformers_[0][2]
    cat_feature_names = preprocessor.named_transformers_['cat'].named_steps['onehot'].get_feature_names_out()
    all_feature_names = np.concatenate([num_feature_names, cat_feature_names])
    importances = pd.DataFrame({'feature': all_feature_names, 'importance': regressor.feature_importances_}).sort_values('importance', ascending=False).head(20)
    plt.figure(figsize=(12, 8)); sns.barplot(x='importance', y='feature', data=importances, palette='viridis')
    plt.xlabel(f"XGBoost Feature Importance"); plt.ylabel("Feature")
    plt.title(f"Top 20 Feature Importances (Optimized XGBoost)"); plt.tight_layout()
    plot_path = os.path.join(output_dir, f'feature_importance_optimized_xgb.png')
    plt.savefig(plot_path)
    plt.close()
    print(f"   -> Feature importance plot saved to '{plot_path}'")
    


# --- 3. MODEL TRAINING & EVALUATION WORKFLOW ---
def main():
    output_dir = setup_environment()
    df_raw = fetch_supabase_data()
    if df_raw.empty:
        print("No data fetched from Supabase. Exiting.")
        return

    df_raw.drop_duplicates(subset=['id'], inplace=True)
    df_featured = prepare_apartment_data(df_raw)

    # ### ADDED: The new features are now included in the model's training data ###
    TARGET = 'log_price'
    # NUMERICAL_FEATURES = [
    #     'log_size_m2', 'bedrooms', 'bathrooms', 'latitude', 'longitude',
    #     'bed_bath_ratio', 'size_per_bedroom', 'avg_price_in_district', 'avg_size_in_district'
    # ]
    NUMERICAL_FEATURES = [
        'log_size_m2', 'bedrooms', 'bathrooms', 'latitude', 'longitude',
        # These features are great because they can be calculated from user input
        'bed_bath_ratio', 
        'size_per_bedroom'
        # DO NOT include 'avg_price_in_district' or 'avg_size_in_district'
    ]
    CATEGORICAL_FEATURES = ['province', 'district', 'city']
    FEATURES = NUMERICAL_FEATURES + CATEGORICAL_FEATURES

    X = df_featured[FEATURES]
    y = df_featured[TARGET]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

    preprocessor = ColumnTransformer(
        transformers=[
            ('num', Pipeline(steps=[('imputer', SimpleImputer(strategy='median')), ('scaler', StandardScaler())]), NUMERICAL_FEATURES),
            ('cat', Pipeline(steps=[('imputer', SimpleImputer(strategy='most_frequent')), ('onehot', OneHotEncoder(handle_unknown='ignore', sparse_output=False))]), CATEGORICAL_FEATURES)
        ])

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

    print("\n--- Step 4: Running GridSearchCV for XGBoost (with 5-Fold CV) ---")
    grid_search_xgb = GridSearchCV(pipeline_xgb, param_grid_xgb, cv=5, scoring='r2', n_jobs=-1, verbose=2)
    grid_search_xgb.fit(X_train, y_train)

    best_xgb_model = grid_search_xgb.best_estimator_

    y_pred_log = best_xgb_model.predict(X_test)
    y_test_actual = np.expm1(y_test)
    y_pred_actual = np.expm1(y_pred_log)
    test_r2 = r2_score(y_test, y_pred_log)
    test_mae = mean_absolute_error(y_test_actual, y_pred_actual)

    plot_regression_results(y_test_actual, y_pred_actual, output_dir)
    plot_feature_importance(best_xgb_model, output_dir)

    # model_path = os.path.join(output_dir, 'final_apartment_price_model_v2.joblib')
    # joblib.dump(best_xgb_model, model_path)
    model_path = os.path.join(output_dir, 'final_api_ready_model.joblib')
    joblib.dump(best_xgb_model, model_path)
    print(f"\n--- Final production model saved to '{model_path}' ---")

    print("\n" + "="*60)
    print("--- FINAL XGBOOST MODEL PERFORMANCE SUMMARY ---")
    print(f"   -> Best CV R²: {grid_search_xgb.best_score_:.4f}")
    print(f"   -> Test Set R²:  {test_r2:.4f}")
    print(f"   -> Test Set MAE: ${test_mae:,.2f}")
    print("="*60 + "\n--- WORKFLOW COMPLETE ---")

if __name__ == '__main__':
    main()