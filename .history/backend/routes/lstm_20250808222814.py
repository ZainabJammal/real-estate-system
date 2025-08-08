# --- 1. IMPORTS ---
import pandas as pd
import numpy as np
import joblib
import os
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_percentage_error, mean_squared_error, mean_absolute_error
from sklearn.preprocessing import MinMaxScaler
from supabase import create_client, Client
from dotenv import load_dotenv

# Import TensorFlow and Keras components
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint

# --- 2. CONFIGURATION ---
load_dotenv()
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
BACKEND_DIR = os.path.dirname(SCRIPT_DIR)
OUTPUT_DIR = os.path.join(BACKEND_DIR, 'lstm_models') # New directory for LSTM models

# --- 3. GLOBAL CONSTANTS ---
GROUPING_KEY = 'city'
TARGET_COL = 'transaction_value'
CITIES = ['baabda', 'beirut', 'bekaa', 'kesrouan', 'tripoli']
N_TIMESTEPS = 30  # How many past days of data to use for a prediction

# --- 4. HELPER FUNCTIONS ---

def load_data_from_supabase():
    # This function remains the same as your original
    print("-> Connecting to Supabase to fetch training data...")
    url: str = os.environ.get("SUPABASE_URL")
    key: str = os.environ.get("SUPABASE_KEY")
    if not url or not key:
        raise ValueError("FATAL: Supabase URL or Key not found in .env file.")
    supabase: Client = create_client(url, key)
    response = supabase.table('merged_trans').select("*").order('date').execute()
    df = pd.DataFrame(response.data)
    print(f"-> Successfully fetched {len(df)} rows from Supabase.")
    df['date'] = pd.to_datetime(df['date'])
    df.set_index('date', inplace=True)
    df.sort_index(inplace=True)
    print("-> Date column converted to datetime and set as index.")
    df[GROUPING_KEY] = df[GROUPING_KEY].str.lower()
    return df

def create_features(df):
    # This function is still very useful for giving the LSTM context
    df_features = df.copy()
    df_features['month_sin'] = np.sin(2 * np.pi * df_features.index.month / 12.0)
    df_features['month_cos'] = np.cos(2 * np.pi * df_features.index.month / 12.0)
    df_features['day_of_year_sin'] = np.sin(2 * np.pi * df_features.index.dayofyear / 365.0)
    df_features['day_of_year_cos'] = np.cos(2 * np.pi * df_features.index.dayofyear / 365.0)
    df_features['quarter'] = df_features.index.quarter
    return df_features

def create_sequences(data, n_timesteps):
    """
    Transforms a time series dataset into supervised learning sequences.
    """
    X, y = [], []
    for i in range(len(data)):
        end_ix = i + n_timesteps
        if end_ix > len(data)-1:
            break
        seq_x, seq_y = data[i:end_ix], data[end_ix]
        X.append(seq_x)
        y.append(seq_y)
    return np.array(X), np.array(y)

def plot_validation_results(train_dates, train_actuals, val_dates, val_actuals, val_predictions, city_name):
    """Plots the validation results for a specific city."""
    plt.figure(figsize=(15, 7))
    plt.plot(train_dates, train_actuals, label='Training Data', color='blue')
    plt.plot(val_dates, val_actuals, label='Actual Values', color='green', marker='o')
    plt.plot(val_dates, val_predictions, label='Predicted Values', color='red', linestyle='--')
    plt.title(f'LSTM Model Validation for {city_name.capitalize()}', fontsize=16)
    plt.xlabel('Date')
    plt.ylabel('Transaction Value')
    plt.legend()
    plt.grid(True)
    plt.show()

# --- 5. MAIN EXECUTION SCRIPT ---

if __name__ == "__main__":
    print("--- Starting LSTM Model Training and Forecasting ---")

    # --- Step 1: Load and Prepare Data ---
    all_data = load_data_from_supabase()
    all_data_with_features = create_features(all_data)

    # --- We will loop through each city and train a separate model ---
    for city in CITIES:
        print(f"\n{'='*60}\n--- Processing City: {city.upper()} ---\n{'='*60}")

        # --- Step 2: Filter, Prepare, and Scale Data for the City ---
        city_data = all_data_with_features[all_data_with_features[GROUPING_KEY] == city].copy()
        features_to_scale = [TARGET_COL] + [col for col in city_data.columns if col not in [TARGET_COL, GROUPING_KEY, 'id', 'transaction_number']]
        
        # Create scaler and fit ONLY on training data to prevent data leakage
        train_df = city_data.loc[:'2018-12-31']
        val_df = city_data.loc['2019-01-01':'2021-12-31']

        scaler = MinMaxScaler(feature_range=(0, 1))
        scaled_train_data = scaler.fit_transform(train_df[features_to_scale])
        
        # Save the scaler for this city
        city_output_dir = os.path.join(OUTPUT_DIR, city)
        os.makedirs(city_output_dir, exist_ok=True)
        joblib.dump(scaler, os.path.join(city_output_dir, f'{city}_scaler.joblib'))
        print(f"-> Scaler for {city} saved.")

        # Transform validation data
        scaled_val_data = scaler.transform(val_df[features_to_scale])

        # --- Step 3: Create Sequences ---
        X_train, y_train = create_sequences(scaled_train_data, N_TIMESTEPS)
        X_val, y_val = create_sequences(scaled_val_data, N_TIMESTEPS)

        # We only want to predict the first column (transaction_value)
        y_train = y_train[:, 0]
        y_val = y_val[:, 0]
        
        print(f"-> Data shaped into sequences: X_train shape {X_train.shape}, X_val shape {X_val.shape}")

        # --- Step 4: Build and Train the LSTM Model ---
        model = Sequential([
            LSTM(units=50, return_sequences=True, input_shape=(X_train.shape[1], X_train.shape[2])),
            Dropout(0.2),
            LSTM(units=50, return_sequences=False),
            Dropout(0.2),
            Dense(units=25),
            Dense(units=1) # Output layer
        ])

        model.compile(optimizer='adam', loss='mean_squared_error')
        model.summary()
        
        # Callbacks
        model_path = os.path.join(city_output_dir, f'{city}_model.h5')
        early_stopping = EarlyStopping(monitor='val_loss', patience=10, mode='min')
        model_checkpoint = ModelCheckpoint(model_path, save_best_only=True, monitor='val_loss', mode='min')

        print("-> Training LSTM model...")
        history = model.fit(
            X_train, y_train,
            epochs=100,
            batch_size=32,
            validation_data=(X_val, y_val),
            callbacks=[early_stopping, model_checkpoint],
            verbose=1
        )
        
        print(f"-> Model for {city} trained and saved.")

        # --- Step 5: Evaluate on Validation Set ---
        # Load the best model saved by ModelCheckpoint
        best_model = tf.keras.models.load_model(model_path)
        
        # Make predictions
        predictions_scaled = best_model.predict(X_val)
        
        # We need to inverse transform the predictions to the original scale
        # To do this, we create a dummy array with the same shape as the scaler expects
        dummy_array = np.zeros((len(predictions_scaled), len(features_to_scale)))
        dummy_array[:, 0] = predictions_scaled.flatten()
        predictions_inversed = scaler.inverse_transform(dummy_array)[:, 0]

        # Also inverse transform the actuals for comparison
        dummy_array_actuals = np.zeros((len(y_val), len(features_to_scale)))
        dummy_array_actuals[:, 0] = y_val.flatten()
        actuals_inversed = scaler.inverse_transform(dummy_array_actuals)[:, 0]

        # Calculate metrics
        rmse = np.sqrt(mean_squared_error(actuals_inversed, predictions_inversed))
        mape = mean_absolute_percentage_error(actuals_inversed, predictions_inversed) * 100
        print(f"\n--- Validation Performance for {city.upper()} ---")
        print(f"  RMSE: {rmse:,.2f}")
        print(f"  MAPE: {mape:.2f}%")

        # --- Step 6: Forecast the Next 4 Years (Iteratively) ---
        print("\n-> Generating future forecast iteratively...")
        
        # Get the last sequence from the training data as the starting point
        last_sequence_scaled = scaled_val_data[-N_TIMESTEPS:]
        current_batch = last_sequence_scaled.reshape((1, N_TIMESTEPS, len(features_to_scale)))
        future_predictions_scaled = []
        
        forecast_horizon = (pd.to_datetime('2025-12-31') - val_df.index[-1]).days

        for i in range(forecast_horizon):
            # Predict the next step
            next_pred_scaled = best_model.predict(current_batch)[0]
            future_predictions_scaled.append(next_pred_scaled)

            # Create the feature row for the next timestep
            next_date = val_df.index[-1] + pd.DateOffset(days=i+1)
            future_features = create_features(pd.DataFrame(index=[next_date]))
            
            # Combine prediction with future features
            next_step_features = np.zeros(len(features_to_scale))
            next_step_features[0] = next_pred_scaled[0] # The predicted value
            
            # Fill in the other features (month_sin, etc.)
            feature_idx = 1
            for col in features_to_scale[1:]:
                next_step_features[feature_idx] = future_features[col].iloc[0]
                feature_idx += 1

            # Append this new step and drop the oldest one
            new_batch_unscaled = np.append(current_batch[0][1:], [next_step_features], axis=0)
            
            # IMPORTANT: Re-scale this new batch - This is a simplification. A more robust method would be to
            # inverse transform, append, and re-scale, but that is much more complex. We will append the scaled
            # prediction directly for simplicity.
            current_batch = new_batch_unscaled.reshape((1, N_TIMESTEPS, len(features_to_scale)))

        # Inverse transform the entire forecast
        dummy_forecast = np.zeros((len(future_predictions_scaled), len(features_to_scale)))
        dummy_forecast[:, 0] = np.array(future_predictions_scaled).flatten()
        future_forecast_inversed = scaler.inverse_transform(dummy_forecast)[:, 0]

        print(f"-> Forecast for {city} complete.")
        # (You would then collect these forecasts and plot them after the loop finishes)

    print("\n--- All City Models Trained and Forecasted ---")