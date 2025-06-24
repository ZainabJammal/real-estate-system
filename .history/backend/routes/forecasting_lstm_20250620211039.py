# # forecasting_lstm.py

# import os
# import json # Not currently used, but often handy
# import numpy as np
# import pandas as pd
# from sklearn.preprocessing import MinMaxScaler
# from tensorflow.keras.models import Sequential # load_model might be needed if you save/load pre-trained models
# from tensorflow.keras.layers import LSTM, Dense # Consider Input layer for Keras 3+ if issues arise
# from tensorflow.keras.optimizers import Adam
# import joblib # For saving/loading the scaler


# class LSTMPredictor:
#     def __init__(self, look_back=12):
#         # Initialize the look_back parameter
#         self.look_back = look_back
#         # Initialize the model parameter
#         self.model = None
#         # Initialize the scaler parameter
#         self.scaler = MinMaxScaler(feature_range=(0, 1))
#         # Initialize the is_trained parameter
#         self.is_trained = False
#         # Print a debug message
#         print(f"DEBUG: LSTMPredictor initialized with look_back={look_back}")

#     def _create_dataset(self, scaled_dataset): # Changed parameter name for clarity
#         dataX, dataY = [], []
#         if len(scaled_dataset) <= self.look_back:
#             # This condition means not enough data to form even one X, Y pair.
#             print(f"DEBUG: _create_dataset - Not enough data. Scaled dataset length: {len(scaled_dataset)}, look_back: {self.look_back}")
#             return np.array(dataX), np.array(dataY) # Return empty arrays

#         for i in range(len(scaled_dataset) - self.look_back):
#             a = scaled_dataset[i:(i + self.look_back), 0]
#             dataX.append(a)
#             dataY.append(scaled_dataset[i + self.look_back, 0])
        
#         if not dataX: # Should not happen if len(scaled_dataset) > self.look_back
#             print("DEBUG: _create_dataset - dataX is empty after loop, this is unexpected.")
#         return np.array(dataX), np.array(dataY)

#     def train(self, df_history, value_column='value', epochs=50, batch_size=1):
#         print(f"DEBUG: LSTMPredictor.train called. df_history length: {len(df_history)}, value_column: '{value_column}'")
#         if value_column not in df_history.columns:
#             raise ValueError(f"Value column '{value_column}' not found in df_history.")
#         if df_history[value_column].isnull().any():
#             print(f"WARNING: df_history['{value_column}'] contains NaN values before training. This might cause issues.")
#             # df_history = df_history.dropna(subset=[value_column]) # Optionally drop NaNs here
#             # if df_history.empty:
#             #     raise ValueError("DataFrame is empty after dropping NaNs from value column.")

#         dataset = df_history[value_column].values.astype('float32').reshape(-1, 1)
        
#         # Check for NaNs or Infs after reshape and before scaling
#         if np.isnan(dataset).any() or np.isinf(dataset).any():
#             raise ValueError("Dataset contains NaN or Inf values before scaling. Check data preparation.")

#         scaled_dataset = self.scaler.fit_transform(dataset)
#         trainX, trainY = self._create_dataset(scaled_dataset)

#         if len(trainX) == 0: # trainX could be empty if _create_dataset returns empty
#             raise ValueError(f"Not enough data to create training sequences (trainX is empty). "
#                              f"Need at least {self.look_back + 1} data points after processing, "
#                              f"got {len(df_history)} in df_history which resulted in {len(scaled_dataset)} scaled points.")

#         trainX = np.reshape(trainX, (trainX.shape[0], trainX.shape[1], 1))

#         self.model = Sequential([
#             LSTM(64, input_shape=(self.look_back, 1)), # For Keras 2. For Keras 3, consider using Input layer first
#             Dense(32, activation='relu'),
#             Dense(1)
#         ])
#         self.model.compile(loss='mean_squared_error', optimizer=Adam(learning_rate=0.001))

#         print(f"DEBUG: Starting LSTM model.fit with trainX shape: {trainX.shape}, trainY shape: {trainY.shape}, epochs: {epochs}, batch_size: {batch_size}")
#         self.model.fit(trainX, trainY, epochs=epochs, batch_size=batch_size, verbose=1) # verbose=1 for server logs
#         self.is_trained = True
#         print("DEBUG: LSTM training complete.")

#     def predict(self, df_history, future_periods=60, value_column='value'):
#         print(f"DEBUG: LSTMPredictor.predict called. df_history length: {len(df_history)}, future_periods: {future_periods}")
#         if not self.is_trained or self.model is None:
#             raise ValueError("Model has not been trained. Call train() first.")
#         if value_column not in df_history.columns:
#             raise ValueError(f"Value column '{value_column}' not found in df_history for prediction.")

#         historical_values = df_history[value_column].values.astype('float32')
#         if len(historical_values) < self.look_back:
#             # This check is important. The input sequence must be of length look_back.
#             raise ValueError(f"Not enough historical data for prediction input sequence. "
#                              f"Need at least {self.look_back} data points, got {len(historical_values)}")

#         # Take the LAST `look_back` points from historical data
#         input_sequence = historical_values[-self.look_back:].reshape(-1, 1)

#         # Check for NaNs or Infs in input_sequence before scaling
#         if np.isnan(input_sequence).any() or np.isinf(input_sequence).any():
#             raise ValueError("Input sequence for prediction contains NaN or Inf values. Check data preparation or historical data.")

#         scaled_sequence = self.scaler.transform(input_sequence) # Use the FITTED scaler
#         current_input = scaled_sequence.reshape((1, self.look_back, 1))

#         future_predictions_scaled = []
#         for i in range(future_periods):
#             # print(f"DEBUG: Predicting period {i+1}/{future_periods}. current_input shape: {current_input.shape}")
#             pred_scaled = self.model.predict(current_input, verbose=0)
#             future_predictions_scaled.append(pred_scaled[0, 0])
#             new_sequence_member = pred_scaled.reshape(1, 1, 1) # Reshape for appending
#             # Append the new prediction and slide the window: new input is last (look_back-1) from old + new prediction
#             current_input = np.append(current_input[:, 1:, :], new_sequence_member, axis=1)

#         future_predictions_scaled = np.array(future_predictions_scaled).reshape(-1, 1)
#         final_predictions = self.scaler.inverse_transform(future_predictions_scaled)
#         print(f"DEBUG: LSTM prediction complete. Generated {len(final_predictions)} future values.")
#         return final_predictions.flatten().tolist()



# async def fetch_and_prepare_transaction_data(
#     supabase_client, # Expecting an initialized Supabase client instance
#     city_name=None,
#     min_date_str=None # Currently not used in query, but kept for potential future use
#     ):
#     table_name = "transactions"
#     date_column_db = "date"
#     value_column_db = "transaction_value"
#     city_column_db = "city"

#     print(f"DEBUG: fetch_and_prepare_transaction_data called. City: '{city_name}', Min Date: '{min_date_str}'")

#     query = supabase_client.from_(table_name) \
#                            .select(f"{date_column_db}, {value_column_db}, {city_column_db}") \
#                            .order(date_column_db, desc=False) # Fetch oldest first

   

#     if city_name and city_name.strip().lower() != 'all' and city_name.strip() != "":
#         print(f"DEBUG: Applying city filter in DB query: '{city_name}'")
#         query = query.eq(city_column_db, city_name)
#     else:
#         print(f"DEBUG: No city filter applied or 'all' cities selected.")


#     try:
#         res = await query.execute()
#     except Exception as e:
#         print(f"ERROR: Supabase query execution failed: {e}")
#         import traceback
#         traceback.print_exc()
#         return pd.DataFrame()


#     if not res.data:
#         print(f"DEBUG: No data returned from Supabase for table '{table_name}' with city filter '{city_name}'.")
#         return pd.DataFrame()

#     print(f"DEBUG: {len(res.data)} raw rows received from Supabase. Sample: {res.data[:2] if res.data else 'None'}")
#     df_raw = pd.DataFrame(res.data)
    
#     # Rename to standard names used internally
#     df_raw.rename(columns={date_column_db: "date", value_column_db: "value_orig"}, inplace=True)

#     # Date Parsing
#     try:
#         # Attempt precise format first
#         df_raw['date'] = pd.to_datetime(df_raw['date'], format="%m/%Y")
#         print(f"DEBUG: Successfully parsed 'date' column with format %m/%Y. Null count after parse: {df_raw['date'].isnull().sum()}")
#     except ValueError as ve: # Catch specific error if format fails
#         print(f"WARNING: Failed to parse 'date' with format %m/%Y (Error: {ve}). Attempting generic parsing with errors='coerce'.")
#         df_raw['date'] = pd.to_datetime(df_raw['date'], errors='coerce')
#         print(f"DEBUG: Parsed 'date' column with generic parsing. Null count after coerce: {df_raw['date'].isnull().sum()}")
    
#     # Value Parsing
#     if 'value_orig' not in df_raw.columns:
#         print(f"ERROR: 'value_orig' (expected from '{value_column_db}') not found in DataFrame columns: {df_raw.columns}")
#         return pd.DataFrame()
        
#     df_raw['value_orig'] = pd.to_numeric(df_raw['value_orig'], errors='coerce')
#     print(f"DEBUG: Parsed 'value_orig' column to numeric. Null count: {df_raw['value_orig'].isnull().sum()}")

#     # Drop rows where essential columns became NaT/NaN after parsing
#     df_raw.dropna(subset=['date', 'value_orig'], inplace=True)
#     if df_raw.empty:
#         print(f"DEBUG: DataFrame became empty after dropping NaNs/NaTs post-parsing.")
#         return pd.DataFrame()

#     df_raw = df_raw.sort_values('date')
#     print(f"DEBUG: DataFrame sorted by date. Length: {len(df_raw)}. Date range: {df_raw['date'].min()} to {df_raw['date'].max()}")

#     # Resample to Month End ('ME') frequency
#     # Ensure 'date' is the index for resampling
#     if not pd.api.types.is_datetime64_any_dtype(df_raw['date']):
#         print(f"ERROR: 'date' column is not datetime type before resampling. Type: {df_raw['date'].dtype}")
#         return pd.DataFrame()
        
#     df_monthly = df_raw.set_index('date')['value_orig'].resample('ME').mean().reset_index()
#     df_monthly.rename(columns={'value_orig': 'value'}, inplace=True) # Final column name for LSTM
    
#     # Fill missing values after resampling (common for time series)
#     df_monthly['value'] = df_monthly['value'].ffill().bfill()
#     print(f"DEBUG: Resampled to monthly, filled NaNs. Final df_monthly length: {len(df_monthly)}. Null values in 'value': {df_monthly['value'].isnull().sum()}")
    
#     if df_monthly.empty:
#         print("DEBUG: df_monthly is empty after resampling and fillna.")
#         return pd.DataFrame()
#     if df_monthly['value'].isnull().any():
#         print("WARNING: df_monthly 'value' column still contains NaNs after ffill/bfill. This might indicate all-NaN groups during resampling.")
#         # Optionally, you could drop these or raise an error:
#         # df_monthly.dropna(subset=['value'], inplace=True)
#         # if df_monthly.empty:
#         #     print("DEBUG: df_monthly became empty after dropping final NaNs in 'value'.")
#         #     return pd.DataFrame()

#     return df_monthly



# forecasting_lstm.py
# forecasting_lstm.py

import os
import pandas as pd
import numpy as np
import joblib
import json
import matplotlib.pyplot as plt

from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Input, Dropout # Import Dropout
from tensorflow.keras.optimizers import Adam

# 1. CONFIGURATION AND PATH SETUP
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
CSV_FILE_PATH = os.path.join(SCRIPT_DIR, 'transactions.csv')
MODEL_FILE_PATH = os.path.join(SCRIPT_DIR, 'forecast_lstm_model.keras') # .keras is preferred format
FEATURE_SCALER_PATH = os.path.join(SCRIPT_DIR, 'lstm_feature_scaler.joblib')
TARGET_SCALER_PATH = os.path.join(SCRIPT_DIR, 'lstm_target_scaler.joblib')
MODEL_COLS_PATH = os.path.join(SCRIPT_DIR, 'lstm_model_columns.json')
TARGET_COL = 'transaction_value'
LOOK_BACK = 12 # How many previous months to use for predicting the next month

# 2. HELPER FUNCTIONS
def load_and_preprocess_data(filepath):
    """Loads and performs initial date parsing and indexing."""
    print("-> Loading and preprocessing data...")
    df = pd.read_csv(filepath)
    # This date parsing is specific to the 'YYYY-Month' format in transactions.csv
    parts = df['date'].str.split('-', expand=True)
    df['date_str'] = '01-' + parts[1] + '-' + parts[0]
    df['date'] = pd.to_datetime(df['date_str'], format='%d-%b-%y')
    df = df.set_index('date').drop(columns=['id', 'date_str'])
    df.sort_index(inplace=True)
    return df

def create_features(df):
    """Creates time-series features from the datetime index and lags."""
    print("-> Creating features (lags, rolling stats, time components)...")
    df['month'] = df.index.month
    df['year'] = df.index.year
    df['quarter'] = df.index.quarter
    # Lags and rolling features must be grouped by city to prevent data leakage
    df['lag_1'] = df.groupby('city')[TARGET_COL].shift(1)
    df['lag_3'] = df.groupby('city')[TARGET_COL].shift(3)
    df['lag_12'] = df.groupby('city')[TARGET_COL].shift(12)
    df['rolling_mean_3'] = df.groupby('city')[TARGET_COL].shift(1).rolling(window=3).mean()
    df['rolling_std_3'] = df.groupby('city')[TARGET_COL].shift(1).rolling(window=3).std()
    return df

def create_lstm_sequences(X_data, y_data, look_back):
    """Converts 2D feature/target data into 3D sequences for LSTM."""
    X_seq, y_seq = [], []
    # Use .iloc for pandas Series/DataFrame, standard indexing for numpy array
    is_pandas = hasattr(y_data, 'iloc')
    
    # The loop must stop look_back steps before the end of the data
    for i in range(len(X_data) - look_back):
        X_seq.append(X_data[i:(i + look_back)])
        if is_pandas:
            # Explicitly use positional indexing
            y_seq.append(y_data.iloc[i + look_back])
        else:
            # Standard indexing for NumPy arrays
            y_seq.append(y_data[i + look_back])
            
    return np.array(X_seq), np.array(y_seq)


# 3. EVALUATION AND PLOTTING FUNCTIONS
def evaluate_model(true_values, predicted_values):
    """Calculates and prints MAE and RMSE."""
    mae = mean_absolute_error(true_values, predicted_values)
    rmse = np.sqrt(mean_squared_error(true_values, predicted_values))
    print("\n--- LSTM Model Performance on Held-Out Validation Set (2016) ---")
    print(f"  Root Mean Squared Error (RMSE): {rmse:,.2f}")
    print(f"  Mean Absolute Error (MAE):      {mae:,.2f}")
    print("----------------------------------------------------------------")

def plot_validation_results(train_df, validation_df, predictions):
    """Plots training data, actual validation data, and predicted data."""
    plt.figure(figsize=(15, 7))
    plt.plot(train_df.index, train_df[TARGET_COL], label='Training Data (2011-2015)', color='blue')
    plt.plot(validation_df.index, validation_df[TARGET_COL], label='Actual Values (2016)', color='green', marker='o', linestyle='-')
    plt.plot(validation_df.index, predictions, label='Predicted Values (2016)', color='red', linestyle='--')
    plt.title('LSTM Model Validation: Actual vs. Predicted for 2016', fontsize=16)
    plt.xlabel('Date')
    plt.ylabel('Transaction Value')
    plt.legend()
    plt.grid(True)
    print("\n-> Displaying validation plot. Close the plot window to continue the script.")
    plt.show()

# 4. MAIN WORKFLOW
if __name__ == "__main__":
    print("--- Starting LSTM Forecasting Workflow: Train, Validate, Re-train ---")

    # --- Step 1: Load, Prepare, and Split Data ---
    all_data = load_and_preprocess_data(CSV_FILE_PATH)
    data_with_features = create_features(all_data)

    split_date = '2016-01-01'
    train_set_raw = data_with_features[data_with_features.index < split_date]
    validation_set_raw = data_with_features[data_with_features.index >= split_date]

    # One-hot encode the 'city' column and handle potential NaNs from feature creation
    train_final = pd.get_dummies(train_set_raw, columns=['city'], prefix='city').dropna()
    validation_final_raw = pd.get_dummies(validation_set_raw, columns=['city'], prefix='city').dropna()
    
    FEATURES = [col for col in train_final.columns if col != TARGET_COL]
    
    # Align columns: ensure validation set has same columns as training set
    validation_final = validation_final_raw.reindex(columns=FEATURES, fill_value=0)
    
    print(f"-> Data prepared. {len(FEATURES)} features created. Training set shape: {train_final.shape}")
    
    # --- Step 2: Scale the Data ---
    feature_scaler = MinMaxScaler(feature_range=(0, 1))
    target_scaler = MinMaxScaler(feature_range=(0, 1))

    # Fit scalers ONLY on the training data to avoid data leakage
    X_train_scaled = feature_scaler.fit_transform(train_final[FEATURES])
    y_train_scaled = target_scaler.fit_transform(train_final[[TARGET_COL]])
    
    # Transform the validation data using the FITTED scalers
    X_val_scaled = feature_scaler.transform(validation_final)
    y_val_unscaled = validation_final_raw.loc[validation_final.index][TARGET_COL] # Get original y-values for evaluation

    # --- Step 3: Create LSTM Sequences ---
    print(f"-> Creating LSTM sequences with look_back={LOOK_BACK}...")
    X_train_seq, y_train_seq = create_lstm_sequences(X_train_scaled, y_train_scaled, LOOK_BACK)
    X_val_seq, y_val_seq = create_lstm_sequences(X_val_scaled, y_val_unscaled.values, LOOK_BACK) 

    if X_train_seq.shape[0] == 0:
        raise ValueError("Training sequences could not be created. Check data length and look_back period.")

    # --- Step 4: Build, Train, and Validate the LSTM Model ---
    print(f"-> Building LSTM model. Input shape: {X_train_seq.shape[1:]}")
    # *** MODIFICATION: Added Dropout layers to combat overfitting ***
    model = Sequential([
        Input(shape=(X_train_seq.shape[1], X_train_seq.shape[2])), # (look_back, num_features)
        LSTM(64, return_sequences=True),
        Dropout(0.2), # REGULARIZATION: Added Dropout
        LSTM(32),
        Dropout(0.2), # REGULARIZATION: Added Dropout
        Dense(16, activation='relu'),
        Dense(1)
    ])
    model.compile(optimizer=Adam(learning_rate=0.001), loss='mean_squared_error')
    model.summary()
    
    print(f"-> Training model on {X_train_seq.shape[0]} sequences...")
    # *** MODIFICATION: Increased epochs slightly to give regularized model a chance to learn ***
    model.fit(X_train_seq, y_train_seq, epochs=75, batch_size=32, validation_split=0.1, verbose=1)
    
    print("-> Evaluating the model on unseen 2016 data...")
    predictions_scaled = model.predict(X_val_seq)
    predictions_unscaled = target_scaler.inverse_transform(predictions_scaled)

    evaluate_model(y_val_seq, predictions_unscaled.flatten())

    # --- Step 5: Plot Validation Results for One City ---
    city_to_plot = 'Beirut'
    print(f"\n-> Generating validation plot for a single city: {city_to_plot}")

    all_city_data_raw = data_with_features[data_with_features['city'] == city_to_plot].copy()
    city_full_history_df = pd.get_dummies(all_city_data_raw, columns=['city'], prefix='city')
    city_full_history_df.dropna(inplace=True)
    city_target_unscaled = city_full_history_df[TARGET_COL]
    city_features_aligned = city_full_history_df.reindex(columns=FEATURES, fill_value=0)
    city_features_scaled = feature_scaler.transform(city_features_aligned)
    city_X_seq, city_y_true = create_lstm_sequences(city_features_scaled, city_target_unscaled, LOOK_BACK)
    validation_city_df = validation_set_raw[validation_set_raw['city'] == city_to_plot].dropna()
    num_validation_points = len(validation_city_df)
    
    if len(city_X_seq) < num_validation_points:
        raise ValueError(f"Not enough historical data for {city_to_plot} to generate {num_validation_points} validation predictions.")

    sequences_to_predict = city_X_seq[-num_validation_points:]
    city_plot_predictions_scaled = model.predict(sequences_to_predict)
    city_plot_predictions_unscaled = target_scaler.inverse_transform(city_plot_predictions_scaled).flatten()
    train_city_df = train_set_raw[train_set_raw['city'] == city_to_plot]
    plot_validation_results(train_city_df, validation_city_df, city_plot_predictions_unscaled)


    # --- Step 6: Re-train Final Production Model on ALL Data ---
    print("-> Validation and plotting complete. Re-training final model on ALL data (2011-2016)...")
    all_data_final_raw = pd.get_dummies(data_with_features, columns=['city'], prefix='city').dropna()
    all_data_final = all_data_final_raw.reindex(columns=FEATURES, fill_value=0)

    X_all_scaled = feature_scaler.transform(all_data_final)
    y_all_scaled = target_scaler.transform(all_data_final_raw[[TARGET_COL]])
    X_all_seq, y_all_seq = create_lstm_sequences(X_all_scaled, y_all_scaled, LOOK_BACK)
    
    print(f"-> Building and training final production model on {X_all_seq.shape[0]} total sequences...")
    # *** MODIFICATION: Added Dropout layers to the production model as well for consistency ***
    production_model = Sequential([
        Input(shape=(X_all_seq.shape[1], X_all_seq.shape[2])),
        LSTM(64, return_sequences=True),
        Dropout(0.2), # REGULARIZATION: Added Dropout
        LSTM(32),
        Dropout(0.2), # REGULARIZATION: Added Dropout
        Dense(16, activation='relu'),
        Dense(1)
    ])
    production_model.compile(optimizer=Adam(learning_rate=0.001), loss='mean_squared_error')
    # *** MODIFICATION: Increased epochs for final training run ***
    production_model.fit(X_all_seq, y_all_seq, epochs=75, batch_size=32, verbose=1)
    print("-> Final production model trained.")

    # --- Step 7: Save the Production-Ready Model and Scalers ---
    print(f"-> Saving final LSTM model and supporting files...")
    production_model.save(MODEL_FILE_PATH)
    joblib.dump(feature_scaler, FEATURE_SCALER_PATH)
    joblib.dump(target_scaler, TARGET_SCALER_PATH)
    pd.Series(FEATURES).to_json(MODEL_COLS_PATH, indent=4)
    
    print("\n--- LSTM Workflow Complete ---")
  


  