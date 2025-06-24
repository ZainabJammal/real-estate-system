# forecasting_lstm.py

import os
import json 
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential # load_model might be needed if you save/load pre-trained models
from tensorflow.keras.layers import LSTM, Dense # Consider Input layer for Keras 3+ if issues arise
from tensorflow.keras.optimizers import Adam
import joblib # For saving/loading the scaler


class LSTMPredictor:
    def __init__(self, look_back=12):
        self.look_back = look_back
        self.model = None
        self.scaler = MinMaxScaler(feature_range=(0, 1))
        self.is_trained = False
        print(f"DEBUG: LSTMPredictor initialized with look_back={look_back}")

    def _create_dataset(self, scaled_dataset): # Changed parameter name for clarity
        dataX, dataY = [], []
        if len(scaled_dataset) <= self.look_back:
            # This condition means not enough data to form even one X, Y pair.
            print(f"DEBUG: _create_dataset - Not enough data. Scaled dataset length: {len(scaled_dataset)}, look_back: {self.look_back}")
            return np.array(dataX), np.array(dataY) # Return empty arrays

        for i in range(len(scaled_dataset) - self.look_back):
            a = scaled_dataset[i:(i + self.look_back), 0]
            dataX.append(a)
            dataY.append(scaled_dataset[i + self.look_back, 0])
        
        if not dataX: # Should not happen if len(scaled_dataset) > self.look_back
            print("DEBUG: _create_dataset - dataX is empty after loop, this is unexpected.")
        return np.array(dataX), np.array(dataY)

    def train(self, df_history, value_column='value', epochs=50, batch_size=1):
        print(f"DEBUG: LSTMPredictor.train called. df_history length: {len(df_history)}, value_column: '{value_column}'")
        if value_column not in df_history.columns:
            raise ValueError(f"Value column '{value_column}' not found in df_history.")
        if df_history[value_column].isnull().any():
            print(f"WARNING: df_history['{value_column}'] contains NaN values before training. This might cause issues.")
            # df_history = df_history.dropna(subset=[value_column]) # Optionally drop NaNs here
            # if df_history.empty:
            #     raise ValueError("DataFrame is empty after dropping NaNs from value column.")

        dataset = df_history[value_column].values.astype('float32').reshape(-1, 1)
        
        # Check for NaNs or Infs after reshape and before scaling
        if np.isnan(dataset).any() or np.isinf(dataset).any():
            raise ValueError("Dataset contains NaN or Inf values before scaling. Check data preparation.")

        scaled_dataset = self.scaler.fit_transform(dataset)
        trainX, trainY = self._create_dataset(scaled_dataset)

        if len(trainX) == 0: # trainX could be empty if _create_dataset returns empty
            raise ValueError(f"Not enough data to create training sequences (trainX is empty). "
                             f"Need at least {self.look_back + 1} data points after processing, "
                             f"got {len(df_history)} in df_history which resulted in {len(scaled_dataset)} scaled points.")

        trainX = np.reshape(trainX, (trainX.shape[0], trainX.shape[1], 1))

        self.model = Sequential([
            LSTM(64, input_shape=(self.look_back, 1)), # For Keras 2. For Keras 3, consider using Input layer first
            Dense(32, activation='relu'),
            Dense(1)
        ])
        self.model.compile(loss='mean_squared_error', optimizer=Adam(learning_rate=0.001))

        print(f"DEBUG: Starting LSTM model.fit with trainX shape: {trainX.shape}, trainY shape: {trainY.shape}, epochs: {epochs}, batch_size: {batch_size}")
        self.model.fit(trainX, trainY, epochs=epochs, batch_size=batch_size, verbose=1) # verbose=1 for server logs
        self.is_trained = True
        print("DEBUG: LSTM training complete.")

    def predict(self, df_history, future_periods=60, value_column='value'):
        print(f"DEBUG: LSTMPredictor.predict called. df_history length: {len(df_history)}, future_periods: {future_periods}")
        if not self.is_trained or self.model is None:
            raise ValueError("Model has not been trained. Call train() first.")
        if value_column not in df_history.columns:
            raise ValueError(f"Value column '{value_column}' not found in df_history for prediction.")

        historical_values = df_history[value_column].values.astype('float32')
        if len(historical_values) < self.look_back:
            # This check is important. The input sequence must be of length look_back.
            raise ValueError(f"Not enough historical data for prediction input sequence. "
                             f"Need at least {self.look_back} data points, got {len(historical_values)}")

        # Take the LAST `look_back` points from historical data
        input_sequence = historical_values[-self.look_back:].reshape(-1, 1)

        # Check for NaNs or Infs in input_sequence before scaling
        if np.isnan(input_sequence).any() or np.isinf(input_sequence).any():
            raise ValueError("Input sequence for prediction contains NaN or Inf values. Check data preparation or historical data.")

        scaled_sequence = self.scaler.transform(input_sequence) # Use the FITTED scaler
        current_input = scaled_sequence.reshape((1, self.look_back, 1))

        future_predictions_scaled = []
        for i in range(future_periods):
            # print(f"DEBUG: Predicting period {i+1}/{future_periods}. current_input shape: {current_input.shape}")
            pred_scaled = self.model.predict(current_input, verbose=0)
            future_predictions_scaled.append(pred_scaled[0, 0])
            new_sequence_member = pred_scaled.reshape(1, 1, 1) # Reshape for appending
            # Append the new prediction and slide the window: new input is last (look_back-1) from old + new prediction
            current_input = np.append(current_input[:, 1:, :], new_sequence_member, axis=1)

        future_predictions_scaled = np.array(future_predictions_scaled).reshape(-1, 1)
        final_predictions = self.scaler.inverse_transform(future_predictions_scaled)
        print(f"DEBUG: LSTM prediction complete. Generated {len(final_predictions)} future values.")
        return final_predictions.flatten().tolist()



async def fetch_and_prepare_transaction_data(
    supabase_client, # Expecting an initialized Supabase client instance
    city_name=None,
    min_date_str=None # Currently not used in query, but kept for potential future use
    ):
    table_name = "transactions"
    date_column_db = "date"
    value_column_db = "transaction_value"
    city_column_db = "city"

    print(f"DEBUG: fetch_and_prepare_transaction_data called. City: '{city_name}', Min Date: '{min_date_str}'")

    query = supabase_client.from_(table_name) \
                           .select(f"{date_column_db}, {value_column_db}, {city_column_db}") \
                           .order(date_column_db, desc=False) # Fetch oldest first

   

    if city_name and city_name.strip().lower() != 'all' and city_name.strip() != "":
        print(f"DEBUG: Applying city filter in DB query: '{city_name}'")
        query = query.eq(city_column_db, city_name)
    else:
        print(f"DEBUG: No city filter applied or 'all' cities selected.")


    try:
        res = await query.execute()
    except Exception as e:
        print(f"ERROR: Supabase query execution failed: {e}")
        import traceback
        traceback.print_exc()
        return pd.DataFrame()


    if not res.data:
        print(f"DEBUG: No data returned from Supabase for table '{table_name}' with city filter '{city_name}'.")
        return pd.DataFrame()

    print(f"DEBUG: {len(res.data)} raw rows received from Supabase. Sample: {res.data[:2] if res.data else 'None'}")
    df_raw = pd.DataFrame(res.data)
    
    # Rename to standard names used internally
    df_raw.rename(columns={date_column_db: "date", value_column_db: "value_orig"}, inplace=True)

    # Date Parsing
    try:
        # Attempt precise format first
        df_raw['date'] = pd.to_datetime(df_raw['date'], format="%m/%Y")
        print(f"DEBUG: Successfully parsed 'date' column with format %m/%Y. Null count after parse: {df_raw['date'].isnull().sum()}")
    except ValueError as ve: # Catch specific error if format fails
        print(f"WARNING: Failed to parse 'date' with format %m/%Y (Error: {ve}). Attempting generic parsing with errors='coerce'.")
        df_raw['date'] = pd.to_datetime(df_raw['date'], errors='coerce')
        print(f"DEBUG: Parsed 'date' column with generic parsing. Null count after coerce: {df_raw['date'].isnull().sum()}")
    
    # Value Parsing
    if 'value_orig' not in df_raw.columns:
        print(f"ERROR: 'value_orig' (expected from '{value_column_db}') not found in DataFrame columns: {df_raw.columns}")
        return pd.DataFrame()
        
    df_raw['value_orig'] = pd.to_numeric(df_raw['value_orig'], errors='coerce')
    print(f"DEBUG: Parsed 'value_orig' column to numeric. Null count: {df_raw['value_orig'].isnull().sum()}")

    # Drop rows where essential columns became NaT/NaN after parsing
    df_raw.dropna(subset=['date', 'value_orig'], inplace=True)
    if df_raw.empty:
        print(f"DEBUG: DataFrame became empty after dropping NaNs/NaTs post-parsing.")
        return pd.DataFrame()

    df_raw = df_raw.sort_values('date')
    print(f"DEBUG: DataFrame sorted by date. Length: {len(df_raw)}. Date range: {df_raw['date'].min()} to {df_raw['date'].max()}")

    # Resample to Month End ('ME') frequency
    # Ensure 'date' is the index for resampling
    if not pd.api.types.is_datetime64_any_dtype(df_raw['date']):
        print(f"ERROR: 'date' column is not datetime type before resampling. Type: {df_raw['date'].dtype}")
        return pd.DataFrame()
        
    df_monthly = df_raw.set_index('date')['value_orig'].resample('ME').mean().reset_index()
    df_monthly.rename(columns={'value_orig': 'value'}, inplace=True) # Final column name for LSTM
    
    # Fill missing values after resampling (common for time series)
    df_monthly['value'] = df_monthly['value'].ffill().bfill()
    print(f"DEBUG: Resampled to monthly, filled NaNs. Final df_monthly length: {len(df_monthly)}. Null values in 'value': {df_monthly['value'].isnull().sum()}")
    
    if df_monthly.empty:
        print("DEBUG: df_monthly is empty after resampling and fillna.")
        return pd.DataFrame()
    if df_monthly['value'].isnull().any():
        print("WARNING: df_monthly 'value' column still contains NaNs after ffill/bfill. This might indicate all-NaN groups during resampling.")
        # Optionally, you could drop these or raise an error:
        # df_monthly.dropna(subset=['value'], inplace=True)
        # if df_monthly.empty:
        #     print("DEBUG: df_monthly became empty after dropping final NaNs in 'value'.")
        #     return pd.DataFrame()

    return df_monthly