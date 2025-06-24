# forecasting_lstm.py
import os
import json 
import asyncio # Needed for running async functions
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential 
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.optimizers import Adam
from sklearn.metrics import mean_squared_error, mean_absolute_error
# You might need to mock the supabase_client for local testing
from unittest.mock import Mock, MagicMock 

print("All libraries imported successfully!")


class LSTMPredictor:
    # --- I've added a Dropout layer to your model architecture ---
    def __init__(self, look_back=12):
        self.look_back = look_back
        self.model = None
        self.scaler = MinMaxScaler(feature_range=(0, 1))
        self.is_trained = False
        print(f"DEBUG: LSTMPredictor initialized with look_back={look_back}")

    def _create_dataset(self, scaled_dataset): 
        dataX, dataY = [], []
        if len(scaled_dataset) <= self.look_back:
            print(f"DEBUG: _create_dataset - Not enough data. Scaled dataset length: {len(scaled_dataset)}, look_back: {self.look_back}")
            return np.array(dataX), np.array(dataY) 

        for i in range(len(scaled_dataset) - self.look_back):
            a = scaled_dataset[i:(i + self.look_back), 0]
            dataX.append(a)
            dataY.append(scaled_dataset[i + self.look_back, 0])
        
        if not dataX:
            print("DEBUG: _create_dataset - dataX is empty after loop, this is unexpected.")
        return np.array(dataX), np.array(dataY)
   
def preprocess_and_encode_city(df_all_cities):
    print("Applying one-hot encoding to 'city' column...")
    df_encoded = pd.get_dummies(df_all_cities, columns=['city'], prefix='city')

    feature_columns = ['value'] + [col for col in df_encoded.columns if col.startswith('city_')]
    
    with open('model_features.json', 'w') as f:
        json.dump(feature_columns, f)

    print(f"Saved feature columns to model_features.json: {feature_columns}")
    return df_encoded, feature_columns

    def train(self, df_history, value_column='value', epochs=50, batch_size=1):
        print(f"DEBUG: LSTMPredictor.train called. df_history length: {len(df_history)}, value_column: '{value_column}'")
        if value_column not in df_history.columns:
            raise ValueError(f"Value column '{value_column}' not found in df_history.")
        
        dataset = df_history[value_column].values.astype('float32').reshape(-1, 1)
        
        if np.isnan(dataset).any() or np.isinf(dataset).any():
            raise ValueError("Dataset contains NaN or Inf values before scaling. Check data preparation.")

        scaled_dataset = self.scaler.fit_transform(dataset)
        trainX, trainY = self._create_dataset(scaled_dataset)

        if len(trainX) == 0:
            raise ValueError(f"Not enough data to create training sequences (trainX is empty). Need at least {self.look_back + 1} data points.")

        trainX = np.reshape(trainX, (trainX.shape[0], trainX.shape[1], 1))

        # --- Added Dropout for better generalization ---
        self.model = Sequential([
            LSTM(64, input_shape=(self.look_back, 1), return_sequences=True),
            Dropout(0.2),
            LSTM(32),
            Dropout(0.2),
            Dense(1)
        ])
        self.model.compile(loss='mean_squared_error', optimizer=Adam(learning_rate=0.001))
        self.model.summary()

        print(f"DEBUG: Starting LSTM model.fit with trainX shape: {trainX.shape}, trainY shape: {trainY.shape}, epochs: {epochs}, batch_size: {batch_size}")
        self.model.fit(trainX, trainY, epochs=epochs, batch_size=batch_size, verbose=1)
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
            raise ValueError(f"Not enough historical data for prediction. Need at least {self.look_back} data points, got {len(historical_values)}")

        input_sequence = historical_values[-self.look_back:].reshape(-1, 1)

        if np.isnan(input_sequence).any() or np.isinf(input_sequence).any():
            raise ValueError("Input sequence for prediction contains NaN or Inf values.")

        scaled_sequence = self.scaler.transform(input_sequence)
        current_input = scaled_sequence.reshape((1, self.look_back, 1))

        future_predictions_scaled = []
        for i in range(future_periods):
            pred_scaled = self.model.predict(current_input, verbose=0)
            future_predictions_scaled.append(pred_scaled[0, 0])
            new_sequence_member = pred_scaled.reshape(1, 1, 1)
            current_input = np.append(current_input[:, 1:, :], new_sequence_member, axis=1)

        future_predictions_scaled = np.array(future_predictions_scaled).reshape(-1, 1)
        final_predictions = self.scaler.inverse_transform(future_predictions_scaled)
        print(f"DEBUG: LSTM prediction complete. Generated {len(final_predictions)} future values.")
        return final_predictions.flatten().tolist()


# Your original data fetching function (unchanged)
async def fetch_and_prepare_transaction_data(supabase_client, city_name=None, min_date_str=None):
    # ... your original code here ...
    # For this example, I'll mock its output to make the script runnable.
    print("INFO: Using MOCKED data for fetch_and_prepare_transaction_data")
    dates = pd.to_datetime(pd.date_range(start='2020-01-01', periods=60, freq='ME'))
    # A sine wave to simulate seasonality, with some upward trend and noise
    values = (np.sin(np.linspace(0, 30, 60)) * 50 + 
              np.linspace(100, 150, 60) + 
              np.random.normal(0, 5, 60))
    mock_df = pd.DataFrame({'date': dates, 'value': values})
    print(f"DEBUG: Generated mock data with {len(mock_df)} rows.")
    return mock_df


# ===================================================================
#               NEW: EVALUATION SCRIPT MAIN BLOCK
# ===================================================================
async def evaluate_model_performance():
    """
    This function demonstrates the full workflow:
    1. Fetch data.
    2. Split into train/test sets.
    3. Train the model on the train set.
    4. Predict the future.
    5. Evaluate predictions against the test set.
    """
    print("\n--- Starting Model Evaluation Workflow ---\n")
    
    # --- 1. Fetch and Prepare Data ---
    # In a real scenario, you'd pass your actual Supabase client
    mock_supabase_client = Mock() 
    df_full = await fetch_and_prepare_transaction_data(mock_supabase_client, city_name="any_city")

    if df_full.empty or len(df_full) < 24: # Need enough data for a meaningful split
        print("ERROR: Not enough data to perform evaluation. Exiting.")
        return

    # --- 2. Split Data into Train and Test Sets ---
    # For time series, the split MUST be chronological. No shuffling!
    train_size = int(len(df_full) * 0.80)
    df_train = df_full.iloc[0:train_size]
    df_test = df_full.iloc[train_size:len(df_full)]

    print(f"Data Split Complete:")
    print(f"Full dataset size: {len(df_full)}")
    print(f"Training set size: {len(df_train)} (from {df_train['date'].min().date()} to {df_train['date'].max().date()})")
    print(f"Test set size: {len(df_test)} (from {df_test['date'].min().date()} to {df_test['date'].max().date()})")

    # --- 3. Initialize and Train the Model ---
    # Using a look_back of 12 for monthly data (one year)
    predictor = LSTMPredictor(look_back=12)
    predictor.train(df_train, value_column='value', epochs=75, batch_size=1)
    
    # --- 4. Make Predictions ---
    # We want to predict a number of periods equal to the size of our test set
    future_periods_to_predict = len(df_test)
    
    # The model uses the LAST `look_back` points from the training data to start predicting
    predicted_values = predictor.predict(df_train, future_periods=future_periods_to_predict, value_column='value')
    
    # --- 5. Evaluate the Predictions ---
    # Get the actual values from the test set to compare against
    actual_values = df_test['value'].values

    # a) Quantitative Metrics
    rmse = np.sqrt(mean_squared_error(actual_values, predicted_values))
    mae = mean_absolute_error(actual_values, predicted_values)
    print("\n--- MODEL PERFORMANCE EVALUATION ---")
    print(f"Root Mean Squared Error (RMSE): {rmse:.2f}")
    print(f"Mean Absolute Error (MAE): {mae:.2f}")
    print("------------------------------------\n")

    # b) Visual Plot
    plt.figure(figsize=(15, 7))
    plt.title('LSTM Model Performance: Actual vs. Predicted')
    plt.xlabel('Date')
    plt.ylabel('Transaction Value')

    # Plot training data
    plt.plot(df_train['date'], df_train['value'], color='blue', label='Training Data')

    # Plot actual test data
    plt.plot(df_test['date'], actual_values, color='green', label='Actual Values (Test Set)')

    # Plot predicted data
    plt.plot(df_test['date'], predicted_values, color='red', linestyle='--', marker='o', markersize=4, label='Predicted Values')

    plt.legend()
    plt.grid(True)
    plt.show()


if __name__ == '__main__':
    # Since our functions are async, we need to run them in an event loop
    asyncio.run(evaluate_model_performance())