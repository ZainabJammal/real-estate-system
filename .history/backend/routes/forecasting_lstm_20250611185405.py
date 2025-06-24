# # forecasting_lstm.py
# import os
# import json 
# import asyncio # Needed for running async functions
# import numpy as np
# import pandas as pd
# import matplotlib.pyplot as plt
# from sklearn.preprocessing import MinMaxScaler
# from tensorflow.keras.models import Sequential 
# from tensorflow.keras.layers import LSTM, Dense, Dropout
# from tensorflow.keras.optimizers import Adam
# from sklearn.metrics import mean_squared_error, mean_absolute_error
# # You might need to mock the supabase_client for local testing
# from unittest.mock import Mock, MagicMock 

# print("All libraries imported successfully!")


# class LSTMPredictor:
#     # --- I've added a Dropout layer to your model architecture ---
#     def __init__(self, look_back=12):
#         self.look_back = look_back
#         self.model = None
#         self.scaler = MinMaxScaler(feature_range=(0, 1))
#         self.is_trained = False
#         print(f"DEBUG: LSTMPredictor initialized with look_back={look_back}")

#     def _create_dataset(self, scaled_dataset): 
#         dataX, dataY = [], []
#         if len(scaled_dataset) <= self.look_back:
#             print(f"DEBUG: _create_dataset - Not enough data. Scaled dataset length: {len(scaled_dataset)}, look_back: {self.look_back}")
#             return np.array(dataX), np.array(dataY) 

#         for i in range(len(scaled_dataset) - self.look_back):
#             a = scaled_dataset[i:(i + self.look_back), 0]
#             dataX.append(a)
#             dataY.append(scaled_dataset[i + self.look_back, 0])
        
#         if not dataX:
#             print("DEBUG: _create_dataset - dataX is empty after loop, this is unexpected.")
#         return np.array(dataX), np.array(dataY)

#     def train(self, df_history, value_column='value', epochs=50, batch_size=1):
#         print(f"DEBUG: LSTMPredictor.train called. df_history length: {len(df_history)}, value_column: '{value_column}'")
#         if value_column not in df_history.columns:
#             raise ValueError(f"Value column '{value_column}' not found in df_history.")

#         dataset = df_history[value_column].values.astype('float32').reshape(-1, 1)

#         if np.isnan(dataset).any() or np.isinf(dataset).any():
#             raise ValueError("Dataset contains NaN or Inf values before scaling. Check data preparation.")

#         scaled_dataset = self.scaler.fit_transform(dataset)
#         trainX, trainY = self._create_dataset(scaled_dataset)

#         if len(trainX) == 0:
#             raise ValueError(f"Not enough data to create training sequences (trainX is empty). Need at least {self.look_back + 1} data points.")

#         trainX = np.reshape(trainX, (trainX.shape[0], trainX.shape[1], 1))

#         # --- Added Dropout for better generalization ---
#         self.model = Sequential([
#             LSTM(64, input_shape=(self.look_back, 1), return_sequences=True),
#             Dropout(0.2),
#             LSTM(32),
#             Dropout(0.2),
#             Dense(1)
#         ])
#         self.model.compile(loss='mean_squared_error', optimizer=Adam(learning_rate=0.001))
#         self.model.summary()

#         print(f"DEBUG: Starting LSTM model.fit with trainX shape: {trainX.shape}, trainY shape: {trainY.shape}, epochs: {epochs}, batch_size: {batch_size}")
#         self.model.fit(trainX, trainY, epochs=epochs, batch_size=batch_size, verbose=1)
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
#             raise ValueError(f"Not enough historical data for prediction. Need at least {self.look_back} data points, got {len(historical_values)}")

#         input_sequence = historical_values[-self.look_back:].reshape(-1, 1)

#         if np.isnan(input_sequence).any() or np.isinf(input_sequence).any():
#             raise ValueError("Input sequence for prediction contains NaN or Inf values.")

#         scaled_sequence = self.scaler.transform(input_sequence)
#         current_input = scaled_sequence.reshape((1, self.look_back, 1))

#         future_predictions_scaled = []
#         for i in range(future_periods):
#             pred_scaled = self.model.predict(current_input, verbose=0)
#             future_predictions_scaled.append(pred_scaled[0, 0])
#             new_sequence_member = pred_scaled.reshape(1, 1, 1)
#             current_input = np.append(current_input[:, 1:, :], new_sequence_member, axis=1)

#         future_predictions_scaled = np.array(future_predictions_scaled).reshape(-1, 1)
#         final_predictions = self.scaler.inverse_transform(future_predictions_scaled)
#         print(f"DEBUG: LSTM prediction complete. Generated {len(final_predictions)} future values.")
#         return final_predictions.flatten().tolist()


# # Your original data fetching function (unchanged)
# async def fetch_and_prepare_transaction_data(supabase_client, city_name=None, min_date_str=None):
#     # ... your original code here ...
#     # For this example, I'll mock its output to make the script runnable.
#     print("INFO: Using MOCKED data for fetch_and_prepare_transaction_data")
#     dates = pd.to_datetime(pd.date_range(start='2020-01-01', periods=60, freq='ME'))
#     # A sine wave to simulate seasonality, with some upward trend and noise
#     values = (np.sin(np.linspace(0, 30, 60)) * 50 + 
#               np.linspace(100, 150, 60) + 
#               np.random.normal(0, 5, 60))
#     mock_df = pd.DataFrame({'date': dates, 'value': values})
#     print(f"DEBUG: Generated mock data with {len(mock_df)} rows.")
#     return mock_df


# # ===================================================================
# #               NEW: EVALUATION SCRIPT MAIN BLOCK
# # ===================================================================
# async def evaluate_model_performance():
#     """
#     This function demonstrates the full workflow:
#     1. Fetch data.
#     2. Split into train/test sets.
#     3. Train the model on the train set.
#     4. Predict the future.
#     5. Evaluate predictions against the test set.
#     """
#     print("\n--- Starting Model Evaluation Workflow ---\n")
    
#     # --- 1. Fetch and Prepare Data ---
#     # In a real scenario, you'd pass your actual Supabase client
#     mock_supabase_client = Mock() 
#     df_full = await fetch_and_prepare_transaction_data(mock_supabase_client, city_name="any_city")

#     if df_full.empty or len(df_full) < 24: # Need enough data for a meaningful split
#         print("ERROR: Not enough data to perform evaluation. Exiting.")
#         return

#     # --- 2. Split Data into Train and Test Sets ---
#     # For time series, the split MUST be chronological. No shuffling!
#     train_size = int(len(df_full) * 0.80)
#     df_train = df_full.iloc[0:train_size]
#     df_test = df_full.iloc[train_size:len(df_full)]

#     print(f"Data Split Complete:")
#     print(f"Full dataset size: {len(df_full)}")
#     print(f"Training set size: {len(df_train)} (from {df_train['date'].min().date()} to {df_train['date'].max().date()})")
#     print(f"Test set size: {len(df_test)} (from {df_test['date'].min().date()} to {df_test['date'].max().date()})")

#     # --- 3. Initialize and Train the Model ---
#     # Using a look_back of 12 for monthly data (one year)
#     predictor = LSTMPredictor(look_back=12)
#     predictor.train(df_train, value_column='value', epochs=75, batch_size=1)
    
#     # --- 4. Make Predictions ---
#     # We want to predict a number of periods equal to the size of our test set
#     future_periods_to_predict = len(df_test)
    
#     # The model uses the LAST `look_back` points from the training data to start predicting
#     predicted_values = predictor.predict(df_train, future_periods=future_periods_to_predict, value_column='value')
    
#     # --- 5. Evaluate the Predictions ---
#     # Get the actual values from the test set to compare against
#     actual_values = df_test['value'].values

#     # a) Quantitative Metrics
#     rmse = np.sqrt(mean_squared_error(actual_values, predicted_values))
#     mae = mean_absolute_error(actual_values, predicted_values)
#     print("\n--- MODEL PERFORMANCE EVALUATION ---")
#     print(f"Root Mean Squared Error (RMSE): {rmse:.2f}")
#     print(f"Mean Absolute Error (MAE): {mae:.2f}")
#     print("------------------------------------\n")

#     # b) Visual Plot
#     plt.figure(figsize=(15, 7))
#     plt.title('LSTM Model Performance: Actual vs. Predicted')
#     plt.xlabel('Date')
#     plt.ylabel('Transaction Value')

#     # Plot training data
#     plt.plot(df_train['date'], df_train['value'], color='blue', label='Training Data')

#     # Plot actual test data
#     plt.plot(df_test['date'], actual_values, color='green', label='Actual Values (Test Set)')

#     # Plot predicted data
#     plt.plot(df_test['date'], predicted_values, color='red', linestyle='--', marker='o', markersize=4, label='Predicted Values')

#     plt.legend()
#     plt.grid(True)
#     plt.show()


# if __name__ == '__main__':
#     # Since our functions are async, we need to run them in an event loop
#     asyncio.run(evaluate_model_performance())
import os
import pandas as pd
import numpy as np
import json
import pickle
import asyncio
import traceback
import matplotlib.pyplot as plt
from dotenv import load_dotenv
from tensorflow.keras.models import load_model
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error
from supabase import create_client, Client

# # ===================================================================
# #                          CONFIGURATION
# # ===================================================================
# # This MUST match the granularity of the model you want to evaluate
# GRANULARITY = 'M' # 'M' for Monthly, 'Y' for Yearly
# # How many of the most recent periods to use for the test set
# TEST_PERIODS = 6 # e.g., use the last 6 months for testing

# # ===================================================================
# #                       1. SETUP AND DATA LOADING
# # ===================================================================

def get_supabase_client() -> Client: # The return type hint should be Client
    """Connects to Supabase using credentials from .env file."""
    load_dotenv()
    url = os.environ.get("SUPABASE_URL")
    key = os.environ.get("SUPABASE_KEY")
    if not url or not key:
        raise ValueError("Supabase URL and Key must be set in your .env file.")
    # This creates the SYNCHRONOUS client
    return create_client(url, key)

class ModelEvaluator:
    def __init__(self, granularity: str):
        self.granularity = granularity
        self.suffix = "monthly" if granularity == 'M' else "yearly"
        self.artifacts_path = "C:\\Users\\user\\Documents\\Real Estate SPF\\real-estate-system\\real-estate-system\\backend\\model_artifacts"

        # Load all the artifacts created during training
        self._load_artifacts()
        # NEW, CORRECTED LINE
        self.look_back = self.model.input_shape[1]  # Auto-detect look_back from model
        print(f"Successfully loaded model with look_back = {self.look_back}")

    def _load_artifacts(self):
        """Loads the trained model, scaler, and feature list."""
        print(f"Loading artifacts for granularity: {self.suffix.capitalize()}...")
        
        # 1. Load Model
        model_path = os.path.join(self.artifacts_path, f"master_model_{self.suffix}.h5")
        self.model = load_model(model_path)
        
        # 2. Load Scaler
        scaler_path = os.path.join(self.artifacts_path, f"master_scaler_{self.suffix}.pkl")
        with open(scaler_path, 'rb') as f:
            self.scaler = pickle.load(f)
            
        # 3. Load Feature List
        features_path = os.path.join(self.artifacts_path, f"model_features_{self.suffix}.json")
        with open(features_path, 'r') as f:
            self.feature_columns = json.load(f)

    def prepare_evaluation_data(self):
        supabase = get_supabase_client()
        """Fetches all data and splits it into training and testing sets."""
        supabase = get_supabase_client()
        # This reuses the data prep logic from the training script
        response = supabase.table('transactions').select("trans_date, city, transaction_value").execute() # No await
        df = pd.DataFrame(response.data)
        df['date'] = pd.to_datetime(df['trans_date'])

        df_grouped = df.set_index('date').groupby('city')['transaction_value'].resample(self.granularity).sum()
        self.full_df = df_grouped.reset_index()
        
        # Chronological Split
        unique_dates = sorted(self.full_df['date'].unique())
        if len(unique_dates) < TEST_PERIODS + self.look_back:
             raise ValueError(f"Not enough unique dates ({len(unique_dates)}) to perform a train/test split.")
        
        split_date = unique_dates[-TEST_PERIODS]
        
        self.train_df = self.full_df[self.full_df['date'] < split_date]
        self.test_df = self.full_df[self.full_df['date'] >= split_date]

        print("\n--- Data Split for Evaluation ---")
        print(f"Training data from {self.train_df['date'].min().date()} to {self.train_df['date'].max().date()}")
        print(f"Testing data from {self.test_df['date'].min().date()} to {self.test_df['date'].max().date()}")
        return

    def _make_predictions_for_test_set(self):
        """Uses the last part of the training data to forecast the test period."""
        all_predictions = []

        for city in self.test_df['city'].unique():
            print(f"Generating forecast for city: {city}")
            
            # Get the last 'look_back' periods from the training data for this city
            history = self.train_df[self.train_df['city'] == city].tail(self.look_back)
            if len(history) < self.look_back:
                print(f"  WARNING: Not enough history in training set for {city}. Skipping.")
                continue

            # Create the input dataframe
            input_df = pd.DataFrame(columns=self.feature_columns)
            input_df['transaction_value'] = history['transaction_value'].values
            input_df.fillna(0, inplace=True)
            city_col = f"city_{city}"
            if city_col in input_df.columns:
                input_df[city_col] = 1
            
            # Scale the input data
            scaled_input = self.scaler.transform(input_df[self.feature_columns])
            current_input = scaled_input.reshape((1, self.look_back, len(self.feature_columns)))
            
            # Iteratively predict
            city_preds_scaled = []
            for _ in range(TEST_PERIODS):
                pred_scaled = self.model.predict(current_input, verbose=0)[0, 0]
                city_preds_scaled.append(pred_scaled)
                
                new_step = current_input[0, -1, :].copy()
                new_step[0] = pred_scaled
                new_step = new_step.reshape(1, 1, len(self.feature_columns))
                current_input = np.append(current_input[:, 1:, :], new_step, axis=1)

            # Inverse transform the predictions
            dummy_array = np.zeros((len(city_preds_scaled), len(self.feature_columns)))
            dummy_array[:, 0] = city_preds_scaled
            city_predictions = self.scaler.inverse_transform(dummy_array)[:, 0]

            # Get the corresponding dates from the test set for this city
            test_dates = self.test_df[self.test_df['city'] == city]['date']
            
            for i, pred in enumerate(city_predictions):
                all_predictions.append({'date': test_dates.iloc[i], 'city': city, 'predicted_value': pred})
        
        return pd.DataFrame(all_predictions)

    def evaluate(self):
        """Performs the evaluation and prints metrics and plots."""
        # Get predictions
        df_predictions = self._make_predictions_for_test_set()
        
        # Merge predictions with actual values from the test set
        df_results = pd.merge(self.test_df, df_predictions, on=['date', 'city'])
        
        if df_results.empty:
            print("\n--- ERROR: No matching predictions and actuals. Check city names. ---")
            return

        # 1. --- Quantitative Metrics ---
        rmse = np.sqrt(mean_squared_error(df_results['transaction_value'], df_results['predicted_value']))
        mae = mean_absolute_error(df_results['transaction_value'], df_results['predicted_value'])
        
        # Calculate Mean Absolute Percentage Error (MAPE)
        mape = np.mean(np.abs((df_results['transaction_value'] - df_results['predicted_value']) / df_results['transaction_value'])) * 100

        print("\n--- ✅ Overall Model Performance Metrics ---")
        print(f"  Root Mean Squared Error (RMSE): {rmse:,.2f}")
        print(f"  Mean Absolute Error (MAE):      {mae:,.2f}")
        print(f"  Mean Absolute Percentage Error (MAPE): {mape:.2f}%")
        print("------------------------------------------\n")
        print("Interpretation:")
        print("  - RMSE/MAE: The average error in transaction value units. Lower is better.")
        print("  - MAPE: The average percentage error. Good for comparing across cities of different scales. Lower is better.")

        # 2. --- Visual Evaluation (Plot) ---
        plt.style.use('seaborn-v0_8-whitegrid')
        
        # Get total transaction values per date for plotting
        train_totals = self.train_df.groupby('date')['transaction_value'].sum()
        test_totals = self.test_df.groupby('date')['transaction_value'].sum()
        predicted_totals = df_results.groupby('date')['predicted_value'].sum()

        plt.figure(figsize=(18, 8))
        plt.plot(train_totals.index, train_totals.values, color='royalblue', label='Training Data (Actual)')
        plt.plot(test_totals.index, test_totals.values, color='green', marker='.', markersize=10, linestyle='-', label='Test Data (Actual)')
        plt.plot(predicted_totals.index, predicted_totals.values, color='red', linestyle='--', marker='o', label='Forecasted Data (Predicted)')
        
        plt.title(f'Model Performance Evaluation ({self.suffix.capitalize()})', fontsize=18)
        plt.xlabel('Date', fontsize=12)
        plt.ylabel('Total Transaction Value', fontsize=12)
        plt.legend(fontsize=12)
        plt.tight_layout()
        plt.show()

# ===================================================================
#                       3. MAIN EXECUTION
# ===================================================================

async def main():
    print("--- Starting Model Evaluation Script ---")
    try:
        evaluator = ModelEvaluator(granularity=GRANULARITY)
        evaluator.prepare_evaluation_data() # No await
        evaluator.evaluate()
    except Exception as e:
        print(f"\n--- ❌ An error occurred during evaluation! ---")
        traceback.print_exc()

if __name__ == '__main__':
    asyncio.run(main())