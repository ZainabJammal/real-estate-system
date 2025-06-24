# # train_master_model.py
# import os
# import pandas as pd
# import numpy as np
# import json
# import asyncio
# from dotenv import load_dotenv
# from tensorflow.keras.models import Sequential
# from tensorflow.keras.layers import LSTM, Dense, Dropout
# from sklearn.preprocessing import MinMaxScaler
# # Assuming your original classes/functions are in forecasting_lstm.py in the root of the backend folder
# # Adjust the import path if you've moved the file
# from routes.forecasting_lstm import LSTMPredictor, fetch_and_prepare_transaction_data
# from supabase.client import Client, AsyncClient 
# # We don't need create_client anymore for async

# async def create_supabase() -> AsyncClient:
#     load_dotenv()
#     url = os.environ.get("SUPABASE_URL")
#     key = os.environ.get("SUPABASE_KEY")
#     if not url or not key:
#         raise ValueError("Supabase URL and Key must be set in your .env file.")
    
#     # --- THIS IS THE FIX ---
#     # Instantiate the AsyncClient class directly. This is now the correct way.
#     # The instantiation itself is synchronous, so we remove 'await'.
#     return AsyncClient(supabase_url=url, supabase_key=key)# --- 1. Data Preparation (Same as before, this part is good) ---
# async def fetch_and_prepare_for_master_model(supabase_client):
#     print("Fetching data for all cities...")
#     # Passing city_name=None fetches data for ALL cities
#     df_all_cities = await fetch_and_prepare_transaction_data(supabase_client, city_name=None)
    
#     if df_all_cities.empty:
#         raise ValueError("No data returned for any city.")
#     print(f"Original data shape for all cities: {df_all_cities.shape}")

#     # Keep track of the original city column for grouping before it gets encoded
#     # Ensure you are using the correct column name from the fetch function.
#     # The fetch function returns a 'city' column.
#     df_all_cities['city'] = df_all_cities['city'] 
    
#     print("Applying one-hot encoding to 'city' column...")
#     df_encoded = pd.get_dummies(df_all_cities, columns=['city'], prefix='city')
    
#     feature_columns = ['value'] + [col for col in df_encoded if col.startswith('city_')]
    
#     # Save files to the root of the backend folder
#     with open('model_features.json', 'w') as f:
#         json.dump(feature_columns, f)
#     print(f"Saved feature columns to model_features.json: {feature_columns}")

#     return df_encoded, feature_columns

# # --- 2. Improved MultivariateLSTMPredictor Class ---
# class MultivariateLSTMPredictor(LSTMPredictor):
#     def _create_sequences_for_city(self, city_data, n_features):
#         """Helper to create sequences for a single city's data."""
#         dataX, dataY = [], []
#         target_col_index = 0  # 'value' is the first column
#         for i in range(len(city_data) - self.look_back):
#             a = city_data[i:(i + self.look_back), :]
#             dataX.append(a)
#             dataY.append(city_data[i + self.look_back, target_col_index])
#         return np.array(dataX), np.array(dataY)

#     def train(self, df, feature_columns, epochs=50, batch_size=32):
#         n_features = len(feature_columns)
#         dataset = df[feature_columns].values.astype('float32')
        
#         # Fit the scaler on the entire dataset
#         self.scaler.fit(dataset)
#         # We will save the scaler later

#         # --- CRITICAL IMPROVEMENT: Create sequences per city ---
#         all_X, all_Y = [], []
        
#         # Group by the original city name before it was one-hot encoded
#         for city_name, city_group in df.groupby('city_name_original'):
#             print(f"Creating sequences for city: {city_name}...")
#             city_dataset = city_group[feature_columns].values.astype('float32')
            
#             # Use the already fitted scaler to transform this city's data
#             scaled_city_dataset = self.scaler.transform(city_dataset)
            
#             city_X, city_Y = self._create_sequences_for_city(scaled_city_dataset, n_features)
            
#             if len(city_X) > 0:
#                 all_X.append(city_X)
#                 all_Y.append(city_Y)

#         if not all_X:
#             raise ValueError("Could not create any training sequences. Check data length for each city.")
            
#         # Combine sequences from all cities into one large training set
#         trainX = np.concatenate(all_X, axis=0)
#         trainY = np.concatenate(all_Y, axis=0)
#         print(f"Total training sequences from all cities: {len(trainX)}")

#         self.model = Sequential([
#             LSTM(64, input_shape=(self.look_back, n_features), return_sequences=True),
#             Dropout(0.2),
#             LSTM(32),
#             Dropout(0.2),
#             Dense(1)
#         ])
#         self.model.compile(loss='mean_squared_error', optimizer='adam')
#         self.model.summary()

#         print(f"Starting master model training with final trainX shape: {trainX.shape}")
#         self.model.fit(trainX, trainY, epochs=epochs, batch_size=batch_size, verbose=1)
#         self.is_trained = True
        
#         # --- 3. SAVE THE MODEL AND THE SCALER ---
#         self.model.save("master_transaction_model.h5")
#         import pickle
#         with open('master_scaler.pkl', 'wb') as f:
#             pickle.dump(self.scaler, f)
            
#         print("Master model and scaler saved successfully.")

# # --- 3. Main Execution Block to Run Training ---
# async def main():
#     """Main function to connect to DB and run the training pipeline."""
#     supabase = None
#     try:
#         supabase = await create_supabase()
#         df_master, feature_cols = await fetch_and_prepare_for_master_model(supabase)
        
#         predictor = MultivariateLSTMPredictor(look_back=12)
#         predictor.train(df_master, feature_columns=feature_cols, epochs=25) # Use desired epochs

#     except Exception as e:
#         print(f"An error occurred during the training process: {e}")
#         import traceback
#         traceback.print_exc()
#     finally:
#         if supabase:
#             # await supabase.close()
#             pass

# if __name__ == '__main__':
#     print("--- Starting Master Model Training Script ---")
#     asyncio.run(main())
#     print("--- Training Script Finished ---")
import os
import pandas as pd
import numpy as np
import argparse  
import json
import pickle
import asyncio
import traceback
import matplotlib.pyplot as plt
from dotenv import load_dotenv
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import LSTM, Dense, Dropout
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error
from supabase import create_async_client, AsyncClient

# ===================================================================
#                          CONFIGURATION
# ===================================================================
LOOK_BACK = 5      # Use 5 periods (years) of data to predict the next one.
GRANULARITY = 'Y'   # 'Y' for Yearly.
EPOCHS = 100         # Number of training cycles. 50-100 is a good starting range.
BATCH_SIZE = 8     # Number of training samples to work through before updating weights.
TEST_PERIODS = 1 # e.g., use the last 1 year for testing

# ===================================================================
#                       1. DATABASE AND DATA PREPARATION
# ===================================================================

def get_supabase_client() -> AsyncClient:
    """Connects to Supabase using credentials from .env file."""
    load_dotenv()
    url = os.environ.get("SUPABASE_URL")
    key = os.environ.get("SUPABASE_KEY")
    if not url or not key:
        raise ValueError("Supabase URL and Key must be set in your .env file.")
    return create_async_client(url, key)

async def fetch_and_prepare_data(supabase_client: AsyncClient, granularity: str, start_year: int, end_year: int):
    """
    Fetches all transaction data, handles text dates, and aggregates by granularity.
    Filters data for the specified year range.
    """
    print(f"Fetching data for all cities with granularity: {granularity} from {start_year} to {end_year}...")
    
    response = await supabase_client.table('transactions').select(
        "date, city, transaction_value"
    ).execute()
    
    if not response.data:
        raise ValueError("No data returned from Supabase. Check table name and permissions.")
        
    df = pd.DataFrame(response.data)
    
    print("Converting 'date' text column to datetime objects...")
    df['date'] = pd.to_datetime(df['date'])
    
    # Filter data for the specified year range
    df = df[(df['date'].dt.year >= start_year) & (df['date'].dt.year <= end_year)]
    
    if df.empty:
        raise ValueError(f"No data found for the years {start_year}-{end_year}. Adjust your Supabase data or date range.")

    df_grouped = df.set_index('date').groupby('city')['transaction_value'].resample(granularity).sum()
    df_resampled = df_grouped.reset_index()

    df_resampled['city_name_original'] = df_resampled['city']
    
    print(f"Data shape after resampling and filtering: {df_resampled.shape}")

    print("Applying one-hot encoding to 'city' column...")
    df_encoded = pd.get_dummies(df_resampled, columns=['city'], prefix='city')
    
    target_variable = 'transaction_value'
    one_hot_columns = [col for col in df_encoded.columns if col.startswith('city_')]
    feature_columns = [target_variable] + one_hot_columns
    
    print(f"✅ Model will be trained on these {len(feature_columns)} numerical features: {feature_columns}")
    
    return df_encoded, feature_columns

# ===================================================================
#                       2. THE LSTM PREDICTOR CLASS
# ===================================================================

class MultivariateLSTMPredictor:
    def __init__(self, look_back=12):
        self.look_back = look_back
        self.model = None
        self.scaler = MinMaxScaler(feature_range=(0, 1))
        self.is_trained = False

    def _create_sequences_per_city(self, city_data, n_features):
        """Helper to create sequences for a single city's data, avoiding data leakage."""
        dataX, dataY = [], []
        target_col_index = 0  # 'value' is always the first column in our features list
        for i in range(len(city_data) - self.look_back):
            sequence_in = city_data[i:(i + self.look_back), :]
            sequence_out = city_data[i + self.look_back, target_col_index]
            dataX.append(sequence_in)
            dataY.append(sequence_out)
        return np.array(dataX), np.array(dataY)

    
    def train(self, df, feature_columns, epochs, batch_size):
        """Trains the multivariate LSTM model with aggressive type conversion."""
        print("--- Starting train method ---")
        n_features = len(feature_columns)
        
        df_features_only = df[feature_columns].copy()

        print("Forcing all feature columns to numeric type...")
        for col in df_features_only.columns:
            df_features_only[col] = pd.to_numeric(df_features_only[col], errors='coerce')

        if df_features_only.isnull().sum().sum() > 0:
            print("\n--- WARNING: Found non-numeric data that was converted to NaN ---")
            print("Columns with NaN values after conversion:")
            print(df_features_only.isnull().sum())
            print("\nRows containing NaN values:")
            print(df_features_only[df_features_only.isnull().any(axis=1)])
            print("Filling NaN values with 0...")
            df_features_only.fillna(0, inplace=True)
        else:
            print("All feature columns converted to numeric successfully.")
            
        print("Fitting the scaler...")
        self.scaler.fit(df_features_only.values)

        all_X, all_Y = [], []
        for city_name, city_group in df.groupby('city_name_original'):
            print(f"Creating training sequences for city: {city_name}...")
            
            city_group_features = city_group[feature_columns].copy()
            for col in city_group_features.columns:
                city_group_features[col] = pd.to_numeric(city_group_features[col], errors='coerce')
            city_group_features.fillna(0, inplace=True)

            scaled_city_dataset = self.scaler.transform(city_group_features.values)
            
            city_X, city_Y = self._create_sequences_per_city(scaled_city_dataset, n_features)
            
            if len(city_X) > 0:
                all_X.append(city_X)
                all_Y.append(city_Y)

        if not all_X:
            raise ValueError("Could not create any training sequences. Check data length for each city.")
            
        trainX = np.concatenate(all_X, axis=0)
        trainY = np.concatenate(all_Y, axis=0)
        print(f"\nTotal training sequences from all cities: {len(trainX)}")

        self.model = Sequential([
            LSTM(100, input_shape=(self.look_back, n_features), return_sequences=True),
            Dropout(0.2),
            LSTM(50),
            Dropout(0.2),
            Dense(1)
        ])
        self.model.compile(loss='mean_squared_error', optimizer='adam')
        self.model.summary()

        print(f"\nStarting master model training... (trainX shape: {trainX.shape})")
        self.model.fit(trainX, trainY, epochs=epochs, batch_size=batch_size, verbose=1)
        self.is_trained = True

    def save_artifacts(self, granularity: str, feature_columns: list):
        """Saves the trained model, the data scaler, and the feature list."""
        if not self.is_trained:
            raise ValueError("Model must be trained before saving.")

        os.makedirs("model_artifacts", exist_ok=True)

        suffix = "monthly" if granularity == 'M' else "yearly"

        model_path = os.path.join("model_artifacts", f"master_model_{suffix}.h5")
        self.model.save(model_path)
        print(f"✅ Model saved to: {model_path}")
        
        scaler_path = os.path.join("model_artifacts", f"master_scaler_{suffix}.pkl")
        with open(scaler_path, 'wb') as f:
            pickle.dump(self.scaler, f)
        print(f"✅ Scaler saved to: {scaler_path}")
            
        features_path = os.path.join("model_artifacts", f"model_features_{suffix}.json")
        with open(features_path, 'w') as f:
            json.dump(feature_columns, f)
        print(f"✅ Feature columns saved to: {features_path}")

# ===================================================================
#                       3. MODEL EVALUATION CLASS
# ===================================================================

class ModelEvaluator:
    def __init__(self, granularity: str, artifacts_path: str = "model_artifacts"):
        self.granularity = granularity
        self.suffix = "monthly" if granularity == 'M' else "yearly"
        self.artifacts_path = artifacts_path

        self._load_artifacts()
        self.look_back = self.model.input_shape[1]
        print(f"Successfully loaded model with look_back = {self.look_back}")

    def _load_artifacts(self):
        """Loads the trained model, scaler, and feature list."""
        print(f"Loading artifacts for granularity: {self.suffix.capitalize()}...")
        
        model_path = os.path.join(self.artifacts_path, f"master_model_{self.suffix}.h5")
        self.model = load_model(model_path)
        
        scaler_path = os.path.join(self.artifacts_path, f"master_scaler_{self.suffix}.pkl")
        with open(scaler_path, 'rb') as f:
            self.scaler = pickle.load(f)
            
        features_path = os.path.join(self.artifacts_path, f"model_features_{self.suffix}.json")
        with open(features_path, 'r') as f:
            self.feature_columns = json.load(f)

    async def prepare_evaluation_data(self, supabase_client: AsyncClient, start_year: int, end_year: int, test_periods: int):
        """Fetches all data, filters by year, and splits it into training and testing sets chronologically."""
        print(f"Fetching data for evaluation from {start_year} to {end_year}...")
        response = await supabase_client.table('transactions').select("date, city, transaction_value").execute()
        df = pd.DataFrame(response.data)
        df['date'] = pd.to_datetime(df['date'])

        # Filter data for the specified year range
        df = df[(df['date'].dt.year >= start_year) & (df['date'].dt.year <= end_year)]
        
        if df.empty:
            raise ValueError(f"No data found for evaluation in the years {start_year}-{end_year}.")

        df_grouped = df.set_index('date').groupby('city')['transaction_value'].resample(self.granularity).sum()
        self.full_df = df_grouped.reset_index()
        
        # Chronological Split
        unique_dates = sorted(self.full_df['date'].unique())
        if len(unique_dates) < test_periods + self.look_back:
             raise ValueError(f"Not enough unique dates ({len(unique_dates)}) to perform a train/test split. Need at least {test_periods + self.look_back}.")
        
        split_date = unique_dates[-test_periods]
        
        self.train_df = self.full_df[self.full_df['date'] < split_date]
        self.test_df = self.full_df[self.full_df["date"] >= split_date].rename(columns={
            "transaction_value": "transaction_value_actual"
        })

        print("\n--- Data Split for Evaluation ---")
        print(f"Training data from {self.train_df['date'].min().date()} to {self.train_df['date'].max().date()}")
        print(f"Testing data from {self.test_df['date'].min().date()} to {self.test_df['date'].max().date()}")
        print(f"Columns in self.test_df: {self.test_df.columns.tolist()}")
        print(f"Sample of self.test_df:\n{self.test_df.head()}")

    def _make_predictions_for_test_set(self, test_periods: int):
        """Uses the last part of the training data to forecast the test period."""
        all_predictions = []

        for city in self.test_df['city'].unique():
            print(f"Generating forecast for city: {city}")
            
            history = self.train_df[self.train_df['city'] == city].tail(self.look_back)
            if len(history) < self.look_back:
                print(f"  WARNING: Not enough history in training set for {city}. Skipping.")
                continue

            input_df = pd.DataFrame(columns=self.feature_columns)
            input_df['transaction_value'] = history['transaction_value'].values
            input_df.fillna(0, inplace=True)
            city_col = f"city_{city}"
            if city_col in self.feature_columns:
                input_df[city_col] = 1
            else:
                print(f"  WARNING: City {city} not found in model's feature columns. Skipping.")
                continue
            
            scaled_input = self.scaler.transform(input_df[self.feature_columns])
            current_input = scaled_input.reshape((1, self.look_back, len(self.feature_columns)))
            
            city_preds_scaled = []
            for _ in range(test_periods):
                pred_scaled = self.model.predict(current_input, verbose=0)[0, 0]
                city_preds_scaled.append(pred_scaled)
                
                new_step = current_input[0, -1, :].copy()
                new_step[0] = pred_scaled
                # Ensure city one-hot encoding is maintained for the new step
                if city_col in self.feature_columns:
                    city_col_index = self.feature_columns.index(city_col)
                    new_step[city_col_index] = 1

                new_step = new_step.reshape(1, 1, len(self.feature_columns))
                current_input = np.append(current_input[:, 1:, :], new_step, axis=1)

            dummy_array = np.zeros((len(city_preds_scaled), len(self.feature_columns)))
            dummy_array[:, 0] = city_preds_scaled
            city_predictions = self.scaler.inverse_transform(dummy_array)[:, 0]

            # Get the corresponding dates from the test set for this city
            test_dates_for_city = self.test_df[self.test_df['city'] == city]['date'].sort_values().tolist()
            
            # Ensure the number of predictions matches the number of test periods
            if len(city_predictions) != len(test_dates_for_city):
                print(f"  WARNING: Mismatch in prediction count ({len(city_predictions)}) and test dates ({len(test_dates_for_city)}) for {city}. Truncating predictions.")
                city_predictions = city_predictions[:len(test_dates_for_city)]

            for date, pred_value in zip(test_dates_for_city, city_predictions):
                all_predictions.append({
                    'city': city,
                    'date': date,
                    'predicted_value': pred_value
                })
        return pd.DataFrame(all_predictions)

    def evaluate_and_plot(self, test_periods: int):
        """Evaluates the model's performance and generates plots."""
        print("\n--- Starting Model Evaluation ---")
        predicted_df = self._make_predictions_for_test_set(test_periods)

        if predicted_df.empty:
            print("No predictions were made for evaluation. Skipping metrics and plots.")
            return

        # Merge actual and predicted values for comparison
        evaluation_df = pd.merge(
            self.test_df,
            predicted_df,
            on=["date", "city"],
            how="inner",
            suffixes=("_actual", "_predicted")
        )
        print(f"Columns in evaluation_df: {evaluation_df.columns.tolist()}")
        print(f"Sample of evaluation_df:\n{evaluation_df.head()}")
        
        if evaluation_df.empty:
            print("No overlapping data between actual and predicted values for evaluation. Skipping metrics and plots.")
            return

        # Calculate metrics
        rmse = np.sqrt(mean_squared_error(evaluation_df["transaction_value_actual"], evaluation_df["predicted_value"]))
        mae = mean_absolute_error(evaluation_df["transaction_value_actual"], evaluation_df["predicted_value"])
        print("\n--- MODEL PERFORMANCE METRICS ---")
        print(f"Root Mean Squared Error (RMSE): {rmse:.2f}")
        print(f"Mean Absolute Error (MAE): {mae:.2f}")
        print("------------------------------------\n")

        # Plotting
        plt.figure(figsize=(18, 8))
        for city in evaluation_df['city'].unique():
            city_df = evaluation_df[evaluation_df['city'] == city]
            plt.plot(city_df['date'], city_df['transaction_value_actual'], label=f'{city} Actual', marker='o', linestyle='-')
            plt.plot(city_df['date'], city_df['predicted_value'], label=f'{city} Predicted', marker='x', linestyle='--')
        
        plt.title('LSTM Model Performance: Actual vs. Predicted by City')
        plt.xlabel('Date')
        plt.ylabel('Transaction Value')
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        plot_path = os.path.join("model_artifacts", "evaluation_plot.png")
        plt.savefig(plot_path)
        print(f"✅ Evaluation plot saved to: {plot_path}")
        plt.close()

# ===================================================================
#                       4. MAIN EXECUTION BLOCK
# ===================================================================

async def main():
    print("--- Starting Master Model Training and Evaluation Script ---")
    supabase = None
    try:
        supabase = await get_supabase_client()
        
        # 1. Fetch and Prepare Data
        df_encoded, feature_columns = await fetch_and_prepare_data(supabase, GRANULARITY, 2011, 2016)
        
        # 2. Train the Model
        predictor = MultivariateLSTMPredictor(look_back=LOOK_BACK)
        predictor.train(df_encoded, feature_columns, EPOCHS, BATCH_SIZE)
        predictor.save_artifacts(GRANULARITY, feature_columns)

        # 3. Evaluate the Model
        evaluator = ModelEvaluator(GRANULARITY)
        await evaluator.prepare_evaluation_data(supabase, 2011, 2016, TEST_PERIODS)
        evaluator.evaluate_and_plot(TEST_PERIODS)

    except ValueError as e:
        print(f"Configuration Error: {e}")
    except Exception as e:
        print(f"An unexpected error occurred: {e}")
        traceback.print_exc()
    finally:
        if supabase:
            print("Supabase client closed.")

if __name__ == "__main__":
    asyncio.run(main())

