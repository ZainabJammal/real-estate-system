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
import json
import pickle
import asyncio
import traceback
from dotenv import load_dotenv
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from sklearn.preprocessing import MinMaxScaler
from supabase import create_client, Client

# ===================================================================
#                          CONFIGURATION
# # ===================================================================
# # Adjust these parameters as needed
# LOOK_BACK = 12      # Use 12 periods (months/years) of data to predict the next one.
# GRANULARITY = 'M'   # 'M' for Monthly, 'Y' for Yearly.
# EPOCHS = 75         # Number of training cycles. 50-100 is a good starting range.
# BATCH_SIZE = 32     # Number of training samples to work through before updating weights.

# # ===================================================================
# #                       1. DATABASE AND DATA PREPARATION
# ===================================================================

def get_supabase_client() -> Client:
    """Connects to Supabase using credentials from .env file."""
    load_dotenv()
    url = os.environ.get("SUPABASE_URL")
    key = os.environ.get("SUPABASE_KEY")
    if not url or not key:
        raise ValueError("Supabase URL and Key must be set in your .env file.")
    return create_client(url, key)

async def fetch_and_prepare_data(supabase_client: Client, granularity: str):
    """
    Fetches all transaction data, handles text dates, and aggregates by granularity.
    This version includes aggressive debugging to find data type issues.
    """
    print("--- Starting fetch_and_prepare_data ---")
    response = supabase_client.table('transactions').select("trans_date, city, transaction_value").execute()
    
    if not response.data:
        raise ValueError("No data returned from Supabase.")
        
    df = pd.DataFrame(response.data)

    df['date'] = pd.to_datetime(df['trans_date'])
    df.drop(columns=['trans_date'], inplace=True)

    df_grouped = df.set_index('date').groupby('city')['transaction_value'].resample(granularity).sum()
    df_resampled = df_grouped.reset_index()

    df_resampled['city_name_original'] = df_resampled['city']
    
    print("Applying one-hot encoding...")
    df_encoded = pd.get_dummies(df_resampled, columns=['city'], prefix='city')
    
    target_variable = 'transaction_value'
    one_hot_columns = [col for col in df_encoded.columns if isinstance(col, str) and col.startswith('city_')]
    feature_columns = [target_variable] + one_hot_columns
    
    print("\n--- DEBUGGING DATA BEFORE RETURN ---")
    print(f"Final DataFrame columns: {df_encoded.columns.tolist()}")
    print(f"Final feature_columns list: {feature_columns}")
    
    # Check for non-numeric types in the columns that are SUPPOSED to be numeric
    df_to_check = df_encoded[feature_columns]
    print("\nData types of the columns being sent to the model:")
    print(df_to_check.dtypes)
    
    # Explicitly check if any non-numeric data exists where it shouldn't
    for col in df_to_check.columns:
        if df_to_check[col].dtype == 'object':
            print(f"WARNING: Column '{col}' is of type 'object' (likely string). This should not happen.")

    print("--- Finished fetch_and_prepare_data ---\n")
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
        
        # --- AGGRESSIVE FIX & DEBUGGING ---
        # 1. Select only the feature columns.
        df_features_only = df[feature_columns].copy() # Use .copy() to avoid SettingWithCopyWarning

        # 2. **FORCE** conversion to numeric.
        # `pd.to_numeric` will turn any non-numeric values into `NaN` (Not a Number).
        # This is better than crashing.
        print("Forcing all feature columns to numeric type...")
        for col in df_features_only.columns:
            df_features_only[col] = pd.to_numeric(df_features_only[col], errors='coerce')

        # 3. Check for any values that failed to convert (they are now NaN).
        if df_features_only.isnull().sum().sum() > 0:
            print("\n--- WARNING: Found non-numeric data that was converted to NaN ---")
            print("Columns with NaN values after conversion:")
            print(df_features_only.isnull().sum())
            # Let's see the rows that have problems
            print("\nRows containing NaN values:")
            print(df_features_only[df_features_only.isnull().any(axis=1)])
            # We must fill these NaN values before scaling. Let's fill with 0.
            print("Filling NaN values with 0...")
            df_features_only.fillna(0, inplace=True)
        else:
            print("All feature columns converted to numeric successfully.")
            
        # 4. Fit the scaler on the now guaranteed-to-be-numeric data.
        print("Fitting the scaler...")
        self.scaler.fit(df_features_only.values) # .astype('float32') is not strictly needed here but is good practice

        all_X, all_Y = [], []
        for city_name, city_group in df.groupby('city_name_original'):
            print(f"Creating training sequences for city: {city_name}...")
            
            # We need to perform the same safe conversion for each group
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
            
        # Combine sequences from all cities into one large training set.
        trainX = np.concatenate(all_X, axis=0)
        trainY = np.concatenate(all_Y, axis=0)
        print(f"\nTotal training sequences from all cities: {len(trainX)}")

        # --- Define the LSTM Model Architecture ---
        self.model = Sequential([
            LSTM(100, input_shape=(self.look_back, n_features), return_sequences=True),
            Dropout(0.2),
            LSTM(50),
            Dropout(0.2),
            Dense(1) # Output layer with 1 neuron to predict the 'transaction_value'
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

        # Create the artifacts directory if it doesn't exist
        os.makedirs("model_artifacts", exist_ok=True)
        
        suffix = "monthly" if granularity == 'M' else "yearly"
        
        # 1. Save Trained Model
        model_path = os.path.join("model_artifacts", f"master_model_{suffix}.h5")
        self.model.save(model_path)
        print(f"✅ Model saved to: {model_path}")
        
        # 2. Save Data Scaler
        scaler_path = os.path.join("model_artifacts", f"master_scaler_{suffix}.pkl")
        with open(scaler_path, 'wb') as f:
            pickle.dump(self.scaler, f)
        print(f"✅ Scaler saved to: {scaler_path}")
            
        # 3. Save Feature Column List
        features_path = os.path.join("model_artifacts", f"model_features_{suffix}.json")
        with open(features_path, 'w') as f:
            json.dump(feature_columns, f)
        print(f"✅ Feature list saved to: {features_path}")

# ===================================================================
#                       3. MAIN EXECUTION SCRIPT
# ===================================================================

async def main(config):
   # Unpack configuration
    granularity = config['granularity']
    look_back = config['look_back']
    epochs = config['epochs']
    batch_size = config['batch_size']
    
    print(f"--- Starting Master Model Training Pipeline for Granularity: '{granularity}' ---")
    print(f"Configuration: LOOK_BACK={look_back}, EPOCHS={epochs}, BATCH_SIZE={batch_size}")
    
    supabase = get_supabase_client()
    try:
        # Step 1: Fetch and prepare the data
        df_master, feature_cols = await fetch_and_prepare_data(supabase, granularity)
            
        # Step 2: Initialize and train the predictor with the dynamic config
        predictor = MultivariateLSTMPredictor(look_back=look_back)
        predictor.train(df_master, feature_columns=feature_cols, epochs=epochs, batch_size=batch_size)
        
        # Step 3: Save all the necessary artifacts for later use in predictions
        predictor.save_artifacts(granularity=granularity, feature_columns=feature_cols)
        
        print(f"\n--- ✅ Pipeline for Granularity '{granularity}' Finished Successfully! ---")

    except Exception as e:
        print("\n--- ❌ An error occurred during the training pipeline! ---")
        traceback.print_exc()

if __name__ == '__main__':
    # --- THIS IS THE NEW PART: COMMAND-LINE ARGUMENT PARSING ---
    
    # 1. Create a parser object
    parser = argparse.ArgumentParser(description="Train a forecasting model with a specific granularity.")
    
    # 2. Add an argument for granularity
    parser.add_argument(
        "granularity", 
        type=str, 
        choices=['M', 'Y'],
        help="The granularity for the model training: 'M' for Monthly or 'Y' for Yearly."
    )
    
    # 3. Parse the arguments from the command line
    args = parser.parse_args()

    # 4. Define the configurations for each granularity
    if args.granularity == 'M':
        TRAINING_CONFIG = {
            'granularity': 'M',
            'look_back': 12,
            'epochs': 75,
            'batch_size': 32
        }
    elif args.granularity == 'Y':
        TRAINING_CONFIG = {
            'granularity': 'Y',
            'look_back': 5,  # Shorter look_back for yearly data
            'epochs': 100, # More epochs might be needed for less data
            'batch_size': 8 # Smaller batch size for a smaller dataset
        }
        
    # 5. Run the main async function with the selected configuration
    asyncio.run(main(config=TRAINING_CONFIG))

if __name__ == '__main__':
    # We use asyncio.run() because our data fetching function is asynchronous
    asyncio.run(main(c))