# train_master_model.py
import os
import pandas as pd
import numpy as np
import json
import asyncio
from dotenv import load_dotenv
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from sklearn.preprocessing import MinMaxScaler
# Assuming your original classes/functions are in forecasting_lstm.py in the root of the backend folder
# Adjust the import path if you've moved the file
from routes.forecasting_lstm import LSTMPredictor, fetch_and_prepare_transaction_data
from supabase.client import Client, AsyncClient 
# We don't need create_client anymore for async

async def create_supabase() -> AsyncClient:
    load_dotenv()
    url = os.environ.get("SUPABASE_URL")
    key = os.environ.get("SUPABASE_KEY")
    if not url or not key:
        raise ValueError("Supabase URL and Key must be set in your .env file.")
    
    # --- THIS IS THE FIX ---
    # Instantiate the AsyncClient class directly. This is now the correct way.
    # The instantiation itself is synchronous, so we remove 'await'.
    return AsyncClient(supabase_url=url, supabase_key=key)# --- 1. Data Preparation (Same as before, this part is good) ---
async def fetch_and_prepare_for_master_model(supabase_client):
    print("Fetching data for all cities...")
    # Passing city_name=None fetches data for ALL cities
    df_all_cities = await fetch_and_prepare_transaction_data(supabase_client, city_name=None)
    
    if df_all_cities.empty:
        raise ValueError("No data returned for any city.")
    print(f"Original data shape for all cities: {df_all_cities.shape}")

    # Keep track of the original city column for grouping before it gets encoded
    # Ensure you are using the correct column name from the fetch function.
    # The fetch function returns a 'city' column.
    df_all_cities['city_name_original'] = df_all_cities['city'] 
    
    print("Applying one-hot encoding to 'city' column...")
    df_encoded = pd.get_dummies(df_all_cities, columns=['city'], prefix='city')
    
    feature_columns = ['value'] + [col for col in df_encoded if col.startswith('city_')]
    
    # Save files to the root of the backend folder
    with open('model_features.json', 'w') as f:
        json.dump(feature_columns, f)
    print(f"Saved feature columns to model_features.json: {feature_columns}")

    return df_encoded, feature_columns

# --- 2. Improved MultivariateLSTMPredictor Class ---
class MultivariateLSTMPredictor(LSTMPredictor):
    def _create_sequences_for_city(self, city_data, n_features):
        """Helper to create sequences for a single city's data."""
        dataX, dataY = [], []
        target_col_index = 0  # 'value' is the first column
        for i in range(len(city_data) - self.look_back):
            a = city_data[i:(i + self.look_back), :]
            dataX.append(a)
            dataY.append(city_data[i + self.look_back, target_col_index])
        return np.array(dataX), np.array(dataY)

    def train(self, df, feature_columns, epochs=50, batch_size=32):
        n_features = len(feature_columns)
        dataset = df[feature_columns].values.astype('float32')
        
        # Fit the scaler on the entire dataset
        self.scaler.fit(dataset)
        # We will save the scaler later

        # --- CRITICAL IMPROVEMENT: Create sequences per city ---
        all_X, all_Y = [], []
        
        # Group by the original city name before it was one-hot encoded
        for city_name, city_group in df.groupby('city_name_original'):
            print(f"Creating sequences for city: {city_name}...")
            city_dataset = city_group[feature_columns].values.astype('float32')
            
            # Use the already fitted scaler to transform this city's data
            scaled_city_dataset = self.scaler.transform(city_dataset)
            
            city_X, city_Y = self._create_sequences_for_city(scaled_city_dataset, n_features)
            
            if len(city_X) > 0:
                all_X.append(city_X)
                all_Y.append(city_Y)

        if not all_X:
            raise ValueError("Could not create any training sequences. Check data length for each city.")
            
        # Combine sequences from all cities into one large training set
        trainX = np.concatenate(all_X, axis=0)
        trainY = np.concatenate(all_Y, axis=0)
        print(f"Total training sequences from all cities: {len(trainX)}")

        self.model = Sequential([
            LSTM(64, input_shape=(self.look_back, n_features), return_sequences=True),
            Dropout(0.2),
            LSTM(32),
            Dropout(0.2),
            Dense(1)
        ])
        self.model.compile(loss='mean_squared_error', optimizer='adam')
        self.model.summary()

        print(f"Starting master model training with final trainX shape: {trainX.shape}")
        self.model.fit(trainX, trainY, epochs=epochs, batch_size=batch_size, verbose=1)
        self.is_trained = True
        
        # --- 3. SAVE THE MODEL AND THE SCALER ---
        self.model.save("master_transaction_model.h5")
        import pickle
        with open('master_scaler.pkl', 'wb') as f:
            pickle.dump(self.scaler, f)
            
        print("Master model and scaler saved successfully.")

# --- 3. Main Execution Block to Run Training ---
async def main():
    """Main function to connect to DB and run the training pipeline."""
    supabase = None
    try:
        supabase = await create_supabase()
        df_master, feature_cols = await fetch_and_prepare_for_master_model(supabase)
        
        predictor = MultivariateLSTMPredictor(look_back=12)
        predictor.train(df_master, feature_columns=feature_cols, epochs=25``) # Use desired epochs
        
    except Exception as e:
        print(f"An error occurred during the training process: {e}")
        import traceback
        traceback.print_exc()
    finally:
        if supabase:
            # await supabase.close()
            pass

if __name__ == '__main__':
    print("--- Starting Master Model Training Script ---")
    asyncio.run(main())
    print("--- Training Script Finished ---")