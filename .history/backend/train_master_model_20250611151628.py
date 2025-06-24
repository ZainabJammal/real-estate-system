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
# ===================================================================
# Adjust these parameters as needed
LOOK_BACK = 12      # Use 12 periods (months/years) of data to predict the next one.
GRANULARITY = 'M'   # 'M' for Monthly, 'Y' for Yearly.
EPOCHS = 75         # Number of training cycles. 50-100 is a good starting range.
BATCH_SIZE = 32     # Number of training samples to work through before updating weights.

# ===================================================================
#                       1. DATABASE AND DATA PREPARATION
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
    """
    print(f"Fetching data for all cities with granularity: {granularity}...")
    
    # Assumes table is 'transactions' with 'created_at'(text), 'city'(text), 'value'(number)
    response = supabase_client.table('transactions').select("created_at, city, value").execute()
    
    if not response.data:
        raise ValueError("No data returned from Supabase. Check table name and permissions.")
        
    df = pd.DataFrame(response.data)
    
    # --- CRITICAL: Convert Text Dates to Datetime Objects ---
    print("Converting 'created_at' text column to datetime objects...")
    # `pd.to_datetime` is smart and can parse most common date formats.
    df['date'] = pd.to_datetime(df['created_at'])
    df.drop(columns=['created_at'], inplace=True) # Drop the old text column
    
    # Sort by date to ensure the timeline is correct before processing.
    df.sort_values(by='date', inplace=True)
    
    # --- Aggregate Data by Granularity ---
    print(f"Aggregating transaction values by {'Month' if granularity == 'M' else 'Year'}...")
    df_resampled = df.groupby('city').apply(
        lambda x: x.set_index('date').resample(granularity)['value'].sum()
    ).reset_index()
    
    df_resampled.rename(columns={'level_1': 'date'}, inplace=True)
    
    # Keep the original city name for creating sequences correctly later.
    df_resampled['city_name_original'] = df_resampled['city']
    
    print(f"Data shape after resampling: {df_resampled.shape}")

    # --- One-Hot Encoding for the 'city' feature ---
    print("Applying one-hot encoding to 'city' column...")
    df_encoded = pd.get_dummies(df_resampled, columns=['city'], prefix='city')
    
    # Define the final list of features the model will be trained on.
    feature_columns = ['value'] + [col for col in df_encoded if col.startswith('city_')]
    
    print(f"Model will be trained on these {len(feature_columns)} features: {feature_columns}")
    
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
        """Trains the multivariate LSTM model."""
        n_features = len(feature_columns)
        
        # Fit the scaler on the 'value' and all one-hot-encoded city columns.
        # This ensures consistent scaling across all data.
        self.scaler.fit(df[feature_columns].values.astype('float32'))
        
        all_X, all_Y = [], []
        # Group by the original city name to create valid, continuous sequences.
        for city_name, city_group in df.groupby('city_name_original'):
            print(f"Creating training sequences for city: {city_name}...")
            # Use the already-fitted scaler to transform this city's data.
            scaled_city_dataset = self.scaler.transform(city_group[feature_columns].values.astype('float32'))
            
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
            Dense(1) # Output layer with 1 neuron to predict the 'value'
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

async def main():
    """Main function to orchestrate the entire training pipeline."""
    print("--- Starting Master Model Training Pipeline ---")
    supabase = get_supabase_client()
    try:
        # Step 1: Fetch and prepare the data
        df_master, feature_cols = await fetch_and_prepare_data(supabase, GRANULARITY)
            
        # Step 2: Initialize and train the predictor
        predictor = MultivariateLSTMPredictor(look_back=LOOK_BACK)
        predictor.train(df_master, feature_columns=feature_cols, epochs=EPOCHS, batch_size=BATCH_SIZE)
        
        # Step 3: Save all the necessary artifacts for later use in predictions
        predictor.save_artifacts(granularity=GRANULARITY, feature_columns=feature_cols)
        
        print("\n--- ✅ Pipeline Finished Successfully! Model artifacts are ready. ---")

    except Exception as e:
        print("\n--- ❌ An error occurred during the training pipeline! ---")
        traceback.print_exc()

if __name__ == '__main__':
    # We use asyncio.run() because our data fetching function is asynchronous
    asyncio.run(main())