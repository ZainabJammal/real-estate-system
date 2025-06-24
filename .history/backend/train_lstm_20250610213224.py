# train_master_model.py
import pandas as pd
import numpy as np
import json
from forecasting_lstm import LSTMPredictor, fetch_and_prepare_transaction_data # Assuming your class is in this file


# --- 1. Modify Data Preparation to handle all cities and One-Hot Encode ---

async def fetch_and_prepare_for_master_model(supabase_client):
    # Fetch data for ALL cities by passing city_name=None
    print("Fetching data for all cities...")
    df_all_cities = await fetch_and_prepare_transaction_data(supabase_client, city_name=None)
    
    if df_all_cities.empty:
        raise ValueError("No data returned for any city.")

    print(f"Original data shape for all cities: {df_all_cities.shape}")

    # --- THIS IS THE KEY STEP: ONE-HOT ENCODING ---
    # We need to keep track of the original city column for grouping
    df_all_cities['city_name'] = df_all_cities['city'] 
    
    # Use get_dummies for one-hot encoding
    print("Applying one-hot encoding to 'city' column...")
    df_encoded = pd.get_dummies(df_all_cities, columns=['city'], prefix='city')
    
    # Save the order of the encoded columns. This is CRITICAL for prediction.
    # The features will be the value + the one-hot encoded columns
    feature_columns = ['value'] + [col for col in df_encoded if col.startswith('city_')]
    with open('model_features.json', 'w') as f:
        json.dump(feature_columns, f)
    print(f"Saved feature columns to model_features.json: {feature_columns}")

    return df_encoded, feature_columns


# --- 2. Modify LSTMPredictor to handle multivariate input ---
# You need to update your LSTMPredictor class to handle multiple features.
# I'll write the necessary modifications here for clarity.

class MultivariateLSTMPredictor(LSTMPredictor):
    def _create_dataset(self, dataset, n_features):
        """
        Modified to handle multiple input features.
        `dataset` is now a 2D numpy array of shape (n_samples, n_features)
        """
        dataX, dataY = [], []
        # We use the first column (value) as the target to predict
        target_col_index = 0 
        
        for i in range(len(dataset) - self.look_back):
            # Input sequence includes all features
            a = dataset[i:(i + self.look_back), :] 
            dataX.append(a)
            # Output is just the value from the target column
            dataY.append(dataset[i + self.look_back, target_col_index])
        
        return np.array(dataX), np.array(dataY)

    def train(self, df, feature_columns, value_column='value', epochs=50, batch_size=32):
        # The number of features is the number of columns we are using
        n_features = len(feature_columns)
        
        # Scale all feature columns
        dataset = df[feature_columns].values.astype('float32')
        scaled_dataset = self.scaler.fit_transform(dataset)
        
        trainX, trainY = self._create_dataset(scaled_dataset, n_features)
        
        # Reshape is already correct from _create_dataset for LSTM
        # trainX shape will be (samples, look_back, n_features)

        # Build the model with the correct input shape
        self.model = Sequential([
            LSTM(64, input_shape=(self.look_back, n_features)), # Key change here
            Dense(32, activation='relu'),
            Dense(1)
        ])
        self.model.compile(loss='mean_squared_error', optimizer='adam')
        self.model.summary()

        print(f"Starting model.fit with trainX shape: {trainX.shape}")
        self.model.fit(trainX, trainY, epochs=epochs, batch_size=batch_size, verbose=1)
        self.is_trained = True
        
        # --- 3. SAVE THE MODEL ---
        self.model.save("master_transaction_model.h5")
        print("Master model saved to master_transaction_model.h5")

# --- Main Execution Block ---
async def main():
    mock_supabase_client = Mock() # Replace with your actual client
    df_master, feature_cols = await fetch_and_prepare_for_master_model(mock_supabase_client)
    
    # Note: A proper implementation would need to handle sequences on a per-city basis
    # so that a sequence doesn't cross from one city's data to another's.
    # For this example, we'll train on the whole shuffled dataset.
    # A robust solution would group by city, create sequences for each, then concatenate.
    
    predictor = MultivariateLSTMPredictor(look_back=12)
    predictor.train(df_master, feature_columns=feature_cols, epochs=10)

if __name__ == '__main__':
    import asyncio
    # Mocking the data fetching for a runnable example
    async def mock_fetch(*args, **kwargs):
        dates = pd.to_datetime(pd.date_range(start='2022-01-01', periods=24, freq='ME'))
        df1 = pd.DataFrame({'date': dates, 'value': np.sin(np.linspace(0, 12, 24))*20+50, 'city': 'London'})
        df2 = pd.DataFrame({'date': dates, 'value': np.cos(np.linspace(0, 12, 24))*15+40, 'city': 'Paris'})
        return pd.concat([df1, df2]).sort_values('date').reset_index(drop=True)
    fetch_and_prepare_transaction_data = mock_fetch

    asyncio.run(main())