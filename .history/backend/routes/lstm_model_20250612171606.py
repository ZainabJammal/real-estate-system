import traceback
import os
import pickle
import pandas as pd
import numpy as np
from quart import Blueprint, jsonify, request
from tensorflow.keras.models import load_model
import json
from db_connect import create_supabase

# Create a Blueprint for LSTM routes
lstm_routes = Blueprint('lstm', __name__)

# Cache for loaded artifacts
LOADED_ARTIFACTS = {}

def get_artifacts(granularity: str):
    """Loads and caches the model, scaler, and features for a given granularity."""
    if granularity in LOADED_ARTIFACTS:
        return LOADED_ARTIFACTS[granularity]

    print(f"Loading artifacts for granularity: {granularity} for the first time...")
    suffix = "monthly" if granularity == 'M' else "yearly"
    
    # Use the model_artifacts directory created by train_master_model.py
    artifacts_path = "model_artifacts"
    
    # 1. Load Model
    model_path = os.path.join(artifacts_path, f"master_model_{suffix}.h5")
    model = load_model(model_path)
    
    # 2. Load Scaler
    scaler_path = os.path.join(artifacts_path, f"master_scaler_{suffix}.pkl")
    with open(scaler_path, 'rb') as f:
        scaler = pickle.load(f)
        
    # 3. Load Feature List
    features_path = os.path.join(artifacts_path, f"model_features_{suffix}.json")
    with open(features_path, 'r') as f:
        features = json.load(f)

    artifacts = {
        "model": model,
        "scaler": scaler,
        "features": features,
        "look_back": model.input_shape[1]
    }
    LOADED_ARTIFACTS[granularity] = artifacts
    print(f"✅ Artifacts for '{granularity}' loaded and cached.")
    return artifacts

@lstm_routes.route("/api/historical_data", methods=["POST"])
async def get_historical_data():
    """
    Fetch historical data for a specific city.
    """
    print("\n--- HISTORICAL DATA REQUEST RECEIVED ---")
    try:
        data = await request.get_json()
        city_name = data.get('city')
        granularity = data.get('granularity', 'Y')
        start_year = data.get('start_year', 2012)
        end_year = data.get('end_year', 2016)
        
        if not city_name:
            return jsonify({"error": "City name is required"}), 400
        
        print(f"[DEBUG] Fetching historical data for {city_name}, {granularity}, {start_year}-{end_year}")
        
        # Fetch data from Supabase
        supabase = await create_supabase()
        response = await supabase.table('transactions').select("date, transaction_value").eq("city", city_name).execute()
        
        if not response.data:
            return jsonify({"historical": []}), 200
        
        # Process data
        df = pd.DataFrame(response.data)
        df['date'] = pd.to_datetime(df['date'])
        
        # Filter by year range
        df = df[(df['date'].dt.year >= start_year) & (df['date'].dt.year <= end_year)]
        
        # Resample by granularity
        df_resampled = df.set_index('date').resample(granularity)['transaction_value'].sum().reset_index()
        
        # Format response
        historical_data = [
            {"date": row['date'].strftime('%Y-%m-%d'), "transaction_value": row['transaction_value']}
            for _, row in df_resampled.iterrows()
        ]
        
        print(f"[DEBUG] Returning {len(historical_data)} historical data points")
        return jsonify({"historical": historical_data})
        
    except Exception as e:
        print(f"\n--- ❌ HISTORICAL DATA ERROR ---")
        traceback.print_exc()
        return jsonify({"error": f"An internal server error occurred: {str(e)}"}), 500

@lstm_routes.route("/api/all_historical_data", methods=["POST"])
async def get_all_historical_data():
    """
    Fetch historical data for all cities.
    """
    print("\n--- ALL HISTORICAL DATA REQUEST RECEIVED ---")
    try:
        data = await request.get_json()
        granularity = data.get('granularity', 'Y')
        start_year = data.get('start_year', 2012)
        end_year = data.get('end_year', 2016)
        
        print(f"[DEBUG] Fetching all historical data, {granularity}, {start_year}-{end_year}")
        
        # Fetch data from Supabase
        supabase = await create_supabase()
        response = await supabase.table('transactions').select("date, city, transaction_value").execute()
        
        if not response.data:
            return jsonify({"historical": {}}), 200
        
        # Process data
        df = pd.DataFrame(response.data)
        df['date'] = pd.to_datetime(df['date'])
        
        # Filter by year range
        df = df[(df['date'].dt.year >= start_year) & (df['date'].dt.year <= end_year)]
        
        # Resample by granularity for each city
        df_grouped = df.set_index('date').groupby('city')['transaction_value'].resample(granularity).sum()
        df_resampled = df_grouped.reset_index()
        
        # Group by city
        historical_by_city = {}
        for city in df_resampled['city'].unique():
            city_data = df_resampled[df_resampled['city'] == city]
            historical_by_city[city] = [
                {"date": row['date'].strftime('%Y-%m-%d'), "transaction_value": row['transaction_value']}
                for _, row in city_data.iterrows()
            ]
        
        print(f"[DEBUG] Returning historical data for {len(historical_by_city)} cities")
        return jsonify({"historical": historical_by_city})
        
    except Exception as e:
        print(f"\n--- ❌ ALL HISTORICAL DATA ERROR ---")
        traceback.print_exc()
        return jsonify({"error": f"An internal server error occurred: {str(e)}"}), 500

@lstm_routes.route("/api/lstm_forecast", methods=["POST"])
async def lstm_forecast():
    """
    LSTM forecasting endpoint that predicts next 5 years of transaction values for all cities.
    """
    print("\n--- LSTM FORECAST REQUEST RECEIVED ---")
    try:
        # Parse request data
        data = await request.get_json()
        granularity = data.get('granularity', 'Y')  # Default to yearly
        
        # Validate granularity
        if granularity not in ['Y', 'M']:
            return jsonify({"error": "Invalid granularity. Must be 'Y' for Yearly or 'M' for Monthly."}), 400
        
        # Set forecast periods
        future_periods = 5 if granularity == 'Y' else 60  # 5 years or 60 months
        
        print(f"[DEBUG] LSTM Forecast - granularity={granularity}, periods={future_periods}")
        
        # Load artifacts
        artifacts = get_artifacts(granularity)
        model = artifacts['model']
        scaler = artifacts['scaler']
        feature_columns = artifacts['features']
        look_back = artifacts['look_back']
        
        # Get all cities from feature columns
        city_columns = [col for col in feature_columns if col.startswith('city_')]
        cities = [col.replace('city_', '') for col in city_columns]
        
        print(f"[DEBUG] Found {len(cities)} cities in model: {cities}")
        
        # Fetch data from Supabase
        supabase = await create_supabase()
        response = await supabase.table('transactions').select("date, city, transaction_value").execute()
        
        if not response.data:
            return jsonify({"error": "No data found in transactions table."}), 404
        
        # Process data
        df = pd.DataFrame(response.data)
        df['date'] = pd.to_datetime(df['date'])
        
        # Resample data by granularity
        df_grouped = df.set_index('date').groupby('city')['transaction_value'].resample(granularity).sum()
        df_resampled = df_grouped.reset_index()
        
        # Generate forecasts for all cities
        all_forecasts = {}
        
        for city in cities:
            print(f"[DEBUG] Generating forecast for city: {city}")
            
            # Get historical data for this city
            city_data = df_resampled[df_resampled['city'] == city].sort_values('date')
            
            if len(city_data) < look_back:
                print(f"[WARNING] Not enough data for {city}. Skipping.")
                continue
            
            # Get last look_back periods
            historical_sequence = city_data.tail(look_back)
            last_date = historical_sequence['date'].max()
            
            # Prepare input for model
            input_df = pd.DataFrame(columns=feature_columns)
            input_df['transaction_value'] = historical_sequence['transaction_value'].values
            input_df.fillna(0, inplace=True)
            
            # Set city one-hot encoding
            city_col_name = f"city_{city}"
            if city_col_name in feature_columns:
                input_df[city_col_name] = 1
            
            # Scale input
            scaled_input = scaler.transform(input_df[feature_columns])
            current_input = scaled_input.reshape((1, look_back, len(feature_columns)))
            
            # Generate predictions
            future_predictions_scaled = []
            for i in range(future_periods):
                pred_scaled = model.predict(current_input, verbose=0)[0, 0]
                future_predictions_scaled.append(pred_scaled)
                
                # Update input for next prediction
                new_step = current_input[0, -1, :].copy()
                new_step[0] = pred_scaled
                # Maintain city encoding
                if city_col_name in feature_columns:
                    city_col_index = feature_columns.index(city_col_name)
                    new_step[city_col_index] = 1
                
                new_step = new_step.reshape(1, 1, len(feature_columns))
                current_input = np.append(current_input[:, 1:, :], new_step, axis=1)
            
            # Inverse transform predictions
            dummy_array = np.zeros((len(future_predictions_scaled), len(feature_columns)))
            dummy_array[:, 0] = future_predictions_scaled
            final_predictions = scaler.inverse_transform(dummy_array)[:, 0]
            
            # Generate future dates
            future_dates = []
            current_date = last_date
            for _ in range(future_periods):
                if granularity == 'M':
                    current_date = current_date + pd.DateOffset(months=1) + pd.tseries.offsets.MonthEnd(0)
                elif granularity == 'Y':
                    current_date = current_date + pd.DateOffset(years=1) + pd.tseries.offsets.YearEnd(0)
                future_dates.append(current_date)
            
            # Format forecast for this city
            city_forecast = [
                {"date": date.strftime('%Y-%m-%d'), "predicted_value": round(value, 2)}
                for date, value in zip(future_dates, final_predictions)
            ]
            
            all_forecasts[city] = city_forecast
        
        print(f"[DEBUG] Generated forecasts for {len(all_forecasts)} cities")
        print("--- LSTM FORECAST REQUEST COMPLETED SUCCESSFULLY ---")
        
        return jsonify({
            "granularity": granularity,
            "forecast_periods": future_periods,
            "forecasts": all_forecasts
        })
        
    except Exception as e:
        print("\n--- ❌ LSTM FORECAST ERROR ---")
        traceback.print_exc()
        return jsonify({"error": f"An internal server error occurred: {str(e)}"}), 500

