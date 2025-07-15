# import os
# import joblib
# import pandas as pd
# import numpy as np
# import traceback
# from quart import Blueprint, request, jsonify
# from supabase import create_client, Client
# from dotenv import load_dotenv

# # This global dictionary will be populated by the init function
# FORECAST_MODELS = {}
# supabase = None
# forecasting_bp = Blueprint('forecasting', __name__)
# # --- NEW: An 'init' function to set up the blueprint and load everything ---
# def init_forecasting_routes(model_dir):
#     """Initializes the blueprint, loads models, and sets up the Supabase client."""
    
#     forecasting_bp = Blueprint('forecasting', __name__)
#     # global supabase # Use the global supabase variable
   
    
#     # --- Load Models ---
#     try:
#         print("--- Loading Forecasting Models ---")
#         print(f"Attempting to load models from: {os.path.abspath(model_dir)}")
        
#         FORECAST_MODELS['forecast_model'] = joblib.load(os.path.join(model_dir, 'forecast_model.joblib'))
#         FORECAST_MODELS['forecast_columns'] = list(pd.read_json(os.path.join(model_dir, 'model_columns.json'), typ='series'))
#         print("-> Forecasting models loaded successfully.")

#     except FileNotFoundError as e:
#         print(f"FATAL ERROR loading forecast models: {e}")
#         # We don't need to return None, just leave the dictionary empty
    
#     # --- Supabase Connection ---
#     # Construct path to the .env file in the backend root
#     backend_dir = os.path.dirname(model_dir) # Go up one level from 'models/forecasting'
#     load_dotenv(os.path.join(backend_dir, '.env')) 
    
#     SUPABASE_URL = os.environ.get("SUPABASE_URL")
#     SUPABASE_KEY = os.environ.get("SUPABASE_KEY")
#     if SUPABASE_URL and SUPABASE_KEY:
#         supabase = create_client(SUPABASE_URL, SUPABASE_KEY)
#         print("-> Supabase client for forecasting initialized.")
#     else:
#         print("-> Supabase credentials not found, forecasting endpoint will be disabled.")


# # Define global constants
# # GROUPING_KEY must match the column name in your training data AND your database
# GROUPING_KEY = 'city' 
# TARGET_COL = 'transaction_value'
# FORECAST_HORIZON_MONTHS = 60

# def map_user_selection_to_city(selection_string: str) -> str:
#     """
#     Maps the user's dropdown selection to the specific custom region names
#     used in the database, ensuring it's lowercase for matching.
#     """
#     if "Tripoli, Akkar" in selection_string:
#         return "Tripoli"
#     if "Baabda, Aley, Chouf" in selection_string:
#         return "Baabda"
#     if "Kesrouan, Jbeil" in selection_string:
#         return "Kesrouan"
#     return selection_string.lower()

# def create_features(df):
#     """Optimal features for a small dataset with a long forecast horizon."""
#     df_features = df.copy()
#     df_features['year'] = df_features.index.year
#     df_features['month_sin'] = np.sin(2 * np.pi * df_features.index.month / 12.0)
#     df_features['month_cos'] = np.cos(2 * np.pi * df_features.index.month / 12.0)
#     df_features['is_december'] = (df_features.index.month == 12).astype(int)
#     df_features['is_summer_peak'] = (df_features.index.month.isin([7, 8])).astype(int)
#     return df_features



#     # # --- Define the endpoint within the init function ---
#     # @forecasting_bp.route("/forecast/xgboost/<string:city_name>", methods=["GET"])
#     # async def forecast_with_xgboost(city_name: str):
#     #     if not FORECAST_MODELS or not supabase:
#     #         return jsonify({"error": "Forecasting service or database is not available."}), 503

#     #     try:
#     #         city_name_lower = city_name.lower()
#     #         response = supabase.table('agg_trans').select('date').ilike('city', city_name_lower).order('date', desc=True).limit(1).execute()
            
#     #         if not response.data:
#     #             return jsonify({"error": f"No historical data for '{city_name}'."}), 404
            
#     #         last_date_str = response.data[0]['date']
#     #         parts = last_date_str.split('-')
#     #         last_historical_date = pd.to_datetime('01-' + parts[1] + '-' + parts[0], format='%d-%b-%y')

#     #         future_dates = pd.date_range(start=last_historical_date + pd.DateOffset(months=1), periods=60, freq='MS')
#     #         future_df = pd.DataFrame(index=future_dates)
#     #         future_df['city'] = city_name_lower
#     #         features_for_pred = create_forecast_features(future_df)
            
#     #         features_encoded = pd.get_dummies(features_for_pred, columns=['city'])
#     #         features_aligned = features_encoded.reindex(columns=FORECAST_MODELS['forecast_columns'], fill_value=0)
            
#     #         predictions = FORECAST_MODELS['forecast_model'].predict(features_aligned)

#     #         forecast_json = [{'date': date.strftime('%Y-%m-%d'), 'predicted_value': float(val)} for date, val in zip(future_dates, predictions)]
            
#     #         return jsonify({ "city_forecasted": city_name, "monthly_forecast": forecast_json })

#     #     except Exception as e:
#     #         traceback.print_exc()
#     #         return jsonify({"error": "An internal server error occurred."}), 500
            
#     # return forecasting_bp


# # --- 4. API ENDPOINT (Simplified for "Best Guess" model) ---
# @forecasting_bp.route("/forecast/xgboost/<string:user_selection>", methods=["GET"])
# async def forecast_with_xgboost(user_selection: str):
#     """
#     Generates a 5-year 'Best Guess' forecast by predicting all future
#     months in a single batch.
#     """
#     print(f"\nReceived 'Best Guess' forecast request for: '{user_selection}'")
#     # if 'forecast_model' not in ML_ARTIFACTS or not supabase:
#     #     return jsonify({"error": "Forecasting service or database is not available."}), 503


#     try:
#         city_name = map_user_selection_to_city(user_selection)
#         print(f"-> Mapped to city: '{city_name}'")

#         # step A: fetch histo data first
#         print(f"-> Querying Supabase for '{city_name}' historical data...")
#         response = supabase.table('agg_trans').select('*').ilike(GROUPING_KEY, city_name).order('date').execute()
        
#           # --- START: NEW "SEE EVERYTHING" DEBUG BLOCK ---
#         print("\n--- RAW SUPABASE RESPONSE DEBUG ---")
#         print(f"Type of response object: {type(response)}")
#         print(f"Response has 'data' attribute: {'data' in dir(response)}")
        
#         # Check if the response contains data and print it
#         if response.data:
#             print(f"Number of records returned: {len(response.data)}")
#             # Print the first record to see its structure
#             print(f"First record received from Supabase: {response.data[0]}") 
#         else:
#             print("!!! CRITICAL: `response.data` is EMPTY or does not exist.")
#             print(f"Full response object: {response}")
        
#         print("-----------------------------------\n")
#         # --- END OF DEBUG BLOCK ---

#         if not response.data:
#             return jsonify({"error": f"No historical data found for city '{city_name}' in the database."}), 404
        
#         # Prepare the historical data DataFrame
#         hist_df = pd.DataFrame(response.data)

#         parts = hist_df['date'].str.split('-', expand=True)
#         hist_df['date'] = pd.to_datetime('01-' + parts[1] + '-' + parts[0], format='%d-%b-%y')
#         city_history_df = hist_df.set_index('date')

#         # Standardize the grouping key column to lowercase to prevent case-mismatch
#         city_history_df[GROUPING_KEY] = city_history_df[GROUPING_KEY].str.lower()


#         # Step B: Create a DataFrame for all 60 future months

#          # Find the last date from the historical data we just fetched.
#         last_historical_date = hist_df['date'].iloc[-1].year

#          # Start the forecast on the first day of the month AFTER the last historical date.
#         start_year = last_historical_date + 1
#         start_date = pd.to_datetime(f'{start_year}-01-01')
#         future_dates = pd.date_range(start=start_date, periods=FORECAST_HORIZON_MONTHS, freq='MS')

#         future_df = pd.DataFrame(index=future_dates)
#         # Create features for the future DataFrame
#         future_df[GROUPING_KEY] = city_name
#         features_for_pred = create_features(future_df)

        
#         # Step 4: One-hot encode the city and align columns
#         features_encoded = pd.get_dummies(features_for_pred, columns=[GROUPING_KEY])
#         features_aligned = features_encoded.reindex(columns=FORECAST_MODELS, fill_value=0)
        
#         # Step 5: Make predictions for ALL 60 months at once
#         predictions = FORECAST_MODELS['forecast_model'].predict(features_aligned)
#         print(f"-> Generated {len(predictions)} future predictions in a single batch.")

#         # --- Part C: Format the Final JSON Response ---

#         historical_json = hist_df.to_dict(orient='records')
        
#         monthly_forecast_df = pd.DataFrame({'date': future_dates, 'predicted_value': predictions})
#         yearly_forecast_df = monthly_forecast_df.resample('YE', on='date')['predicted_value'].sum().to_frame()
#         # features_aligned = features_encoded.reindex(columns=ML_ARTIFACTS['forecast_columns'], fill_value=0)
        
#         # predictions = ML_ARTIFACTS['forecast_model'].predict(features_aligned)

#         monthly_json = monthly_forecast_df.to_dict(orient='records')
#         yearly_json = yearly_forecast_df.reset_index().rename(columns={'date': 'year', 'predicted_value': 'total_value'}).to_dict(orient='records')
#         for item in yearly_json: item['year'] = item['year'].year

#         print(f"-> Successfully generated forecast for '{city_name}'.")
#         return jsonify({
#             "city_forecasted": user_selection,
#             "historical_data": historical_json,
#             "monthly_forecast": monthly_json,
#             "yearly_forecast": yearly_json
#         })

#     except Exception as e:
#         traceback.print_exc()
#         return jsonify({"error": "An internal server error occurred."}), 500


