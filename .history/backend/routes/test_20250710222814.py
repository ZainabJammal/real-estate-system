import os
import traceback
import joblib
import warnings
import pandas as pd
import numpy as np
from quart import Blueprint, request, jsonify
from dotenv import load_dotenv
from supabase import create_client, Client
from quart import Quart, jsonify, request
from quart_cors import cors


SUPABASE_URL: str = os.environ.get("SUPABASE_URL")
SUPABASE_KEY: str = os.environ.get("SUPABASE_KEY")

if not SUPABASE_URL or not SUPABASE_KEY:
    raise ValueError("FATAL ERROR: SUPABASE_URL and SUPABASE_KEY must be set in your .env file.")

# Initialize the Supabase client
supabase: Client = create_client(SUPABASE_URL, SUPABASE_KEY)
print("-> Supabase client initialized.")


load_dotenv() 


def fetchinfo():
    # Fetch the data from the Supabase table
    data = supabase.table('properties').select('*').execute()
    return data.data

price_estimator_bp = Blueprint("price_estimator_bp", __name__)
    # --- 4. DEFINE THE API ENDPOINT ---
@price_estimator_bp.route("/estimate_fair_price", methods=["POST"])
async def predict_apartment_price():
    """Endpoint specifically for apartment price prediction."""
    

    print("--- Received request at /estimate_fair_price ---")
    print("\n--- [START] New Apartment Price Prediction Request ---")

    if PIPELINE is None:
        print("[ERROR] Model is not loaded.")
        return jsonify({"error": "Price estimation model is not available due to a startup error."}), 503

    try:
        # --- a. Get User Input ---
        data = await request.get_json()
        print(f"[INPUT] Received data: {data}")
        input_df = pd.DataFrame([data])
        

        print("[PROCESS] Performing feature engineering on input data...")

        # Convert types to numeric, just in case
        numeric_cols = ['size_m2', 'bedrooms', 'bathrooms', 'latitude', 'longitude']
        for col in numeric_cols:
            input_df[col] = pd.to_numeric(input_df[col], errors='coerce')

        # Create 'size_per_room' feature
        input_df['size_per_room'] = input_df['size_m2'] / (input_df['bedrooms'] + input_df['bathrooms'] + 1)
        
        trained_districts = ['Beirut', 'El Metn', 'Kesrouane', 'Jbeil', 'Baabda', 'Aley', 'Batroun', 'Other']
        
        user_district = input_df['district'].iloc[0]

        # If the user's district is not one of the known major ones, classify it as 'Other'
        if user_district not in trained_districts:
            user_district_mapped = 'Other'
            print(f"[PROCESS] User district '{user_district}' mapped to 'Other'")
        else:
            user_district_mapped = user_district

        # NOW, create ALL interaction columns that the model expects.
        for dist in trained_districts:
            interaction_col_name = f'size_x_dist_{dist}'
            # If the user's mapped district matches this column, calculate the interaction.
            if dist == user_district_mapped:
                input_df[interaction_col_name] = input_df['size_m2']
            # Otherwise, set it to 0.
            else:
                input_df[interaction_col_name] = 0
        
        print("[PROCESS] Feature engineering complete. DataFrame has all required columns.")

        # --- c. Make Prediction ---
        print(f"[PROCESS] Input DataFrame for prediction:\n{input_df.head()}")
        print(f"[PROCESS] Making prediction for {user_district_mapped} district...")
        prediction_log = PIPELINE.predict(input_df)

        # Convert prediction back to dollars
        prediction_dollars = np.expm1(prediction_log)[0]
        print(f"[RESULT] Predicted Price: ${prediction_dollars:,.2f}")

        # --- d. Format and Return Response ---

        
        response_data = {
            "status": "success",
            "predicted_price_$": round(prediction_dollars, -3), # Use the calculated price
            "district": user_district_mapped
        }

        
        print(f"[OUTPUT] Sending response: {response_data}")
        print("--- [END] Prediction Request Finished ---\n")

        return jsonify(response_data) 
    
    except Exception as e:
        print(f"[ERROR] An exception occurred: {e}")
        traceback.print_exc()
        return jsonify({"error": "An internal server error occurred.", "details": str(e)}), 500