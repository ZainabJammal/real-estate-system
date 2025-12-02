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


warnings.filterwarnings("ignore", category=UserWarning)


PIPELINE = None

SUPABASE_URL: str = os.environ.get("SUPABASE_URL")
SUPABASE_KEY: str = os.environ.get("SUPABASE_KEY")

if not SUPABASE_URL or not SUPABASE_KEY:
    raise ValueError("FATAL ERROR: SUPABASE_URL and SUPABASE_KEY must be set in your .env file.")


supabase: Client = create_client(SUPABASE_URL, SUPABASE_KEY)
print("-> Supabase client initialized.")


load_dotenv() 


price_estimator_bp = Blueprint('price_estimator', __name__)


def init_price_estimator_routes(model_dir):
    """Initializes the blueprint and loads the trained apartment model."""
    global PIPELINE
    try:
      
        script_path = os.path.abspath(__file__)
        routes_dir = os.path.dirname(script_path)
        backend_dir = os.path.dirname(routes_dir)
        model_dir = os.path.join(backend_dir, 'price_models')
        
     
        model_path = os.path.join(model_dir, 'final_api_ready_model.joblib')
        
        print("--- Loading Price Estimation Model ---")
        print(f"-> Attempting to load model from: {model_path}")
        
        PIPELINE = joblib.load(model_path)
        
        print("-> Successfully loaded: 'final_api_ready_model.joblib'")
        
    except FileNotFoundError as e:
        print(f"FATAL ERROR: Could not find the model file. {e}")
    except Exception as e:
        print(f"FATAL ERROR during model loading: {e}")
        traceback.print_exc()
       
    return price_estimator_bp



@price_estimator_bp.route("/estimate_fair_price", methods=["POST"])
async def predict_apartment_price():
    """
    Endpoint for apartment price prediction using the size-category model.
    This is much simpler as no feature engineering is needed at prediction time.
    """
    print("\n--- [START] New Apartment Price Prediction Request (Size Category Model) ---")

    if PIPELINE is None:
        print("[ERROR] Model is not loaded.")
        return jsonify({"error": "Price estimation model is not available."}), 503

    try:
       
        data = await request.get_json()
        print(f"[INPUT] Received data: {data}")

       
        required_fields = [
            'province', 'district', 'city', 'size_m2', 
            'bedrooms', 'bathrooms', 'latitude', 'longitude'
        ]
        
        if not all(field in data for field in required_fields):
            print(f"[ERROR] Missing required fields. Expected: {required_fields}")
            return jsonify({"error": "Missing required fields in request body."}), 400

        input_df = pd.DataFrame([data])
        
       
        print(f"[PROCESS] Data prepared for pipeline:\n{input_df}")

        for col in ['size_m2', 'bedrooms', 'bathrooms']:
            input_df[col] = pd.to_numeric(input_df[col], errors='coerce')

        print(f"[PROCESS] Engineering features from input...")

        input_df['log_size_m2'] = np.log1p(input_df['size_m2'])
        input_df['bed_bath_ratio'] = input_df['bedrooms'] / input_df['bathrooms']
        input_df['size_per_bedroom'] = input_df['size_m2'] / input_df['bedrooms']
   
        input_df.replace([np.inf, -np.inf], np.nan, inplace=True)
        
        print(f"[PROCESS] DataFrame after engineering:\n{input_df[['log_size_m2', 'bed_bath_ratio', 'size_per_bedroom']]}")

     
        prediction_log = PIPELINE.predict(input_df)

      
        prediction_dollars = np.expm1(prediction_log)[0]
        print(f"[RESULT] Predicted Log Price: {prediction_log[0]:.4f}, Converted Price: ${prediction_dollars:,.2f}")

       
        price_range_low = prediction_dollars * 0.90 
        price_range_high = prediction_dollars * 1.10 # e.g., +10%
        
        response_data = {
            "status": "success",
            "estimated_price": int(round(prediction_dollars, -2)), # Converted to a standard int
            "price_range_low": int(round(price_range_low, -2)),   # Converted to a standard int
            "price_range_high": int(round(price_range_high, -2)), # Converted to a standard int
        }
        
        print(f"[OUTPUT] Sending response: {response_data}")
        print("--- [END] Prediction Request Finished ---\n")

        return jsonify(response_data) 
    
    except Exception as e:
        print(f"[ERROR] An exception occurred during prediction: {e}")
        traceback.print_exc()
        return jsonify({"error": "An internal server error occurred.", "details": str(e)}), 500