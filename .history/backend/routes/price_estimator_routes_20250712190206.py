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

# This will hold our single, loaded model pipeline
PIPELINE = None

SUPABASE_URL: str = os.environ.get("SUPABASE_URL")
SUPABASE_KEY: str = os.environ.get("SUPABASE_KEY")

if not SUPABASE_URL or not SUPABASE_KEY:
    raise ValueError("FATAL ERROR: SUPABASE_URL and SUPABASE_KEY must be set in your .env file.")

# Initialize the Supabase client
supabase: Client = create_client(SUPABASE_URL, SUPABASE_KEY)
print("-> Supabase client initialized.")


load_dotenv() 

# SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
# BACKEND_DIR = os.path.dirname(SCRIPT_DIR)


price_estimator_bp = Blueprint('price_estimator', __name__)

# --- 3. BLUEPRINT INITIALIZATION AND MODEL LOADING ---
def init_price_estimator_routes(model_dir):
    """Initializes the blueprint and loads the trained apartment model."""
    global PIPELINE
    try:
      
        script_path = os.path.abspath(__file__)
        routes_dir = os.path.dirname(script_path)
        backend_dir = os.path.dirname(routes_dir)
        model_dir = os.path.join(backend_dir, 'final_model_output_v2')
        
        # --- Load the specific model you trained ---
        model_path = os.path.join(model_dir, 'model_apartment_size_category.joblib')
        
        print("--- Loading Price Estimation Model ---")
        print(f"-> Attempting to load model from: {model_path}")
        
        PIPELINE = joblib.load(model_path)
        
        print("-> Successfully loaded: 'model_apartment_size_category.joblib'")
        
    except FileNotFoundError as e:
        print(f"FATAL ERROR: Could not find the model file. {e}")
    except Exception as e:
        print(f"FATAL ERROR during model loading: {e}")
        traceback.print_exc()
       
    return price_estimator_bp


    # --- 4. DEFINE THE API ENDPOINT ---
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
        # --- a. Get User Input ---
        data = await request.get_json()
        print(f"[INPUT] Received data: {data}")

        # --- b. Validate Input ---
        # These are the exact features the new model was trained on.
        required_fields = [
            'province', 'district', 'size_category', 
            'bedrooms', 'bathrooms', 'latitude', 'longitude'
        ]
        
        if not all(field in data for field in required_fields):
            print(f"[ERROR] Missing required fields. Expected: {required_fields}")
            return jsonify({"error": "Missing required fields in request body."}), 400

        # --- c. Prepare Data for Model ---
        # The frontend sends all the necessary data. We just need to put it in a DataFrame.
        # No complex feature engineering is needed here anymore!
        input_df = pd.DataFrame([data])
        
        # The pipeline's preprocessor will handle everything (one-hot encoding, scaling).
        print(f"[PROCESS] Data prepared for pipeline:\n{input_df}")

        # --- d. Make Prediction ---
        # The pipeline automatically selects the correct columns and processes them.
        prediction_log = PIPELINE.predict(input_df)

        # Convert the log-price prediction back to dollars
        prediction_dollars = np.expm1(prediction_log)[0]
        print(f"[RESULT] Predicted Log Price: {prediction_log[0]:.4f}, Converted Price: ${prediction_dollars:,.2f}")

        # --- e. Format and Return Response ---
        # Create a plausible price range for the UI
        price_range_low = prediction_dollars * 0.90 # e.g., -10%
        price_range_high = prediction_dollars * 1.10 # e.g., +10%
        
        response_data = {
            "status": "success",
            "estimated_price": round(prediction_dollars, -2), # Round to nearest 100
            "price_range_low": round(price_range_low, -2),
            "price_range_high": round(price_range_high, -2),
        }
        
        print(f"[OUTPUT] Sending response: {response_data}")
        print("--- [END] Prediction Request Finished ---\n")

        return jsonify(response_data) 
    
    except Exception as e:
        print(f"[ERROR] An exception occurred during prediction: {e}")
        traceback.print_exc()
        return jsonify({"error": "An internal server error occurred.", "details": str(e)}), 500