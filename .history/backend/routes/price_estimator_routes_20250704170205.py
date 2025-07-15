import os
import traceback
import joblib
import warnings
import pandas as pd
import numpy as np
from quart import Blueprint, request, jsonify
from db_connect import create_supabase
from supabase import create_client, Client
from dotenv import load_dotenv

# warnings.filterwarnings("ignore", category=UserWarning, message="X does not have valid feature names")
load_dotenv() 
SUPABASE_URL: str = os.environ.get("SUPABASE_URL")
SUPABASE_KEY: str = os.environ.get("SUPABASE_KEY")

if not SUPABASE_URL or not SUPABASE_KEY:
    raise ValueError("FATAL ERROR: SUPABASE_URL and SUPABASE_KEY must be set in your .env file.")

# Initialize the Supabase client
supabase: Client = create_client(SUPABASE_URL, SUPABASE_KEY)
print("-> Supabase client initialized.")

# SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
# BACKEND_DIR = os.path.dirname(SCRIPT_DIR)

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
BACKEND_DIR = os.path.dirname(SCRIPT_DIR)

FORECAST_MODEL_DIR = os.path.join(BACKEND_DIR, 'forecasting_models')

# SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(SCRIPT_DIR, 'forecast_model.joblib')
MODEL_COLS_PATH = os.path.join(SCRIPT_DIR, 'model_columns.json')

# This global dictionary will be populated by the init function
PRICE_MODELS = {}

def setup_environment():
    """Sets up directories and configurations for the script."""
    warnings.filterwarnings('ignore', category=UserWarning)
    pd.options.mode.chained_assignment = None
    
    # Get the absolute path of the currently running script (e.g., .../backend/routes/your_script.py)
    script_path = os.path.abspath(__file__)
    
    # Get the directory the script is in (e.g., .../backend/routes/)
    routes_dir = os.path.dirname(script_path)
    
    # Go UP ONE LEVEL to get the backend directory (e.g., .../backend/)
    backend_dir = os.path.dirname(routes_dir)

    # Now, build the paths relative to the `backend_dir`
    output_dir = os.path.join(backend_dir, 'final_model_output')
    data_path = os.path.join(backend_dir, 'properties.csv')
    
    os.makedirs(output_dir, exist_ok=True)
    
    print("="*60)
    print("Environment Setup Complete")
    print(f"-> Model artifacts will be saved to: '{os.path.abspath(output_dir)}'")
    print("="*60)
    
    return data_path, output_dir

# --- NEW: An 'init' function to set up the blueprint and load everything ---
def init_price_estimator_routes(model_dir):
    """Initializes the blueprint and loads models from the provided directory."""
    
    price_estimator_bp = Blueprint('price_estimator', __name__)
    
    # --- Load Models into the global dictionary ---
    try:
        print("--- Loading Price Estimation Models ---")
        print(f"Attempting to load models from: {os.path.abspath(model_dir)}")
        
        PRICE_MODELS['model_general_fallback'] = joblib.load(os.path.join(model_dir, 'model_general_fallback.joblib'))
        PRICE_MODELS['kmeans_model'] = joblib.load(os.path.join(model_dir, 'kmeans_model.joblib'))
        
        specialist_types = ['apartment', 'office', 'shop']
        for prop_type in specialist_types:
            model_path = os.path.join(model_dir, f'model_{prop_type}.joblib')
            if os.path.exists(model_path):
                PRICE_MODELS[f'model_{prop_type}'] = joblib.load(model_path)
                print(f"-> Loaded specialist model: '{prop_type}'")
        
    except FileNotFoundError as e:
        print(f"FATAL ERROR loading price models: {e}")
        # If loading fails, we will return the blueprint but the endpoint will fail gracefully
    finally:
        print("--- Price Estimation Models Loaded ---")

    @price_estimator_bp.route("/estimate_fair_price", methods=["POST"])
    async def estimate_fair_price():
        """Endpoint for property price prediction using the hybrid model strategy."""
        # DEBUG: Announce that a new request has started.
        print("\n--- [START] New Price Estimation Request ---")

        if not PRICE_MODELS:
            print("[ERROR] Price models not loaded.") # DEBUG: Note the failure reason
            return jsonify({"error": "Price estimation models are not available due to a startup error."}), 503

        try:
            data = await request.get_json()
            # DEBUG: Show the raw data received from the user's browser.
            print(f"[INPUT] Received data: {data}")

            input_df = pd.DataFrame([data])
            
            property_type = input_df['type'].iloc[0].lower().replace('/', '_')
            # DEBUG: Log the processed property type.
            print(f"[PROCESS] Processed property type: '{property_type}'")
            
            # Add the crucial location_cluster feature
            cluster_prediction = PRICE_MODELS['kmeans_model'].predict(input_df[['latitude', 'longitude']])
            input_df['location_cluster'] = cluster_prediction.astype(str)
            # DEBUG: Show the result of the clustering step.
            print(f"[PROCESS] Assigned location cluster: {input_df['location_cluster'].iloc[0]}")
            
            # Router logic
            model_key = f'model_{property_type}'
            model_to_use = PRICE_MODELS.get(model_key, PRICE_MODELS['model_general_fallback'])
            is_specialist = model_key in PRICE_MODELS
            
            # DEBUG: Announce which model was selected. This is a critical step.
            model_name = "Specialist Model" if is_specialist else "General Fallback Model"
            print(f"[LOGIC] Selected model: {model_name} (key: '{model_key}')")
            
            # Make prediction
            print("[PROCESS] Estimation in progress...")
            prediction_log = model_to_use.predict(input_df)
            prediction_dollars = np.expm1(prediction_log)[0]

            # DEBUG: Show the raw prediction values before formatting.
            print(f"[RESULT] Raw log prediction: {prediction_log[0]}")
            print(f"[RESULT] Converted dollar prediction (pre-rounding): {prediction_dollars}")

            # Format response
            if is_specialist:
                response_data = {"status": "high_confidence", "prediction": round(prediction_dollars, -3)}
            else:
                mae_fallback = 224000  # From our training analysis
                lower_bound = max(0, prediction_dollars - mae_fallback)
                upper_bound = prediction_dollars + mae_fallback
                response_data = {
                    "status": "low_confidence_range",
                    "estimated_center": round(prediction_dollars, -3),
                    "estimated_range": [round(lower_bound, -3), round(upper_bound, -3)]
                }
            
            # DEBUG: Show the final data that will be sent back to the front end.
            print(f"[OUTPUT] Sending response: {response_data}")
            print("--- [END] Price Estimation Request Finished ---\n")
            
            return jsonify(response_data)

        except Exception as e:
            # DEBUG: This is also a form of debugging for when things go wrong.
            print(f"[ERROR] An exception occurred: {e}")
            traceback.print_exc() # This is excellent for debugging, keep it!
            print("--- [END] Price Estimation Request FAILED ---\n")
            return jsonify({"error": "An internal server error occurred.", "details": str(e)}), 500
            
        # --- THE FIX: Return the blueprint object, not the dictionary ---
    return price_estimator_bp
