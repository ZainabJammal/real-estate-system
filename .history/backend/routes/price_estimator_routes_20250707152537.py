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

# def setup_environment():
#     """Sets up directories and configurations for the script."""
#     warnings.filterwarnings('ignore', category=UserWarning)
#     pd.options.mode.chained_assignment = None
    
    # # Get the absolute path of the currently running script (e.g., .../backend/routes/your_script.py)
    # script_path = os.path.abspath(__file__)
    
    # # Get the directory the script is in (e.g., .../backend/routes/)
    # routes_dir = os.path.dirname(script_path)
    
    # # Go UP ONE LEVEL to get the backend directory (e.g., .../backend/)
    # backend_dir = os.path.dirname(routes_dir)

    # # Now, build the paths relative to the `backend_dir`
    # output_dir = os.path.join(backend_dir, 'final_model_output')
    # data_path = os.path.join(backend_dir, 'properties.csv')
        
       
    # print("="*60)
    # print("Environment Setup Complete")
    # print(f"-> Model artifacts will be saved to: '{os.path.abspath(output_dir)}'")
    # print("="*60)
    
    # return data_path, output_dir



# # --- NEW: An 'init' function to set up the blueprint and load everything ---
# def init_price_estimator_routes(model_dir):
#     """Initializes the blueprint and loads models from the provided directory."""
    
#     price_estimator_bp = Blueprint('price_estimator', __name__)
    
#     # --- Load Models into the global dictionary ---
#     try:
#         print("--- Loading Price Estimation Models ---")
#         print(f"Attempting to load models from: {os.path.abspath(model_dir)}")
        
#         PRICE_MODELS['model_general_fallback'] = joblib.load(os.path.join(model_dir, 'model_general_fallback.joblib'))
#         PRICE_MODELS['kmeans_model'] = joblib.load(os.path.join(model_dir, 'kmeans_model.joblib'))
        
#         specialist_types = ['apartment', 'office', 'shop']
#         for prop_type in specialist_types:
#             model_path = os.path.join(model_dir, f'model_{prop_type}.joblib')
#             if os.path.exists(model_path):
#                 PRICE_MODELS[f'model_{prop_type}'] = joblib.load(model_path)
#                 print(f"-> Loaded specialist model: '{prop_type}'")
        
#     except FileNotFoundError as e:
#         print(f"FATAL ERROR loading price models: {e}")
#         # If loading fails, we will return the blueprint but the endpoint will fail gracefully
#     finally:
#         print("--- Price Estimation Models Loaded ---")


price_estimator_bp = Blueprint('price_estimator', __name__)

# --- 3. BLUEPRINT INITIALIZATION AND MODEL LOADING ---
def init_price_estimator_routes(model_dir):
    """Initializes the blueprint and loads the trained apartment model."""
    # global PIPELINE
    try:
      
        script_path = os.path.abspath(__file__)
        routes_dir = os.path.dirname(script_path)
        backend_dir = os.path.dirname(routes_dir)
        model_dir = os.path.join(backend_dir, 'final_model_output')
        
        # --- Load the specific model you trained ---
        model_path = os.path.join(model_dir, 'model_apartment_advanced.joblib')
        
        print("--- Loading Price Estimation Model ---")
        print(f"-> Attempting to load model from: {model_path}")
        
        PIPELINE = joblib.load(model_path)
        
        print("-> Successfully loaded: 'model_apartment_advanced.joblib'")
        
    except FileNotFoundError as e:
        print(f"FATAL ERROR: Could not find the model file. {e}")
    except Exception as e:
        print(f"FATAL ERROR during model loading: {e}")
        traceback.print_exc()
       
    return price_estimator_bp


# price_estimator_bp = cors(price_estimator_bp, allow_origin=[
#       "http://localhost:3000",  # Your Vite app's origin
#     "http://127.0.0.1:3000"  # Also good to include this
# ])
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
            print("[PROCESS] Making prediction...")
            prediction_log = PIPELINE.predict(input_df)
            
            # Convert prediction back to dollars
            prediction_dollars = np.expm1(prediction_log)[0]
            print(f"[RESULT] Predicted Price: ${prediction_dollars:,.2f}")

            # --- d. Format and Return Response ---
            response_data = {
                "status": "success",
                "predicted_price_$": round(prediction_dollars, -3)
            }
            
            print(f"[OUTPUT] Sending response: {response_data}")
            print("--- [END] Prediction Request Finished ---\n")
            
            return jsonify(response_data)

        except Exception as e:
            print(f"[ERROR] An exception occurred: {e}")
            traceback.print_exc()
            return jsonify({"error": "An internal server error occurred.", "details": str(e)}), 500