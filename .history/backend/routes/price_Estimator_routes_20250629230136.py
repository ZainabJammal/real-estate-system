import os
import joblib
import pandas as pd
import numpy as np
import traceback
from quart import Blueprint, request, jsonify

# This global dictionary will be populated by the init function
PRICE_MODELS = {}

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
    
    # --- Define the endpoint within the init function ---
    @price_estimator_bp.route("/estimate_fair_price", methods=["POST"])
    async def estimate_fair_price():
        """Endpoint for property price prediction using the hybrid model strategy."""
        if not PRICE_MODELS:
            return jsonify({"error": "Price estimation models are not available due to a startup error."}), 503

        try:
            data = await request.get_json()
            input_df = pd.DataFrame([data])
            
            property_type = input_df['type'].iloc[0].lower().replace('/', '_')
            
            # Add the crucial location_cluster feature
            input_df['location_cluster'] = PRICE_MODELS['kmeans_model'].predict(input_df[['latitude', 'longitude']]).astype(str)
            
            # Router logic
            model_key = f'model_{property_type}'
            model_to_use = PRICE_MODELS.get(model_key, PRICE_MODELS['model_general_fallback'])
            is_specialist = model_key in PRICE_MODELS
            if response.data:
            print(f"Number of records returned: {len(response.data)}")
            print(f"First record received from Supabase: {response.data[0]}") 
        else:
            print("!!! CRITICAL: `response.data` is EMPTY or does not exist.")
            print(f"Full response object: {response}")
        
        print("-----------------------------------\n"
            # Make prediction
            prediction_log = model_to_use.predict(input_df)
            prediction_dollars = np.expm1(prediction_log)[0]

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
            
            return jsonify(response_data)

        except Exception as e:
            traceback.print_exc()
            return jsonify({"error": "An internal server error occurred.", "details": str(e)}), 500
            
    # --- THE FIX: Return the blueprint object, not the dictionary ---
    return price_estimator_bp