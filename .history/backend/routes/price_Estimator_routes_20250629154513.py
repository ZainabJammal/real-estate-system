import os
import joblib
import pandas as pd
import numpy as np
import traceback
from quart import Blueprint, request, jsonify

# --- 1. SETUP AND MODEL LOADING ---
PRICE_MODELS = {}

# Define the Blueprint for these routes


# Construct correct paths to the models, assuming this file is in the 'routes' folder
# SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
# BACKEND_DIR = os.path.dirname(SCRIPT_DIR)
# MODEL_DIR = os.path.join(BACKEND_DIR, 'final_models')

def init_price_estimator_routes(model_dir):
    """Loads all necessary price estimation models and objects into memory."""
       price_estimator_bp = Blueprint('price_estimator', __name__)
    artifacts = {}
    try:
        print("--- Loading Price Estimation Models ---")
        artifacts['model_general_fallback'] = joblib.load(os.path.join(MODEL_DIR, 'model_general_fallback.joblib'))
        artifacts['kmeans_model'] = joblib.load(os.path.join(MODEL_DIR, 'kmeans_model.joblib'))
        
        specialist_types = ['apartment', 'office', 'shop']
        for prop_type in specialist_types:
            model_path = os.path.join(MODEL_DIR, f'model_{prop_type}.joblib')
            if os.path.exists(model_path):
                artifacts[f'model_{prop_type}'] = joblib.load(model_path)
                print(f"-> Loaded specialist model: '{prop_type}'")
        
        return artifacts
    except FileNotFoundError as e:
        print(f"FATAL ERROR loading price models: {e}")
        return None

# Load models into a dictionary when the server starts
PRICE_MODELS = load_price_artifacts()

# --- 2. API ENDPOINT ---

@price_estimator_bp.route("/predict_price", methods=["POST"])
async def predict_price():
    """Endpoint for property price prediction using the hybrid model strategy."""
    if not PRICE_MODELS:
        return jsonify({"error": "Price estimation models are not available."}), 503

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