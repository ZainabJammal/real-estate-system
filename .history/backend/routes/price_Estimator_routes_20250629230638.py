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
    
    price_estimator_bp