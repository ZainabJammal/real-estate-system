import os
import joblib
from sklearn.preprocessing import MinMaxScaler
import pandas as pd
import numpy as np
import traceback
from supabase import create_client, Client
from dotenv import load_dotenv
from quart import Blueprint, jsonify, request
from db_connect import create_supabase
from quart_cors import cors



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


market_comparison = Blueprint('market_comparison', __name__)

@market_comparison.route('/api/v1/compare', methods=['GET'])
def market_compare():
    """
    Endpoint for the "Market Comparison" feature.
    Accepts query parameters: districts, type, bedrooms
    """
    if not supabase:
        return jsonify({"error": "Database connection not configured"}), 500
    
        # In your data loading script
    df = pd.read_csv('properties.csv')
    print(f"Original rows: {len(df)}")
    df.drop_duplicates(inplace=True)
    print(f"Rows after removing duplicates: {len(df)}")
        
    # 1. Get and Validate User Input from URL query parameters
    districts_str = request.args.get('districts')
    prop_type = request.args.get('type')
    bedrooms_str = request.args.get('bedrooms', '1,2,3,4,5') # Default bedrooms if not provided

    if not districts_str or not prop_type:
        return jsonify({"error": "Missing required parameters: 'districts' and 'type'"}), 400

    districts_list = [d.strip() for d in districts_str.split(',')]
    bedrooms_list = [int(b.strip()) for b in bedrooms_str.split(',')]

    if len(districts_list) < 2:
        return jsonify({"error": "Please provide at least two districts to compare."}), 400
        
    # 2. Call the SQL Function in Supabase using RPC
    try:
        print(f"Executing RPC for districts: {districts_list}, type: {prop_type}, bedrooms: {bedrooms_list}")
        
        params = {
            'districts_in': districts_list,
            'type_in': prop_type,
            'bedrooms_in': bedrooms_list
        }
        
        response = supabase.rpc('market_comparison_query', params).execute()
        
        if response.data:
            # 3. Format the data for the frontend
            return jsonify({"comparison_results": response.data})
        else:
            return jsonify({"error": "No matching data found for the selected criteria."}), 404

    except Exception as e:
        print(f"An error occurred: {e}")
        return jsonify({"error": "An internal server error occurred while fetching data."}), 500





