import os
from quart import Blueprint, jsonify, request
from supabase import create_client, Client
from dotenv import load_dotenv
import traceback # Useful for debugging

# Load environment variables from .env file
load_dotenv()

SUPABASE_URL: str = os.environ.get("SUPABASE_URL")
SUPABASE_KEY: str = os.environ.get("SUPABASE_KEY")

if not SUPABASE_URL or not SUPABASE_KEY:
    raise ValueError("FATAL ERROR: SUPABASE_URL and SUPABASE_KEY must be set in your .env file.")

# Initialize the Supabase client
supabase: Client = create_client(SUPABASE_URL, SUPABASE_KEY)
print("-> Supabase client initialized.")

# Create a Blueprint for our routes
market_comparison_bp = Blueprint('market_comparison', __name__)

@market_comparison_bp.route('/compare', methods=['GET'])
async def market_compare():
    """
    Endpoint for the "Market Comparison" feature.
    Accepts query parameters: districts (comma-separated string)
    """
    if not supabase:
        return jsonify({"error": "Database connection not configured"}), 500

    # 1. Get and Validate User Input from URL query parameters
    districts_str = request.args.get('districts')

    if not districts_str:
        return jsonify({"error": "Missing required parameter: 'districts'"}), 400

    # Convert comma-separated string to a list of strings
    districts_list = [d.strip() for d in districts_str.split(',')]

    if len(districts_list) < 2:
        return jsonify({"error": "Please provide at least two districts to compare."}), 400

    # 2. Call the SQL Function in Supabase using RPC (Remote Procedure Call)
    try:
        print(f"Executing RPC 'market_comparison_query' for districts: {districts_list}")
        
        params = {'districts_in': districts_list}
        
        # --- THE FIX IS HERE ---
        # The .execute() method is synchronous, so we do NOT use 'await'.
        response = supabase.rpc('market_comparison_query', params).execute()
        # --- END OF FIX ---
        
         # --- d. Format and Return Response ---

        # (1) CREATE the response dictionary FIRST.
        response_data = {
            "status": "success",
            "districts_in": round(prediction_dollars, -3), # Use the calculated price
            
        }

        # (2) NOW you can safely print it and send it.
        print(f"[OUTPUT] Sending response: {response_data}")
        print("--- [END] Prediction Request Finished ---\n")
        if response.data:
            # 3. Format the data for the frontend
            return jsonify({"comparison_results": response.data})
        else:
            # Return an empty list for consistency, along with an info message.
            return jsonify({"info": "No matching data found for the selected criteria.", "comparison_results": []})

    except Exception as e:
        print(f"An error occurred during Supabase RPC call: {e}")
        traceback.print_exc() # This provides a more detailed error log in your terminal
        return jsonify({"error": "An internal server error occurred while fetching data."}), 500