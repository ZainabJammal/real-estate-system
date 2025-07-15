import os
import pandas as pd
import numpy as np
import traceback
from supabase import create_client, Client
from dotenv import load_dotenv
from quart import Blueprint, jsonify, request





# --- Configuration: Load data once when the server starts ---
# In a real app, this would be loaded once, not in every function call.
try:
    df = pd.read_csv('properties.csv')
    # --- Initial Data Cleaning and Preparation ---
    property_types = ['Apartment', 'House/Villa', 'Chalet']
    properties_df = df[df['type'].isin(property_types)].copy()
    properties_df = properties_df[(properties_df['price_$'] > 0) & (properties_df['size_m2'] > 0)]
    properties_df['district'] = properties_df['district'].str.title().str.strip()
    properties_df['price_per_sqm'] = properties_df['price_$'] / properties_df['size_m2']
    print("Data loaded and prepared successfully.")
except FileNotFoundError:
    print("Error: properties.csv not found. Please ensure the file is in the correct directory.")
    properties_df = pd.DataFrame()

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

@market_comparison.route('/compare', methods=['GET'])
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

 # Create an empty DataFrame to avoid errors

def compare_districts(selected_districts: List[str]) -> Dict:
    """
    Analyzes and compares real estate data for a given list of districts.

    Args:
        selected_districts (List[str]): A list of district names to compare.

    Returns:
        Dict: A dictionary where keys are district names and values are their
              market analysis data. Returns an empty dictionary if no data is found.
    """
    if properties_df.empty:
        return {"error": "Data could not be loaded."}

    # Standardize the input list to match the data's format
    selected_districts_standardized = [d.title().strip() for d in selected_districts]

    # Filter the main DataFrame to only include the user's selected districts
    filtered_df = properties_df[properties_df['district'].isin(selected_districts_standardized)]

    if filtered_df.empty:
        return {} # Return empty if no properties match the selection

    # --- Aggregation and Calculation ---
    # Define custom functions for 10th and 90th percentiles
    p10 = lambda x: x.quantile(0.10)
    p90 = lambda x: x.quantile(0.90)

    analysis = filtered_df.groupby('district').agg(
        Number_of_Listings=('id', 'count'),
        Median_Price=('price_$', 'median'),
        Price_p10=('price_$', p10), # Calculate 10th percentile
        Price_p90=('price_$', p90), # Calculate 90th percentile
        Median_Price_per_m2=('price_per_sqm', 'median'),
        Median_Size_m2=('size_m2', 'median'),
        Median_Bedrooms=('bedrooms', 'median')
    )

    # --- Formatting the Output for the Frontend ---
    # Create the "Price Range" string
    analysis['Price_Range'] = analysis.apply(
        lambda row: f"${int(row['Price_p10']):,} - ${int(row['Price_p90']):,}",
        axis=1
    )

    # Drop the intermediate percentile columns
    analysis = analysis.drop(columns=['Price_p10', 'Price_p90'])

    # Format numbers for better display
    analysis['Median_Price'] = analysis['Median_Price'].apply(lambda x: f"${int(x):,}")
    analysis['Median_Price_per_m2'] = analysis['Median_Price_per_m2'].apply(lambda x: f"${int(x):,}/m²")
    analysis['Median_Size_m2'] = analysis['Median_Size_m2'].apply(lambda x: f"{int(x)} m²")
    
    # Sort by median price per m2 by default
    # To sort, we need the raw number, so we do it before formatting
    # Let's adjust the flow slightly for sorting
    analysis_sorted = analysis.sort_values(by='Median_Price_per_m2', key=lambda col: col.str.replace(r'[$,/m²]', '', regex=True).astype(float), ascending=False)
    
    # Convert the final DataFrame to a dictionary, which is easily converted to JSON
    return analysis_sorted.to_dict('index')


# --- Example of How to Use This Function ---
if __name__ == "__main__":
    # This simulates the user entering districts on the frontend
    user_selection = ['Beirut', 'Jbeil', 'El Metn']
    
    # The backend calls the function with the user's selection
    comparison_data = compare_districts(user_selection)
    
    # The backend would then return this data as JSON to the frontend
    import json
    print("\n--- Example API JSON Output ---")
    print(json.dumps(comparison_data, indent=4))


