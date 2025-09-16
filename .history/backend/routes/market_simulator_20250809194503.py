import pandas as pd
from quart import Blueprint, jsonify, request
from supabase import create_client, Client
from dotenv import load_dotenv
import os

# --- 1. SETUP & INITIALIZATION ---

load_dotenv()

# Initialize Supabase client ONCE
SUPABASE_URL: str = os.environ.get("SUPABASE_URL")
SUPABASE_KEY: str = os.environ.get("SUPABASE_KEY")
if not SUPABASE_URL or not SUPABASE_KEY:
    raise ValueError("FATAL: SUPABASE_URL and SUPABASE_KEY must be set in your .env file.")
supabase: Client = create_client(SUPABASE_URL, SUPABASE_KEY)
print("-> Supabase client initialized.")

# Create the Quart Blueprint
market_simulator_bp = Blueprint('market_simulator', __name__)


# --- 2. DATA LOADING (BEST PRACTICE: Load once at startup) ---

def fetch_and_prepare_all_data():
    """
    Connects to Supabase, fetches the ENTIRE synthetic_data table,
    and prepares it for use. This should be called only once.
    """
    print("-> Connecting to Supabase to fetch ALL synthetic data for caching...")
    response = supabase.table('synthetic_data').select("*").order('date').execute()
    
    if not response.data:
        raise ConnectionError("Failed to fetch data from Supabase or table is empty.")
        
    df = pd.DataFrame(response.data)
    
    # Prepare the data
    df['date'] = pd.to_datetime(df['date'])
    df.rename(columns={'transaction_value': 'price'}, inplace=True)
    df.sort_values(by=['city', 'date'], inplace=True)
    df.reset_index(drop=True, inplace=True)
    
    print(f"-> Successfully fetched and cached {len(df)} rows from Supabase.")
    return df

# Fetch and store the data in a global variable when the server starts
# This is our in-memory cache.
ALL_SYNTHETIC_DATA = fetch_and_prepare_all_data()


# --- 3. HELPER & SCENARIO FUNCTIONS ---

def map_user_selection_to_city(selection_string: str) -> str:
    """Maps user-friendly frontend names to database city names."""
    # This function is great, no changes needed.
    # Convert to lowercase for robust matching
    selection_lower = selection_string.lower() 
    if "tripoli" in selection_lower: return "Tripoli"
    if "baabda" in selection_lower: return "Baabda"
    if "kesrouan" in selection_lower: return "Kesrouan"
    if "bekaa" in selection_lower: return "Bekaa"
    if "beirut" in selection_lower: return "Beirut"
    return selection_string # Fallback

# Your scenario functions (boom_scenario, crash_scenario_grouped) are well-defined.
# We will rename crash_scenario_grouped to crash_scenario for consistency.
def boom_scenario(df, annual_growth=0.15):
    """Applies a boom scenario to each city group."""
    # ... (No changes needed, this code is correct)
    df_boom = df.copy()
    m = (1 + annual_growth) ** (1/12)
    def calculate_boom(group):
        initial_price = group['price'].iloc[0]
        growth_multipliers = [m ** i for i in range(len(group))]
        group['price'] = initial_price * pd.Series(growth_multipliers, index=group.index)
        return group
    df_boom = df_boom.groupby('city').apply(calculate_boom)
    df_boom["scenario"] = "Boom"
    return df_boom.reset_index(drop=True)

def crash_scenario(df, crash_start="2018-06-01", drop=-0.20, drop_months=6, recovery_annual=0.05):
    """Applies a crash-and-recovery scenario to each city group."""
    # ... (No changes needed, this code is correct)
    df_crash = df.copy()
    m_drop = (1 + drop) ** (1/drop_months)
    m_recover = (1 + recovery_annual) ** (1/12)
    def calculate_crash(group):
        crash_date_ts = pd.Timestamp(crash_start)
        crash_idx_series = group.index[group['date'] == crash_date_ts]
        crash_idx = crash_idx_series[0] if not crash_idx_series.empty else group.index[len(group) // 2]
        crash_start_row = group.index.get_loc(crash_idx)
        new_prices = []
        current_price = 0
        for i, row in enumerate(group.itertuples()):
            if i < crash_start_row:
                current_price = row.price
            elif crash_start_row <= i < crash_start_row + drop_months:
                current_price *= m_drop
            else:
                current_price *= m_recover
            new_prices.append(current_price)
        group['price'] = new_prices
        return group
    df_crash = df_crash.groupby('city').apply(calculate_crash)
    df_crash["scenario"] = "Crash"
    return df_crash.reset_index(drop=True)


# --- 4. API ENDPOINT DEFINITION ---

@market_simulator_bp.route("/api/market-simulator/<string:user_selection>", methods=["GET"])
async def get_market_scenarios(user_selection: str):
    """
    Returns market scenarios for a given city selection.
    e.g., /market-simulator/city_name
    """
   # Step 1: Get user's selection and VALIDATE IT
    user_selection = request.args.get('selection')
    
    # --- THIS IS THE CRITICAL FIX ---
    if not user_selection:
        # If the 'selection' parameter is missing or empty, return a clear 400 error
        print("-> ERROR: Request received with no 'selection' parameter.")
        return jsonify({"error": "The 'selection' query parameter is required and cannot be empty."}), 400

    print(f"\n-> Request received for market simulation: '{user_selection}'")

    # Step 2: Map the user selection to a known city name
    city_name = map_user_selection_to_city(user_selection)
    print(f"-> Mapped to city: '{city_name}'")

    # Step 3: Filter the PRE-LOADED DataFrame (this is very fast)
    base_df = ALL_SYNTHETIC_DATA[ALL_SYNTHETIC_DATA['city'] == city_name].copy()
    
    if base_df.empty:
        print(f"-> ERROR: No data found for city '{city_name}'.")
        return jsonify({"error": f"No data available for city '{city_name}'."}), 404

    # Step 4: Generate the three scenarios
    print(f"-> Generating scenarios for {len(base_df)} rows...")
    baseline = base_df.copy()
    baseline["scenario"] = "Baseline"
    
    boom = boom_scenario(base_df)
    crash = crash_scenario(base_df)

    # Step 5: Combine, format, and return the result
    df_scenarios = pd.concat([baseline, boom, crash]).reset_index(drop=True)
    df_scenarios['date'] = df_scenarios['date'].dt.strftime('%Y-%m-%d')
    df_scenarios.rename(columns={'price': 'transaction_value'}, inplace=True)
    
    print("-> Successfully generated and returning scenarios.")
    return jsonify(df_scenarios.to_dict(orient='records'))



@market_simulator_bp.route("/ping")
async def ping():
    print("-> PING route was successfully hit!")
    return {"message": "pong"}