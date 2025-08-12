import pandas as pd
from quart import Blueprint, jsonify, request
from supabase import create_client, Client
from dotenv import load_dotenv
import os
import random

# --- 1. SETUP & INITIALIZATION ---
load_dotenv()
SUPABASE_URL = os.environ.get("SUPABASE_URL")
SUPABASE_KEY = os.environ.get("SUPABASE_KEY")
if not SUPABASE_URL or not SUPABASE_KEY:
    raise ValueError("FATAL: SUPABASE_URL/KEY not set in .env")
supabase: Client = create_client(SUPABASE_URL, SUPABASE_KEY)
print("-> Supabase client initialized.")

market_simulator_bp = Blueprint('market_simulator', __name__)

# --- 2. DATA LOADING (Load once at startup) ---
def fetch_and_prepare_all_data():
    print("-> Connecting to Supabase to fetch ALL synthetic data for caching...")
    response = supabase.table('synthetic_data').select("*").order('date').execute()
    if not response.data:
        raise ConnectionError("Failed to fetch data from Supabase.")
    df = pd.DataFrame(response.data)
    df['date'] = pd.to_datetime(df['date'])
    df.rename(columns={'transaction_value': 'price'}, inplace=True)
    df.sort_values(by=['city', 'date'], inplace=True)
    print(f"-> Successfully fetched and cached {len(df)} rows.")
    return df

ALL_SYNTHETIC_DATA = fetch_and_prepare_all_data()

# --- 3. HELPER & SCENARIO FUNCTIONS ---
def map_user_selection_to_city(selection_string: str) -> str:
    selection_lower = selection_string.lower()
    if "Tripoli, Akkar" in selection_string:
        return "Tripoli"
    if "Baabda, Aley, Chouf" in selection_string:
        return "Baabda"
    if "Kesrouan, Jbeil" in selection_string:
        return "Kesrouan"
    return selection_string

# We use the midpoint of the estimated ranges you provided.
SCENARIO_RATES = {
    'Beirut':   {'boom': 0.065, 'crash': -0.125}, # 6.5% boom, -12.5% crash
    'Baabda':   {'boom': 0.05,  'crash': -0.08},  # 5% boom, -8% crash
    'Tripoli':  {'boom': 0.03,  'crash': -0.045}, # 3% boom, -4.5% crash
    'Kesrouan': {'boom': 0.055, 'crash': -0.065}, # 5.5% boom, -6.5% crash
    'Bekaa':    {'boom': 0.02,  'crash': -0.03},  # 2% boom, -3% crash
    'default':  {'boom': 0.04,  'crash': -0.05}   # Fallback for any other region
}

# --- 4. SCENARIO GENERATION (NOW REGION-AWARE) ---

def boom_scenario_stochastic(df):
    df_boom = df.copy()

    def calculate_boom(group):
        city_name = group.name
        annual_growth = SCENARIO_RATES.get(city_name, SCENARIO_RATES['default'])['boom']
        m = (1 + annual_growth) ** (1/12)

        initial_price = group['price'].iloc[0]
        
        new_prices = []
        for i, row in enumerate(group.itertuples()):
            # 1. Calculate the main trend effect
            growth_effect = (initial_price * (m ** i)) - initial_price
            
            # 2. Add a small, random fluctuation (e.g., between -2% and +4%)
            # This makes the boom scenario slightly more volatile and optimistic
            random_factor = 1 + random.uniform(-0.02, 0.04) 
            
            # 3. Apply both to the original baseline price
            new_price = (row.price + growth_effect) * random_factor
            new_prices.append(new_price)
            
        group['price'] = new_prices
        return group

    df_boom = df_boom.groupby('city').apply(calculate_boom)
    df_boom["scenario"] = "Boom"
    return df_boom.reset_index(drop=True)

def crash_scenario_regional(df, crash_start="2019-10-01"): # Updated crash start date to be more realistic
    """Applies a region-specific 'crash' trend on top of the baseline data."""
    df_crash = df.copy()
    
    def calculate_crash(group):
        city_name = group.name
        # Get the specific annual decline rate for this city
        annual_decline = SCENARIO_RATES.get(city_name, SCENARIO_RATES['default'])['crash']
        # For a crash, we'll model a sharp 2-year drop followed by stabilization
        m_drop = (1 + annual_decline) ** (1/12) 

        crash_date_ts = pd.Timestamp(crash_start)
        crash_idx_series = group.index[group['date'] >= crash_date_ts]
        if crash_idx_series.empty: # If no dates after crash start, return original
            return group
        
        crash_idx = crash_idx_series[0]
        price_at_crash = group.loc[crash_idx, 'price']
        new_prices = []

        for row in group.itertuples():
            if row.Index < crash_idx:
                new_prices.append(row.price) # Price is same as baseline before crash
            else:
                months_after_crash = (row.date.year - crash_date_ts.year) * 12 + (row.date.month - crash_date_ts.month)
                smooth_price = price_at_crash * (m_drop ** months_after_crash)
                crash_effect = smooth_price - price_at_crash
                random_factor = 1 + random.uniform(-0.03, 0.01)
                new_price = (row.price + crash_effect) * random_factor
                new_prices.append(new_price)

        group['price'] = new_prices
        return group

    df_crash = df_crash.groupby('city').apply(calculate_crash)
    df_crash["scenario"] = "Crash"
    return df_crash.reset_index(drop=True)

# --- 5. API ENDPOINT (Now calls the regional functions) ---
@market_simulator_bp.route("/api/market-simulator", methods=["GET"])
async def get_market_scenarios():
    user_selection = request.args.get('selection')
    if not user_selection:
        return jsonify({"error": "Missing 'selection' parameter."}), 400
    
    city_name = map_user_selection_to_city(user_selection)
    base_df = ALL_SYNTHETIC_DATA[ALL_SYNTHETIC_DATA['city'] == city_name].copy()
    
    if base_df.empty:
        return jsonify({"error": f"No data for city '{city_name}'."}), 404
        
    print(f"-> Generating REGION-AWARE scenarios for '{city_name}'...")
    
    baseline = base_df.copy()
    baseline["scenario"] = "Baseline"
    
    # --- CALL THE NEW, REGIONAL FUNCTIONS ---
    boom = boom_scenario_stochastic(base_df)
    crash = crash_scenario_stochastic(base_df)
    
    df_scenarios = pd.concat([baseline, boom, crash]).reset_index(drop=True)
    df_scenarios['date'] = df_scenarios['date'].dt.strftime('%Y-%m-%d')
    df_scenarios.rename(columns={'price': 'transaction_value'}, inplace=True)
    
    return jsonify(df_scenarios.to_dict(orient='records'))