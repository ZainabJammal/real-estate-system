import pandas as pd
from quart import Quart, jsonify
from quart_cors import cors 
from supabase import create_client, Client
from dotenv import load_dotenv
import os

load_dotenv()

SUPABASE_URL: str = os.environ.get("SUPABASE_URL")
SUPABASE_KEY: str = os.environ.get("SUPABASE_KEY")

if not SUPABASE_URL or not SUPABASE_KEY:
    raise ValueError("FATAL ERROR: SUPABASE_URL and SUPABASE_KEY must be set in your .env file.")

supabase: Client = create_client(SUPABASE_URL, SUPABASE_KEY)
print("-> Supabase client initialized.")

market_simulator = Blueprint('market_simulator', __name__)


def fetch_supabase_data():
    """Connects to Supabase and fetches all properties data."""
    print("-> Connecting to Supabase to fetch training data...")
    load_dotenv()
    url: str = os.environ.get("SUPABASE_URL")
    key: str = os.environ.get("SUPABASE_KEY")
    if not url or not key:
        raise ValueError("FATAL: Supabase URL or Key not found in .env file.")
    supabase: Client = create_client(url, key)
    response = supabase.table('synthetic_data').select("*").order('date').execute()
    df = pd.DataFrame(response.data)
    print(f"-> Successfully fetched {len(df)} rows from Supabase.")
    return df
def prepare_data(df):
    """Prepares the DataFrame for market simulation."""
    if df is None or df.empty:
        raise ValueError("DataFrame is empty or not loaded properly.")
    df['date'] = pd.to_datetime(df['date'])
    # 2. We'll work with 'transaction_value' as our price metric
    df.rename(columns={'transaction_value': 'price'}, inplace=True)
    # 3. Sort by date to ensure chronological order
    df.sort_values(by='date', inplace=True)
    # 4. Reset index for clean DataFrame
    df.reset_index(drop=True, inplace=True)
    print("-> Data preparation complete.")
    return df


def map_user_selection_to_city(selection_string: str) -> str:
    """
    Maps the user's dropdown selection to the specific custom region names
    used in the database, ensuring it's lowercase for matching.
    """
    if "Tripoli, Akkar" in selection_string:
        return "Tripoli"
    if "Baabda, Aley, Chouf" in selection_string:
        return "Baabda"
    if "Kesrouan, Jbeil" in selection_string:
        return "Kesrouan"
    return selection_string.lower()

# --- Scenario Generation Functions (from our previous discussion) ---
def boom_scenario(df, annual_growth=0.15):
    """Applies a boom scenario to each city group."""
    df_boom = df.copy()
    m = (1 + annual_growth) ** (1/12)

    # This function will be applied to each city's sub-dataframe
    def calculate_boom(group):
        initial_price = group['price'].iloc[0]
        # Create a growth factor for each month in the group
        growth_multipliers = [m ** i for i in range(len(group))]
        group['price'] = initial_price * pd.Series(growth_multipliers, index=group.index)
        return group

    # Group by city, apply the calculation, and combine results
    df_boom = df_boom.groupby('city').apply(calculate_boom)
    df_boom["scenario"] = "Boom"
    return df_boom.reset_index(drop=True)

def crash_scenario_grouped(df, crash_start="2018-06-01", drop=-0.20, drop_months=6, recovery_annual=0.05):
    """Applies a crash-and-recovery scenario to each city group."""
    df_crash = df.copy()
    m_drop = (1 + drop) ** (1/drop_months)
    m_recover = (1 + recovery_annual) ** (1/12)

    # This function operates on one city at a time
    def calculate_crash(group):
        # Find the index for the crash start date WITHIN THE GROUP
        crash_date_ts = pd.Timestamp(crash_start)
        crash_idx_series = group.index[group['date'] == crash_date_ts]
        
        # If the specific date isn't found, default to a midpoint
        crash_idx = crash_idx_series[0] if not crash_idx_series.empty else group.index[len(group) // 2]
        crash_start_row = group.index.get_loc(crash_idx)
        
        new_prices = []
        current_price = 0

        for i, row in enumerate(group.itertuples()):
            # Use the baseline price until the crash
            if i < crash_start_row:
                current_price = row.price
            # During the crash period
            elif crash_start_row <= i < crash_start_row + drop_months:
                current_price *= m_drop
            # After the crash, during recovery
            else:
                current_price *= m_recover
            new_prices.append(current_price)
            
        group['price'] = new_prices
        return group

    df_crash = df_crash.groupby('city').apply(calculate_crash)
    df_crash["scenario"] = "Crash"
    return df_crash.reset_index(drop=True)
# --- API Endpoint Definition ---
@market_simulator.route("/api/market-simulator", methods=["GET"])
async def get_market_scenarios():
    """
    Returns market scenarios. Can be filtered by city.
    e.g., /api/market-scenarios?city=Beirut
    """

    print(f"\nReceived 'Best Guess' forecast request for: '{user_selection}'")

    try: 
        city_name = map_user_selection_to_city(user_selection)
        print(f"-> Mapped to city: '{city_name}'")

        # step A: fetch histo data first
        print(f"-> Querying Supabase for '{city_name}' historical data...")
        response = supabase.table('agg_trans').select('*').ilike(GROUPING_KEY, city_name).order('date').execute()
        
    # Get the city from query args, if provided
    city_filter = request.args.get('city')
    
    if city_filter:
        # Filter the main DataFrame for the selected city
        base_df = df[df['city'] == city_filter].copy()
        if base_df.empty:
            return jsonify({"error": f"City '{city_filter}' not found."}), 404
    else:
        # Use the entire DataFrame if no city is specified
        base_df = df.copy()

    # 1. Create Baseline
    baseline = base_df.copy()
    baseline["scenario"] = "Baseline"
    
    # 2. Generate other scenarios on the (potentially filtered) data
    boom = boom_scenario(base_df)
    crash = crash_scenario_grouped(base_df)

    # 3. Combine into one DataFrame
    df_scenarios = pd.concat([baseline, boom, crash]).reset_index(drop=True)
    
    # 4. Convert date to string and return as JSON records
    df_scenarios['date'] = df_scenarios['date'].dt.strftime('%Y-%m-%d')
    
    # We rename 'price' back to 'transaction_value' for API consistency
    df_scenarios.rename(columns={'price': 'transaction_value'}, inplace=True)
    
    return jsonify(df_scenarios.to_dict(orient='records'))