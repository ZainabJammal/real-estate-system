import pandas as pd
from quart import Blueprint, jsonify, request
from supabase import create_client, Client
from dotenv import load_dotenv
import os

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
    if "tripoli" in selection_lower: return "Tripoli"
    if "baabda" in selection_lower: return "Baabda"
    if "kesrouan" in selection_lower: return "Kesrouan"
    if "bekaa" in selection_lower: return "Bekaa"
    if "beirut" in selection_lower: return "Beirut"
    return selection_string

def boom_scenario(df):
    df_boom = df.copy()
    m = (1 + 0.15) ** (1/12)
    def calculate_boom(group):
        initial_price = group['price'].iloc[0]
        multipliers = [m ** i for i in range(len(group))]
        group['price'] = initial_price * pd.Series(multipliers, index=group.index)
        return group
    df_boom = df_boom.groupby('city').apply(calculate_boom)
    df_boom["scenario"] = "Boom"
    return df_boom.reset_index(drop=True)

def crash_scenario(df):
    df_crash = df.copy()
    m_drop = (1 - 0.20) ** (1/6)
    m_recover = (1 + 0.05) ** (1/12)
    def calculate_crash(group):
        crash_date_ts = pd.Timestamp("2018-06-01")
        crash_idx_series = group.index[group['date'] == crash_date_ts]
        crash_idx = crash_idx_series[0] if not crash_idx_series.empty else group.index[len(group) // 2]
        crash_start_row = group.index.get_loc(crash_idx)
        new_prices = []
        current_price = 0
        for i, row in enumerate(group.itertuples()):
            if i < crash_start_row: current_price = row.price
            elif crash_start_row <= i < crash_start_row + 6: current_price *= m_drop
            else: current_price *= m_recover
            new_prices.append(current_price)
        group['price'] = new_prices
        return group
    df_crash = df_crash.groupby('city').apply(calculate_crash)
    df_crash["scenario"] = "Crash"
    return df_crash.reset_index(drop=True)

# --- 4. API ENDPOINT DEFINITION ---
@market_simulator_bp.route("/api/market-simulator", methods=["GET"])
async def get_market_scenarios():
    user_selection = request.args.get('selection')
    if not user_selection:
        return jsonify({"error": "Missing 'selection' parameter."}), 400
    print(f"\n-> Request for: '{user_selection}'")
    city_name = map_user_selection_to_city(user_selection)
    print(f"-> Mapped to: '{city_name}'")
    base_df = ALL_SYNTHETIC_DATA[ALL_SYNTHETIC_DATA['city'] == city_name].copy()
    if base_df.empty:
        return jsonify({"error": f"No data for city '{city_name}'."}), 404
    print(f"-> Found {len(base_df)} rows. Generating scenarios...")
    baseline = base_df.copy()
    baseline["scenario"] = "Baseline"
    boom = boom_scenario(base_df)
    crash = crash_scenario(base_df)
    df_scenarios = pd.concat([baseline, boom, crash]).reset_index(drop=True)
    df_scenarios['date'] = df_scenarios['date'].dt.strftime('%Y-%m-%d')
    df_scenarios.rename(columns={'price': 'transaction_value'}, inplace=True)
    print("-> Scenarios generated. Sending JSON response.")
    return jsonify(df_scenarios.to_dict(orient='records'))