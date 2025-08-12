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

DYNAMIC_RATES = {
    # Year: { Region: {'crash': rate, 'boom': rate} }
    2019: {'Beirut': {'crash': -0.07, 'boom': 0.03}, 'Bekaa': {'crash': -0.05, 'boom': 0.02}, 'Tripoli': {'crash': -0.06, 'boom': 0.03}, 'Baabda': {'crash': -0.04, 'boom': 0.03}, 'Kesrouan': {'crash': -0.03, 'boom': 0.04}},
    2020: {'Beirut': {'crash': -0.175, 'boom': 0.05}, 'Bekaa': {'crash': -0.125, 'boom': 0.04}, 'Tripoli': {'crash': -0.15, 'boom': 0.05}, 'Baabda': {'crash': -0.12, 'boom': 0.06}, 'Kesrouan': {'crash': -0.10, 'boom': 0.07}},
    2021: {'Beirut': {'crash': -0.10, 'boom': 0.075}, 'Bekaa': {'crash': -0.08, 'boom': 0.065}, 'Tripoli': {'crash': -0.09, 'boom': 0.075}, 'Baabda': {'crash': -0.07, 'boom': 0.085}, 'Kesrouan': {'crash': -0.06, 'boom': 0.095}},
    2022: {'Beirut': {'crash': -0.035, 'boom': 0.10}, 'Bekaa': {'crash': -0.025, 'boom': 0.085}, 'Tripoli': {'crash': -0.035, 'boom': 0.095}, 'Baabda': {'crash': -0.02, 'boom': 0.11}, 'Kesrouan': {'crash': -0.02, 'boom': 0.12}},
    2023: {'Beirut': {'crash': -0.01, 'boom': 0.125}, 'Bekaa': {'crash': -0.005, 'boom': 0.10}, 'Tripoli': {'crash': -0.01, 'boom': 0.11}, 'Baabda': {'crash': -0.005, 'boom': 0.12}, 'Kesrouan': {'crash': 0.02, 'boom': 0.14}},
    2024: {'Beirut': {'crash': 0.02, 'boom': 0.15}, 'Bekaa': {'crash': 0.03, 'boom': 0.125}, 'Tripoli': {'crash': 0.02, 'boom': 0.135}, 'Baabda': {'crash': 0.03, 'boom': 0.145}, 'Kesrouan': {'crash': 0.04, 'boom': 0.17}},
    2025: {'Beirut': {'crash': 0.03, 'boom': 0.17}, 'Bekaa': {'crash': 0.04, 'boom': 0.15}, 'Tripoli': {'crash': 0.03, 'boom': 0.16}, 'Baabda': {'crash': 0.04, 'boom': 0.17}, 'Kesrouan': {'crash': 0.05, 'boom': 0.185}},
}

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
# SCENARIO_RATES = {
#     'Beirut':   {'boom': 0.065, 'crash': -0.125}, # 6.5% boom, -12.5% crash
#     'Baabda':   {'boom': 0.05,  'crash': -0.08},  # 5% boom, -8% crash
#     'Tripoli':  {'boom': 0.03,  'crash': -0.045}, # 3% boom, -4.5% crash
#     'Kesrouan': {'boom': 0.055, 'crash': -0.065}, # 5.5% boom, -6.5% crash
#     'Bekaa':    {'boom': 0.02,  'crash': -0.03},  # 2% boom, -3% crash
#     'default':  {'boom': 0.04,  'crash': -0.05}   # Fallback for any other region
# }

