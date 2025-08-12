import pandas as pd
from quart import Blueprint, jsonify, request
from supabase import create_client, Client
from dotenv import load_dotenv
import os
import random


load_dotenv()
SUPABASE_URL = os.environ.get("SUPABASE_URL")
SUPABASE_KEY = os.environ.get("SUPABASE_KEY")
if not SUPABASE_URL or not SUPABASE_KEY:
    raise ValueError("FATAL: SUPABASE_URL/KEY not set in .env")
supabase: Client = create_client(SUPABASE_URL, SUPABASE_KEY)
print("-> Supabase client initialized.")

market_simulator_bp = Blueprint('market_simulator', __name__)

DYNAMIC_RATES = {
    2019: {'Beirut': {'crash': -0.07, 'boom': 0.03}, 
                'Bekaa': {'crash': -0.05, 'boom': 0.02}, 
                 'Tripoli': {'crash': -0.06, 'boom': 0.03}, 
                    'Baabda': {'crash': -0.04, 'boom': 0.03}, 
                        'Kesrouan': {'crash': -0.03, 'boom': 0.04}},

    2020: {'Beirut': {'crash': -0.175, 'boom': 0.05}, 
              'Bekaa': {'crash': -0.125, 'boom': 0.04}, 
                'Tripoli': {'crash': -0.15, 'boom': 0.05}, 
                    'Baabda': {'crash': -0.12, 'boom': 0.06}, 
                        'Kesrouan': {'crash': -0.10, 'boom': 0.07}},

    2021: {'Beirut': {'crash': -0.10, 'boom': 0.075}, 
             'Bekaa': {'crash': -0.08, 'boom': 0.065}, 
                'Tripoli': {'crash': -0.09, 'boom': 0.075}, 
                    'Baabda': {'crash': -0.07, 'boom': 0.085}, 
                        'Kesrouan': {'crash': -0.06, 'boom': 0.095}},

    2022: {'Beirut': {'crash': -0.035, 'boom': 0.10}, 
                'Bekaa': {'crash': -0.025, 'boom': 0.085}, 
                    'Tripoli': {'crash': -0.035, 'boom': 0.095}, 
                        'Baabda': {'crash': -0.02, 'boom': 0.11}, 
                             'Kesrouan': {'crash': -0.02, 'boom': 0.12}},

    2023: {'Beirut': {'crash': -0.01, 'boom': 0.125}, 
                'Bekaa': {'crash': -0.005, 'boom': 0.10}, 
                    'Tripoli': {'crash': -0.01, 'boom': 0.11}, 
                        'Baabda': {'crash': -0.005, 'boom': 0.12}, 
                            'Kesrouan': {'crash': 0.02, 'boom': 0.14}},

    2024: {'Beirut': {'crash': 0.02, 'boom': 0.15}, 
                'Bekaa': {'crash': 0.03, 'boom': 0.125}, 
                    'Tripoli': {'crash': 0.02, 'boom': 0.135}, 'Baabda': {'crash': 0.03, 'boom': 0.145}, 'Kesrouan': {'crash': 0.04, 'boom': 0.17}},
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

def generate_scenario(df, scenario_type):
    """
    Generates a scenario by applying dynamic, year-by-year rates to the baseline.
    This single function replaces the two older boom/crash functions.
    """
    df_scenario = df.copy()

    def calculate_dynamic_trend(group):
        city_name = group.name
        new_prices = []
        
        for row in group.itertuples():
            year = row.date.year
            baseline_price = row.price
            
            year_rates = DYNAMIC_RATES.get(year)
            if not year_rates:
                new_prices.append(baseline_price)
                continue
                
            region_rates = year_rates.get(city_name, year_rates.get('Beirut')) # Default to Beirut
            annual_rate = region_rates[scenario_type]
            monthly_rate = (1 + annual_rate) ** (1/12)
            
            start_of_year_price = group[group['date'].dt.year == year]['price'].iloc[0]
            months_into_year = row.date.month - 1
            
            trend_price = start_of_year_price * (monthly_rate ** months_into_year)
            trend_effect = trend_price - start_of_year_price

            # Apply a small amount of randomness to make plots feel more unique
            random_factor = 1 + random.uniform(-0.015, 0.015)
            
            new_prices.append((baseline_price + trend_effect) * random_factor)

        group['price'] = new_prices
        return group

    df_scenario = df_scenario.groupby('city').apply(calculate_dynamic_trend)
    df_scenario["scenario"] = scenario_type.capitalize()
    return df_scenario.reset_index(drop=True)

# --- 5. API ENDPOINT ---
@market_simulator_bp.route("/api/market-simulator", methods=["GET"])
async def get_market_scenarios():
    user_selection = request.args.get('selection')
    if not user_selection:
        return jsonify({"error": "Missing 'selection' parameter."}), 400
    
    city_name = map_user_selection_to_city(user_selection)
    base_df = ALL_SYNTHETIC_DATA[ALL_SYNTHETIC_DATA['city'] == city_name].copy()
    
    if base_df.empty:
        return jsonify({"error": f"No data for city '{city_name}'."}), 404
        
    print(f"-> Generating DYNAMIC scenarios for '{city_name}'...")
    
    baseline = base_df.copy()
    baseline["scenario"] = "Baseline"
    
    boom = generate_scenario(base_df, 'boom')
    crash = generate_scenario(base_df, 'crash')
    
    df_scenarios = pd.concat([baseline, boom, crash]).reset_index(drop=True)
    df_scenarios['date'] = df_scenarios['date'].dt.strftime('%Y-%m-%d')
    df_scenarios.rename(columns={'price': 'transaction_value'}, inplace=True)
    
    return jsonify(df_scenarios.to_dict(orient='records'))