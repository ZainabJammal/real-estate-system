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

def boom_scenario_enhanced(df, annual_growth=0.15):
    """
    Applies a 'boom' trend ON TOP OF the baseline data, preserving its volatility.
    """
    df_boom = df.copy()
    m = (1 + annual_growth) ** (1/12)

    def calculate_boom(group):
        initial_price = group['price'].iloc[0]
        
        # Calculate the "growth effect": the difference between a smooth boom
        # curve and the initial price.
        growth_effect = [(initial_price * (m ** i)) - initial_price for i in range(len(group))]
        
        # Add this growth effect to the original baseline price for each month.
        group['price'] = group['price'] + pd.Series(growth_effect, index=group.index)
        return group

    df_boom = df_boom.groupby('city').apply(calculate_boom)
    df_boom["scenario"] = "Boom"
    return df_boom.reset_index(drop=True)

def crash_scenario_enhanced(df, crash_start="2018-06-01", drop=-0.20, drop_months=6, recovery_annual=0.05):
    """
    Applies a 'crash' trend ON TOP OF the baseline data, preserving volatility.
    """
    df_crash = df.copy()
    m_drop = (1 + drop) ** (1/drop_months)
    m_recover = (1 + recovery_annual) ** (1/12)

    def calculate_crash(group):
        crash_date_ts = pd.Timestamp(crash_start)
        crash_idx_series = group.index[group['date'] == crash_date_ts]
        crash_idx = crash_idx_series[0] if not crash_idx_series.empty else group.index[len(group) // 2]
        
        # Price at the moment of the crash
        price_at_crash = group.loc[crash_idx, 'price']
        
        new_prices = []
        
        for i, row in enumerate(group.itertuples()):
            # Find the row's position relative to the crash start
            row_position = group.index.get_loc(row.Index)
            crash_start_position = group.index.get_loc(crash_idx)

            # Before the crash, the price is identical to the baseline.
            if row_position < crash_start_position:
                new_prices.append(row.price)
            else:
                # After the crash, calculate the crash/recovery effect.
                # This is the difference between a smooth curve starting from the crash price
                # and the crash price itself.
                months_after_crash = row_position - crash_start_position
                
                if months_after_crash < drop_months:
                    # During the drop period
                    smooth_price = price_at_crash * (m_drop ** (months_after_crash + 1))
                else:
                    # During the recovery period
                    price_at_bottom = price_at_crash * (m_drop ** drop_months)
                    months_into_recovery = months_after_crash - drop_months
                    smooth_price = price_at_bottom * (m_recover ** (months_into_recovery + 1))
                
                crash_effect = smooth_price - price_at_crash
                
                # Add the calculated effect to the actual baseline price for that month.
                new_prices.append(row.price + crash_effect)

        group['price'] = new_prices
        return group

    df_crash = df_crash.groupby('city').apply(calculate_crash)
    df_crash["scenario"] = "Crash"
    return df_crash.reset_index(drop=True)


# --- API ENDPOINT DEFINITION (Now calls the enhanced functions) ---
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
        
    print(f"-> Found {len(base_df)} rows. Generating enhanced scenarios...")
    
    # Create the scenarios
    baseline = base_df.copy()
    baseline["scenario"] = "Baseline"
    
    # --- CALL THE NEW, ENHANCED FUNCTIONS ---
    boom = boom_scenario_enhanced(base_df)
    crash = crash_scenario_enhanced(base_df)
    
    # Combine and return
    df_scenarios = pd.concat([baseline, boom, crash]).reset_index(drop=True)
    df_scenarios['date'] = df_scenarios['date'].dt.strftime('%Y-%m-%d')
    df_scenarios.rename(columns={'price': 'transaction_value'}, inplace=True)
    
    print("-> Scenarios generated. Sending JSON response.")
    return jsonify(df_scenarios.to_dict(orient='records'))