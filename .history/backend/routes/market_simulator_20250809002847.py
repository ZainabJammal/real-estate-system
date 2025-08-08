import pandas as pd
from quart import Quart, jsonify
from quart_cors import cors # Essential for cross-domain requests from React dev server
from supabase import create_client, Client
from dotenv import load_dotenv
# --- Initialize the Quart App ---
app = Quart(__name__)
# Allow your React app (e.g., running on localhost:3000) to access the API
app = cors(app, allow_origin="*") 

# --- Load your data once on startup (or cache it) ---
# For performance, we load the base synthetic data into memory when the server starts.
try:
    SYNTHETIC_DF = pd.read_csv("synthetic_2017_2021.csv", parse_dates=["date"])
except FileNotFoundError:
    print("ERROR: synthetic_2017_2021.csv not found. Please ensure the file is in the correct directory.")
    SYNTHETIC_DF = None

# --- Scenario Generation Functions (from our previous discussion) ---
def boom_scenario(df, annual_growth=0.15):
    # This function creates the "Boom" data
    m = (1 + annual_growth) ** (1/12)
    df_boom = df.copy()
    
    # We base the growth on the *first* price point to create a smooth curve
    initial_price = df_boom["price"].iloc[0]
    growth_multipliers = [m ** i for i in range(len(df_boom))]
    df_boom["price"] = initial_price * pd.Series(growth_multipliers)
    
    df_boom["scenario"] = "Boom"
    return df_boom

def crash_scenario(df, crash_start="2018-06-01", drop=-0.20, drop_months=6, recovery_annual=0.05):
    # This function creates the "Crash" data
    m_drop = (1 + drop) ** (1/drop_months)
    m_recover = (1 + recovery_annual) ** (1/12)

    df_crash = df.copy()
    df_crash["scenario"] = "Crash"

    crash_idx_series = df_crash.index[df_crash["date"] == pd.Timestamp(crash_start)]
    if not crash_idx_series.empty:
        crash_idx = crash_idx_series[0]
    else: # Fallback if date not found
        crash_idx = len(df) // 2 

    prices = []
    # Start with the initial price from the baseline data
    price = df_crash["price"].iloc[0] 

    for i in range(len(df_crash)):
        current_price = df_crash["price"].iloc[i]
        # Before the crash, you can choose to follow the baseline or a stable path
        # Here we follow the baseline until the crash point
        if i < crash_idx:
            price = current_price
        # During the crash period
        elif crash_idx <= i < crash_idx + drop_months:
            price *= m_drop
        # After the crash, during recovery
        else:
            price *= m_recover
        prices.append(price)

    df_crash["price"] = prices
    return df_crash

# --- API Endpoint Definition ---
@app.route("/api/market-scenarios")
async def get_market_scenarios():
    if SYNTHETIC_DF is None:
        return jsonify({"error": "Data not available"}), 500

    # 1. Create Baseline
    baseline = SYNTHETIC_DF.copy()
    baseline["scenario"] = "Baseline"
    
    # 2. Generate other scenarios on-the-fly
    boom = boom_scenario(SYNTHETIC_DF)
    crash = crash_scenario(SYNTHETIC_DF)

    # 3. Combine into one DataFrame
    df_scenarios = pd.concat([baseline, boom, crash]).reset_index(drop=True)
    
    # 4. Convert date to string for JSON compatibility and return
    df_scenarios['date'] = df_scenarios['date'].dt.strftime('%Y-%m-%d')
    
    # The .to_dict(orient='records') is the perfect format for React
    return jsonify(df_scenarios.to_dict(orient='records'))

