import os
import traceback
import joblib
import warnings
import pandas as pd
import numpy as np
from quart import Blueprint, request, jsonify
from dotenv import load_dotenv
from supabase import create_client, Client
from quart import Quart, jsonify, request
from quart_cors import cors


SUPABASE_URL: str = os.environ.get("SUPABASE_URL")
SUPABASE_KEY: str = os.environ.get("SUPABASE_KEY")

if not SUPABASE_URL or not SUPABASE_KEY:
    raise ValueError("FATAL ERROR: SUPABASE_URL and SUPABASE_KEY must be set in your .env file.")

# Initialize the Supabase client
supabase: Client = create_client(SUPABASE_URL, SUPABASE_KEY)
print("-> Supabase client initialized.")


load_dotenv() 


import sqlite3
import pandas as pd

# Imagine the sales data is in a database table named 'sales'

# --- CONNECT TO DATABASE ---
# For SQLite, this creates a file-based database.
# For other DBs, you'd provide host, user, password, etc.
conn = sqlite3.connect('my_database.db') 

# For this example, let's first load our CSV into the DB
# (You would normally already have data in your database)
df_from_csv = pd.read_csv('sales_data.csv')
df_from_csv.to_sql('sales', conn, if_exists='replace', index=False)


# --- WRITE AND EXECUTE A SQL QUERY ---

# Define your SQL query as a string
sql_query = """
SELECT
    Category,
    SUM(Price * Quantity) as TotalRevenue,
    AVG(Price) as AveragePrice
FROM
    sales
WHERE
    Category = 'Electronics'
GROUP BY
    Category;
"""

# --- LOAD QUERY RESULTS INTO A PANDAS DATAFRAME ---
# This is the magic! Pandas handles the connection and fetching.
electronics_summary_df = pd.read_sql_query(sql_query, conn)

print("\n--- Results of SQL Query loaded into Pandas ---")
print(electronics_summary_df)

# Close the connection when you're done
conn.close()