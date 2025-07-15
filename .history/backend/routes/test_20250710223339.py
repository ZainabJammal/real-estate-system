import os
import traceback
import joblib
import warnings
import pandas as pd
import numpy as np
from quart import Blueprint, request, jsonify
from dotenv import load_dotenv
from quart import Quart, jsonify, request
from quart_cors import cors

import pandas as pd

# --- LOAD DATA ---
# Load the data from the CSV file into a DataFrame
df = pd.read_csv('properties.csv')

# --- VIEWING & INSPECTING DATA ---
print("--- Initial Data ---")
print(df.head())  # Show the first 5 rows

print("\n--- Data Info ---")
print(df.info())  # Get a summary of columns and data types

# --- QUERYING (SELECTING & FILTERING) ---

# 1. Select specific columns
size_property = df[['city','district', 'province','Price','size_m2']]
print("\n--- Selecting Columns (Product and Price) ---")
print(size_property.head())

# 2. Filter rows based on a condition (like a SQL WHERE clause)
# Get all orders from the 'Electronics' category
electronics_sales = df[df['type'] == 'size_property']
print("\n--- Filtering for Electronics ---")
print(electronics_sales)

# 3. Combine multiple conditions (use & for AND, | for OR)
# Get expensive electronics (Price > 100 AND Category is Electronics)
expensive_electronics = df[(df['Price'] > 100) & (df['Category'] == 'Electronics')]
print("\n--- Filtering for Expensive Electronics ---")
print(expensive_electronics)


# --- MANIPULATING DATA ---

# 1. Add a new column
# Calculate the total price for each order
df['TotalPrice'] = df['Price'] * df['Quantity']
print("\n--- Added a 'TotalPrice' column ---")
print(df.head())

# 2. Sort the data
# Sort by the new TotalPrice column in descending order
sorted_df = df.sort_values(by='TotalPrice', ascending=False)
print("\n--- Sorted by Total Price ---")
print(sorted_df.head())

# 3. Grouping and Aggregating data (like SQL GROUP BY)
# What is the total revenue per category?
category_revenue = df.groupby('Category')['TotalPrice'].sum()
print("\n--- Total Revenue by Category ---")
print(category_revenue)

# You can also do multiple aggregations
category_summary = df.groupby('Category').agg(
    TotalRevenue=('TotalPrice', 'sum'),
    AveragePrice=('Price', 'mean'),
    NumberOfOrders=('OrderID', 'count')
)
print("\n--- Detailed Summary by Category ---")
print(category_summary)