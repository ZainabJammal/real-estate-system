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
size_property = df[['city','district', 'province','size_m2']]
outputdf= df[df['type'] == 'Apartment'].sort_values(by='size_m2')
print(outputdf)

max_apartment_size = df[df['type'] == 'Apartment']['size_m2'].max()
print(max_apartment_size)
largest_apartment_row = df[df['type'] == 'Apartment'].sort_values('size_m2', ascending=False).head(1)

print("\n--- Entire row for the largest apartment (Method A) ---")
print(largest_apartment_row)


# Find the 99th percentile for apartment sizes
p99 = df[df['type'] == 'Apartment']['size_m2'].quantile(0.99)
print(f"99% of apartments in this dataset are smaller than {p99:.2f} m².")

# Find the 99th percentile for apartment sizes
p10 = df[df['type'] == 'Apartment']['size_m2'].quantile(0.10)
print(f"99% of apartments in this dataset are smaller than {p10:.2f} m².")
# print(size_property.head())



# # 2. Filter rows based on a condition (like a SQL WHERE clause)
# # Get all orders from the 'Electronics' category
# # sizes = df[df['type'] == 'size_property']
# # print("\n")
# # print(sizes)

# # # 3. Combine multiple conditions (use & for AND, | for OR)
# # # Get expensive electronics (Price > 100 AND Category is Electronics)
# # expensive_electronics = df[(df['Price'] > 100) & (df['Category'] == 'Electronics')]
# # print("\n--- Filtering for Expensive Electronics ---")
# # print(expensive_electronics)


# # --- MANIPULATING DATA ---

# # # 1. Add a new column
# # # Calculate the total price for each order
# # df['TotalPrice'] = df['Price'] * df['Quantity']
# # print("\n--- Added a 'TotalPrice' column ---")
# # print(df.head())

# # 2. Sort the data
# # Sort by the new TotalPrice column in descending order
# sorted_df = df.sort_values(by='size_m2', ascending=False)
# print("\n--- Sorted by Total Price ---")
# print(sorted_df.head())

# # # 3. Grouping and Aggregating data (like SQL GROUP BY)
# # # What is the total revenue per category?
# # category_revenue = df.groupby('Category')['TotalPrice'].sum()
# # print("\n--- Total Revenue by Category ---")
# # print(category_revenue)

# # # You can also do multiple aggregations
# # sorted_df = df.orderby('Category').agg(
# #     TotalRevenue=('TotalPrice', 'sum'),
# #     AveragePrice=('Price', 'mean'),
# #     NumberOfOrders=('OrderID', 'count')
# # )
# print("\n--- Detailed Summary by Category ---")
# print(sorted_df)