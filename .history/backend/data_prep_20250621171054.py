import pandas as pd
import os

def create_regional_data(input_path, output_path):
    """
    Reads raw data, correctly converts dates, aggregates cities into regions,
    and saves a new, clean CSV file with dates in 'YYYY-MM-DD' format.
    """
    print(f"-> Reading raw data from: {input_path}")
    if not os.path.exists(input_path):
        print(f"!! ERROR: Raw data file not found. Make sure 'transactions.csv' is inside the 'data' folder.")
        return

    df = pd.read_csv(input_path)

    # --- THIS IS THE NEW, CRITICAL STEP ---
    # Convert the original 'Mon-YY' format to a proper datetime object first.
    print("-> Converting original date strings to datetime objects...")
    parts = df['date'].str.split('-', expand=True)
    df['date'] = pd.to_datetime('01-' + parts[1] + '-' + parts[0], format='%d-%b-%y')

    # Now the 'date' column is a true datetime object, e.g., 2011-04-01 00:00:00

    # 1. Explode the city strings into individual rows
    df['city'] = df['city'].str.split(',').apply(lambda x: [c.strip() for c in x])
    df = df.explode('city')

    # 2. Define the region mapping logic
    def map_city_to_region(city_name):
        if city_name in ["Tripoli", "Akkar"]: return "North"
        if city_name in ["Baabda", "Aley", "Chouf"]: return "Mount Lebanon South"
        if city_name in ["Kesrouan", "Jbeil", "Metn"]: return "Mount Lebanon North"
        return city_name

    df['region'] = df['city'].apply(map_city_to_region)
    print("-> Region mapping applied.")

    # 4. Group by date and the new region, summing the values
    print("-> Aggregating data by new regions...")
    regional_df = df.groupby(['date', 'region']).agg(
        transaction_value=('transaction_value', 'sum'),
        id=('id', 'first')
    ).reset_index()

    # 5. Save the new, clean DataFrame. Because 'date' is a proper datetime object,
    #    pandas will automatically save it in the standard 'YYYY-MM-DD' format.
    regional_df.to_csv(output_path, index=False)
    print(f"-> Successfully created regional data at: {output_path}")
    print("\nColumns in new file:", regional_df.columns.tolist())
    print("Sample of new regional data:\n", regional_df.head())