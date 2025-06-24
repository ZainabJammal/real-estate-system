import pandas as pd
import os

def create_regional_data(input_path, output_path):
    """
    Reads the raw transaction data, aggregates cities into regions,
    and saves a new, clean CSV file ready for modeling.
    """
    print(f"-> Reading raw data from: {input_path}")
    if not os.path.exists(input_path):
        print(f"!! ERROR: Raw data file not found. Make sure 'transactions.csv' is inside the 'data' folder.")
        return

    df = pd.read_csv(input_path)

    # 1. Explode the city strings into individual rows
    df['city'] = df['city'].str.split(',').apply(lambda x: [c.strip() for c in x])
    df = df.explode('city')

    # 2. Define the region mapping logic - THIS IS THE SINGLE SOURCE OF TRUTH
    def map_city_to_region(city_name):
        if city_name in ["Tripoli", "Akkar"]:
            return "North"
        if city_name in ["Baabda", "Aley", "Chouf"]:
            return "Mount Lebanon South"
        if city_name in ["Kesrouan", "Jbeil", "Metn"]:
            return "Mount Lebanon North"
        # Default: city is its own region (e.g., "Beirut" -> "Beirut")
        return city_name

    # 3. Create the 'region' column
    df['region'] = df['city'].apply(map_city_to_region)
    print("-> Region mapping applied successfully.")

    # 4. Group by date and the new region, summing the values
    print("-> Aggregating data by new regions...")
    # We keep the 'id' column from the first entry as a placeholder
    regional_df = df.groupby(['date', 'region']).agg(
        transaction_value=('transaction_value', 'sum'),
        id=('id', 'first') # This ensures the 'id' column exists in the output file
    ).reset_index()

    # 5. Save the new, clean DataFrame to a new CSV file inside the 'data' folder
    regional_df.to_csv(output_path, index=False)
    print(f"-> Successfully created regional data at: {output_path}")
    print("\nColumns in new file:", regional_df.columns.tolist())
    print("Sample of new regional data:\n", regional_df.head())


if __name__ == "__main__":
    # Define the project's root directory (assuming this script is in the root)
    PROJECT_ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
    
    # Define the path to the 'data' folder
    DATA_FOLDER_PATH = os.path.join(PROJECT_ROOT_DIR, 'data')
    
    # Ensure the data folder exists
    os.makedirs(DATA_FOLDER_PATH, exist_ok=True)
    
    # Define the full paths for the input and output files
    RAW_CSV_PATH = os.path.join(DATA_FOLDER_PATH, 'transactions.csv')
    REGIONAL_CSV_PATH = os.path.join(DATA_FOLDER_PATH, 'regional_transactions.csv')
    
    create_regional_data(RAW_CSV_PATH, REGIONAL_CSV_PATH)