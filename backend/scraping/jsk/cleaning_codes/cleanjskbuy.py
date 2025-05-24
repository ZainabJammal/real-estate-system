import os
import pandas as pd

# # Get the absolute path of the current script
# BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# # Build the full path to the CSV file relative to the script location
# fullData_path = os.path.join(BASE_DIR, "../scraping/jsk/full_data/JSKREfullbuy.csv")
# cleanData_path = os.path.join(BASE_DIR, "../scraping/jsk/clean_data/cleanjskbuy.csv")

# # Read the CSV
# df = pd.read_csv(fullData_path)

df = pd.read_csv('JSKREfullbuy.csv')


df = df[~df['Address'].str.contains('France', case=False)]


df['Size'] = df['Size'].str.replace(' m²', '').str.replace(',', '')


df['Price'] = df['Price'].replace({'\$': '', ',': ''}, regex=True)


df.rename(columns={'Price': 'Price $'}, inplace=True)
df.rename(columns={'Size': 'Size m²'}, inplace=True)


df['Bedrooms'] = df['Bedrooms'].apply(lambda x: 'N/A' if x == 'N/A' else (int(x) if pd.notna(x) else 'N/A'))
df['Bathrooms'] = df['Bathrooms'].apply(lambda x: 'N/A' if x == 'N/A' else (int(x) if pd.notna(x) else 'N/A'))

# df.to_csv(cleanData_path, index=False)
df.to_csv('cleanjskbuy.csv', index=False)

print("CSV file cleaned and saved as 'cleanjskbuy.csv'.")