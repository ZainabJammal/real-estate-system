import pandas as pd


df = pd.read_csv('JSKREfullbuy.csv')


df = df[~df['Address'].str.contains('France', case=False)]


df['Size'] = df['Size'].str.replace(' m²', '').str.replace(',', '')


df['Price'] = df['Price'].replace({'\$': '', ',': ''}, regex=True)


df.rename(columns={'Price': 'Price $'}, inplace=True)
df.rename(columns={'Size': 'Size m²'}, inplace=True)


df['Bedrooms'] = df['Bedrooms'].apply(lambda x: 'N/A' if x == 'N/A' else (int(x) if pd.notna(x) else 'N/A'))
df['Bathrooms'] = df['Bathrooms'].apply(lambda x: 'N/A' if x == 'N/A' else (int(x) if pd.notna(x) else 'N/A'))


df.to_csv('cleanjskbuy.csv', index=False)

print("CSV file cleaned and saved as 'cleanjskbuy.csv'.")