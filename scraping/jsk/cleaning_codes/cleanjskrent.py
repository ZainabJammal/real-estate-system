import pandas as pd


df = pd.read_csv('JSKREfullrent.csv')


df['Price'] = df['Price'].replace({'\$': '', ' /Year': '', ',': ''}, regex=True)


df['Size'] = df['Size'].str.replace(' m²', '').str.replace(',', '')


df.rename(columns={'Price': 'Rent $/year'}, inplace=True)


df.rename(columns={'Size': 'Size m²'}, inplace=True)


df['Bedrooms'] = df['Bedrooms'].apply(lambda x: 'N/A' if x == 'N/A' else (int(x) if pd.notna(x) else 'N/A'))
df['Bathrooms'] = df['Bathrooms'].apply(lambda x: 'N/A' if x == 'N/A' else (int(x) if pd.notna(x) else 'N/A'))


df.to_csv('cleanjskrent.csv', index=False)

print("CSV file cleaned and saved as 'cleanjskrent.csv'.")