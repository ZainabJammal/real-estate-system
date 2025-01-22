import pandas as pd
import matplotlib.pyplot as plt

file_path = 'JSKRE.csv'  
df = pd.read_csv(file_path)


print(df.head())


print(df.info())
print(df.isnull().sum())


df['Price'] = df['Price'].replace('[^\d]', '', regex=True).astype(float)


df = df.dropna(subset=['Price'])


df['Type'] = df['Type'].str.strip().str.lower()


print("Cleaned Data:")
print(df.head())

avg_prices = df.groupby('Type')['Price'].mean()
print("Average Prices by Property Type:")
print(avg_prices)



avg_prices.plot(kind='line', marker='o', title='Average Prices by Property Type', color='blue')
plt.xlabel('Property Type')
plt.ylabel('Average Price')
plt.grid(True)
plt.savefig('avg_prices_by_type.png')
plt.show()