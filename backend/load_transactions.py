import pandas as pd
from db_connect import create_supabase
import asyncio

# Open the csv files and save it into a DataFrame
df = pd.read_csv("../Transactions/merged_all_cities.csv")

supabase = None

async def establish_connection():
    global supabase
    supabase = await create_supabase()
    print("\nDatabase connection established...\n")

async def insert_data_from_csv():
    data = None
    for i in range(len(df["Date"])):
        object = { "date": df["Date"][i],
                    "transaction_number": int(df["Transaction_Number"][i]),
                    "city": str(df["City"][i]),
                    "transaction_value": df["Transaction_Value"][i], }
        print(type(object["transaction_number"]))
        try:
            data = await supabase.from_("transactions").upsert(object).execute()
            print(f"Inserted row: {i}")

        except Exception as e:
            print(f"Error occured: {e}")
            return
        
    print("Data inserted successfully!")

async def main():
    print("Main function is running")
    results = await asyncio.gather(establish_connection(), insert_data_from_csv())
    print("Main function is done")
    print(results)

if __name__ == '__main__':
    asyncio.run(main())