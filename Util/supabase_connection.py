import os
import asyncio
from supabase import create_client, acreate_client
from dotenv import load_dotenv
load_dotenv()



url = os.environ.get("SUPABASE_URL")
key = os.environ.get("SUPABASE_KEY")

# Establish connection to the database
async def create_supabase():
    return await acreate_client(
        url,
        key,
    )

supabase = asyncio.run(create_supabase())

print("\n\nSupabase client created...\n\n")

# Function to get rows by ID
async def select_row_by_id(pr_key: int)-> list:
    if pr_key == None:
        print("No primary key provided! No rows selected.")
        return []
    
    data = await supabase.from_("properties").select("*").eq("id", pr_key).execute()

    if data == None:
        print("Empty data!")
        return []
    
    return data.data

# Function to get rows by Name
async def select_row_by_name(name: str)-> list:
    if name == None:
        print("No title provided! No rows selected.")
        return []
    
    data = await supabase.from_("properties").select("*").eq("property_title", name).execute()

    if data == None:
        print("Empty data!")
        return []
    
    return data.data

# Function to insert rows by passing property object
async def insert_row(property: dict)-> any:
    if property == None:
        return print("Empty data! No rows inserted.")
    
    data = await supabase.from_("properties").insert(property).execute()

    if data == None:
        return print("Error happened while inserting data!")
    return print("1 row inserted.\n\n")

# Dummy data to test the database
property = {
    "property_title":"Villa",
    "location":"Bekaa",
    "rooms":12,
    "floor":3,
    "negotiable":True,
    "price":300000,
    }
