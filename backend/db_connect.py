import os
from dotenv import load_dotenv
from supabase import create_client, acreate_client
load_dotenv()

url = os.environ.get("SUPABASE_URL")
key = os.environ.get("SUPABASE_KEY")

# Establish connection to the database
async def create_supabase():
    return await acreate_client(
        url,
        key,
    )