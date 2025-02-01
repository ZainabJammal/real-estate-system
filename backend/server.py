from quart import Quart, request, jsonify
import os
from dotenv import load_dotenv
from supabase import create_client, acreate_client
from quart_cors import cors
load_dotenv()

app = Quart(__name__)
app = cors(app)

url = os.environ.get("SUPABASE_URL")
key = os.environ.get("SUPABASE_KEY")

# Establish connection to the database
async def create_supabase():
    return await acreate_client(
        url,
        key,
    )

supabase = None  # Declare the Supabase client variable

@app.before_serving
async def startup():
    global supabase
    supabase = await create_supabase()
    print("\n\nSupabase client created...\n\n")

@app.route("/user", methods=["POST", "GET"])
async def register():
    if request.method == "POST":
        data = await request.get_json()
        try:
            await supabase.from_("user").insert(data).execute()
            return jsonify({"Success": "Data Inserted"}), 201
        except Exception as e:
            return jsonify({"Error":str(e)}), 400
    else:
        response = await supabase.from_("user").select().execute()
        return jsonify(response.data), 200
    
if __name__ == "__main__":
    app.run(debug=True)
    