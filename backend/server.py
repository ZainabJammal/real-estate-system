from quart import Quart, request, jsonify
from quart_cors import cors
from db_connect import create_supabase

app = Quart(__name__)
app = cors(app)

supabase = None  # Declare the Supabase client variable

@app.before_serving
async def startup():
    global supabase
    supabase = await create_supabase()
    print("\n\nSupabase client created...\n\n")

# POST, GET users to/from the server
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
    
# GET users to/from the server
@app.route("/transactions", methods=["GET"])
async def transactions():
    if request.method == "GET":
        try:
            await supabase.from_("transactions").select().execute()
            return jsonify({"Success": "Data Inserted"}), 201
        except Exception as e:
            return jsonify({"Error":str(e)}), 400
    else:
        response = await supabase.from_("user").select().execute()
        return jsonify(response.data), 200


    
if __name__ == "__main__":
    app.run(debug=True)
    