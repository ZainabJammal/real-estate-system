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
    try:
        res = await supabase.from_("transactions").select().execute()
        return jsonify(res.data[0]), 200
    except Exception as e:
        return jsonify({"Error":str(e)}), 400

# GET users to/from the server
@app.route("/max_price", methods=["GET"])
async def get_max_prices():
    try:
        res = await supabase.from_("district_prices").select("city, max_price_$").order("max_price_$", desc=True).limit(1).execute()
        return jsonify({"title": "Max Price", "num": res.data[0]["max_price_$"], "region": res.data[0]["city"]}), 200
    except Exception as e:
        return jsonify({"Error":str(e)}), 400
    
# GET users to/from the server
@app.route("/min_price", methods=["GET"])
async def get_min_prices():
    try:
        res = await supabase.from_("district_prices").select("city, min_price_$").order("min_price_$", desc=False).limit(1).execute()
        return jsonify({"title": "Min Price", "num": res.data[0]["min_price_$"], "region": res.data[0]["city"]}), 200
    except Exception as e:
        return jsonify({"Error":str(e)}), 400
    
# GET users to/from the server
@app.route("/list_num", methods=["GET"])
async def get_list_num():
    try:
        res = await supabase.from_("district_prices").select("listings_count.sum()").execute()
        print(res.data)
        return jsonify({"title": "Number of Lists", "num": res.data[0]["sum"], "region": "All regions"}), 200
    except Exception as e:
        return jsonify({"Error":str(e)}), 400

    
if __name__ == "__main__":
    app.run(debug=True)
    