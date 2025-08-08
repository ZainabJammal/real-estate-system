import json
import os
import joblib
from sklearn.preprocessing import MinMaxScaler
import pandas as pd
import numpy as np
import traceback
from supabase import create_client, Client
from dotenv import load_dotenv
from models import get_models_path, get_enc_paths
from quart import Blueprint, jsonify, request,  current_app
from statsmodels.tsa.holtwinters import ExponentialSmoothing
from tensorflow.keras.models import load_model
from db_connect import create_supabase

# Create a Blueprint for your main routes
ml_routes = Blueprint('ml', __name__)


# Get models and encoders paths
trans_path, prop_path = get_models_path()
city_trans_enc_path, city_enc_path, dis_enc_path, prov_enc_path, type_enc_path = get_enc_paths()

# Import models and encoders
trans_model = joblib.load(trans_path)
prop_model = joblib.load(prop_path)
city_t_enc = joblib.load(city_trans_enc_path)
city_enc = joblib.load(city_enc_path)
dis_enc = joblib.load(dis_enc_path)
prov_enc = joblib.load(prov_enc_path)
type_enc = joblib.load(type_enc_path)

@ml_routes.route("/ml/city_circles", methods=["GET"])
async def city_circles():
    try:
        supabase = current_app.supabase
        result = await supabase.table("city_prices").select("city, listings_count, latitude, longitude").execute()
        data = result.data
        return jsonify(data)
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@ml_routes.route("/ml/city_price_trend", methods=["GET"])
async def city_price_trend():
    try:
        supabase = current_app.supabase

        # Call the SQL function we created
        result = await supabase.rpc("city_price_trend_medians").select("*").execute()
        query_result = result.data

        final_result = []

        def get_coordinates_for_city(city_name):
            # Static mapping — update with real coordinates!
            coords = {
                "Beirut": (33.8965, 35.4829),
                "Bekaa": (33.9105, 35.9631),
                "Baabda, Aley, Chouf": (33.7779, 35.5737),
                "Kesrouan, Jbeil": (34.1145, 35.6634), 
                "Tripoli, Akkar": (34.4887, 35.9544)
            }

            if not city_name:
                return (None, None)
            
            return coords.get(city_name.strip(), (None, None))

        for row in query_result:
            p2015 = row["median_price_2015"]
            p2016 = row["median_price_2016"]
            city = row["city"]

            if p2015 is None or p2016 is None:
                continue

            change = p2016 - p2015
            percent = (change / p2015) * 100

            direction = "neutral"
            if percent > 0.5: #use small threshold to avoid "up" for tiny changes
                direction = "up"
            elif percent < -0.5:
                direction = "down"

            lat, lng = get_coordinates_for_city(city)
            if lat is None or lng is None:
                continue

            final_result.append({
                "city": city,
                "direction": direction,
                "change_percent": round(percent, 2),
                "current_price": p2016, #price for most recent period
                "latitude": lat,
                "longitude": lng,
            })

        return jsonify(final_result)

    except Exception as e:
        import traceback
        traceback.print_exc()
        return jsonify({"error": str(e)}), 500


@ml_routes.route("/predict_transaction", methods=["GET","POST"])
async def predict_trans():
    """API endpoint for making predictions."""
    try:
        data = await request.get_json()
        
        # Convert input JSON to DataFrame
        input_data = pd.DataFrame([data])
        input_data["City"] = city_t_enc.transform([input_data["City"].iloc[0]])[0]

        # Convert data to float
        input_data = input_data.astype(float)
        # input_array = input_data.values.reshape(1, -1)

        # Make prediction
        prediction = trans_model.predict(input_data)
        print(prediction)

        return jsonify({"prediction": float(prediction[0])}) # Changed to prediction[0] for consistency

    except Exception as e:
        return jsonify({"error": str(e)}), 500
    
    
@ml_routes.route("/predict_property", methods=["POST"])
async def predict_prop():
    """API endpoint for making predictions."""
    try:
        data = await request.get_json()
        
        # Convert JSON to DataFrame
        input_data = pd.DataFrame([data])  

        # Encode categorical values correctly
        input_data["City"] = city_enc.transform([input_data["City"].iloc[0]])[0]
        input_data["District"] = dis_enc.transform([input_data["District"].iloc[0]])[0]
        input_data["Province"] = prov_enc.transform([input_data["Province"].iloc[0]])[0]
        input_data["Type"] = type_enc.transform([input_data["Type"].iloc[0]])[0]

        # Ensure the shape is correct
        input_data = input_data.astype(float)  # Convert to float for ML model
        input_array = input_data.values.reshape(1, -1)  # Ensure (1, n_features) shape

        print("Processed input:", input_array)  # Debugging

        # Make prediction
        prediction = prop_model.predict(input_array)
        print("Prediction:", prediction)

        return jsonify({"prediction": float(prediction[0])})

    except Exception as e:
        return jsonify({"error": str(e)}), 500



