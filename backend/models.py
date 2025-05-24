import os
import requests
from dotenv import load_dotenv
load_dotenv()

SUPABASE_URL = os.environ.get("SUPABASE_URL")  # Load from environment
SUPABASE_KEY = os.environ.get("SUPABASE_KEY")
SUPABASE_BUCKET = "models"  # Bucket where models are stored

# Model file name (must match the name in Supabase Storage)
models = ["lgbm_model.pkl", "lgbm_property_model.pkl"]


# Import Encoders
encoders = ["city_trans_enc.pkl",
"city_prop_enc.pkl",
"dis_prop_enc.pkl",
"prov_prop_enc.pkl",
"type_prop_enc.pkl"]

# Local model storage path
model_paths = [f"./models/{model}" for model in models]

# Local city encoder storage path
encoders_paths = [f"./models/encoders/{encoder}" for encoder in encoders]

# Headers for Supabase requests
headers = {
    "apikey": SUPABASE_KEY,
    "Authorization": f"Bearer {SUPABASE_KEY}",
}


# Function to download model from Supabase
def download_model():
    i = 0
    while(i < len(model_paths)):
        url = f"{SUPABASE_URL}/storage/v1/object/public/{SUPABASE_BUCKET}/{models[i]}"
        response = requests.get(url, headers=headers)
        if response.status_code == 200:
            with open(model_paths[i], "wb") as f:
                f.write(response.content)
            print(f"Model {models[i]} downloaded successfully!")
            i += 1
        else:
            print("Failed to download model:", response.json())
    i = 0
    while(i < len(encoders_paths)):
        url = f"{SUPABASE_URL}/storage/v1/object/public/{SUPABASE_BUCKET}/{encoders[i]}"
        response = requests.get(url, headers=headers)
        if response.status_code == 200:
            with open(encoders_paths[i], "wb") as f:
                f.write(response.content)
            print(f"Encoder {encoders[i]} downloaded successfully!")
            i += 1
        else:
            print("Failed to download model:", response.json())

# Return the models' paths
def get_models_path():
    return (model_paths[0], model_paths[1])

# Return the encoders' paths
def get_enc_paths():
    return (encoders_paths[0], encoders_paths[1], encoders_paths[2], encoders_paths[3], encoders_paths[4])

# Download the model before starting the API
download_model()