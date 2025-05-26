import json
import joblib
from quart import Quart, Response, request, jsonify
from quart_cors import cors
from db_connect import create_supabase
from routes.routes import main_routes
from routes.ml_routes import ml_routes
from routes.chat_routes import chat_routes


app = Quart(__name__)
app = cors(app, allow_origin="http://localhost:3000")
app.register_blueprint(main_routes)
app.register_blueprint(ml_routes)
app.register_blueprint(chat_routes, url_prefix="/api")


supabase = None  # Declare the Supabase client variable

@app.before_serving
async def startup():
    global supabase
    supabase = await create_supabase()
    print("\n\nSupabase client created...\n\n")
    
if __name__ == "__main__":
    app.run(debug=True)
    