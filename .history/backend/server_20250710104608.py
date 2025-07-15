import json
import os 
import joblib
from quart import Quart, Response, current_app, request, jsonify
from quart_cors import cors
from db_connect import create_supabase
from routes.routes import main_routes
from routes.ml_routes import ml_routes
from routes.chat_routes import chat_routes
from routes.price_estimator_routes import init_price_estimator_routes
from routes.market_comparison import market_comparison_bp
 
app = Quart(__name__)
app = cors(app)

app = cors(app, allow_origin=[
    "http://localhost:3000",
    "http://127.0.0.1:3000"
])

BACKEND_DIR = os.path.dirname(os.path.abspath(__file__))
PRICE_MODEL_DIR = os.path.join(BACKEND_DIR, 'final_model_output')

price_estimator_bp = init_price_estimator_routes(PRICE_MODEL_DIR)



app.register_blueprint(main_routes)
app.register_blueprint(ml_routes)
app.register_blueprint(chat_routes, url_prefix="/api")
app.register_blueprint(price_estimator_bp)
app.register_blueprint(market_comparison, url_prefix="/compare")


supabase = None  # Declare the Supabase client variable

@app.before_serving
async def startup():
    global supabase
    current_app.supabase = await create_supabase()
    print("\n\nSupabase client created...\n\n")
    print("--- Registered API Routes ---")
    for rule in app.url_map.iter_rules():
        # Filter out the 'static' endpoint and OPTIONS method for clarity
        if rule.endpoint != 'static':
            methods = [method for method in rule.methods if method != 'OPTIONS']
            print(f"Endpoint: {rule.endpoint}, Methods: {methods}, URL: {rule.rule}")
    print("---------------------------")

# if __name__ == "__main__":
#     app.run(debug=True)

if __name__ == '__main__':
    app.run(port=5000, debug=True)