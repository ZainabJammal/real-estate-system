# routes/chat_routes.py
import asyncio
from quart import Blueprint, request, jsonify
from dotenv import load_dotenv
import httpx
import os

load_dotenv()

chat_routes = Blueprint('chat_routes', __name__)

HEADERS = {
    "Authorization": f"Bearer {os.getenv('OPENROUTER_API_KEY')}",
    "Content-Type": "application/json",
    "X-Title": "LebaneseRealEstateAI"
}

# System prompt with real estate focus
REAL_ESTATE_SYSTEM_PROMPT = """You are an expert Lebanese real estate assistant. 
Provide accurate, concise answers about property prices, trends, and recommendations. 
When discussing prices, always mention:
- City/district
- Price range (USD)
- Property type
- Key metrics (price per m², ROI)"""

@chat_routes.route("/chat", methods=["POST"])
async def chat_with_ai():
    try:
        data = await request.get_json()
        messages = data.get("messages", [])

        # Validate messages input
        if not isinstance(messages, list):
            return jsonify({"error": "Invalid 'messages' list"}), 400
        
        # Inject real estate context if missing
        if not any(m.get("role") == "system" for m in messages):
            messages.insert(0, {"role": "system", "content": REAL_ESTATE_SYSTEM_PROMPT})
        
        payload = {
            "model": "mistralai/mistral-7b-instruct",  # Consider gpt-3.5-turbo for better quality
            "messages": messages,
            "temperature": 0.3,  # More factual responses
            "max_tokens": 500
        }

        async with httpx.AsyncClient(timeout=30.0) as client:
            response = await client.post(
                "https://openrouter.ai/api/v1/chat/completions",
                headers=HEADERS,
                json=payload
            )
        response.raise_for_status()
        result = response.json()
        return jsonify({
                    "reply": result["choices"][0]["message"]["content"],
                    "usage": result.get("usage")  # Return token usage for monitoring
                })

    except httpx.RequestError as e:
        return jsonify({"error": f"Network error: {str(e)}"}), 502
    except Exception as e:
        return jsonify({"error": f"Processing error: {str(e)}"}), 500