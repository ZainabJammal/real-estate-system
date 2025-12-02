# routes/chat_routes.py
import asyncio
from quart import Blueprint, request, jsonify
from dotenv import load_dotenv
import httpx
import os
from uuid import uuid4
import json
from quart import current_app


load_dotenv()

chat_routes = Blueprint('chat_routes', __name__)

HEADERS = {
    "Authorization": f"Bearer {os.getenv('OPENROUTER_API_KEY')}",
    "Content-Type": "application/json",
    "X-Title": "LebaneseRealEstateAI"
}

# System prompt with real estate focus
REAL_ESTATE_SYSTEM_PROMPT = """
You are a Lebanese real estate assistant. 
Automatically detect the language of the user's question.

If they write in Arabic, reply in Arabic.
If they write in English, reply in English.

Be clear, concise, and always mention:
- City or district
- Price range in USD
- Property type
- Price per m² and ROI if relevant
"""

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
            "model": "meta-llama/llama-3-8b-instruct",  
            "messages": messages,
            "temperature": 0.3,  # More factual responses
            "max_tokens": 2048,
        }

        try:
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
                        "usage": result.get("usage")
                    })

        except httpx.HTTPStatusError as e:
            # 🔍 Log more about OpenRouter's response
            error_content = e.response.text
            return jsonify({
                "error": f"OpenRouter HTTP error: {e.response.status_code}",
                "details": error_content
            }), 500

    except httpx.RequestError as e:
        return jsonify({"error": f"Network error: {str(e)}"}), 502
    except Exception as e:
        return jsonify({"error": f"Processing error: {str(e)}"}), 500
    
@chat_routes.route("/chat/last", methods=["GET"])
async def get_last_session():
    supabase = current_app.supabase
    try:
        response = await supabase.table("chat_sessions") \
            .select("id") \
            .order("created_at", desc=True) \
            .limit(1) \
            .execute()

        if response.data and len(response.data) > 0:
            return jsonify({"session_id": response.data[0]["id"]})
        else:
            return jsonify({"session_id": None})
    except Exception as e:
        return jsonify({"error": str(e)}), 500

    
@chat_routes.route("/chat/save", methods=["POST"])
async def save_chat_history():
    supabase = current_app.supabase
    data = await request.get_json()
    session_id = data.get("session_id") or str(uuid4())
    messages = data.get("messages", [])

    try:
        # Upsert chat session
        await supabase.table("chat_sessions").upsert({
            "id": session_id,
            "messages": messages
        }).execute()
        return jsonify({"session_id": session_id})
    except Exception as e:
        return jsonify({"error": str(e)}), 500
    
@chat_routes.route("/chat/history/<session_id>", methods=["GET"])
async def get_chat_history(session_id):
    supabase = current_app.supabase
    try:
        print(f"[INFO] Loading history for session_id: {session_id}")
        response = await supabase.table("chat_sessions").select("messages").eq("id", session_id).execute()
        if response.data and len(response.data) > 0:
            messages = response.data[0]["messages"]
        else:
            messages = []
        return jsonify({"messages": messages})
    except Exception as e:
        print(f"[ERROR] Chat history fetch failed: {e}")
        return jsonify({"messages": [], "error": str(e)}), 500


