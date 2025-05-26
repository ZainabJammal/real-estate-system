# routes/chat_routes.py

import asyncio
from quart import Blueprint, request, jsonify
from dotenv import load_dotenv
import httpx
import os

load_dotenv()

chat_routes = Blueprint('chat_routes', __name__)

# # Optional: Set your OpenAI API key directly here or from env variable
# openai.api_key = os.getenv("OPENAI_API_KEY", "sk-...")  # Use your key or set in .env

# Initialize OpenAI client with API key
# client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY")

HEADERS = {
    "Authorization": f"Bearer {OPENROUTER_API_KEY}",
    "Content-Type": "application/json",
    "HTTP-Referer": "http://localhost:3000/",  # Replace with your deployed frontend URL
    "X-Title": "RealEstateAssistant"
}

@chat_routes.route("/chat", methods=["POST"])
async def chat_with_ai():
    data = await request.get_json()
    messages = data.get("messages")

    if not messages or not isinstance(messages, list):
        return jsonify({"error": "Invalid or missing 'messages' list"}), 400
    
    # Make sure system prompt exists, or add default
    system_prompt_exists = any(m.get("role") == "system" for m in messages)
    if not system_prompt_exists:
        messages.insert(0, {"role": "system", "content": "You are a helpful real estate assistant."})
    
    payload = {
    "model": "mistralai/mistral-7b-instruct",  # You can change the model below
    "messages": messages
}

    try:
        async with httpx.AsyncClient() as client:
            response = await client.post(
                "https://openrouter.ai/api/v1/chat/completions",
                headers=HEADERS,
                json=payload,
                timeout=15
            )

        # response = await client.chat.completions.create(
        #     model="gpt-3.5-turbo",
        #     messages=messages,
        #     temperature=0.7,
        #     max_tokens=300,
        # )
        # reply = response.choices[0].message.content
        # return jsonify({"reply": reply})

        response.raise_for_status()
        result = response.json()
        reply = result["choices"][0]["message"]["content"]
        return jsonify({"reply": reply})

    # except Exception as e:
    #     return jsonify({"error": str(e)}), 500

    except httpx.RequestError as e:
        return jsonify({"error": f"Request error: {str(e)}"}), 500
    except httpx.HTTPStatusError as e:
        return jsonify({"error": f"Error code: {e.response.status_code} - {e.response.text}"}), e.response.status_code
