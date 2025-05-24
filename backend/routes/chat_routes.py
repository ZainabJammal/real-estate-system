# routes/chat_routes.py

import asyncio
from quart import Blueprint, request, jsonify
from dotenv import load_dotenv
from openai import OpenAI
from openai.types.chat import ChatCompletionMessageParam
import os

load_dotenv()

chat_routes = Blueprint('chat_routes', __name__)

# # Optional: Set your OpenAI API key directly here or from env variable
# openai.api_key = os.getenv("OPENAI_API_KEY", "sk-...")  # Use your key or set in .env

# Initialize OpenAI client with API key
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

@chat_routes.route("/chat", methods=["POST"])
async def chat_with_ai():
    data = await request.get_json()
    user_message = data.get("message")

    if not user_message:
        return jsonify({"error": "No message provided"}), 400

    try:
        messages: list[ChatCompletionMessageParam] = [
            {"role": "system", "content": "You are a real estate assistant."},
            {"role": "user", "content": user_message},
        ]

        response = await client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=messages,
            temperature=0.7,
            max_tokens=300,
        )
        reply = response.choices[0].message.content
        return jsonify({"reply": reply})

    except Exception as e:
        return jsonify({"error": str(e)}), 500
