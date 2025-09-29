# server/model_load.py

import os
from openai import OpenAI

# Load Hugging Face API token from environment (Required)
HF_API_TOKEN = os.getenv("HF_API_TOKEN", "").strip()
if not HF_API_TOKEN:
    print(" Warning: No Hugging Face API token found! Please set HF_API_TOKEN in your environment.")

# Initialize Hugging Face Router Client using OpenAI interface
client = OpenAI(
    base_url="https://router.huggingface.co/v1",
    api_key=HF_API_TOKEN,
)

MODEL_NAME = "meta-llama/Meta-Llama-3-8B-Instruct"


def chat_with_model(prompt: str, context: str = "") -> str:
    try:
        messages = [
            {
                "role": "system",
                "content": (
                    "You are a helpful AI assistant. Use the context below to answer the user's question. "
                    "If the context is irrelevant, ignore it and answer normally."
                ),
            },
            {
                "role": "user",
                "content": f"Context:\n{context}\n\nQuestion:\n{prompt}\n\nAnswer in a helpful and concise way:",
            },
        ]

        response = client.chat.completions.create(
            model=MODEL_NAME,
            messages=messages,
            max_tokens=250,
        )

        return response.choices[0].message.content.strip()

    except Exception as e:
        return f" Error communicating with model: {str(e)}"


