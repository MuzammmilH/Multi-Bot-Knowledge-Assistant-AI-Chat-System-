import os
import requests

HF_API_TOKEN = os.getenv("HF_API_TOKEN", "").strip()
if not HF_API_TOKEN:
    raise ValueError("HF_API_TOKEN not set!")

# Query Hugging Face router models
url = "https://router.huggingface.co/v1/models"
headers = {"Authorization": f"Bearer {HF_API_TOKEN}"}
response = requests.get(url, headers=headers)

if response.status_code == 200:
    models = [m["id"] for m in response.json()["data"]]
    print(f" You have access to {len(models)} models:")
    for m in models[:20]:  # show first 20
        print("-", m)
else:
    print(f"Failed to list models: {response.status_code} - {response.text}")
