import chainlit as cl
import os
import pickle
import json
import faiss
import numpy as np
from server.embeddings import Embedder
from server.api import build_context_and_ask
from server.config import settings

INDEX_DIR = "server/indexes"
DATA_DIR = "data"
os.makedirs(INDEX_DIR, exist_ok=True)

embedder = Embedder(settings.EMBEDDING_MODEL)

# Load bots config
with open("bots.json", "r", encoding="utf-8") as f:
    BOTS = json.load(f)

# Session state
session_state = {
    "selected_bot": BOTS[0],  # Default to first bot
    "index": None,
    "documents": []
}

def load_faiss_index(index_file):
    if not os.path.exists(index_file):
        return None, []
    with open(index_file, "rb") as f:
        index, docs = pickle.load(f)
    print(f"🔄 Loaded FAISS index for bot ({index_file}). Total docs: {len(docs)}")
    return index, docs

def rebuild_index(bot, docs):
    if not docs:
        print(f"⚠️ No documents found for bot {bot['name']}.")
        return None, []
    vectors = embedder.encode(docs).astype("float32")
    index = faiss.IndexFlatL2(vectors.shape[1])
    index.add(vectors)
    with open(bot["index_file"], "wb") as f:
        pickle.dump((index, docs), f)
    print(f"✅ Rebuilt FAISS index for {bot['name']} with {len(docs)} docs.")
    return index, docs

@cl.on_chat_start
async def start():
    # Let user pick a bot at the start
    options = [b["name"] for b in BOTS]
    selected = await cl.AskActionMessage(
        content="🤖 Select a bot to chat with:",
        actions=[cl.Action(name=o, label=o, payload={"bot_name": o}) for o in options]
    ).send()

# selected will be a dict with "name" and "payload"
    selected_bot_name = selected.get("payload", {}).get("bot_name", options[0])
    for bot in BOTS:
        if bot["name"] == selected_bot_name:
            session_state["selected_bot"] = bot
            break



    # Set selected bot
    for bot in BOTS:
        if bot["name"] == selected.get("name"):
            session_state["selected_bot"] = bot
            break

    # Load index for that bot
    session_state["index"], session_state["documents"] = load_faiss_index(bot["index_file"])

    await cl.Message(content=f"✅ You are now chatting with **{bot['name']}**!").send()

@cl.on_message
async def on_message(message: cl.Message):
    bot = session_state["selected_bot"]

    # Handle file uploads
    if message.elements:
        docs = session_state["documents"]
        for file in message.elements:
            if not file.name.endswith(".txt"):
                await cl.Message(content=f"⚠️ Unsupported file type: {file.name}. Only .txt is supported.").send()
                return

            bot_data_dir = os.path.join(DATA_DIR, bot["name"].replace(" ", "_"))
            os.makedirs(bot_data_dir, exist_ok=True)
            save_path = os.path.join(bot_data_dir, file.name)

            with open(file.path, "r", encoding="utf-8") as src, open(save_path, "w", encoding="utf-8") as dst:
                dst.write(src.read())
            docs.append(open(save_path, "r", encoding="utf-8").read())

        session_state["index"], session_state["documents"] = rebuild_index(bot, docs)
        await cl.Message(content=f"✅ Documents indexed for **{bot['name']}**! Now ask questions.").send()
        return

    # Handle normal user messages
    index, documents = session_state["index"], session_state["documents"]
    if index is None or len(documents) == 0:
        await cl.Message(content=f"⚠️ No documents found for {bot['name']}. Please upload a file first.").send()
        return

    try:
        q_vec = embedder.encode([message.content])
        D, I = index.search(q_vec, k=3)
        context_docs = [{"source": f"doc_{i}", "text": documents[i]} for i in I[0] if i < len(documents)]
        answer = build_context_and_ask(message.content, context_docs, persona=bot["persona"])
        await cl.Message(content=answer).send()
    except Exception as e:
        await cl.Message(content=f"⚠️ Error: {str(e)}").send()
