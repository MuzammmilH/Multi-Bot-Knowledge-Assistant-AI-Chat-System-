# rag.py

from fastapi import FastAPI
from pydantic import BaseModel
from server.config import settings
from server.embeddings import Embedder
from server.vectorstore import FaissStore
from server.retriever import Retriever
from server.rag import build_context_and_ask

app = FastAPI(title="Intelligent Chatbot Builder - API")

class ChatRequest(BaseModel):
    query: str
    persona: str | None = "Neutral Assistant"
    top_k: int | None = 5

class ChatResponse(BaseModel):
    answer: str
    sources: list

# --- Initialization ---
try:
    # Load embedder and detect dimension dynamically
    embedder = Embedder(settings.EMBEDDING_MODEL)
    dim = embedder.encode(["test"]).shape[1]

    # Try to load FAISS store
    try:
        store = FaissStore(dim, settings.FAISS_INDEX_PATH, settings.METADATA_PATH)
    except Exception as e:
        print(f"⚠️ Warning: Could not load FAISS index. Running in 'no context' mode.\nError: {e}")
        store = None

    retriever = Retriever(embedder=embedder, store=store)

except Exception as e:
    print(f" Critical: Failed to initialize embedder or retriever: {e}")
    retriever = None
    embedder = None


@app.post("/chat", response_model=ChatResponse)
def chat(req: ChatRequest):
    """
    Chat endpoint that retrieves relevant docs (if available) and queries the model.
    Falls back to general knowledge if FAISS or retriever fails.
    """
    retrieved = []
    if retriever:
        try:
            retrieved = retriever.get_relevant(req.query)
        except Exception as e:
            print(f"⚠️ Retriever failed: {e}")

    answer = build_context_and_ask(req.query, retrieved, persona=req.persona)
    sources = [{"source": r.get("source"), "score": r.get("score", None)} for r in retrieved]

    return ChatResponse(answer=answer, sources=sources)
