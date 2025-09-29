# app.py

from server.embeddings import Embedder
from server.api import build_context_and_ask
from fastapi import FastAPI, UploadFile, File
from pydantic import BaseModel
import faiss
import pickle
from sentence_transformers import SentenceTransformer
from server.api import build_context_and_ask

INDEX_PATH = "server/faiss_index.pkl"

# ---- FastAPI App ----
app = FastAPI(title="Intelligent Chatbot Builder - API")

# ---- Load FAISS Index ----
try:
    with open("server/faiss_index.pkl", "rb") as f:
        index, documents = pickle.load(f)
    print(f" FAISS index loaded successfully. Total docs: {len(documents)}")
except Exception as e:
    print(f" Could not load FAISS index. Context search will be disabled.\nError: {e}")
    index, documents = None, []

# ---- Load Embedding Model ----
try:
    embedder = Embedder()  # Uses your embeddings.py
    print(" Local embedding model loaded successfully.")
except Exception as e:
    print(f" Failed to load embedding model: {e}")
    embedder = None


class Query(BaseModel):
    question: str


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/chat")
def chat(query: Query):
    try:
        context_docs = []

        if index is not None and embedder:
            try:
                q_vector = embedder.encode(query.question)
                q_vector = q_vector.astype("float32").reshape(1, -1)
                D, I = index.search(q_vector, k=3)

                for idx in I[0]:
                    if idx < len(documents):
                        context_docs.append({
                            "source": f"doc_{idx}",
                            "text": documents[idx]
                        })
            except Exception as search_err:
                print(f"⚠️ FAISS search failed: {search_err}")

        # Ask the model with or without context
        answer = build_context_and_ask(query.question, context_docs)

        # Handle bad responses (timeouts / HTML error pages)
        if isinstance(answer, str) and (
            answer.strip().startswith("<!DOCTYPE html>") or "504" in answer
        ):
            answer = " Model request timed out or failed. Please try again later."

        return {
            "question": query.question,
            "answer": answer,
            "context_docs": context_docs
        }

    except Exception as e:
        return {
            "question": query.question,
            "answer": f" Internal server error: {str(e)}",
            "context_docs": []
        }


@app.post("/upload")
async def upload(file: UploadFile = File(...)):
    try:
        # Save file temporarily
        file_path = f"data/{file.filename}"
        with open(file_path, "wb") as f:
            f.write(await file.read())

        # Call ingest script logic to rebuild FAISS
        from server.ingest import main as ingest_main
        ingest_main()

        return {"message": f"File {file.filename} indexed successfully!", "total_docs": len(documents)}

    except Exception as e:
        return {"error": str(e)}