# server/ingest.py

import os
import faiss
import numpy as np
import pickle
from server.embeddings import Embedder

DATA_DIR = "data"


def load_documents():
    docs = []
    for filename in os.listdir(DATA_DIR):
        filepath = os.path.join(DATA_DIR, filename)
        if filename.endswith(".txt"):
            with open(filepath, "r", encoding="utf-8") as f:
                docs.append(f.read())
    return docs


def main():
    embedder = Embedder()
    docs = load_documents()

    if not docs:
        print("No documents found in data/ folder")
        return

    print(f"Loaded {len(docs)} documents")

    # Create embeddings
    vectors = embedder.encode(docs)
    vectors = np.array(vectors).astype("float32")

    # Store in FAISS
    index = faiss.IndexFlatL2(vectors.shape[1])
    index.add(vectors)

    print(f"Indexed {index.ntotal} documents into FAISS")

    # Save index + documents
    with open("server/faiss_index.pkl", "wb") as f:
        pickle.dump((index, docs), f)

    print("Indexed documents saved to server/faiss_index.pkl")


if __name__ == "__main__":
    main()
