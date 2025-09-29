# server/retriever.py

import os
from server.embeddings import Embedder
from server.vectorstore import FaissStore

class Retriever:
    def __init__(
        self,
        embedder: Embedder = None,
        store: FaissStore = None,
        top_k: int = 5,
        index_dir: str = "data"
    ):
        self.embedder = embedder or Embedder()

        # Determine embedding dimension dynamically
        try:
            dim = self.embedder.encode(["test"]).shape[1]
        except Exception:
            dim = 384  # fallback
            print("⚠️ Could not auto-detect embedding dimension. Using 384 as default.")

        if store:
            self.store = store
        else:
            os.makedirs(index_dir, exist_ok=True)
            index_path = os.path.join(index_dir, "faiss.index")
            meta_path = os.path.join(index_dir, "metadata.json")
            try:
                self.store = FaissStore(dim=dim, index_path=index_path, meta_path=meta_path)
            except Exception as e:
                print(f"⚠️ Failed to initialize FaissStore: {e}")
                self.store = None

        self.top_k = top_k

    def get_relevant(self, query: str):
        """
        Returns the top-k most relevant text chunks for a given query.
        """
        if not self.store:
            return []

        try:
            q_vec = self.embedder.encode([query])
            if q_vec is None or q_vec.shape[1] == 0:
                return []

            results = self.store.search(q_vec, top_k=self.top_k)

            formatted_results = []
            for res in results:
                if isinstance(res, dict):
                    formatted_results.append(res)
                elif hasattr(res, "text"):
                    formatted_results.append({
                        "source": getattr(res, "source", "unknown"),
                        "text": res.text
                    })
                else:
                    formatted_results.append({
                        "source": "unknown",
                        "text": str(res)
                    })

            return formatted_results

        except Exception as e:
            print(f"Retriever error: {e}")
            return []
