import faiss, os, pickle
from typing import List, Dict
import numpy as np

class FaissStore:
    def __init__(self, dim: int, index_path: str, meta_path: str):
        self.dim = dim
        self.index_path = index_path
        self.meta_path = meta_path
        os.makedirs(os.path.dirname(index_path), exist_ok=True)

        if os.path.exists(index_path) and os.path.exists(meta_path):
            try:
                self.index = faiss.read_index(index_path)
            except Exception:
                # fallback to new index if read fails
                self.index = faiss.IndexFlatIP(dim)
            with open(meta_path, "rb") as f:
                self.metadata = pickle.load(f)
        else:
            self.index = faiss.IndexFlatIP(dim)
            self.metadata = []

    def add(self, vectors: np.ndarray, metadatas: List[Dict]):
        if vectors is None or len(vectors) == 0:
            return
        self.index.add(vectors.astype('float32'))
        self.metadata.extend(metadatas)
        self._save()

    def search(self, q_vec: np.ndarray, top_k: int = 5):
        if self.index.ntotal == 0:
            return []
        D, I = self.index.search(q_vec.astype('float32'), top_k)
        results = []
        for score, idx in zip(D[0], I[0]):
            if idx < 0 or idx >= len(self.metadata):
                continue
            m = self.metadata[idx].copy()
            m["score"] = float(score)
            results.append(m)
        return results

    def _save(self):
        faiss.write_index(self.index, self.index_path)
        with open(self.meta_path, "wb") as f:
            pickle.dump(self.metadata, f)
