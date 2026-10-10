"""Rule retriever with a Windows-safe native-library loading order."""

import json
import threading

import numpy as np


class RuleRetriever:
    def __init__(self, index_path: str, document_path: str, top_k: int = 5):
        from src.embedding.embedder import Embedder

        # On Windows, initializing Torch/BGE before importing FAISS avoids an
        # OpenMP/native-runtime collision observed with faiss-cpu 1.15.0.
        try:
            self.embedder = Embedder()
        except OSError as exc:
            if getattr(exc, "winerror", None) == 1455 or "1455" in str(exc):
                raise RuntimeError(
                    "BGE model loading failed because the Windows paging file "
                    "is too small (error 1455). Increase Windows virtual "
                    "memory and restart before running the experiment again."
                ) from exc
            raise

        import faiss

        print("Loading FAISS Index...")
        self.index = faiss.read_index(index_path)
        print("Loading Rule Documents...")
        with open(document_path, "r", encoding="utf-8") as file:
            self.rules = json.load(file)
        self.top_k = top_k
        self._search_lock = threading.Lock()
        print("Rule Retriever Ready.")

    def retrieve(self, query: str):
        # All workers share one BGE model and one read-only FAISS index. Keep
        # local inference/search serialized while remote LLM requests run in
        # parallel, avoiding model thread-safety and memory-pressure issues.
        with self._search_lock:
            query_vector = np.asarray(
                self.embedder.encode(query), dtype=np.float32
            ).reshape(1, -1)
            distances, indices = self.index.search(query_vector, self.top_k)

        results = []
        for score, index in zip(distances[0], indices[0]):
            if index == -1:
                continue
            rule = self.rules[index].copy()
            rule["similarity"] = float(score)
            results.append(rule)
        return results
