from typing import List
import numpy as np
import faiss

from app.config import TOP_K


class InMemoryRetriever:
    def __init__(self, dimension: int):
        self._index = faiss.IndexFlatIP(dimension)
        self._chunks: List[dict] = []

    def index(self, embeddings: np.ndarray, chunks: List[dict]) -> None:
        if len(embeddings) == 0:
            return
        self._index.add(embeddings.astype(np.float32))
        self._chunks = chunks

    def retrieve(self, query_embedding: np.ndarray, k: int = TOP_K) -> List[dict]:
        if self._index.ntotal == 0:
            return []
        k = min(k, self._index.ntotal)
        scores, indices = self._index.search(query_embedding.astype(np.float32), k)
        results = []
        for i, idx in enumerate(indices[0]):
            results.append({
                **self._chunks[idx],
                "score": float(scores[0][i]),
            })
        return results

    @property
    def size(self) -> int:
        return self._index.ntotal
