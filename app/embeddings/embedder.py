from typing import List
import numpy as np
from fastembed import TextEmbedding, SparseTextEmbedding

from app.config import EMBEDDING_MODEL


class Embedder:
    def __init__(self, model_name: str = EMBEDDING_MODEL):
        self._dense_model = TextEmbedding(model_name)
        self._sparse_model = SparseTextEmbedding("Qdrant/bm25")

    def embed(self, texts: List[str]) -> np.ndarray:
        return np.array(list(self._dense_model.embed(texts)), dtype=np.float32)

    def embed_sparse(self, texts: List[str]) -> List[dict]:
        return [
            dict(zip(s.indices.tolist(), s.values.tolist()))
            for s in self._sparse_model.embed(texts)
        ]
