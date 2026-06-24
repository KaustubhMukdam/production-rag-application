from typing import List
import numpy as np
from sentence_transformers import SentenceTransformer

from app.config import EMBEDDING_MODEL


class Embedder:
    def __init__(self, model_name: str = EMBEDDING_MODEL):
        self._model = SentenceTransformer(model_name)

    def embed(self, texts: List[str]) -> np.ndarray:
        if not texts:
            return np.empty((0, self._model.get_sentence_embedding_dimension()), dtype=np.float32)
        return self._model.encode(texts, normalize_embeddings=True, convert_to_numpy=True)
