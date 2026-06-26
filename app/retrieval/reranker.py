from typing import List
from sentence_transformers import CrossEncoder

from app.config import RERANKER_MODEL


class Reranker:
    def __init__(self, model_name: str = RERANKER_MODEL):
        self._model = CrossEncoder(model_name)

    def rerank(self, query: str, chunks: List[dict]) -> List[dict]:
        if not chunks:
            return []

        pairs = [(query, chunk["text"]) for chunk in chunks]
        scores = self._model.predict(pairs)

        for chunk, score in zip(chunks, scores):
            chunk["rerank_score"] = float(score)

        chunks.sort(key=lambda x: x["rerank_score"], reverse=True)
        return chunks
