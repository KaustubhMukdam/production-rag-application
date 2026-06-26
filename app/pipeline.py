import logging

from app.config import TOP_K, TOP_N
from app.ingestion.chunker import Chunker
from app.embeddings.embedder import Embedder
from app.retrieval.retriever import QdrantHybridRetriever
from app.retrieval.reranker import Reranker
from app.generation.generator import create_generator
from app.generation.schema import Answer, parse_answer
from app.generation.validation import CitationValidator

logger = logging.getLogger(__name__)


class Pipeline:
    def __init__(self):
        self.chunker = Chunker()
        self.embedder = Embedder()
        self.retriever = QdrantHybridRetriever()
        self.reranker = Reranker()
        self.generator = create_generator()
        self._indexed = False

    def index_documents(self, documents: list) -> None:
        chunks = self.chunker.chunk_documents(documents)
        texts = [c["text"] for c in chunks]
        dense_vecs = self.embedder.embed(texts)
        sparse_vecs = self.embedder.embed_sparse(texts)
        self.retriever.index(dense_vecs, sparse_vecs, chunks)
        self._indexed = True

    def query(self, question: str) -> dict:
        if not self._indexed:
            raise RuntimeError("No documents indexed. Call index_documents() first.")
        dense = self.embedder.embed([question])[0]
        sparse = self.embedder.embed_sparse([question])[0]
        retrieved = self.retriever.retrieve(dense, sparse, TOP_N)
        reranked = self.reranker.rerank(question, retrieved)[:TOP_K]

        raw = self.generator.generate(question, reranked)
        structured = self._parse_and_validate(raw, reranked, question)

        return {
            "question": question,
            "answer": structured.answer,
            "retrieved_chunks": reranked,
            "structured_answer": {
                "answer": structured.answer,
                "source_chunk_ids": structured.source_chunk_ids,
                "supported": structured.supported,
            },
        }

    def _parse_and_validate(self, raw: str, reranked: list, question: str) -> Answer:
        answer = parse_answer(raw)
        if answer and CitationValidator.validate(answer, reranked):
            logger.info("Citation validation passed on first attempt")
            return answer

        if answer:
            logger.warning(
                "Citation validation failed. Retrying with strict prompt. "
                "Cited: %s, Available: %s",
                answer.source_chunk_ids,
                {c["chunk_id"] for c in reranked},
            )
        else:
            logger.warning("Failed to parse LLM response as JSON. Retrying with strict prompt.")

        raw_retry = self.generator.generate(question, reranked, strict=True)
        answer_retry = parse_answer(raw_retry)

        if answer_retry and CitationValidator.validate(answer_retry, reranked):
            logger.info("Citation validation passed on retry")
            return answer_retry

        fallback_text = (
            answer_retry.answer if answer_retry else (answer.answer if answer else raw)
        )
        logger.warning("Citation validation failed after retry. Returning unsupported answer.")
        return Answer.unsupported(fallback_text)
