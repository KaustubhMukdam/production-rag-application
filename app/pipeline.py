"""
RAG pipeline — Phase 4.

Changes from Phase 3:
  - Reranker confidence gate: if the top rerank_score is below
    RERANK_GATE_THRESHOLD the pipeline short-circuits and returns
    "insufficient context" without calling the LLM.  This is the
    forward-compatible CRAG hook documented in Architecture.md.
  - Every response now carries a boolean "gated" key so API callers
    can distinguish a gate-triggered refusal from a citation failure.
"""
import logging

from app.config import TOP_K, TOP_N, RERANK_GATE_THRESHOLD
from app.ingestion.chunker import SemanticChunker
from app.embeddings.embedder import Embedder
from app.retrieval.retriever import QdrantHybridRetriever
from app.retrieval.reranker import Reranker
from app.generation.generator import create_generator
from app.generation.schema import Answer, parse_answer
from app.generation.validation import CitationValidator

logger = logging.getLogger(__name__)

_GATE_MSG = "Insufficient context — no relevant chunks found."


class Pipeline:
    def __init__(self):
        self.chunker = SemanticChunker()
        self.embedder = Embedder()
        self.retriever = QdrantHybridRetriever()
        self.reranker = Reranker()
        self.generator = create_generator()
        # If Qdrant Cloud already has data from a previous session we can
        # serve queries immediately without re-embedding everything.
        existing = self.retriever.size
        self._indexed = existing > 0
        if self._indexed:
            logger.info(
                "Qdrant already has %d chunks — skipping startup re-index.",
                existing,
            )

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

        # ── Confidence gate (mini-CRAG) ──────────────────────────────────
        if not self._confidence_gate(reranked):
            logger.info(
                "Confidence gate fired: top rerank_score=%s is below threshold=%s.",
                reranked[0]["rerank_score"] if reranked else "N/A",
                RERANK_GATE_THRESHOLD,
            )
            fallback = Answer.unsupported(_GATE_MSG)
            return self._build_response(
                question=question,
                answer=fallback,
                provider="none",
                reranked=reranked,
                gated=True,
            )
        # ─────────────────────────────────────────────────────────────────

        raw = self.generator.generate(question, reranked)
        provider = getattr(self.generator, "provider", "unknown")
        structured = self._parse_and_validate(raw, reranked, question)

        return self._build_response(
            question=question,
            answer=structured,
            provider=provider,
            reranked=reranked,
            gated=False,
        )

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _confidence_gate(self, reranked: list) -> bool:
        """Return True when retrieval confidence is high enough to answer.

        The gate passes (returns True) when the top chunk's rerank_score
        is greater than or equal to RERANK_GATE_THRESHOLD.  An empty list
        always fails the gate.
        """
        if not reranked:
            return False
        return reranked[0]["rerank_score"] >= RERANK_GATE_THRESHOLD

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

    @staticmethod
    def _build_response(
        question: str,
        answer: Answer,
        provider: str,
        reranked: list,
        gated: bool,
    ) -> dict:
        return {
            "question": question,
            "answer": answer.answer,
            "provider": provider,
            "retrieved_chunks": reranked,
            "gated": gated,
            "structured_answer": {
                "answer": answer.answer,
                "source_chunk_ids": answer.source_chunk_ids,
                "supported": answer.supported,
            },
        }
