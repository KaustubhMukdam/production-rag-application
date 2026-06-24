# Architecture — Production RAG Application

## System overview

A user submits a natural-language question. The query is embedded and run against a Qdrant collection using hybrid search (dense `bge-m3` vectors + sparse/BM25 signal), returning a candidate set of chunks. These candidates are reranked by a cross-encoder to fix ordering. The top reranked chunks are passed to an LLM (Ollama locally, or Groq for eval/demo) with a strict prompt requiring structured output: an answer plus the exact `source_chunk_id` it used. The system validates that the cited chunk actually exists in the retrieved set; if validation fails once, it retries with a stricter prompt; if it fails again, the system returns "insufficient context" rather than risk an unsupported answer. Every component is swappable behind a thin interface — the embedding model, vector store, reranker, and LLM provider are not hardwired into the pipeline logic.

## Component diagram (ASCII)

```
[User Query]
     |
     v
[Query Embedder] ──(bge-m3, dense + sparse)──┐
     |                                         │
     v                                         v
[Qdrant Hybrid Search] <───────────────────────┘
     |
     v
[Candidate Chunks (top-N)]
     |
     v
[Cross-Encoder Reranker] (ms-marco-MiniLM-L-6-v2)
     |
     v
[Top-K Reranked Chunks]
     |
     v
[LLM Generator] ──(Ollama local | Groq API)──┐
     |                                         │
     v                                         │
[Structured Output: {answer, source_chunk_id, supported}]
     |
     v
[Citation Validator] ── chunk_id not in retrieved set? ──> [Retry once, stricter prompt]
     |                                                              |
     v                                                              v
[Validated Answer] <──────────────────────── still invalid? ──> ["Insufficient context"]
     |
     v
[Response to User]
```

## Data flow

1. Documents are loaded and chunked (chunking strategy TBD in Phase 0 — start simple, revisit if retrieval quality suffers).
2. Each chunk is embedded with `bge-m3` (dense + sparse) and indexed into Qdrant.
3. User query arrives → embedded the same way → Qdrant hybrid search returns top-N candidates.
4. Candidates are reranked by the cross-encoder; top-K are selected for the generation prompt.
5. LLM receives the query + top-K chunks (each tagged with a `chunk_id`) and a system prompt enforcing a JSON schema response.
6. Response is parsed; `source_chunk_id` is checked against the actual retrieved chunk IDs.
7. If citation is invalid, one retry is attempted with a stricter prompt. If still invalid, the system fails closed with "insufficient context."
8. (CI/eval path) A fixed eval question set runs through this entire pipeline via Ragas, scoring Faithfulness / Answer Relevancy / Context Precision; GitHub Actions blocks merge if Faithfulness drops below 0.85.

## Key interfaces

- `Embedder.embed(text) -> Vector` — wraps `bge-m3`; swappable if a different embedding model is tried later.
- `Retriever.retrieve(query) -> List[Chunk]` — wraps Qdrant hybrid search; returns chunks with both hybrid score and (once reranked) rerank score attached.
- `Reranker.rerank(query, chunks) -> List[Chunk]` — wraps the cross-encoder; returns chunks reordered with rerank scores attached.
- `Generator.generate(query, chunks) -> StructuredAnswer` — wraps the LLM call; same interface regardless of whether the backend is Ollama or Groq (provider is a config value, not a code branch in calling code).
- `CitationValidator.validate(answer, chunks) -> ValidatedAnswer | InsufficientContext` — checks the LLM's claimed `source_chunk_id` against the actually-retrieved chunk set.

## Security considerations

- [ ] API keys (Groq) in environment variables only, never committed to the repo.
- [ ] No user-uploaded documents stored beyond the local Qdrant instance for this learning project — no multi-tenant data isolation needed at this scope.
- [ ] Input validation on any document ingestion path (file type, size limits) before chunking.

## Notes on future extension (not built now)

The `Retriever` interface returning both hybrid and rerank scores per chunk means a future CRAG-style grading step (deciding "is this retrieval good enough to answer from, or should we re-retrieve with a rewritten query") can be added as a new component sitting between `Reranker` and `Generator`, without changing either of those interfaces. This is documented here so the decision to defer CRAG (see PRD non-goals) doesn't quietly become "impossible to add later."
