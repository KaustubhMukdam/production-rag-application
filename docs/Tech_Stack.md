# Tech Stack — Production RAG Application

## Backend

| Technology | Version       | Why chosen                                                                              |
| ---------- | ------------- | --------------------------------------------------------------------------------------- |
| Python     | 3.11          | ML/RAG library ecosystem (Qdrant client, sentence-transformers, Ragas all target 3.10+) |
| FastAPI    | latest stable | Async support, automatic OpenAPI docs, minimal boilerplate for a query endpoint         |

## Retrieval

| Technology             | Why chosen                                                                                                                   |
| ---------------------- | ---------------------------------------------------------------------------------------------------------------------------- |
| Qdrant (local Docker)  | Native hybrid search (dense + sparse) without hand-rolling Reciprocal Rank Fusion — reduces custom fusion code in Phase 1    |
| `BAAI/bge-m3`          | Single embedding model that natively supports both dense and sparse retrieval — avoids running two separate embedding models |
| `rank_bm25` (fallback) | If Qdrant's native sparse support proves awkward to configure, fall back to manual BM25 + RRF fusion                         |

## Reranking

| Technology                             | Why chosen                                                                                                                   |
| -------------------------------------- | ---------------------------------------------------------------------------------------------------------------------------- |
| `cross-encoder/ms-marco-MiniLM-L-6-v2` | Small (~22M params), CPU-friendly, well-established baseline reranker — fast enough to rerank top-N candidates without a GPU |

## Generation

| Technology                                    | Why chosen                                                                                                                                                                                                                                 |
| --------------------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------ |
| Ollama + `llama3.2:3b` / `phi3:mini`          | Local, free, fast enough on CPU (i7-1360P, no dedicated GPU) for the dev/debug iteration loop                                                                                                                                              |
| Groq API (free tier) + `llama-3.1-8b-instant` | Used for the Ragas evaluation run (50-100 questions) and demo responsiveness — CPU-bound local 8B inference would be too slow for either. Free tier: 14,400 requests/day, 500K tokens/day — comfortably covers both use cases at zero cost |

## Evaluation

| Technology                    | Why chosen                                                                                                                                          |
| ----------------------------- | --------------------------------------------------------------------------------------------------------------------------------------------------- |
| Ragas (`vibrantlabsai/ragas`) | Computes Faithfulness, Answer Relevancy, Context Precision without requiring human-labeled ground truth — the standard free/open RAG eval framework |

## CI/CD

| Technology     | Why chosen                                                                       |
| -------------- | -------------------------------------------------------------------------------- |
| GitHub Actions | Free for public repos; straightforward to gate PR merges on a script's exit code |

## Alternatives considered

- **ChromaDB instead of Qdrant** — rejected as primary choice; Qdrant's native hybrid search is a better fit, though ChromaDB + manual BM25/RRF remains a documented fallback if Qdrant setup friction is too high.
- **mixedbread-ai/mxbai-rerank-base-v1 instead of ms-marco-MiniLM** — considered; deferred. ms-marco-MiniLM is smaller and faster on CPU, a better fit for this hardware; mxbai can be tried later as a comparison experiment if reranking quality needs investigation.
- **CRAG/Self-RAG as core architecture instead of citation-enforcement-only** — rejected for MVP scope (see PRD non-goals); the corrective/reflective design space is a separate, harder problem better tackled once real failure data exists from this baseline.

## Known tradeoffs

- Running generation locally via Ollama on CPU-only hardware means slow iteration during Phase 2-3 development if not careful to use the smaller 3-4B models for active debugging.
- Groq free tier rate limits (30 RPM) mean the Ragas eval run needs to be paced, not fired as 100 parallel requests.
- `bge-m3` and the cross-encoder reranker both run on CPU — embedding/reranking is cheap relative to generation, so this is expected to be fine, but worth monitoring if the corpus grows large.
