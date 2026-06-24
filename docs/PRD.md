# PRD — Production RAG Application ("Ask My Docs")

## Problem statement

Generic LLM chat doesn't know about a specific, private, or domain-specific document set, and naive RAG implementations (single embedding model, top-k cosine retrieval, no verification) frequently retrieve the wrong chunk or let the model answer beyond what's actually supported — producing confident-sounding but ungrounded answers. This is the most common failure mode blocking RAG systems from being trusted in production.

This project builds the pattern that fixes it: hybrid retrieval (catches both keyword and semantic matches), reranking (fixes ordering when the first-pass retrieval is noisy), citation enforcement (forces the model to point at the exact chunk it used, and refuse when it can't), and a CI-gated evaluation pipeline (prevents silent quality regressions from ever reaching "production").

## Target users

- Primary: the project owner — this is a learning project where the build process matters as much as the output. Secondary audience is anyone reviewing this as a portfolio/resume artifact (recruiters, interviewers for AI/ML engineering roles).
- Implied end user of the deployed demo: someone who wants to ask natural-language questions against a specific document set (e.g. a textbook, a set of research papers, internal docs) and get answers that are traceable back to source.

## Core features (MVP)

- [ ] **Document ingestion & chunking** — load a domain-specific document set, chunk it sensibly (no naive fixed-size splitting that cuts mid-sentence/mid-table where avoidable).
- [ ] **Hybrid retrieval** — BM25 (keyword/sparse) fused with dense vector search (`bge-m3`) via Qdrant's native hybrid support. Acceptance: a query containing an exact technical term retrieves the chunk containing that term even if it's not the top dense-similarity match.
- [ ] **Cross-encoder reranking** — rerank the top-N hybrid candidates with `ms-marco-MiniLM-L-6-v2` before passing to generation. Acceptance: reranked top-1 differs from hybrid top-1 on at least some queries in the eval set, demonstrating the rerank step does real work (not a no-op).
- [ ] **Citation-enforced generation** — LLM returns structured output `{answer, source_chunk_id, supported}`. Acceptance: every returned answer's `source_chunk_id` corresponds to a chunk actually present in the retrieved set; if no retrieved chunk supports the question, the system returns "insufficient context" rather than answering.
- [ ] **CI-gated Ragas evaluation** — a 50-100 question eval set run through Ragas (Faithfulness, Answer Relevancy, Context Precision) on every PR. Acceptance: PR is blocked from merging if `faithfulness < 0.85`.

## Nice-to-have (post-MVP)

- [ ] Provider swap config (Ollama ↔ Groq) exposed as a single environment variable, not a code change.
- [ ] Simple query CLI or minimal API endpoint for asking questions interactively.
- [ ] Logged retrieval scores (hybrid score + rerank score) surfaced per answer, for debugging and future CRAG-style gating.

## Non-goals

- No CRAG / Self-RAG corrective loop in MVP scope (see project_context.md — deferred to Phase 5 stretch).
- No multi-turn conversational memory — single question in, single grounded answer out.
- No fine-tuning of embedding, reranker, or generation models.
- No production deployment / scaling concerns (load balancing, multi-tenancy, auth).
- No web search fallback — closed corpus only.

## Success metrics

- Phase 0 baseline runs end-to-end: question in, answer out, no crashes.
- Hybrid + rerank measurably improves retrieval quality over the Phase 0 baseline on the same test query set (manual inspection, then Ragas `context_precision`).
- Ragas `faithfulness` ≥ 0.85 on the eval set, enforced by CI — not just measured once locally.
- The CI gate actually fails a PR at least once during development (proof it's a real gate, not a no-op check).

## Constraints

- Cost: free tier only — no paid API usage anywhere in the pipeline.
- Hardware: Dell Inspiron 14 (i7-1360P, 16GB RAM, Intel Iris Xe integrated graphics, no dedicated GPU) — all local inference is CPU-bound. Heavy/bulk inference (the 50-100 question eval run) routes to Groq's free tier instead of local Ollama.
- Time: self-paced learning project, phased (Phase 0 → Phase 4 core, Phase 5 optional).
- Process: architecture and design decisions are owned by the project author; AI coding assistance (OpenCode or similar) is used only from Phase 1 onward, against tests written first (TDD), never for Phase 0's hand-coded baseline.
