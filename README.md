# Production RAG Application — "Ask My Docs"

A domain-specific document Q&A system built to demonstrate the production RAG pattern: hybrid retrieval, cross-encoder reranking, citation enforcement, and a CI-gated evaluation pipeline. Built entirely on free-tier tools (Ollama, Qdrant, Groq free tier, Ragas, GitHub Actions).

## What this is
Generic RAG demos retrieve with a single embedding model and answer without verification. This project builds the pattern that fixes both problems:

- **Hybrid retrieval** — BM25 (keyword) + `bge-m3` (dense vector) fused via Qdrant
- **Cross-encoder reranking** — fixes retrieval ordering before generation
- **Citation enforcement** — the model must cite the exact chunk it used; if it can't, the system fails closed with "insufficient context" instead of hallucinating
- **CI-gated evaluation** — every PR runs a Ragas evaluation; merges are blocked if Faithfulness drops below 0.85

## Status
In progress — Phase 0 (naive RAG baseline). See `docs/tasks.md` for the full phased plan.

## Documentation
Full project documentation lives in `docs/`:
- [`project_context.md`](docs/project_context.md) — single source of truth
- [`PRD.md`](docs/PRD.md) — what's being built and why
- [`tech_stack.md`](docs/tech_stack.md) — every technology choice and the reasoning
- [`architecture.md`](docs/architecture.md) — system design and data flow
- [`folder_structure.md`](docs/folder_structure.md) — codebase layout
- [`tasks.md`](docs/tasks.md) — phased task list
- [`eval.md`](docs/eval.md) — evaluation methodology and CI gate thresholds
- [`learnings.md`](docs/learnings.md) / [`debug_log.md`](docs/debug_log.md) / [`experiment_log.md`](docs/experiment_log.md) — build-as-you-learn logs

## Why this exists
This is a learning project first, portfolio artifact second. The architecture and design decisions are owned end-to-end by the author; AI coding assistance is used from Phase 1 onward, against tests written first, never to generate the conceptual core (Phase 0) wholesale.

## Local setup
```
docker-compose up -d        # starts local Qdrant
ollama pull llama3.2:3b      # local dev model
pip install -r requirements.txt
```
Set `GROQ_API_KEY` as an environment variable for eval runs and demo mode (provider selection lives in `app/config.py`).