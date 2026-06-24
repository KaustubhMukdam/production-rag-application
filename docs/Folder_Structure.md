# Folder Structure — Production RAG Application

```
production-rag-application/
├── app/
│   ├── ingestion/
│   │   ├── loader.py          # loads raw documents from the domain corpus
│   │   └── chunker.py         # chunking strategy — kept isolated so it can be swapped/tuned
│   ├── embeddings/
│   │   └── embedder.py        # wraps bge-m3 (dense + sparse embedding calls)
│   ├── retrieval/
│   │   ├── retriever.py       # Qdrant hybrid search wrapper — implements Retriever interface
│   │   └── reranker.py        # cross-encoder reranking wrapper
│   ├── generation/
│   │   ├── providers/
│   │   │   ├── ollama_provider.py
│   │   │   └── groq_provider.py
│   │   ├── generator.py       # provider-agnostic Generator interface, picks backend from config
│   │   └── prompts.py         # system prompts, JSON schema definitions for structured output
│   ├── validation/
│   │   └── citation_validator.py   # checks LLM's claimed source_chunk_id against retrieved set
│   ├── config.py              # central config — provider selection (ollama/groq), model names, thresholds
│   └── main.py                 # FastAPI app entry point / query endpoint
├── eval/
│   ├── eval_questions.json    # the 50-100 question eval set for Ragas
│   ├── run_eval.py            # runs the eval set through the full pipeline + Ragas scoring
│   └── thresholds.py          # the faithfulness/relevancy/precision thresholds CI checks against
├── tests/
│   ├── test_chunker.py
│   ├── test_retriever.py
│   ├── test_reranker.py
│   ├── test_generator.py
│   └── test_citation_validator.py   # one test file per module — written before implementation (TDD) from Phase 1 onward
├── docs/                       # all documentation lives here
│   ├── project_context.md
│   ├── PRD.md
│   ├── tech_stack.md
│   ├── architecture.md
│   ├── folder_structure.md
│   ├── tasks.md
│   ├── learnings.md
│   ├── debug_log.md
│   ├── experiment_log.md
│   ├── data_doc.md
│   └── eval.md
├── .github/
│   └── workflows/
│       └── evaluate.yml       # CI gate — runs eval/run_eval.py on every PR, fails if faithfulness < 0.85
├── docker-compose.yml          # local Qdrant container
├── requirements.txt
└── README.md
```

## Naming conventions

- Python files: snake_case (`citation_validator.py`)
- Classes: PascalCase (`CitationValidator`, `HybridRetriever`)
- Interfaces/abstract base classes: named for the role, not the implementation (`Retriever`, not `QdrantRetriever` — the concrete class is `QdrantRetriever(Retriever)`)
- Config values (model names, provider choice, thresholds): centralized in `app/config.py`, never hardcoded inside pipeline logic
- Test files: `test_<module_name>.py`, one per module under `app/`

## Why this structure

Each pipeline stage (ingestion, embedding, retrieval, reranking, generation, validation) is its own module with a narrow interface. This is deliberate: it's what makes the TDD workflow described in `tasks.md` and the dev system guide actually work — you can write a test against `Retriever.retrieve()` without caring whether it's backed by Qdrant or something else, and an AI coding agent (OpenCode, from Phase 1 onward) can implement against that test without needing the whole system in context.
