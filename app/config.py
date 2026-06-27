import os
from pathlib import Path

from dotenv import load_dotenv

env_path = Path(__file__).resolve().parent.parent / ".env"
if env_path.exists():
    load_dotenv(env_path)

PROJECT_ROOT = Path(__file__).resolve().parent.parent

DATA_DIR = PROJECT_ROOT / "data"
DATA_DIR.mkdir(exist_ok=True)

CORPUS_URL = "https://web.stanford.edu/~jurafsky/slp3/"

EMBEDDING_MODEL = "BAAI/bge-small-en-v1.5"
EMBEDDING_DIM = 384

CHUNK_SIZE = 500
CHUNK_OVERLAP = 50

TOP_K = 5
TOP_N = 20

QDRANT_URL = os.getenv("QDRANT_URL", "http://localhost:6333")
QDRANT_COLLECTION = "slp3_chunks"

RERANKER_MODEL = "cross-encoder/ms-marco-MiniLM-L-6-v2"

OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "llama3.2:3b")

_has_groq_key = bool(os.getenv("GROQ_API_KEY"))
LLM_PROVIDER = os.getenv("LLM_PROVIDER", "groq" if _has_groq_key else "ollama")
GROQ_MODEL = os.getenv("GROQ_MODEL", "llama-3.3-70b-versatile")
