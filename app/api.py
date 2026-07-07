"""
FastAPI application — Phase 4.

Endpoints:
  GET  /health   → system status + indexed chunk count
  POST /query    → question → grounded, citation-enforced answer
  POST /index    → trigger re-indexing in a background task

Rate limiting:
  Per-IP token bucket middleware applied to /query and /index.
  /health is always exempt so load-balancer health checks always succeed.

Startup:
  Documents are loaded and indexed once at startup via the lifespan
  context manager.  /index allows re-indexing on demand without restarting.
"""
import logging
import threading
from contextlib import asynccontextmanager
from typing import Optional

from fastapi import BackgroundTasks, FastAPI, HTTPException, Request
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field, field_validator
from starlette.middleware.base import BaseHTTPMiddleware

from app.config import API_RATE_LIMIT_BURST, API_RATE_LIMIT_RPM
from app.ingestion.loader import load_documents
from app.pipeline import Pipeline
from app.rate_limit import PerIpRateLimiter

logger = logging.getLogger(__name__)

# Rate-limited paths — /health is deliberately excluded
_RATE_LIMITED_PATHS = {"/query", "/index"}


# ---------------------------------------------------------------------------
# Pydantic models
# ---------------------------------------------------------------------------

class QueryRequest(BaseModel):
    question: str = Field(..., min_length=3, max_length=500)

    @field_validator("question")
    @classmethod
    def question_not_blank(cls, v: str) -> str:
        if not v.strip():
            raise ValueError("question must not be blank")
        return v


class ChunkSource(BaseModel):
    chunk_id: str
    page_number: Optional[int] = None
    section_header: Optional[str] = None
    source_url: Optional[str] = None


class QueryResponse(BaseModel):
    question: str
    answer: str
    provider: str
    supported: bool
    gated: bool
    source_chunk_ids: list[str]
    sources: list[ChunkSource]


class HealthResponse(BaseModel):
    status: str
    indexed_chunks: int
    indexing: bool


class IndexResponse(BaseModel):
    status: str
    message: str


# ---------------------------------------------------------------------------
# Rate-limit middleware
# ---------------------------------------------------------------------------

class PerIpRateLimitMiddleware(BaseHTTPMiddleware):
    """Reject requests from IPs that exceed the per-minute limit."""

    async def dispatch(self, request: Request, call_next):
        if request.url.path in _RATE_LIMITED_PATHS:
            limiter: PerIpRateLimiter = request.app.state.limiter
            client_ip = request.client.host if request.client else "unknown"
            if not limiter.is_allowed(client_ip):
                retry_after = max(1, int(60 / API_RATE_LIMIT_RPM))
                return JSONResponse(
                    status_code=429,
                    content={"detail": "Rate limit exceeded", "retry_after": retry_after},
                    headers={"Retry-After": str(retry_after)},
                )
        return await call_next(request)


# ---------------------------------------------------------------------------
# Background indexing
# ---------------------------------------------------------------------------

def _do_index(pipeline: Pipeline, app_state) -> None:
    """Re-index all documents.  Runs in a background thread."""
    logger.info("Background re-index started.")
    app_state.indexing = True
    try:
        docs = load_documents()
        pipeline.index_documents(docs)
        logger.info("Background re-index complete. Indexed %d doc(s).", len(docs))
    except Exception as exc:
        logger.error("Background re-index failed: %s", exc)
    finally:
        app_state.indexing = False


# ---------------------------------------------------------------------------
# Application factory / lifespan
# ---------------------------------------------------------------------------

def create_limiter() -> PerIpRateLimiter:
    """Create the per-IP limiter from config values."""
    rate = API_RATE_LIMIT_RPM / 60.0  # convert RPM → tokens-per-second
    return PerIpRateLimiter(rate=rate, capacity=API_RATE_LIMIT_BURST)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Load and index documents once at startup."""
    app.state.indexing = False
    app.state.limiter = create_limiter()

    pipeline = Pipeline()
    app.state.pipeline = pipeline

    try:
        docs = load_documents()
        if docs:
            pipeline.index_documents(docs)
            logger.info(
                "Startup indexing complete — %d document(s), %d chunk(s).",
                len(docs),
                pipeline.retriever.size,
            )
        else:
            logger.warning(
                "No documents found at startup. "
                "Call POST /index after adding PDFs to data/pdfs/."
            )
    except Exception as exc:
        logger.error("Startup indexing failed: %s", exc)

    yield
    # Shutdown: nothing to clean up (Qdrant client manages its own connection)


app = FastAPI(
    title="Production RAG API",
    description="Ask-my-docs system with hybrid retrieval, reranking, and citation enforcement.",
    version="4.0.0",
    lifespan=lifespan,
)
app.add_middleware(PerIpRateLimitMiddleware)


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------

@app.get("/health", response_model=HealthResponse, tags=["ops"])
async def health(request: Request) -> HealthResponse:
    """Return system status and indexed chunk count."""
    pipeline: Pipeline = request.app.state.pipeline
    chunk_count = pipeline.retriever.size if pipeline._indexed else 0
    return HealthResponse(
        status="ok",
        indexed_chunks=chunk_count,
        indexing=request.app.state.indexing,
    )


@app.post("/query", response_model=QueryResponse, tags=["rag"])
async def query(body: QueryRequest, request: Request) -> QueryResponse:
    """Answer a question grounded in the indexed document corpus.

    Returns a citation-enforced answer with source metadata.
    Sets ``gated=true`` when retrieval confidence is too low to answer.
    """
    pipeline: Pipeline = request.app.state.pipeline
    if not pipeline._indexed:
        raise HTTPException(status_code=503, detail="Index not ready — no documents indexed yet.")

    try:
        result = pipeline.query(body.question)
    except Exception as exc:
        logger.exception("Pipeline query failed: %s", exc)
        raise HTTPException(status_code=503, detail="Generation service unavailable.")

    sources = [
        ChunkSource(
            chunk_id=chunk.get("chunk_id", ""),
            page_number=chunk.get("page_number"),
            section_header=chunk.get("section_header"),
            source_url=chunk.get("source_url"),
        )
        for chunk in result.get("retrieved_chunks", [])
    ]

    return QueryResponse(
        question=result["question"],
        answer=result["answer"],
        provider=result["provider"],
        supported=result["structured_answer"]["supported"],
        gated=result["gated"],
        source_chunk_ids=result["structured_answer"]["source_chunk_ids"],
        sources=sources,
    )


@app.post("/index", status_code=202, response_model=IndexResponse, tags=["ops"])
async def reindex(background_tasks: BackgroundTasks, request: Request) -> IndexResponse:
    """Trigger a background re-index of all documents in data/pdfs/.

    Returns 202 Accepted immediately; use GET /health to monitor progress.
    """
    pipeline: Pipeline = request.app.state.pipeline
    if request.app.state.indexing:
        return IndexResponse(
            status="already_indexing",
            message="A re-index is already in progress.",
        )
    background_tasks.add_task(_do_index, pipeline, request.app.state)
    return IndexResponse(
        status="indexing",
        message="Re-indexing started in the background. Check GET /health for progress.",
    )
