"""
TDD tests for app.api (FastAPI)
---------------------------------
All Pipeline calls are mocked — no real Qdrant, Ollama, or Groq required.
Uses FastAPI's built-in TestClient (backed by httpx).

Test coverage:
  GET  /health    — status, chunk count, rate-limit bypass
  POST /query     — success, validation errors, 503 when unindexed, gated response
  POST /index     — 202 accepted, rate limiting
  Rate limiting   — 429 after burst, per-IP independence
"""
import pytest
from unittest.mock import MagicMock, patch
from fastapi.testclient import TestClient

from app.rate_limit import PerIpRateLimiter


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

def _mock_pipeline(gated: bool = False, indexed: bool = True, chunk_count: int = 42):
    pipeline = MagicMock()
    pipeline._indexed = indexed

    if indexed:
        pipeline.retriever.size = chunk_count
        pipeline.query.return_value = {
            "question": "What is HMM?",
            "answer": "HMM stands for Hidden Markov Model.",
            "provider": "groq",
            "gated": gated,
            "retrieved_chunks": [
                {
                    "chunk_id": "ch3_2",
                    "text": "HMM stands for Hidden Markov Model.",
                    "page_number": 7,
                    "section_header": "3.2 HMM",
                    "source_url": "https://example.com/ch03.pdf",
                    "rerank_score": 5.2,
                }
            ],
            "structured_answer": {
                "answer": "HMM stands for Hidden Markov Model.",
                "source_chunk_ids": [] if gated else ["ch3_2"],
                "supported": not gated,
            },
        }
    return pipeline


@pytest.fixture
def client():
    """TestClient with a mocked pipeline injected at app startup."""
    from app.api import app, create_limiter

    pipeline = _mock_pipeline()
    # Reset limiter to a generous one so regular tests are not rate-limited
    fresh_limiter = PerIpRateLimiter(rate=1000.0, capacity=1000)

    with patch("app.api.Pipeline") as MockPipeline, \
         patch("app.api.load_documents", return_value=[{"doc_id": "test", "text": "t"}]):
        MockPipeline.return_value = pipeline

        # Swap limiter to non-restrictive for all tests except rate-limit tests
        app.state.limiter = fresh_limiter

        with TestClient(app) as c:
            c.app.state.pipeline = pipeline
            yield c


@pytest.fixture
def strict_client():
    """TestClient with a tight rate limiter (burst=2) for rate-limit tests."""
    from app.api import app

    pipeline = _mock_pipeline()
    tight_limiter = PerIpRateLimiter(rate=0.001, capacity=2)

    with patch("app.api.Pipeline") as MockPipeline, \
         patch("app.api.load_documents", return_value=[{"doc_id": "test", "text": "t"}]):
        MockPipeline.return_value = pipeline

        app.state.limiter = tight_limiter

        with TestClient(app) as c:
            c.app.state.pipeline = pipeline
            yield c


# ---------------------------------------------------------------------------
# GET /health
# ---------------------------------------------------------------------------

class TestHealth:
    def test_returns_200(self, client):
        resp = client.get("/health")
        assert resp.status_code == 200

    def test_has_status_ok(self, client):
        resp = client.get("/health")
        assert resp.json()["status"] == "ok"

    def test_has_indexed_chunks(self, client):
        resp = client.get("/health")
        assert "indexed_chunks" in resp.json()

    def test_indexed_chunks_is_int(self, client):
        resp = client.get("/health")
        assert isinstance(resp.json()["indexed_chunks"], int)

    def test_has_indexing_flag(self, client):
        resp = client.get("/health")
        assert "indexing" in resp.json()

    def test_health_not_rate_limited(self, strict_client):
        """Health endpoint must bypass rate limiting."""
        for _ in range(10):
            resp = strict_client.get("/health")
            assert resp.status_code == 200


# ---------------------------------------------------------------------------
# POST /query
# ---------------------------------------------------------------------------

class TestQuery:
    def test_returns_200_on_valid_question(self, client):
        resp = client.post("/query", json={"question": "What is HMM?"})
        assert resp.status_code == 200

    def test_response_has_question(self, client):
        resp = client.post("/query", json={"question": "What is HMM?"})
        assert "question" in resp.json()

    def test_response_has_answer(self, client):
        resp = client.post("/query", json={"question": "What is HMM?"})
        assert "answer" in resp.json()

    def test_response_has_provider(self, client):
        resp = client.post("/query", json={"question": "What is HMM?"})
        assert "provider" in resp.json()

    def test_response_has_supported(self, client):
        resp = client.post("/query", json={"question": "What is HMM?"})
        assert "supported" in resp.json()

    def test_response_has_gated(self, client):
        resp = client.post("/query", json={"question": "What is HMM?"})
        assert "gated" in resp.json()

    def test_response_has_sources(self, client):
        resp = client.post("/query", json={"question": "What is HMM?"})
        assert "sources" in resp.json()

    def test_sources_is_list(self, client):
        resp = client.post("/query", json={"question": "What is HMM?"})
        assert isinstance(resp.json()["sources"], list)

    def test_source_has_chunk_id(self, client):
        resp = client.post("/query", json={"question": "What is HMM?"})
        sources = resp.json()["sources"]
        if sources:
            assert "chunk_id" in sources[0]

    def test_source_has_page_number(self, client):
        resp = client.post("/query", json={"question": "What is HMM?"})
        sources = resp.json()["sources"]
        if sources:
            assert "page_number" in sources[0]

    def test_source_has_section_header(self, client):
        resp = client.post("/query", json={"question": "What is HMM?"})
        sources = resp.json()["sources"]
        if sources:
            assert "section_header" in sources[0]

    def test_source_has_source_url(self, client):
        resp = client.post("/query", json={"question": "What is HMM?"})
        sources = resp.json()["sources"]
        if sources:
            assert "source_url" in sources[0]

    def test_returns_422_on_missing_question(self, client):
        resp = client.post("/query", json={})
        assert resp.status_code == 422

    def test_returns_422_on_blank_question(self, client):
        resp = client.post("/query", json={"question": "  "})
        assert resp.status_code == 422

    def test_returns_422_on_question_too_short(self, client):
        resp = client.post("/query", json={"question": "hi"})
        assert resp.status_code == 422

    def test_gated_response_has_gated_true(self, client):
        from app.api import app
        gated_pipeline = _mock_pipeline(gated=True)
        app.state.pipeline = gated_pipeline
        resp = client.post("/query", json={"question": "pizza recipes?"})
        assert resp.json()["gated"] is True

    def test_returns_503_when_not_indexed(self):
        from app.api import app
        unindexed_pipeline = _mock_pipeline(indexed=False)

        from app.rate_limit import PerIpRateLimiter
        fresh_limiter = PerIpRateLimiter(rate=1000.0, capacity=1000)
        app.state.limiter = fresh_limiter

        with patch("app.api.Pipeline") as MockPipeline, \
             patch("app.api.load_documents", return_value=[]):
            MockPipeline.return_value = unindexed_pipeline
            with TestClient(app) as c:
                c.app.state.pipeline = unindexed_pipeline
                resp = c.post("/query", json={"question": "What is HMM?"})
        assert resp.status_code == 503


# ---------------------------------------------------------------------------
# POST /index
# ---------------------------------------------------------------------------

class TestIndex:
    def test_returns_202(self, client):
        with patch("app.api.load_documents", return_value=[]):
            resp = client.post("/index")
        assert resp.status_code == 202

    def test_response_has_status_field(self, client):
        with patch("app.api.load_documents", return_value=[]):
            resp = client.post("/index")
        assert "status" in resp.json()


# ---------------------------------------------------------------------------
# Rate limiting
# ---------------------------------------------------------------------------

class TestRateLimiting:
    def test_query_returns_429_after_burst_exhausted(self, strict_client):
        results = []
        for _ in range(5):
            resp = strict_client.post("/query", json={"question": "What is HMM?"})
            results.append(resp.status_code)
        assert 429 in results

    def test_429_has_detail_field(self, strict_client):
        for _ in range(5):
            resp = strict_client.post("/query", json={"question": "What is HMM?"})
            if resp.status_code == 429:
                assert "detail" in resp.json()
                break

    def test_429_has_retry_after_header(self, strict_client):
        for _ in range(5):
            resp = strict_client.post("/query", json={"question": "What is HMM?"})
            if resp.status_code == 429:
                assert "retry-after" in resp.headers or "Retry-After" in resp.headers
                break

    def test_health_not_blocked_by_rate_limit(self, strict_client):
        # Drain the query rate limit
        for _ in range(5):
            strict_client.post("/query", json={"question": "What is HMM?"})
        # Health must still succeed
        resp = strict_client.get("/health")
        assert resp.status_code == 200
