import pytest

from app.retrieval.reranker import Reranker


@pytest.fixture
def reranker():
    return Reranker()


@pytest.fixture
def sample_chunks():
    return [
        {"doc_id": "doc1", "chunk_id": "doc1_0", "text": "the cat sat on the mat", "score": 0.7},
        {"doc_id": "doc1", "chunk_id": "doc1_1", "text": "the dog ran in the park", "score": 0.6},
        {"doc_id": "doc2", "chunk_id": "doc2_0", "text": "birds fly in the sky", "score": 0.5},
    ]


def test_rerank_empty_list(reranker):
    results = reranker.rerank("test query", [])
    assert results == []


def test_rerank_preserves_count(reranker, sample_chunks):
    results = reranker.rerank("where is the cat", sample_chunks)
    assert len(results) == len(sample_chunks)


def test_rerank_returns_correct_keys(reranker, sample_chunks):
    results = reranker.rerank("where is the cat", sample_chunks)
    assert "doc_id" in results[0]
    assert "chunk_id" in results[0]
    assert "text" in results[0]
    assert "score" in results[0]
    assert "rerank_score" in results[0]


def test_rerank_scores_descending(reranker, sample_chunks):
    results = reranker.rerank("cat", sample_chunks)
    for i in range(len(results) - 1):
        assert results[i]["rerank_score"] >= results[i + 1]["rerank_score"]


def test_rerank_relevant_chunk_first(reranker):
    chunks = [
        {"doc_id": "d1", "chunk_id": "d1_0", "text": "pizza recipes for dinner", "score": 0.1},
        {"doc_id": "d2", "chunk_id": "d2_0", "text": "machine learning transformers explained", "score": 0.1},
    ]
    results = reranker.rerank("how do transformers work", chunks)
    assert results[0]["chunk_id"] == "d2_0"


def test_rerank_single_chunk(reranker):
    chunks = [{"doc_id": "d1", "chunk_id": "d1_0", "text": "Python is a programming language", "score": 0.5}]
    results = reranker.rerank("what is python", chunks)
    assert len(results) == 1
    assert results[0]["rerank_score"] is not None


def test_rerank_updates_score_field(reranker, sample_chunks):
    results = reranker.rerank("cat", sample_chunks)
    for r in results:
        assert r["score"] != r["rerank_score"]
