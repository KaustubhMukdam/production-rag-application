import numpy as np
import pytest

from app.retrieval.retriever import InMemoryRetriever


@pytest.fixture
def retriever():
    return InMemoryRetriever(dimension=384)


@pytest.fixture
def sample_chunks():
    return [
        {"doc_id": "doc1", "chunk_id": "doc1_0", "text": "the cat sat on the mat"},
        {"doc_id": "doc1", "chunk_id": "doc1_1", "text": "the dog ran in the park"},
        {"doc_id": "doc2", "chunk_id": "doc2_0", "text": "birds fly in the sky"},
        {"doc_id": "doc2", "chunk_id": "doc2_1", "text": "fish swim in the ocean"},
    ]


def test_retrieve_empty_index(retriever):
    query = np.random.randn(1, 384).astype(np.float32)
    results = retriever.retrieve(query, k=5)
    assert results == []


def test_retrieve_returns_k_results(retriever, sample_chunks):
    embeddings = np.random.randn(len(sample_chunks), 384).astype(np.float32)
    retriever.index(embeddings, sample_chunks)
    query = np.random.randn(1, 384).astype(np.float32)
    results = retriever.retrieve(query, k=2)
    assert len(results) == 2


def test_retrieve_k_larger_than_index(retriever, sample_chunks):
    embeddings = np.random.randn(len(sample_chunks), 384).astype(np.float32)
    retriever.index(embeddings, sample_chunks)
    query = np.random.randn(1, 384).astype(np.float32)
    results = retriever.retrieve(query, k=100)
    assert len(results) == len(sample_chunks)


def test_retrieve_most_similar_first(retriever, sample_chunks):
    embeddings = np.zeros((len(sample_chunks), 384), dtype=np.float32)
    embeddings[0] = np.full(384, 0.9, dtype=np.float32)
    embeddings[1] = np.full(384, 0.1, dtype=np.float32)
    retriever.index(embeddings, sample_chunks)
    query = np.full((1, 384), 0.9, dtype=np.float32)
    results = retriever.retrieve(query, k=2)
    assert results[0]["chunk_id"] == "doc1_0"
    assert results[0]["score"] >= results[1]["score"]


def test_retrieve_scores_descending(retriever, sample_chunks):
    rng = np.random.RandomState(42)
    embeddings = rng.randn(len(sample_chunks), 384).astype(np.float32)
    retriever.index(embeddings, sample_chunks)
    query = rng.randn(1, 384).astype(np.float32)
    results = retriever.retrieve(query, k=3)
    for i in range(len(results) - 1):
        assert results[i]["score"] >= results[i + 1]["score"]


def test_retrieve_returns_chunk_metadata(retriever, sample_chunks):
    embeddings = np.random.randn(len(sample_chunks), 384).astype(np.float32)
    retriever.index(embeddings, sample_chunks)
    query = np.random.randn(1, 384).astype(np.float32)
    results = retriever.retrieve(query, k=1)
    assert "doc_id" in results[0]
    assert "chunk_id" in results[0]
    assert "text" in results[0]
    assert "score" in results[0]


def test_retrieve_with_k_one(retriever, sample_chunks):
    embeddings = np.random.randn(len(sample_chunks), 384).astype(np.float32)
    retriever.index(embeddings, sample_chunks)
    query = np.random.randn(1, 384).astype(np.float32)
    results = retriever.retrieve(query, k=1)
    assert len(results) == 1
