import numpy as np
import pytest

from app.retrieval.retriever import QdrantHybridRetriever


@pytest.fixture
def retriever():
    return QdrantHybridRetriever(url=":memory:", collection_name="test_collection")


@pytest.fixture
def sample_chunks():
    return [
        {"doc_id": "doc1", "chunk_id": "doc1_0", "text": "the cat sat on the mat"},
        {"doc_id": "doc1", "chunk_id": "doc1_1", "text": "the dog ran in the park"},
        {"doc_id": "doc2", "chunk_id": "doc2_0", "text": "birds fly in the sky"},
    ]


def test_retrieve_empty_before_index(retriever):
    results = retriever.retrieve(
        dense=np.random.randn(384).astype(np.float32),
        sparse={0: 1.0, 1: 0.5},
        k=5,
    )
    assert results == []


def test_index_and_retrieve(retriever, sample_chunks):
    dense_vecs = np.random.randn(len(sample_chunks), 384).astype(np.float32)
    sparse_vecs = [
        {0: 0.5, 10: 0.3},
        {1: 0.8, 20: 0.2},
        {2: 0.6, 30: 0.4},
    ]
    retriever.index(dense_vecs, sparse_vecs, sample_chunks)
    assert retriever.size == 3


def test_retrieve_returns_k_results(retriever, sample_chunks):
    dense_vecs = np.random.randn(len(sample_chunks), 384).astype(np.float32)
    sparse_vecs = [{0: 0.5}, {1: 0.8}, {2: 0.6}]
    retriever.index(dense_vecs, sparse_vecs, sample_chunks)

    query_dense = np.random.randn(384).astype(np.float32)
    query_sparse = {0: 1.0}
    results = retriever.retrieve(query_dense, query_sparse, k=2)
    assert len(results) == 2


def test_retrieve_k_larger_than_index(retriever, sample_chunks):
    dense_vecs = np.random.randn(len(sample_chunks), 384).astype(np.float32)
    sparse_vecs = [{0: 0.5}, {1: 0.8}, {2: 0.6}]
    retriever.index(dense_vecs, sparse_vecs, sample_chunks)

    query_dense = np.random.randn(384).astype(np.float32)
    query_sparse = {0: 1.0}
    results = retriever.retrieve(query_dense, query_sparse, k=100)
    assert len(results) == len(sample_chunks)


def test_retrieve_returns_chunk_metadata(retriever, sample_chunks):
    dense_vecs = np.random.randn(len(sample_chunks), 384).astype(np.float32)
    sparse_vecs = [{0: 0.5}, {1: 0.8}, {2: 0.6}]
    retriever.index(dense_vecs, sparse_vecs, sample_chunks)

    query_dense = np.random.randn(384).astype(np.float32)
    query_sparse = {0: 1.0}
    results = retriever.retrieve(query_dense, query_sparse, k=1)
    assert "doc_id" in results[0]
    assert "chunk_id" in results[0]
    assert "text" in results[0]
    assert "score" in results[0]


def test_index_twice_clears_old_data(retriever, sample_chunks):
    dense_vecs = np.random.randn(len(sample_chunks), 384).astype(np.float32)
    sparse_vecs = [{0: 0.5}, {1: 0.8}, {2: 0.6}]
    retriever.index(dense_vecs, sparse_vecs, sample_chunks)
    assert retriever.size == 3

    new_chunks = [{"doc_id": "doc3", "chunk_id": "doc3_0", "text": "new data"}]
    retriever.index(
        np.random.randn(1, 384).astype(np.float32),
        [{0: 0.9}],
        new_chunks,
    )
    assert retriever.size == 1


def test_retrieve_returns_empty_when_no_sparse_match(retriever, sample_chunks):
    dense_vecs = np.random.randn(len(sample_chunks), 384).astype(np.float32)
    sparse_vecs = [{0: 0.5}, {1: 0.8}, {2: 0.6}]
    retriever.index(dense_vecs, sparse_vecs, sample_chunks)
    query_dense = np.random.randn(384).astype(np.float32)
    query_sparse = {99999: 1.0}
    results = retriever.retrieve(query_dense, query_sparse, k=5)
    assert len(results) > 0
