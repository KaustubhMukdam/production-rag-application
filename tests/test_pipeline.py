from unittest.mock import MagicMock, patch
import pytest
import numpy as np


@pytest.fixture
def sample_docs():
    return [
        {"doc_id": "test_doc", "text": "the cat sat on the mat in the living room"},
    ]


def _make_mock_pipeline():
    from app.pipeline import Pipeline
    pipeline = Pipeline.__new__(Pipeline)
    pipeline.chunker = MagicMock()
    pipeline.chunker.chunk_documents.return_value = [
        {"doc_id": "test_doc", "chunk_id": "test_doc_0", "text": "the cat sat on the mat"}
    ]
    mock_embedder = MagicMock()
    mock_embedder.embed.return_value = np.random.randn(1, 384).astype(np.float32)
    mock_embedder.embed_sparse.return_value = [{0: 0.5}]
    pipeline.embedder = mock_embedder
    pipeline.retriever = MagicMock()
    pipeline.retriever.retrieve.return_value = [
        {"doc_id": "test_doc", "chunk_id": "test_doc_0", "text": "the cat sat on the mat", "score": 0.9}
    ]
    mock_reranker = MagicMock()
    mock_reranker.rerank.return_value = [
        {"doc_id": "test_doc", "chunk_id": "test_doc_0", "text": "the cat sat on the mat", "score": 0.9, "rerank_score": 8.5}
    ]
    pipeline.reranker = mock_reranker
    mock_generator = MagicMock()
    mock_generator.generate.return_value = "The cat is on the mat."
    pipeline.generator = mock_generator
    pipeline._indexed = False
    return pipeline


def test_pipeline_query_before_index_raises(sample_docs):
    pipeline = _make_mock_pipeline()
    with pytest.raises(RuntimeError, match="No documents indexed"):
        pipeline.query("test question")


def test_pipeline_index_calls_expected_steps(sample_docs):
    pipeline = _make_mock_pipeline()
    pipeline.index_documents(sample_docs)
    assert pipeline._indexed is True
    pipeline.chunker.chunk_documents.assert_called_once_with(sample_docs)
    pipeline.embedder.embed.assert_called_once()
    pipeline.embedder.embed_sparse.assert_called_once()
    pipeline.retriever.index.assert_called_once()


def test_pipeline_query_calls_all_stages(sample_docs):
    pipeline = _make_mock_pipeline()
    pipeline.index_documents(sample_docs)
    result = pipeline.query("Where is the cat?")
    pipeline.embedder.embed.assert_called()
    pipeline.embedder.embed_sparse.assert_called()
    pipeline.retriever.retrieve.assert_called_once()
    pipeline.reranker.rerank.assert_called_once()
    pipeline.generator.generate.assert_called_once()


def test_pipeline_query_returns_expected_structure(sample_docs):
    pipeline = _make_mock_pipeline()
    pipeline.index_documents(sample_docs)
    result = pipeline.query("Where is the cat?")
    assert "question" in result
    assert "answer" in result
    assert "retrieved_chunks" in result
    assert result["question"] == "Where is the cat?"
    assert result["answer"] == "The cat is on the mat."


def test_pipeline_retrieved_chunks_have_rerank_scores(sample_docs):
    pipeline = _make_mock_pipeline()
    pipeline.index_documents(sample_docs)
    result = pipeline.query("query")
    for chunk in result["retrieved_chunks"]:
        assert "rerank_score" in chunk


def test_pipeline_index_can_be_called_twice(sample_docs):
    pipeline = _make_mock_pipeline()
    pipeline.index_documents(sample_docs)
    pipeline.index_documents(sample_docs)
    assert pipeline._indexed is True
    assert pipeline.retriever.index.call_count == 2
