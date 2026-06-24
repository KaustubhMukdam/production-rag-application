from unittest.mock import patch, MagicMock
import pytest
import numpy as np


@pytest.fixture
def sample_docs():
    return [
        {"doc_id": "test_doc", "text": "the cat sat on the mat in the living room"},
    ]


def test_pipeline_query_before_index_raises(sample_docs):
    pipeline = _make_mock_pipeline()
    with pytest.raises(RuntimeError, match="No documents indexed"):
        pipeline.query("test question")


def test_pipeline_index_calls_expected_steps(sample_docs):
    pipeline = _make_mock_pipeline()
    pipeline.index_documents(sample_docs)
    assert pipeline._indexed is True


def test_pipeline_query_returns_expected_structure(sample_docs):
    pipeline = _make_mock_pipeline()
    pipeline.index_documents(sample_docs)
    result = pipeline.query("Where is the cat?")
    assert "question" in result
    assert "answer" in result
    assert "retrieved_chunks" in result
    assert result["question"] == "Where is the cat?"
    assert result["answer"] == "The cat is on the mat."


def test_pipeline_retrieved_chunks_have_scores(sample_docs):
    pipeline = _make_mock_pipeline()
    pipeline.index_documents(sample_docs)
    result = pipeline.query("query")
    for chunk in result["retrieved_chunks"]:
        assert "score" in chunk


def _make_mock_pipeline():
    from app.pipeline import Pipeline
    pipeline = Pipeline.__new__(Pipeline)
    pipeline.chunker = MagicMock()
    pipeline.chunker.chunk_documents.return_value = [
        {"doc_id": "test_doc", "chunk_id": "test_doc_0", "text": "the cat sat on the mat"}
    ]
    mock_embedder = MagicMock()
    mock_embedder.embed.return_value = np.random.randn(1, 384).astype(np.float32)
    pipeline.embedder = mock_embedder
    pipeline.retriever = MagicMock()
    pipeline.retriever.retrieve.return_value = [
        {"doc_id": "test_doc", "chunk_id": "test_doc_0", "text": "the cat sat on the mat", "score": 0.95}
    ]
    mock_generator = MagicMock()
    mock_generator.generate.return_value = "The cat is on the mat."
    pipeline.generator = mock_generator
    pipeline._indexed = False
    return pipeline
