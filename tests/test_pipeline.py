from unittest.mock import MagicMock, patch
import pytest
import numpy as np

from app.config import RERANK_GATE_THRESHOLD


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
        {"doc_id": "test_doc", "chunk_id": "test_doc_0", "text": "the cat sat on the mat",
         "score": 0.9, "page_number": 1, "section_header": "", "source_url": ""}
    ]
    mock_reranker = MagicMock()
    mock_reranker.rerank.return_value = [
        {"doc_id": "test_doc", "chunk_id": "test_doc_0", "text": "the cat sat on the mat",
         "score": 0.9, "rerank_score": 8.5, "page_number": 1, "section_header": "", "source_url": ""}
    ]
    pipeline.reranker = mock_reranker
    mock_generator = MagicMock()
    mock_generator.generate.return_value = (
        '{"answer": "The cat is on the mat.", "source_chunk_ids": ["test_doc_0"], "supported": true}'
    )
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
    assert "provider" in result
    assert "retrieved_chunks" in result
    assert "structured_answer" in result
    assert result["question"] == "Where is the cat?"
    assert result["answer"] == "The cat is on the mat."


def test_pipeline_structured_answer_has_correct_fields(sample_docs):
    pipeline = _make_mock_pipeline()
    pipeline.index_documents(sample_docs)
    result = pipeline.query("Where is the cat?")
    sa = result["structured_answer"]
    assert "answer" in sa
    assert "source_chunk_ids" in sa
    assert "supported" in sa
    assert sa["answer"] == "The cat is on the mat."
    assert sa["source_chunk_ids"] == ["test_doc_0"]
    assert sa["supported"] is True


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


def test_pipeline_invalid_citation_retries(sample_docs):
    pipeline = _make_mock_pipeline()
    from unittest.mock import MagicMock
    generator = MagicMock()
    generator.generate.side_effect = [
        '{"answer": "bad", "source_chunk_ids": ["nonexistent"], "supported": true}',
        '{"answer": "good", "source_chunk_ids": ["test_doc_0"], "supported": true}',
    ]
    pipeline.generator = generator
    pipeline.index_documents(sample_docs)
    result = pipeline.query("query")
    assert result["answer"] == "good"
    assert result["structured_answer"]["supported"] is True
    assert generator.generate.call_count == 2


def test_pipeline_invalid_citation_twice_returns_unsupported(sample_docs):
    pipeline = _make_mock_pipeline()
    generator = MagicMock()
    generator.generate.return_value = (
        '{"answer": "still bad", "source_chunk_ids": ["nonexistent"], "supported": true}'
    )
    pipeline.generator = generator
    pipeline.index_documents(sample_docs)
    result = pipeline.query("query")
    assert result["answer"] == "still bad"
    assert result["structured_answer"]["supported"] is False
    assert generator.generate.call_count == 2


def test_pipeline_unparseable_json_retries(sample_docs):
    pipeline = _make_mock_pipeline()
    generator = MagicMock()
    generator.generate.side_effect = [
        "not json at all",
        '{"answer": "retry worked", "source_chunk_ids": ["test_doc_0"], "supported": true}',
    ]
    pipeline.generator = generator
    pipeline.index_documents(sample_docs)
    result = pipeline.query("query")
    assert result["answer"] == "retry worked"
    assert result["structured_answer"]["supported"] is True


def test_pipeline_unparseable_twice_returns_unsupported(sample_docs):
    pipeline = _make_mock_pipeline()
    generator = MagicMock()
    generator.generate.return_value = "completely broken output"
    pipeline.generator = generator
    pipeline.index_documents(sample_docs)
    result = pipeline.query("query")
    assert "completely broken output" in result["answer"]
    assert result["structured_answer"]["supported"] is False


# ---------------------------------------------------------------------------
# Confidence gate tests (Block B)
# ---------------------------------------------------------------------------

def test_pipeline_query_returns_gated_key_false_on_normal_flow():
    """Every successful query must include gated=False in the response."""
    pipeline = _make_mock_pipeline()
    pipeline.index_documents([{"doc_id": "test_doc", "text": "the cat sat on the mat in the living room"}])
    result = pipeline.query("Where is the cat?")
    assert "gated" in result
    assert result["gated"] is False


def test_pipeline_confidence_gate_fires_when_score_below_threshold():
    """Gate must fire when top rerank_score < RERANK_GATE_THRESHOLD."""
    pipeline = _make_mock_pipeline()
    pipeline.reranker.rerank.return_value = [
        {"doc_id": "test_doc", "chunk_id": "test_doc_0", "text": "irrelevant",
         "score": 0.1, "rerank_score": RERANK_GATE_THRESHOLD - 0.1,
         "page_number": 1, "section_header": "", "source_url": ""}
    ]
    pipeline.index_documents([{"doc_id": "test_doc", "text": "the cat sat on the mat in the living room"}])
    result = pipeline.query("Completely off-topic question")
    assert result["gated"] is True


def test_pipeline_confidence_gate_does_not_call_generator_when_gated():
    """When the gate fires, the LLM generator must NOT be called."""
    pipeline = _make_mock_pipeline()
    pipeline.reranker.rerank.return_value = [
        {"doc_id": "test_doc", "chunk_id": "test_doc_0", "text": "irrelevant",
         "score": 0.1, "rerank_score": RERANK_GATE_THRESHOLD - 1.0,
         "page_number": 1, "section_header": "", "source_url": ""}
    ]
    pipeline.index_documents([{"doc_id": "test_doc", "text": "the cat sat on the mat in the living room"}])
    pipeline.query("Off-topic")
    pipeline.generator.generate.assert_not_called()


def test_pipeline_confidence_gate_passes_when_score_at_threshold():
    """Gate must NOT fire when score equals exactly the threshold (inclusive)."""
    pipeline = _make_mock_pipeline()
    pipeline.reranker.rerank.return_value = [
        {"doc_id": "test_doc", "chunk_id": "test_doc_0", "text": "the cat sat on the mat",
         "score": 0.9, "rerank_score": RERANK_GATE_THRESHOLD,
         "page_number": 1, "section_header": "", "source_url": ""}
    ]
    pipeline.index_documents([{"doc_id": "test_doc", "text": "the cat sat on the mat in the living room"}])
    result = pipeline.query("Where is the cat?")
    assert result["gated"] is False


def test_pipeline_confidence_gate_passes_when_score_above_threshold():
    pipeline = _make_mock_pipeline()
    # rerank_score=8.5 in default mock is well above 0.0
    pipeline.index_documents([{"doc_id": "test_doc", "text": "the cat sat on the mat in the living room"}])
    result = pipeline.query("Where is the cat?")
    assert result["gated"] is False
    pipeline.generator.generate.assert_called()


def test_pipeline_confidence_gate_fires_on_empty_reranked_list():
    """No chunks returned → gate fires → generator not called."""
    pipeline = _make_mock_pipeline()
    pipeline.reranker.rerank.return_value = []
    pipeline.index_documents([{"doc_id": "test_doc", "text": "the cat sat on the mat in the living room"}])
    result = pipeline.query("Where is the cat?")
    assert result["gated"] is True
    pipeline.generator.generate.assert_not_called()


def test_pipeline_gated_answer_is_not_supported():
    """A gated response must have supported=False in structured_answer."""
    pipeline = _make_mock_pipeline()
    pipeline.reranker.rerank.return_value = [
        {"doc_id": "test_doc", "chunk_id": "test_doc_0", "text": "x",
         "score": 0.1, "rerank_score": -5.0,
         "page_number": 1, "section_header": "", "source_url": ""}
    ]
    pipeline.index_documents([{"doc_id": "test_doc", "text": "the cat sat on the mat in the living room"}])
    result = pipeline.query("Off-topic")
    assert result["structured_answer"]["supported"] is False
