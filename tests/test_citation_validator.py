from app.generation.schema import Answer
from app.generation.validation import CitationValidator


def test_valid_all_ids_match():
    retrieved = [
        {"chunk_id": "doc1_0", "text": "foo"},
        {"chunk_id": "doc1_1", "text": "bar"},
    ]
    answer = Answer(answer="text", source_chunk_ids=["doc1_0", "doc1_1"], supported=True)
    assert CitationValidator.validate(answer, retrieved) is True


def test_valid_single_id():
    retrieved = [{"chunk_id": "ch3_2", "text": "n-gram definition"}]
    answer = Answer(answer="text", source_chunk_ids=["ch3_2"], supported=True)
    assert CitationValidator.validate(answer, retrieved) is True


def test_invalid_id_not_in_retrieved():
    retrieved = [{"chunk_id": "doc1_0", "text": "foo"}]
    answer = Answer(answer="text", source_chunk_ids=["doc1_0", "nonexistent"], supported=True)
    assert CitationValidator.validate(answer, retrieved) is False


def test_invalid_empty_retrieved():
    retrieved = []
    answer = Answer(answer="text", source_chunk_ids=["doc1_0"], supported=True)
    assert CitationValidator.validate(answer, retrieved) is False


def test_valid_empty_source_ids():
    retrieved = [{"chunk_id": "doc1_0", "text": "foo"}]
    answer = Answer(answer="text", source_chunk_ids=[], supported=False)
    assert CitationValidator.validate(answer, retrieved) is True


def test_valid_case_sensitive_ids():
    retrieved = [{"chunk_id": "Doc1_0", "text": "foo"}]
    answer = Answer(answer="text", source_chunk_ids=["doc1_0"], supported=True)
    assert CitationValidator.validate(answer, retrieved) is False


def test_validate_returns_bool():
    retrieved = [{"chunk_id": "a", "text": "x"}]
    answer = Answer(answer="x", source_chunk_ids=["a"], supported=True)
    result = CitationValidator.validate(answer, retrieved)
    assert isinstance(result, bool)
