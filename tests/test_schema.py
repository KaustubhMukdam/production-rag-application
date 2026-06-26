import json

from app.generation.schema import Answer, parse_answer


def test_parse_valid_json():
    raw = '{"answer": "The cat is on the mat.", "source_chunk_ids": ["doc1_0"], "supported": true}'
    result = parse_answer(raw)
    assert result is not None
    assert result.answer == "The cat is on the mat."
    assert result.source_chunk_ids == ["doc1_0"]
    assert result.supported is True


def test_parse_with_multiple_chunk_ids():
    raw = '{"answer": "n-grams are sequences.", "source_chunk_ids": ["ch3_1", "ch3_5", "ch4_2"], "supported": true}'
    result = parse_answer(raw)
    assert result is not None
    assert result.source_chunk_ids == ["ch3_1", "ch3_5", "ch4_2"]


def test_parse_supported_false():
    raw = '{"answer": "I am not sure.", "source_chunk_ids": [], "supported": false}'
    result = parse_answer(raw)
    assert result is not None
    assert result.supported is False
    assert result.source_chunk_ids == []


def test_parse_json_in_markdown_fences():
    raw = """```json
{"answer": "Hidden Markov Models are...", "source_chunk_ids": ["ch17_0"], "supported": true}
```"""
    result = parse_answer(raw)
    assert result is not None
    assert "Hidden Markov Models" in result.answer


def test_parse_json_with_trailing_commas():
    raw = '{"answer": "hello", "source_chunk_ids": ["a_1",], "supported": true,}'
    result = parse_answer(raw)
    assert result is not None
    assert result.answer == "hello"


def test_parse_json_with_extra_text_after():
    raw = '{"answer": "yes", "source_chunk_ids": ["x_0"], "supported": false} Hope this helps!'
    result = parse_answer(raw)
    assert result is not None
    assert result.answer == "yes"


def test_parse_supported_as_string_true():
    raw = '{"answer": "text", "source_chunk_ids": ["a_0"], "supported": "true"}'
    result = parse_answer(raw)
    assert result is not None
    assert result.supported is True


def test_parse_supported_as_string_false():
    raw = '{"answer": "text", "source_chunk_ids": ["a_0"], "supported": "false"}'
    result = parse_answer(raw)
    assert result is not None
    assert result.supported is False


def test_parse_invalid_json_returns_none():
    result = parse_answer("this is not json at all")
    assert result is None


def test_parse_empty_string_returns_none():
    result = parse_answer("")
    assert result is None


def test_parse_missing_fields_returns_none():
    raw = '{"answer": "hello"}'
    result = parse_answer(raw)
    assert result is None


def test_answer_unsupported_factory():
    ans = Answer.unsupported("partial answer")
    assert ans.answer == "partial answer"
    assert ans.source_chunk_ids == []
    assert ans.supported is False
