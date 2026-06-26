from unittest.mock import patch, Mock

from app.generation.ollama import OllamaGenerator
from app.generation.prompts import build_prompt


def test_build_prompt_includes_context():
    chunks = [
        {"chunk_id": "doc1_0", "text": "the cat sat on the mat"},
    ]
    prompt = build_prompt("Where is the cat?", chunks)
    assert "the cat sat on the mat" in prompt
    assert "Where is the cat?" in prompt


def test_build_prompt_strict_includes_strict_instruction():
    chunks = [{"chunk_id": "doc1_0", "text": "content"}]
    prompt = build_prompt("query", chunks, strict=True)
    assert "MUST" in prompt
    assert "valid JSON" in prompt


def test_build_prompt_default_not_strict():
    chunks = [{"chunk_id": "doc1_0", "text": "content"}]
    prompt = build_prompt("query", chunks)
    assert "MUST" not in prompt


def test_build_prompt_includes_json_schema():
    chunks = [{"chunk_id": "doc1_0", "text": "content"}]
    prompt = build_prompt("query", chunks)
    assert "source_chunk_ids" in prompt
    assert "\"supported\": true" in prompt


def test_build_prompt_multiple_chunks():
    chunks = [
        {"chunk_id": "doc1_0", "text": "chunk one content"},
        {"chunk_id": "doc1_1", "text": "chunk two content"},
    ]
    prompt = build_prompt("test query", chunks)
    assert "chunk one content" in prompt
    assert "chunk two content" in prompt
    assert "[Chunk doc1_0]" in prompt
    assert "[Chunk doc1_1]" in prompt


@patch("app.generation.ollama.requests.post")
def test_generate_calls_ollama(mock_post):
    mock_response = Mock()
    mock_response.json.return_value = {"response": "The cat is on the mat."}
    mock_response.raise_for_status.return_value = None
    mock_post.return_value = mock_response

    generator = OllamaGenerator(base_url="http://localhost:11434", model="llama3.2:3b")
    chunks = [{"chunk_id": "doc1_0", "text": "the cat sat on the mat"}]
    answer = generator.generate("Where is the cat?", chunks)

    assert answer == "The cat is on the mat."
    mock_post.assert_called_once()


@patch("app.generation.ollama.requests.post")
def test_generate_sends_correct_payload(mock_post):
    mock_response = Mock()
    mock_response.json.return_value = {"response": "answer"}
    mock_response.raise_for_status.return_value = None
    mock_post.return_value = mock_response

    generator = OllamaGenerator(base_url="http://localhost:11434", model="llama3.2:3b")
    chunks = [{"chunk_id": "doc1_0", "text": "context text"}]
    generator.generate("test query", chunks)

    call_kwargs = mock_post.call_args[1]
    assert call_kwargs["json"]["model"] == "llama3.2:3b"
    assert "test query" in call_kwargs["json"]["prompt"]
    assert call_kwargs["json"]["stream"] is False


@patch("app.generation.ollama.requests.post")
def test_generate_uses_base_url(mock_post):
    mock_response = Mock()
    mock_response.json.return_value = {"response": "answer"}
    mock_response.raise_for_status.return_value = None
    mock_post.return_value = mock_response

    generator = OllamaGenerator(base_url="http://custom:11434", model="phi3:mini")
    generator.generate("query", [])

    called_url = mock_post.call_args[0][0]
    assert "http://custom:11434" in called_url
