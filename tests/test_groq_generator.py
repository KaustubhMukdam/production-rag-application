from unittest.mock import patch, MagicMock

import pytest
import requests

from app.generation.groq import GroqGenerator, GroqError
from app.generation.ollama import OllamaGenerator


def test_groq_generator_raises_without_key():
    with patch.dict("os.environ", {}, clear=True):
        with pytest.raises(GroqError, match="GROQ_API_KEY not set"):
            GroqGenerator()


def test_groq_generator_accepts_key_from_env():
    with patch.dict("os.environ", {"GROQ_API_KEY": "sk-test-key"}):
        gen = GroqGenerator()
        assert gen.api_key == "sk-test-key"


def test_groq_generator_sets_api_key():
    with patch.dict("os.environ", {"GROQ_API_KEY": "sk-test-key"}):
        gen = GroqGenerator()
        assert gen.api_key == "sk-test-key"


@patch("app.generation.groq.requests.post")
def test_groq_generate_success(mock_post):
    mock_response = MagicMock()
    mock_response.json.return_value = {
        "choices": [{"message": {"content": "The answer is 42."}}]
    }
    mock_response.raise_for_status.return_value = None
    mock_post.return_value = mock_response

    with patch.dict("os.environ", {"GROQ_API_KEY": "sk-test-key"}):
        gen = GroqGenerator()
        chunks = [{"chunk_id": "doc1_0", "text": "context"}]
        answer = gen.generate("What is the answer?", chunks)

    assert answer == "The answer is 42."
    mock_post.assert_called_once()


@patch("app.generation.groq.requests.post")
def test_groq_generate_sends_correct_payload(mock_post):
    mock_response = MagicMock()
    mock_response.json.return_value = {
        "choices": [{"message": {"content": "answer"}}]
    }
    mock_response.raise_for_status.return_value = None
    mock_post.return_value = mock_response

    with patch.dict("os.environ", {"GROQ_API_KEY": "sk-test-key"}):
        gen = GroqGenerator(model="llama-3.3-70b-versatile")
        gen.generate("test query", [{"chunk_id": "a", "text": "ctx"}])

    call_kwargs = mock_post.call_args[1]
    assert call_kwargs["headers"]["Authorization"] == "Bearer sk-test-key"
    assert call_kwargs["json"]["model"] == "llama-3.3-70b-versatile"


@patch("app.generation.groq.requests.post")
def test_groq_generate_uses_correct_url(mock_post):
    mock_response = MagicMock()
    mock_response.json.return_value = {
        "choices": [{"message": {"content": "answer"}}]
    }
    mock_response.raise_for_status.return_value = None
    mock_post.return_value = mock_response

    with patch.dict("os.environ", {"GROQ_API_KEY": "sk-test-key"}):
        gen = GroqGenerator()
        gen.generate("query", [{"chunk_id": "a", "text": "ctx"}])

    called_url = mock_post.call_args[0][0]
    assert "api.groq.com" in called_url
    assert "chat/completions" in called_url


@patch("app.generation.groq.requests.post")
def test_groq_api_error_falls_back_to_ollama(mock_post):
    mock_post.side_effect = requests.RequestException("API error")

    mock_ollama = MagicMock(spec=OllamaGenerator)
    mock_ollama.generate.return_value = "Ollama fallback answer"

    with patch.dict("os.environ", {"GROQ_API_KEY": "sk-test-key"}):
        gen = GroqGenerator(fallback=mock_ollama)
        answer = gen.generate("query", [{"chunk_id": "a", "text": "ctx"}])

    assert answer == "Ollama fallback answer"
    mock_ollama.generate.assert_called_once()


@patch("app.generation.groq.requests.post")
def test_groq_http_error_falls_back(mock_post):
    http_error = requests.HTTPError("401 Unauthorized")
    mock_response = MagicMock()
    mock_response.raise_for_status.side_effect = http_error
    mock_post.return_value = mock_response

    mock_ollama = MagicMock(spec=OllamaGenerator)
    mock_ollama.generate.return_value = "fallback"

    with patch.dict("os.environ", {"GROQ_API_KEY": "sk-test-key"}):
        gen = GroqGenerator(fallback=mock_ollama)
        answer = gen.generate("query", [{"chunk_id": "a", "text": "ctx"}])

    assert answer == "fallback"
    mock_ollama.generate.assert_called_once()


@patch("app.generation.groq.requests.post")
def test_groq_token_limit_error_falls_back(mock_post):
    mock_response = MagicMock()
    mock_response.raise_for_status.side_effect = requests.HTTPError(
        "413 Request Entity Too Large"
    )
    mock_post.return_value = mock_response

    mock_ollama = MagicMock(spec=OllamaGenerator)
    mock_ollama.generate.return_value = "fallback answer"

    with patch.dict("os.environ", {"GROQ_API_KEY": "sk-test-key"}):
        gen = GroqGenerator(fallback=mock_ollama)
        answer = gen.generate("query", [{"chunk_id": "a", "text": "ctx"}])

    assert answer == "fallback answer"
    mock_ollama.generate.assert_called_once()
