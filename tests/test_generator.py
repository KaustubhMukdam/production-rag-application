from unittest.mock import patch

from app.generation.generator import create_generator, _fallback_generator


def test_create_generator_ollama():
    with patch("app.config.LLM_PROVIDER", "ollama"):
        gen = create_generator()
        from app.generation.ollama import OllamaGenerator
        assert isinstance(gen, OllamaGenerator)


def test_create_generator_explicit_ollama():
    gen = create_generator("ollama")
    from app.generation.ollama import OllamaGenerator
    assert isinstance(gen, OllamaGenerator)


@patch.dict("os.environ", {"GROQ_API_KEY": "sk-test-key"})
def test_create_generator_explicit_groq():
    gen = create_generator("groq")
    from app.generation.groq import GroqGenerator
    assert isinstance(gen, GroqGenerator)


def test_fallback_generator_returns_ollama():
    gen = _fallback_generator()
    from app.generation.ollama import OllamaGenerator
    assert isinstance(gen, OllamaGenerator)


def test_create_generator_unknown_provider_falls_back():
    gen = create_generator("nonexistent")
    from app.generation.ollama import OllamaGenerator
    assert isinstance(gen, OllamaGenerator)
