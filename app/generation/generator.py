from typing import Protocol

from app.config import LLM_PROVIDER


class Generator(Protocol):
    def generate(self, query: str, context_chunks: list, strict: bool = False) -> str:
        ...


def create_generator(provider: str | None = None) -> Generator:
    provider = provider or LLM_PROVIDER
    if provider == "groq":
        from app.generation.groq import GroqGenerator
        return GroqGenerator()
    return _fallback_generator()


def _fallback_generator() -> Generator:
    from app.generation.ollama import OllamaGenerator
    return OllamaGenerator()
