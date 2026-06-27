from typing import Protocol


class Generator(Protocol):
    def generate(self, query: str, context_chunks: list, strict: bool = False) -> str:
        ...


def create_generator(provider: str | None = None) -> Generator:
    if provider is None:
        from app.config import LLM_PROVIDER
        provider = LLM_PROVIDER
    if provider == "groq":
        from app.generation.groq import GroqGenerator
        return GroqGenerator()
    return _fallback_generator()


def _fallback_generator() -> Generator:
    from app.generation.ollama import OllamaGenerator
    return OllamaGenerator()
