import os
import requests

from app.generation.ollama import OllamaGenerator
from app.generation.prompts import build_prompt
from app.config import GROQ_MODEL


class GroqError(Exception):
    pass


class GroqGenerator:
    provider = "groq"

    def __init__(self, model: str = GROQ_MODEL, fallback=None):
        self.api_key = os.getenv("GROQ_API_KEY")
        if not self.api_key:
            raise GroqError("GROQ_API_KEY not set in environment")
        self.model = model
        self.fallback = fallback or OllamaGenerator()

    def generate(self, query: str, context_chunks: list, strict: bool = False) -> str:
        try:
            return self._groq_call(query, context_chunks, strict=strict)
        except requests.RequestException as e:
            self.provider = self.fallback.provider
            return self._fallback(query, context_chunks, f"Groq API error: {e}", strict=strict)

    def _groq_call(self, query: str, context_chunks: list, strict: bool = False) -> str:
        prompt = build_prompt(query, context_chunks, strict=strict)
        response = requests.post(
            "https://api.groq.com/openai/v1/chat/completions",
            headers={
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json",
            },
            json={
                "model": self.model,
                "messages": [
                    {"role": "user", "content": prompt},
                ],
            },
        )
        response.raise_for_status()
        return response.json()["choices"][0]["message"]["content"]

    def _fallback(self, query: str, context_chunks: list, reason: str, strict: bool = False) -> str:
        import logging
        logging.warning(f"GroqGenerator fallback to Ollama: {reason}")
        return self.fallback.generate(query, context_chunks, strict=strict)
