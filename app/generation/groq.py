import os
import time
import requests

from app.generation.ollama import OllamaGenerator
from app.generation.prompts import build_prompt
from app.config import GROQ_MODEL

_RETRYABLE_STATUSES = {429, 500, 502, 503}
_UNSET = object()


class GroqError(Exception):
    pass


class GroqGenerator:
    provider = "groq"

    def __init__(self, model: str = GROQ_MODEL, fallback=_UNSET, max_retries: int = 3):
        self.api_key = os.getenv("GROQ_API_KEY")
        if not self.api_key:
            raise GroqError("GROQ_API_KEY not set in environment")
        self.model = model
        self.fallback = OllamaGenerator() if fallback is _UNSET else fallback
        self.max_retries = max_retries

    def generate(self, query: str, context_chunks: list, strict: bool = False) -> str:
        last_error = None
        for attempt in range(self.max_retries):
            try:
                return self._groq_call(query, context_chunks, strict=strict)
            except requests.HTTPError as e:
                status = e.response.status_code if e.response is not None else 0
                if status in _RETRYABLE_STATUSES and attempt < self.max_retries - 1:
                    wait = 2 ** (attempt + 1)
                    import logging
                    logging.warning(f"Groq {status}, retry {attempt + 1}/{self.max_retries} in {wait}s")
                    time.sleep(wait)
                    last_error = e
                    continue
                last_error = e
                break
            except requests.RequestException as e:
                last_error = e
                break

        if self.fallback is None:
            raise GroqError(f"Groq error after {self.max_retries} tries: {last_error}") from last_error
        self.provider = self.fallback.provider
        return self._fallback(query, context_chunks, f"Groq error after {self.max_retries} tries: {last_error}", strict=strict)

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
