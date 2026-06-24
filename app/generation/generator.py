import requests

from app.config import OLLAMA_BASE_URL, OLLAMA_MODEL
from app.generation.prompts import build_prompt


class Generator:
    def __init__(self, base_url: str = OLLAMA_BASE_URL, model: str = OLLAMA_MODEL):
        self.base_url = base_url.rstrip("/")
        self.model = model

    def generate(self, query: str, context_chunks: list) -> str:
        prompt = build_prompt(query, context_chunks)
        response = requests.post(
            f"{self.base_url}/api/generate",
            json={"model": self.model, "prompt": prompt, "stream": False},
        )
        response.raise_for_status()
        return response.json()["response"]
