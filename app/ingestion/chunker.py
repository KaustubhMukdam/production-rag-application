from typing import List
import tiktoken

from app.config import CHUNK_SIZE, CHUNK_OVERLAP


class Chunker:
    def __init__(self, chunk_size: int = CHUNK_SIZE, chunk_overlap: int = CHUNK_OVERLAP):
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self._tokenizer = tiktoken.get_encoding("cl100k_base")

    def chunk_text(self, text: str) -> List[str]:
        if not text:
            return []
        tokens = self._tokenizer.encode(text)
        chunks = []
        start = 0
        while start < len(tokens):
            end = start + self.chunk_size
            chunk_tokens = tokens[start:end]
            chunk_text = self._tokenizer.decode(chunk_tokens)
            if chunk_text:
                chunks.append(chunk_text)
            if end >= len(tokens):
                break
            start += self.chunk_size - self.chunk_overlap
        return chunks

    def chunk_documents(self, documents: List[dict]) -> List[dict]:
        chunks = []
        for doc in documents:
            doc_chunks = self.chunk_text(doc["text"])
            for i, chunk in enumerate(doc_chunks):
                chunks.append({
                    "doc_id": doc["doc_id"],
                    "chunk_id": f"{doc['doc_id']}_{i}",
                    "text": chunk,
                })
        return chunks
