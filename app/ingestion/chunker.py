"""
SemanticChunker — Phase 4 ingestion.

Two-pass chunking strategy:
  1. Header split — detect section headings in the page stream and split
     there so chunks respect logical document structure.
  2. Token-window fallback — if a section exceeds CHUNK_SIZE tokens the
     existing sliding-window logic is applied *within that section* so no
     chunk ever overflows the embedding model's context window.

Input document format (new):
    {"doc_id": str, "pages": [{"page_num": int, "text": str}], "source_url": str}

Input document format (old — backward compat):
    {"doc_id": str, "text": str}
    Treated as a single page with page_num=1 and source_url="".

Output chunk format:
    {
        "doc_id":         str,
        "chunk_id":       str,   # f"{doc_id}_{global_index}"
        "text":           str,
        "page_number":    int,   # 1-indexed page where chunk starts
        "section_header": str,   # nearest heading above chunk, "" if none
        "source_url":     str,   # URL to the source PDF
    }
"""
import re
import logging
from typing import List

import tiktoken

from app.config import CHUNK_SIZE, CHUNK_OVERLAP

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Header detection patterns
# ---------------------------------------------------------------------------

_NUMBERED_SECTION = re.compile(r"^\d+(\.\d+)*\s+[A-Z\w]")
_ALLCAPS_HEADING = re.compile(r"^[A-Z][A-Z\s\d\-:]{3,}$")
_MAX_HEADER_LEN = 80
_SENTENCE_END = re.compile(r"[.!?,;:]\s*$")


def _is_heading(line: str) -> bool:
    """Return True if *line* looks like a section heading.

    Heuristics (conservative to avoid over-splitting):
    - Matches a numbered section pattern  e.g. "3.2 The Viterbi Algorithm"
    - OR is an all-caps short line        e.g. "WORDS AND TRANSDUCERS"
    - AND is not longer than _MAX_HEADER_LEN chars
    - AND does not end with sentence-terminal punctuation
    """
    stripped = line.strip()
    if not stripped or len(stripped) > _MAX_HEADER_LEN:
        return False
    if _SENTENCE_END.search(stripped):
        return False
    return bool(_NUMBERED_SECTION.match(stripped) or _ALLCAPS_HEADING.match(stripped))


# ---------------------------------------------------------------------------
# SemanticChunker
# ---------------------------------------------------------------------------

class SemanticChunker:
    """Chunk documents by respecting section headers, with token-window fallback."""

    def __init__(self, chunk_size: int = CHUNK_SIZE, chunk_overlap: int = CHUNK_OVERLAP):
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self._tokenizer = tiktoken.get_encoding("cl100k_base")

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    def chunk_documents(self, documents: List[dict]) -> List[dict]:
        """Chunk a list of documents into context-preserving pieces.

        Accepts both the new page-aware format and the old flat-text format
        for backward compatibility with test fixtures and CLI callers that
        have not yet been updated.
        """
        chunks: List[dict] = []
        chunk_idx = 0

        for doc in documents:
            doc_id = doc["doc_id"]
            source_url = doc.get("source_url", "")
            pages = self._normalise_pages(doc)

            if not pages:
                continue

            for section in self._split_into_sections(pages):
                for text in self._split_section_by_tokens(section["text"]):
                    chunks.append({
                        "doc_id": doc_id,
                        "chunk_id": f"{doc_id}_{chunk_idx}",
                        "text": text,
                        "page_number": section["page_num"],
                        "section_header": section["header"],
                        "source_url": source_url,
                    })
                    chunk_idx += 1

        return chunks

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _normalise_pages(doc: dict) -> List[dict]:
        """Convert either input format into a list of page dicts."""
        if "pages" in doc:
            return [p for p in doc["pages"] if p.get("text", "").strip()]
        # Backward compat: flat text → single synthetic page
        text = doc.get("text", "")
        if not text.strip():
            return []
        return [{"page_num": 1, "text": text}]

    def _split_into_sections(self, pages: List[dict]) -> List[dict]:
        """Walk the page stream and split at heading boundaries.

        Returns a list of sections:
            {"page_num": int, "header": str, "text": str}
        """
        sections: List[dict] = []
        current_header = ""
        current_page = pages[0]["page_num"]
        current_lines: List[str] = []

        def _flush():
            text = "\n".join(current_lines).strip()
            if text:
                sections.append({
                    "page_num": current_page,
                    "header": current_header,
                    "text": text,
                })

        for page in pages:
            for line in page["text"].splitlines():
                if _is_heading(line):
                    _flush()
                    current_lines = []
                    current_header = line.strip()
                    current_page = page["page_num"]
                else:
                    # When moving to a new page without hitting a heading,
                    # flush the previous section so chunks carry the correct
                    # page number (rather than inheriting the last heading's page).
                    if current_lines and page["page_num"] != current_page:
                        _flush()
                        current_lines = []
                        # Inherit the section header across the page boundary
                    current_page = page["page_num"]
                    current_lines.append(line)

        _flush()

        # Edge case: no headings found → single section with empty header
        if not sections:
            all_text = "\n".join(p["text"] for p in pages).strip()
            if all_text:
                sections.append({
                    "page_num": pages[0]["page_num"],
                    "header": "",
                    "text": all_text,
                })

        return sections

    def _split_section_by_tokens(self, text: str) -> List[str]:
        """Apply sliding-window token splitting within a single section.

        If the section fits in one chunk it is returned as-is.
        """
        if not text.strip():
            return []

        tokens = self._tokenizer.encode(text)
        if len(tokens) <= self.chunk_size:
            return [text]

        chunks: List[str] = []
        start = 0
        while start < len(tokens):
            end = start + self.chunk_size
            chunk_text = self._tokenizer.decode(tokens[start:end])
            if chunk_text.strip():
                chunks.append(chunk_text)
            if end >= len(tokens):
                break
            start += self.chunk_size - self.chunk_overlap

        return chunks


# ---------------------------------------------------------------------------
# Public alias — keeps old import paths working (used in __main__ and eval)
# ---------------------------------------------------------------------------

Chunker = SemanticChunker
