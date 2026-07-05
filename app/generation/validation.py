import re

from app.generation.schema import Answer


def _strip_prefix(chunk_id: str) -> str:
    """Normalize ch{chapter}_{index}, removing any book prefix and optional dash."""
    m = re.search(r'ch-?(\d+_\d+)$', chunk_id)
    return f"ch{m.group(1)}" if m else chunk_id


class CitationValidator:
    @staticmethod
    def validate(answer: Answer, retrieved_chunks: list[dict]) -> bool:
        chunk_ids = {c["chunk_id"] for c in retrieved_chunks}
        for cid in answer.source_chunk_ids:
            if cid in chunk_ids:
                continue
            normalized = _strip_prefix(cid)
            if any(_strip_prefix(a) == normalized for a in chunk_ids):
                continue
            return False
        return True
