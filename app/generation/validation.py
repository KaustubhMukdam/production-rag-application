from app.generation.schema import Answer


class CitationValidator:
    @staticmethod
    def validate(answer: Answer, retrieved_chunks: list[dict]) -> bool:
        chunk_ids = {c["chunk_id"] for c in retrieved_chunks}
        return all(cid in chunk_ids for cid in answer.source_chunk_ids)
