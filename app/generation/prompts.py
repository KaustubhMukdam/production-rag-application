"""
Prompt templates — Phase 4.

Changes from Phase 3:
  - _format_chunk_header() builds a rich label that includes page number
    and section header when the chunk carries them.  Falls back gracefully
    when those fields are absent so existing tests and eval fixtures work.
"""

SYSTEM_PROMPT = (
    "You are a helpful assistant. Answer the user's question based only on the "
    "provided context. If the context does not contain enough information to "
    "answer the question, say 'I don't have enough information to answer this "
    "question.' Do not make up information."
)

JSON_SCHEMA_INSTRUCTION = (
    'Return your answer as valid JSON with the following fields:\n'
    '- "answer": string, your response to the question\n'
    '- "source_chunk_ids": list of strings, the chunk IDs from the context that support your answer\n'
    '- "supported": boolean, true if your answer is well-supported by the provided context, false otherwise\n'
    '\n'
    'Example:\n'
    '{\n'
    '  "answer": "The capital of France is Paris.",\n'
    '  "source_chunk_ids": ["ch3_2", "ch3_5"],\n'
    '  "supported": true\n'
    '}\n'
    '\n'
    'Return ONLY the JSON object, no other text.'
)

STRICT_JSON_INSTRUCTION = (
    'You MUST return ONLY a valid JSON object with exactly these fields:\n'
    '- "answer": string\n'
    '- "source_chunk_ids": list of strings (must be from the chunk IDs listed below)\n'
    '- "supported": boolean\n'
    '\n'
    'Do NOT include markdown, explanations, or any text outside the JSON object.'
)


def _format_chunk_header(chunk: dict) -> str:
    """Build a rich chunk label including page and section when available.

    Examples:
        "Chunk ch3_2 | Page 47 | Section: 3.2 The Viterbi Algorithm"
        "Chunk ch3_2 | Page 47"
        "Chunk ch3_2"
    """
    parts = [f"Chunk {chunk['chunk_id']}"]

    page = chunk.get("page_number")
    if page is not None:
        parts.append(f"Page {page}")

    header = chunk.get("section_header", "")
    if header:
        parts.append(f"Section: {header}")

    return " | ".join(parts)


def build_prompt(query: str, context_chunks: list, strict: bool = False) -> str:
    context = "\n\n".join(
        f"[{_format_chunk_header(chunk)}]\n{chunk['text']}"
        for chunk in context_chunks
    )
    instruction = STRICT_JSON_INSTRUCTION if strict else JSON_SCHEMA_INSTRUCTION
    return (
        f"{SYSTEM_PROMPT}\n\n"
        f"{instruction}\n\n"
        f"Context:\n{context}\n\n"
        f"Question: {query}\n\n"
        f"Answer:"
    )
