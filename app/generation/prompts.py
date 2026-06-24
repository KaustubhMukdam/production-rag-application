SYSTEM_PROMPT = (
    "You are a helpful assistant. Answer the user's question based only on the "
    "provided context. If the context does not contain enough information to "
    "answer the question, say 'I don't have enough information to answer this "
    "question.' Do not make up information."
)


def build_prompt(query: str, context_chunks: list) -> str:
    context = "\n\n".join(
        f"[Chunk {chunk['chunk_id']}] {chunk['text']}"
        for chunk in context_chunks
    )
    return (
        f"{SYSTEM_PROMPT}\n\n"
        f"Context:\n{context}\n\n"
        f"Question: {query}\n\n"
        f"Answer:"
    )
