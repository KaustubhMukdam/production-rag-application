"""
CLI entrypoint — Phase 4.

Usage:
    python -m app "your question here"

Displays the answer, retrieval confidence (gated or not), and enriched
source citations (page number + section header where available).
"""
import sys

from app.pipeline import Pipeline
from app.ingestion.loader import load_documents


def main():
    if len(sys.argv) < 2:
        print("Usage: python -m app \"your question here\"")
        sys.exit(1)

    question = " ".join(sys.argv[1:])
    print("Loading documents...")
    docs = load_documents()
    if not docs:
        print("No documents found. Download the corpus first.")
        sys.exit(1)

    print(f"Indexing {len(docs)} document(s)...")
    pipeline = Pipeline()
    pipeline.index_documents(docs)

    print(f"Querying: {question}")
    result = pipeline.query(question)

    print(f"\nAnswer ({result['provider']}): {result['answer']}")

    if result.get("gated"):
        print("\nNote: Retrieval confidence was too low — answer blocked by confidence gate.")
        return

    sa = result["structured_answer"]
    if not sa["supported"]:
        print("\nNote: We could not strictly verify this answer against the provided documents.")

    if result.get("retrieved_chunks"):
        print("\nSources:")
        seen = set()
        for chunk in result["retrieved_chunks"]:
            cid = chunk.get("chunk_id", "")
            if cid in seen:
                continue
            seen.add(cid)
            parts = [f"  [{cid}]"]
            if chunk.get("page_number") is not None:
                parts.append(f"Page {chunk['page_number']}")
            if chunk.get("section_header"):
                parts.append(chunk["section_header"])
            if chunk.get("source_url"):
                parts.append(chunk["source_url"])
            print("  " + " | ".join(parts))


if __name__ == "__main__":
    main()
