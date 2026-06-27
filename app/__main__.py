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
    sa = result["structured_answer"]
    if not sa["supported"]:
        print("\nNote: We could not strictly verify this answer against the provided documents.")
    if sa["source_chunk_ids"]:
        print(f"Sources: {', '.join(sa['source_chunk_ids'])}")


if __name__ == "__main__":
    main()
