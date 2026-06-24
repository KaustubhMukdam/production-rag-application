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
        print("No documents found. Download the corpus first (see docs/commands/p0-commands.md).")
        sys.exit(1)
    print(f"Indexing {len(docs)} document(s)...")
    pipeline = Pipeline()
    pipeline.index_documents(docs)
    print(f"Querying: {question}")
    result = pipeline.query(question)
    print(f"\nAnswer: {result['answer']}")


if __name__ == "__main__":
    main()
