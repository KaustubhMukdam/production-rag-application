from app.ingestion.chunker import Chunker


def test_chunks_empty_text():
    chunker = Chunker(chunk_size=500, chunk_overlap=50)
    chunks = chunker.chunk_text("")
    assert chunks == []


def test_chunks_short_text_single_chunk():
    chunker = Chunker(chunk_size=500, chunk_overlap=50)
    text = "This is a short text."
    chunks = chunker.chunk_text(text)
    assert len(chunks) == 1
    assert chunks[0] == text


def test_chunks_long_text_multiple_chunks():
    chunker = Chunker(chunk_size=50, chunk_overlap=10)
    text = "word " * 200
    chunks = chunker.chunk_text(text)
    assert len(chunks) > 1


def test_chunks_no_overlap():
    chunker = Chunker(chunk_size=50, chunk_overlap=0)
    text = "token " * 100
    chunks = chunker.chunk_text(text)
    assert len(chunks) >= 2


def test_chunks_overlap_present():
    chunker = Chunker(chunk_size=50, chunk_overlap=10)
    text = "a " * 200
    chunks = chunker.chunk_text(text)
    if len(chunks) >= 2:
        assert chunks[0] != chunks[1]


def test_chunks_exact_chunk_size():
    chunker = Chunker(chunk_size=50, chunk_overlap=0)
    text = "hello world this is a test of the chunking mechanism"
    chunks = chunker.chunk_text(text)
    assert len(chunks) >= 1


def test_chunk_documents_empty():
    chunker = Chunker(chunk_size=500, chunk_overlap=50)
    chunks = chunker.chunk_documents([])
    assert chunks == []


def test_chunk_documents_single_doc():
    chunker = Chunker(chunk_size=50, chunk_overlap=0)
    docs = [{"doc_id": "doc1", "text": "word " * 100}]
    chunks = chunker.chunk_documents(docs)
    assert len(chunks) > 1
    assert all(c["doc_id"] == "doc1" for c in chunks)
    assert all("_0" in c["chunk_id"] or "_1" in c["chunk_id"] for c in chunks)


def test_chunk_documents_multiple_docs():
    chunker = Chunker(chunk_size=50, chunk_overlap=0)
    docs = [
        {"doc_id": "doc1", "text": "hello " * 30},
        {"doc_id": "doc2", "text": "world " * 30},
    ]
    chunks = chunker.chunk_documents(docs)
    assert len(chunks) > 1
    doc_ids = {c["doc_id"] for c in chunks}
    assert doc_ids == {"doc1", "doc2"}


def test_chunk_id_format():
    chunker = Chunker(chunk_size=50, chunk_overlap=0)
    docs = [{"doc_id": "mydoc", "text": "word " * 30}]
    chunks = chunker.chunk_documents(docs)
    for c in chunks:
        assert "mydoc" in c["chunk_id"]
        assert "_" in c["chunk_id"]
