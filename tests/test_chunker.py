"""
TDD tests for app.ingestion.chunker (SemanticChunker)
------------------------------------------------------
Tests cover both the new semantic-split behaviour and backward-compat with
the old flat-text {doc_id, text} input format used by existing fixtures.

Input format (new): {doc_id, pages: [{page_num, text}], source_url}
Input format (old): {doc_id, text}  ← still accepted, treated as page 1

Output format: {doc_id, chunk_id, text, page_number, section_header, source_url}
"""
import pytest
from app.ingestion.chunker import SemanticChunker


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def chunker():
    return SemanticChunker(chunk_size=500, chunk_overlap=50)


@pytest.fixture
def small_chunker():
    """Chunker with tiny chunk size to force token-window fallback."""
    return SemanticChunker(chunk_size=20, chunk_overlap=5)


def _doc_new(doc_id: str, pages: list[dict], source_url: str = "https://example.com/doc.pdf") -> dict:
    return {"doc_id": doc_id, "pages": pages, "source_url": source_url}


def _doc_old(doc_id: str, text: str) -> dict:
    """Old flat-text format (backward compat)."""
    return {"doc_id": doc_id, "text": text}


def _page(page_num: int, text: str) -> dict:
    return {"page_num": page_num, "text": text}


# ---------------------------------------------------------------------------
# Output schema tests
# ---------------------------------------------------------------------------

class TestOutputSchema:
    """Every chunk must contain the required keys."""

    def test_chunk_has_doc_id(self, chunker):
        docs = [_doc_new("d1", [_page(1, "Some content here.")])]
        chunks = chunker.chunk_documents(docs)
        assert all("doc_id" in c for c in chunks)

    def test_chunk_has_chunk_id(self, chunker):
        docs = [_doc_new("d1", [_page(1, "Some content here.")])]
        chunks = chunker.chunk_documents(docs)
        assert all("chunk_id" in c for c in chunks)

    def test_chunk_has_text(self, chunker):
        docs = [_doc_new("d1", [_page(1, "Some content here.")])]
        chunks = chunker.chunk_documents(docs)
        assert all("text" in c for c in chunks)

    def test_chunk_has_page_number(self, chunker):
        docs = [_doc_new("d1", [_page(3, "Some content on page 3.")])]
        chunks = chunker.chunk_documents(docs)
        assert all("page_number" in c for c in chunks)

    def test_chunk_has_section_header(self, chunker):
        docs = [_doc_new("d1", [_page(1, "Some content.")])]
        chunks = chunker.chunk_documents(docs)
        assert all("section_header" in c for c in chunks)

    def test_chunk_has_source_url(self, chunker):
        docs = [_doc_new("d1", [_page(1, "Some content.")], source_url="https://x.com/d.pdf")]
        chunks = chunker.chunk_documents(docs)
        assert all("source_url" in c for c in chunks)

    def test_source_url_propagates_correctly(self, chunker):
        url = "https://slp3.example.com/ch03.pdf"
        docs = [_doc_new("ch03", [_page(1, "Content.")], source_url=url)]
        chunks = chunker.chunk_documents(docs)
        assert all(c["source_url"] == url for c in chunks)

    def test_page_number_is_int(self, chunker):
        docs = [_doc_new("d1", [_page(5, "Page five content.")])]
        chunks = chunker.chunk_documents(docs)
        assert all(isinstance(c["page_number"], int) for c in chunks)

    def test_chunk_id_contains_doc_id(self, chunker):
        docs = [_doc_new("mydoc", [_page(1, "word " * 10)])]
        chunks = chunker.chunk_documents(docs)
        assert all("mydoc" in c["chunk_id"] for c in chunks)

    def test_chunk_id_has_underscore_separator(self, chunker):
        docs = [_doc_new("mydoc", [_page(1, "word " * 10)])]
        chunks = chunker.chunk_documents(docs)
        assert all("_" in c["chunk_id"] for c in chunks)

    def test_chunk_ids_are_unique(self, chunker):
        docs = [_doc_new("d1", [_page(1, "word " * 200)])]
        chunks = chunker.chunk_documents(docs)
        ids = [c["chunk_id"] for c in chunks]
        assert len(ids) == len(set(ids))


# ---------------------------------------------------------------------------
# Page number propagation
# ---------------------------------------------------------------------------

class TestPageNumber:
    """Chunks must carry the page number of the page they originate from."""

    def test_page_number_matches_source_page(self, chunker):
        docs = [_doc_new("d1", [_page(7, "Content on page seven.")])]
        chunks = chunker.chunk_documents(docs)
        assert chunks[0]["page_number"] == 7

    def test_multipage_doc_chunks_carry_correct_page(self, chunker):
        docs = [_doc_new("d1", [
            _page(1, "First page content."),
            _page(2, "Second page content."),
        ])]
        chunks = chunker.chunk_documents(docs)
        page_nums = {c["page_number"] for c in chunks}
        # must include both pages
        assert 1 in page_nums
        assert 2 in page_nums


# ---------------------------------------------------------------------------
# Section header detection
# ---------------------------------------------------------------------------

class TestSectionHeaderDetection:
    """The chunker must detect heading-like lines and tag chunks with them."""

    def test_numbered_section_detected_as_header(self, chunker):
        text = "3.2 The Viterbi Algorithm\n\nThis algorithm is efficient."
        docs = [_doc_new("d1", [_page(1, text)])]
        chunks = chunker.chunk_documents(docs)
        headers = [c["section_header"] for c in chunks]
        assert any("Viterbi" in h for h in headers)

    def test_allcaps_line_detected_as_header(self, chunker):
        text = "WORDS AND TRANSDUCERS\n\nThis chapter covers transducers."
        docs = [_doc_new("d1", [_page(1, text)])]
        chunks = chunker.chunk_documents(docs)
        headers = headers_of(chunks)
        assert any("WORDS AND TRANSDUCERS" in h for h in headers)

    def test_plain_prose_has_empty_or_inherited_header(self, chunker):
        text = "This is plain prose without any headings. It continues normally."
        docs = [_doc_new("d1", [_page(1, text)])]
        chunks = chunker.chunk_documents(docs)
        # All chunks should have section_header key; value may be ""
        assert all("section_header" in c for c in chunks)

    def test_section_header_is_string(self, chunker):
        docs = [_doc_new("d1", [_page(1, "Some text here.")])]
        chunks = chunker.chunk_documents(docs)
        assert all(isinstance(c["section_header"], str) for c in chunks)

    def test_header_is_assigned_to_subsequent_chunks(self, chunker):
        """Chunks that follow a heading should carry that heading."""
        text = "3.2 The Viterbi Algorithm\n\n" + ("content word " * 60)
        docs = [_doc_new("d1", [_page(1, text)])]
        chunks = chunker.chunk_documents(docs)
        # At least one chunk should carry the header
        assert any("Viterbi" in c["section_header"] for c in chunks)


def headers_of(chunks):
    return [c["section_header"] for c in chunks]


# ---------------------------------------------------------------------------
# Token-window fallback
# ---------------------------------------------------------------------------

class TestTokenWindowFallback:
    """Large sections must be sub-split by the token window."""

    def test_large_section_produces_multiple_chunks(self, small_chunker):
        # 200 words >> chunk_size=20 tokens
        text = "word " * 200
        docs = [_doc_new("d1", [_page(1, text)])]
        chunks = small_chunker.chunk_documents(docs)
        assert len(chunks) > 1

    def test_no_chunk_exceeds_chunk_size(self, small_chunker):
        import tiktoken
        enc = tiktoken.get_encoding("cl100k_base")
        text = "word " * 200
        docs = [_doc_new("d1", [_page(1, text)])]
        chunks = small_chunker.chunk_documents(docs)
        for c in chunks:
            assert len(enc.encode(c["text"])) <= small_chunker.chunk_size

    def test_no_chunk_text_is_empty(self, small_chunker):
        text = "word " * 100
        docs = [_doc_new("d1", [_page(1, text)])]
        chunks = small_chunker.chunk_documents(docs)
        assert all(c["text"].strip() for c in chunks)


# ---------------------------------------------------------------------------
# Backward compatibility (old flat-text format)
# ---------------------------------------------------------------------------

class TestBackwardCompat:
    """Old {doc_id, text} format must still work without crashing."""

    def test_old_format_produces_chunks(self, chunker):
        docs = [_doc_old("doc1", "word " * 100)]
        chunks = chunker.chunk_documents(docs)
        assert len(chunks) > 0

    def test_old_format_chunks_have_all_new_keys(self, chunker):
        docs = [_doc_old("doc1", "word " * 50)]
        chunks = chunker.chunk_documents(docs)
        for c in chunks:
            assert "page_number" in c
            assert "section_header" in c
            assert "source_url" in c

    def test_old_format_page_number_defaults_to_1(self, chunker):
        docs = [_doc_old("doc1", "some text")]
        chunks = chunker.chunk_documents(docs)
        assert chunks[0]["page_number"] == 1

    def test_old_format_source_url_is_string(self, chunker):
        docs = [_doc_old("doc1", "some text")]
        chunks = chunker.chunk_documents(docs)
        assert isinstance(chunks[0]["source_url"], str)

    def test_empty_input_returns_empty_list(self, chunker):
        assert chunker.chunk_documents([]) == []

    def test_empty_text_returns_no_chunks(self, chunker):
        docs = [_doc_old("doc1", "")]
        chunks = chunker.chunk_documents(docs)
        assert chunks == []

    def test_short_text_single_chunk(self, chunker):
        docs = [_doc_old("doc1", "This is a short text.")]
        chunks = chunker.chunk_documents(docs)
        assert len(chunks) == 1
        assert chunks[0]["text"] == "This is a short text."

    def test_multiple_docs_all_have_doc_id(self, chunker):
        docs = [
            _doc_old("doc1", "hello " * 30),
            _doc_old("doc2", "world " * 30),
        ]
        chunks = chunker.chunk_documents(docs)
        doc_ids = {c["doc_id"] for c in chunks}
        assert doc_ids == {"doc1", "doc2"}
