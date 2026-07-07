"""
TDD tests for app.ingestion.loader
-----------------------------------
These tests cover the Phase-4 contract:
  load_documents() -> list[{doc_id, pages, source_url}]
  pages            -> list[{page_num: int, text: str}]
  source_url       -> str  (CORPUS_URL + doc_id + ".pdf")

All fitz/PyMuPDF calls are mocked so no real PDFs are required.
"""
from pathlib import Path
from unittest.mock import MagicMock, patch, call
import pytest


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_fitz_page(text: str) -> MagicMock:
    page = MagicMock()
    page.get_text.return_value = text
    return page


def _make_fitz_doc(page_texts: list[str]) -> MagicMock:
    """Return a mock fitz Document that iterates over fake pages."""
    pages = [_make_fitz_page(t) for t in page_texts]
    doc = MagicMock()
    doc.__iter__ = MagicMock(return_value=iter(pages))
    doc.__len__ = MagicMock(return_value=len(pages))
    return doc


def _fake_pdf_dir(tmp_path: Path, filenames: list[str]) -> Path:
    pdf_dir = tmp_path / "pdfs"
    pdf_dir.mkdir()
    for name in filenames:
        (pdf_dir / name).write_bytes(b"fake-pdf-content")
    return pdf_dir


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

class TestLoadDocumentsInterface:
    """The returned dict shape must include pages and source_url."""

    def test_returns_pages_key(self, tmp_path):
        pdf_dir = _fake_pdf_dir(tmp_path, ["ch01.pdf"])
        mock_doc = _make_fitz_doc(["Page one text."])
        with patch("fitz.open", return_value=mock_doc):
            from app.ingestion.loader import load_documents
            docs = load_documents(pdf_dir)
        assert len(docs) == 1
        assert "pages" in docs[0]

    def test_does_not_return_flat_text_key(self, tmp_path):
        """New interface uses pages list, not a single flat text string."""
        pdf_dir = _fake_pdf_dir(tmp_path, ["ch01.pdf"])
        mock_doc = _make_fitz_doc(["Some text."])
        with patch("fitz.open", return_value=mock_doc):
            from app.ingestion.loader import load_documents
            docs = load_documents(pdf_dir)
        # flat "text" key should not be present in new interface
        assert "text" not in docs[0]

    def test_returns_source_url_key(self, tmp_path):
        pdf_dir = _fake_pdf_dir(tmp_path, ["ch03.pdf"])
        mock_doc = _make_fitz_doc(["Chapter 3 text."])
        with patch("fitz.open", return_value=mock_doc):
            from app.ingestion.loader import load_documents
            docs = load_documents(pdf_dir)
        assert "source_url" in docs[0]

    def test_returns_doc_id_key(self, tmp_path):
        pdf_dir = _fake_pdf_dir(tmp_path, ["mybook.pdf"])
        mock_doc = _make_fitz_doc(["Content."])
        with patch("fitz.open", return_value=mock_doc):
            from app.ingestion.loader import load_documents
            docs = load_documents(pdf_dir)
        assert docs[0]["doc_id"] == "mybook"


class TestPageExtraction:
    """pages list must contain dicts with page_num (int) and text (str)."""

    def test_pages_is_a_list(self, tmp_path):
        pdf_dir = _fake_pdf_dir(tmp_path, ["ch01.pdf"])
        mock_doc = _make_fitz_doc(["Page 1 text.", "Page 2 text."])
        with patch("fitz.open", return_value=mock_doc):
            from app.ingestion.loader import load_documents
            docs = load_documents(pdf_dir)
        assert isinstance(docs[0]["pages"], list)

    def test_page_has_page_num_and_text(self, tmp_path):
        pdf_dir = _fake_pdf_dir(tmp_path, ["ch01.pdf"])
        mock_doc = _make_fitz_doc(["First page."])
        with patch("fitz.open", return_value=mock_doc):
            from app.ingestion.loader import load_documents
            docs = load_documents(pdf_dir)
        page = docs[0]["pages"][0]
        assert "page_num" in page
        assert "text" in page

    def test_page_num_is_int(self, tmp_path):
        pdf_dir = _fake_pdf_dir(tmp_path, ["ch01.pdf"])
        mock_doc = _make_fitz_doc(["Content."])
        with patch("fitz.open", return_value=mock_doc):
            from app.ingestion.loader import load_documents
            docs = load_documents(pdf_dir)
        assert isinstance(docs[0]["pages"][0]["page_num"], int)

    def test_page_numbers_are_1_indexed(self, tmp_path):
        pdf_dir = _fake_pdf_dir(tmp_path, ["ch01.pdf"])
        mock_doc = _make_fitz_doc(["Page A.", "Page B.", "Page C."])
        with patch("fitz.open", return_value=mock_doc):
            from app.ingestion.loader import load_documents
            docs = load_documents(pdf_dir)
        page_nums = [p["page_num"] for p in docs[0]["pages"]]
        assert page_nums == [1, 2, 3]

    def test_page_count_matches_non_empty_pages(self, tmp_path):
        pdf_dir = _fake_pdf_dir(tmp_path, ["ch01.pdf"])
        # One empty page should be filtered out
        mock_doc = _make_fitz_doc(["Content.", "   \n  ", "More content."])
        with patch("fitz.open", return_value=mock_doc):
            from app.ingestion.loader import load_documents
            docs = load_documents(pdf_dir)
        assert len(docs[0]["pages"]) == 2  # empty page filtered

    def test_page_text_matches_extracted_content(self, tmp_path):
        pdf_dir = _fake_pdf_dir(tmp_path, ["ch01.pdf"])
        mock_doc = _make_fitz_doc(["Hello world content."])
        with patch("fitz.open", return_value=mock_doc):
            from app.ingestion.loader import load_documents
            docs = load_documents(pdf_dir)
        assert docs[0]["pages"][0]["text"] == "Hello world content."


class TestSourceUrl:
    """source_url must be a properly formed string."""

    def test_source_url_contains_doc_id(self, tmp_path):
        pdf_dir = _fake_pdf_dir(tmp_path, ["chapter_five.pdf"])
        mock_doc = _make_fitz_doc(["Text."])
        with patch("fitz.open", return_value=mock_doc):
            from app.ingestion.loader import load_documents
            docs = load_documents(pdf_dir)
        assert "chapter_five" in docs[0]["source_url"]

    def test_source_url_is_string(self, tmp_path):
        pdf_dir = _fake_pdf_dir(tmp_path, ["ch01.pdf"])
        mock_doc = _make_fitz_doc(["Text."])
        with patch("fitz.open", return_value=mock_doc):
            from app.ingestion.loader import load_documents
            docs = load_documents(pdf_dir)
        assert isinstance(docs[0]["source_url"], str)

    def test_source_url_ends_with_pdf(self, tmp_path):
        pdf_dir = _fake_pdf_dir(tmp_path, ["ch01.pdf"])
        mock_doc = _make_fitz_doc(["Text."])
        with patch("fitz.open", return_value=mock_doc):
            from app.ingestion.loader import load_documents
            docs = load_documents(pdf_dir)
        assert docs[0]["source_url"].endswith(".pdf")


class TestMultipleDocuments:
    """load_documents must handle multiple PDFs and sort them."""

    def test_loads_multiple_pdfs(self, tmp_path):
        pdf_dir = _fake_pdf_dir(tmp_path, ["ch01.pdf", "ch02.pdf", "ch03.pdf"])
        # Each call to fitz.open() needs its own fresh iterator —
        # a single mock doc is exhausted after the first PDF.
        with patch("fitz.open", side_effect=[
            _make_fitz_doc(["Content ch01."]),
            _make_fitz_doc(["Content ch02."]),
            _make_fitz_doc(["Content ch03."]),
        ]):
            from app.ingestion.loader import load_documents
            docs = load_documents(pdf_dir)
        assert len(docs) == 3

    def test_all_docs_have_source_url(self, tmp_path):
        pdf_dir = _fake_pdf_dir(tmp_path, ["ch01.pdf", "ch02.pdf"])
        mock_doc = _make_fitz_doc(["Content."])
        with patch("fitz.open", return_value=mock_doc):
            from app.ingestion.loader import load_documents
            docs = load_documents(pdf_dir)
        assert all("source_url" in d for d in docs)

    def test_all_docs_have_pages(self, tmp_path):
        pdf_dir = _fake_pdf_dir(tmp_path, ["ch01.pdf", "ch02.pdf"])
        mock_doc = _make_fitz_doc(["Content."])
        with patch("fitz.open", return_value=mock_doc):
            from app.ingestion.loader import load_documents
            docs = load_documents(pdf_dir)
        assert all("pages" in d for d in docs)

    def test_doc_with_all_empty_pages_is_excluded(self, tmp_path):
        pdf_dir = _fake_pdf_dir(tmp_path, ["empty.pdf", "real.pdf"])
        empty_doc = _make_fitz_doc(["   ", "\n\n"])
        real_doc = _make_fitz_doc(["Real content."])
        with patch("fitz.open", side_effect=[empty_doc, real_doc]):
            from app.ingestion.loader import load_documents
            docs = load_documents(pdf_dir)
        assert len(docs) == 1
        assert docs[0]["doc_id"] == "real"


class TestEdgeCases:
    """Boundary and error conditions."""

    def test_missing_directory_raises_file_not_found(self, tmp_path):
        from app.ingestion.loader import load_documents
        with pytest.raises(FileNotFoundError):
            load_documents(tmp_path / "nonexistent")

    def test_empty_directory_returns_empty_list(self, tmp_path):
        pdf_dir = tmp_path / "pdfs"
        pdf_dir.mkdir()
        from app.ingestion.loader import load_documents
        assert load_documents(pdf_dir) == []
