"""
Document loader — Phase 4 (page-aware).

Returns a list of documents where each document exposes its pages
individually so the SemanticChunker can respect section boundaries
and attach page numbers to every chunk.

Output schema:
    {
        "doc_id":     str,
        "pages":      list[{"page_num": int, "text": str}],
        "source_url": str,
    }
"""
import logging
from pathlib import Path
from typing import List

from app.config import CORPUS_URL

logger = logging.getLogger(__name__)


def load_documents(pdf_dir: str | Path = "data/pdfs") -> List[dict]:
    """Load all PDFs from *pdf_dir* and return a page-aware document list.

    Each entry contains the raw per-page text so downstream chunkers can
    split on section boundaries and attach accurate page numbers.

    Args:
        pdf_dir: Path to the directory containing `.pdf` files.

    Returns:
        List of document dicts.  Empty list if the directory has no PDFs.

    Raises:
        FileNotFoundError: If *pdf_dir* does not exist.
    """
    pdf_path = Path(pdf_dir)
    if not pdf_path.exists():
        raise FileNotFoundError(f"PDF directory not found: {pdf_path}")

    documents: List[dict] = []
    for pdf_file in sorted(pdf_path.glob("*.pdf")):
        pages = _extract_pages(pdf_file)
        if not pages:
            logger.warning("Skipping %s — no extractable text found.", pdf_file.name)
            continue
        documents.append({
            "doc_id": pdf_file.stem,
            "pages": pages,
            "source_url": f"{CORPUS_URL.rstrip('/')}/{pdf_file.name}",
        })

    logger.info("Loaded %d document(s) from %s.", len(documents), pdf_path)
    return documents


def _extract_pages(pdf_path: Path) -> List[dict]:
    """Extract per-page text from a PDF using PyMuPDF.

    Pages whose text is empty or whitespace-only are filtered out.

    Args:
        pdf_path: Absolute path to the PDF file.

    Returns:
        List of ``{"page_num": int, "text": str}`` dicts (1-indexed).
    """
    import fitz  # PyMuPDF — imported lazily so the module loads in test envs

    doc = fitz.open(pdf_path)
    pages: List[dict] = []
    try:
        for page_num, page in enumerate(doc, start=1):
            text = page.get_text()
            if text.strip():
                pages.append({"page_num": page_num, "text": text})
    finally:
        doc.close()
    return pages
