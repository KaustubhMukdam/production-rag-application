from pathlib import Path
from typing import List


def load_documents(pdf_dir: str | Path = "data/pdfs") -> List[dict]:
    pdf_path = Path(pdf_dir)
    if not pdf_path.exists():
        raise FileNotFoundError(f"PDF directory not found: {pdf_path}")
    documents = []
    for pdf_file in sorted(pdf_path.glob("*.pdf")):
        text = _extract_text(pdf_file)
        if text.strip():
            documents.append({
                "doc_id": pdf_file.stem,
                "text": text,
            })
    return documents


def _extract_text(pdf_path: Path) -> str:
    import fitz
    doc = fitz.open(pdf_path)
    text = ""
    for page in doc:
        text += page.get_text()
    doc.close()
    return text
