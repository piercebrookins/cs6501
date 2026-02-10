"""
Document loader — extracts text from PDFs (and plain-text files).
"""

from pathlib import Path
from typing import Any, Dict, List

import fitz  # PyMuPDF


def load_pdf(pdf_path: Path) -> List[Dict[str, Any]]:
    """Extract pages from a single PDF."""
    pages: List[Dict[str, Any]] = []
    doc = fitz.open(pdf_path)
    for page_num, page in enumerate(doc):
        text = page.get_text()
        if text.strip():
            pages.append({
                "text": text,
                "source": pdf_path.name,
                "page": page_num + 1,
                "path": str(pdf_path),
            })
    doc.close()
    return pages


def load_text_file(txt_path: Path) -> List[Dict[str, Any]]:
    """Load a plain-text or markdown file as a single 'page'."""
    text = txt_path.read_text(encoding="utf-8", errors="replace")
    if not text.strip():
        return []
    return [{
        "text": text,
        "source": txt_path.name,
        "page": 1,
        "path": str(txt_path),
    }]


def load_documents(folder_path: str) -> List[Dict[str, Any]]:
    """
    Load all supported documents from *folder_path*.

    Supported extensions: .pdf, .txt, .md
    """
    folder = Path(folder_path)
    if not folder.is_dir():
        raise FileNotFoundError(f"Document folder not found: {folder}")

    loaders = {
        ".pdf": load_pdf,
        ".txt": load_text_file,
        ".md": load_text_file,
    }

    documents: List[Dict[str, Any]] = []
    file_count = 0

    for ext, loader in loaders.items():
        for file_path in sorted(folder.glob(f"*{ext}")):
            try:
                docs = loader(file_path)
                documents.extend(docs)
                file_count += 1
            except Exception as exc:  # noqa: BLE001
                print(f"  \u26a0 Error loading {file_path.name}: {exc}")

    print(f"Loaded {len(documents)} pages from {file_count} files in {folder}")
    return documents
