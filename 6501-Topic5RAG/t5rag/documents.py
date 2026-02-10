from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import fitz  # PyMuPDF


@dataclass(frozen=True)
class Document:
    filename: str
    content: str


_TEXT_SUFFIXES = {".txt", ".md", ".text"}
_PDF_SUFFIXES = {".pdf"}


def load_text_file(filepath: Path) -> str:
    with filepath.open("r", encoding="utf-8", errors="ignore") as f:
        return f.read()


def load_pdf_file(filepath: Path) -> str:
    """Extract embedded text from a PDF.

    Note: This does NOT do OCR for scanned PDFs.
    """

    doc = fitz.open(str(filepath))
    text_parts: list[str] = []

    for page_num, page in enumerate(doc):
        text = page.get_text()
        if text.strip():
            text_parts.append(f"\n[Page {page_num + 1}]\n{text}")

    doc.close()
    return "\n".join(text_parts)


def iter_document_paths(doc_folder: Path) -> Iterable[Path]:
    for filepath in doc_folder.rglob("*"):
        try:
            if not filepath.is_file():
                continue
        except OSError:
            continue

        suffix = filepath.suffix.lower()
        if suffix in _PDF_SUFFIXES or suffix in _TEXT_SUFFIXES:
            yield filepath


def load_documents(doc_folder: str | Path) -> list[Document]:
    """Load all documents from a folder."""

    folder = Path(doc_folder)
    if not folder.exists():
        raise FileNotFoundError(f"Document folder does not exist: {folder}")

    documents: list[Document] = []
    for filepath in iter_document_paths(folder):
        suffix = filepath.suffix.lower()
        try:
            if suffix in _PDF_SUFFIXES:
                content = load_pdf_file(filepath)
            elif suffix in _TEXT_SUFFIXES:
                content = load_text_file(filepath)
            else:
                continue

            if content.strip():
                documents.append(Document(filename=filepath.name, content=content))
        except Exception as e:  # noqa: BLE001 - notebook-style resiliency
            # Keep going; a single bad PDF shouldn't nuke the whole run.
            print(f"✗ Error loading {filepath}: {e}")

    return documents
