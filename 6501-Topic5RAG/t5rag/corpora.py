from __future__ import annotations

import zipfile
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class CorpusPaths:
    root: Path

    @property
    def model_t(self) -> Path:
        return self.root / "ModelTService" / "pdf_embedded"

    @property
    def congressional_record(self) -> Path:
        return self.root / "CongressionalRecord" / "pdf_embedded"

    @property
    def learjet(self) -> Path:
        return self.root / "Learjet" / "pdf_embedded"

    @property
    def eu_ai_act(self) -> Path:
        return self.root / "EUAIAct" / "pdf_embedded"


def ensure_corpora_unzipped(repo_root: Path) -> Path:
    """Ensure `Corpora/` exists, unzipping `Corpora.zip` if available."""

    corpora_dir = repo_root / "Corpora"
    if corpora_dir.exists():
        return corpora_dir

    zip_path = repo_root / "Corpora.zip"
    if not zip_path.exists():
        raise FileNotFoundError(
            "Missing corpora. Expected either `Corpora/` directory or `Corpora.zip` at repo root."
        )

    with zipfile.ZipFile(zip_path, "r") as z:
        z.extractall(repo_root)

    if not corpora_dir.exists():
        raise RuntimeError("Unzipped Corpora.zip but `Corpora/` directory still not found.")

    return corpora_dir


def get_corpus_paths(repo_root: str | Path) -> CorpusPaths:
    repo_root = Path(repo_root)
    corpora_root = ensure_corpora_unzipped(repo_root)
    return CorpusPaths(root=corpora_root)


def validate_pdf_embedded_dir(path: Path) -> None:
    if not path.exists() or not path.is_dir():
        raise FileNotFoundError(f"Expected pdf_embedded directory not found: {path}")

    # Basic sanity: should contain at least one PDF.
    has_pdf = any(p.suffix.lower() == ".pdf" for p in path.rglob("*.pdf"))
    if not has_pdf:
        raise FileNotFoundError(f"No PDFs found under: {path}")
