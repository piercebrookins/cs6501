from __future__ import annotations

from pathlib import Path

from t5rag.exercise_runner import run_all_exercises


def main() -> None:
    repo_root = Path(__file__).resolve().parent
    run_all_exercises(repo_root)
    print("Done. Outputs written to:", repo_root / "Topic5RAG")


if __name__ == "__main__":
    main()
