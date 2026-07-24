"""Upload path rules. Dependency-free on purpose so it stays testable without the LLM stack."""
import os
import re
from pathlib import Path

# Absolute default: Streamlit is launched from app/ but the API from the repo root.
DATA_DIR = os.getenv("DATA_DIR") or str(Path(__file__).resolve().parent.parent / "data")

SUPPORTED_EXTS = frozenset({".md", ".pdf", ".docx", ".txt"})


def safe_dest(filename: str, category: str) -> Path:
    """Resolve an upload to DATA_DIR/<category>/<name>. Raises ValueError on untrusted input."""
    name = Path(filename).name          # strips ../ and any directory component
    if not name or name.startswith("."):
        raise ValueError("Invalid filename")
    if Path(name).suffix.lower() not in SUPPORTED_EXTS:
        raise ValueError(f"Unsupported file type (allowed: {', '.join(sorted(SUPPORTED_EXTS))})")
    if not re.fullmatch(r"[A-Za-z0-9_-]{1,40}", category):
        raise ValueError("Invalid category (letters, digits, _ and - only)")
    return Path(DATA_DIR) / category / name
