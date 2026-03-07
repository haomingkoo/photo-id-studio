from __future__ import annotations

from pathlib import Path

APP_ROOT = Path(__file__).resolve().parents[1]
VERSION_FILE = APP_ROOT / "VERSION"


def _load_version(default: str = "0.1.0") -> str:
    try:
        value = VERSION_FILE.read_text(encoding="utf-8").strip()
        return value or default
    except Exception:
        return default


__version__ = _load_version()
