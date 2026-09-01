"""Locate the ffmpeg/ffprobe binaries to run.

Prefers a copy bundled next to the app (for a packaged .exe build that
ships its own ffmpeg so end users don't have to install anything) and
falls back to whatever's on PATH otherwise.
"""

from __future__ import annotations

import sys
from pathlib import Path


def _bundled_dir() -> Path:
    # PyInstaller --onefile extracts bundled data to a temp dir named in
    # sys._MEIPASS; --onedir (and other frozen builds) put it next to the
    # executable. Falling back to the project root covers running from source.
    meipass = getattr(sys, "_MEIPASS", None)
    if meipass:
        return Path(meipass)
    if getattr(sys, "frozen", False):
        return Path(sys.executable).resolve().parent
    return Path(__file__).resolve().parent.parent


def _resolve(name: str) -> str:
    exe_name = f"{name}.exe" if sys.platform == "win32" else name
    bundled = _bundled_dir() / exe_name
    return str(bundled) if bundled.is_file() else name


def ffmpeg_path() -> str:
    return _resolve("ffmpeg")


def ffprobe_path() -> str:
    return _resolve("ffprobe")
