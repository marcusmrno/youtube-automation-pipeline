"""
Metadata & Thumbnail Generator — standalone post-run step.

Produces YouTube titles, description, hashtags, and thumbnails for a
completed run. See docs/superpowers/specs/2026-06-29-metadata-thumbnail-generator-design.md.
"""
from __future__ import annotations

import json
import os
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from profile import Profile

from pipeline import OUTPUT_ROOT


def _run_dir(run_slug: str) -> Path:
    return OUTPUT_ROOT / run_slug


def load_metadata(run_slug: str) -> dict | None:
    """Return parsed metadata.json for the run, or None if it doesn't exist.

    Raises ValueError if the file exists but is malformed.
    """
    path = _run_dir(run_slug) / "metadata.json"
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text())
    except json.JSONDecodeError as e:
        raise ValueError(f"metadata.json for run '{run_slug}' is malformed: {e}") from e


def _save_metadata(run_dir: Path, data: dict) -> None:
    """Atomically write metadata.json to the run directory."""
    final = run_dir / "metadata.json"
    tmp = run_dir / "metadata.json.tmp"
    tmp.write_text(json.dumps(data, indent=2, ensure_ascii=False))
    os.replace(tmp, final)
