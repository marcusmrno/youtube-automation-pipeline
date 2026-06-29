import json
import sys
from pathlib import Path
from unittest.mock import MagicMock

import pytest

if "claude_agent_sdk" not in sys.modules:
    mock_sdk = MagicMock()
    sys.modules["claude_agent_sdk"] = mock_sdk
    sys.modules["claude_agent_sdk.types"] = mock_sdk


def test_load_metadata_missing(tmp_path, monkeypatch):
    import metadata
    monkeypatch.setattr(metadata, "OUTPUT_ROOT", tmp_path)
    (tmp_path / "abc").mkdir()
    assert metadata.load_metadata("abc") is None


def test_load_metadata_run_dir_missing(tmp_path, monkeypatch):
    import metadata
    monkeypatch.setattr(metadata, "OUTPUT_ROOT", tmp_path)
    assert metadata.load_metadata("nope") is None


def test_load_metadata_roundtrip(tmp_path, monkeypatch):
    import metadata
    monkeypatch.setattr(metadata, "OUTPUT_ROOT", tmp_path)
    run_dir = tmp_path / "abc"
    run_dir.mkdir()
    data = {"run_slug": "abc", "titles": [], "chosen_title_index": 0}
    metadata._save_metadata(run_dir, data)
    loaded = metadata.load_metadata("abc")
    assert loaded == data


def test_load_metadata_corrupt_json(tmp_path, monkeypatch):
    import metadata
    monkeypatch.setattr(metadata, "OUTPUT_ROOT", tmp_path)
    run_dir = tmp_path / "abc"
    run_dir.mkdir()
    (run_dir / "metadata.json").write_text("{not json")
    with pytest.raises(ValueError) as exc:
        metadata.load_metadata("abc")
    assert "metadata.json" in str(exc.value)


def test_save_metadata_atomic_no_tmp_left(tmp_path, monkeypatch):
    import metadata
    monkeypatch.setattr(metadata, "OUTPUT_ROOT", tmp_path)
    run_dir = tmp_path / "abc"
    run_dir.mkdir()
    metadata._save_metadata(run_dir, {"x": 1})
    assert (run_dir / "metadata.json").exists()
    assert not (run_dir / "metadata.json.tmp").exists()
