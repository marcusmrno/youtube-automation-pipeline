"""`python pipeline.py ...`, run in-process from a copy so output/ is a temp dir. Paid calls are faked."""
import runpy
import shutil
import sys
from pathlib import Path

import pytest

import metadata
import pipeline
import profile

ROOT = Path(pipeline.__file__).parent


def run_cli(tmp_path, monkeypatch, *argv):
    """Run the CLI with argv; return its exit code (None if it returned without sys.exit)."""
    shutil.copy(ROOT / "pipeline.py", tmp_path / "pipeline.py")   # its output/ is tmp_path/output
    monkeypatch.setattr(sys, "argv", ["pipeline.py", *argv])
    monkeypatch.setenv("ANTHROPIC_API_KEY", "test")
    monkeypatch.setenv("GOOGLE_API_KEY", "test")
    monkeypatch.setenv("ELEVENLABS_API_KEY", "test")
    try:
        runpy.run_path(str(tmp_path / "pipeline.py"), run_name="__main__")
    except SystemExit as e:
        return e.code
    return None


@pytest.fixture
def two_profiles(monkeypatch):
    monkeypatch.setattr(profile, "list_profiles", lambda: ["alpha", "beta"])
    monkeypatch.setattr(profile, "load_profile", lambda name: f"<profile {name}>")


def test_metadata_uses_the_profile_the_run_was_made_with(tmp_path, monkeypatch, two_profiles):
    run = tmp_path / "output" / "r1"
    run.mkdir(parents=True)
    (run / "profile.txt").write_text("beta")
    used = []
    monkeypatch.setattr(metadata, "load_metadata", lambda slug: None)
    monkeypatch.setattr(metadata, "generate_metadata", lambda slug, prof, log_fn, regenerate: used.append(prof))
    assert run_cli(tmp_path, monkeypatch, "metadata", "r1") == 0
    assert used == ["<profile beta>"]
