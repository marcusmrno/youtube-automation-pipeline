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


def test_malformed_metadata_can_be_regenerated(tmp_path, monkeypatch, two_profiles):
    run = tmp_path / "output" / "r1"
    run.mkdir(parents=True)
    (run / "profile.txt").write_text("beta")
    (run / "metadata.json").write_text('{"titles": [')                      # truncated write
    generated = []
    monkeypatch.setattr(metadata, "OUTPUT_ROOT", tmp_path / "output")
    monkeypatch.setattr(metadata, "generate_metadata", lambda slug, *a, **k: generated.append(slug))
    assert run_cli(tmp_path, monkeypatch, "metadata", "r1", "--regenerate") == 0
    assert generated == ["r1"]
    assert run_cli(tmp_path, monkeypatch, "metadata", "r1") not in (0, None)   # a message, not a traceback


def test_preview_never_overwrites_a_runs_prompts(tmp_path, monkeypatch):
    # its own docstring example (output/my-run/script.txt) replaced the prompts that made the images
    import preview_prompts
    run = tmp_path / "my-run"
    run.mkdir()
    (run / "script.txt").write_text("TITLE: T\nbody")
    (run / "image_prompts.txt").write_text("001 | s | ORIGINAL prompt")
    monkeypatch.setattr(preview_prompts, "load_profile", lambda name: f"<profile {name}>")
    monkeypatch.setattr(preview_prompts, "_generate_tts_and_prompts", lambda *a: ("NEW TTS", "001 | s | NEW"))
    monkeypatch.setattr(sys, "argv", ["preview_prompts.py", str(run / "script.txt"), "--profile", "p"])
    with pytest.raises(SystemExit) as exc:
        preview_prompts.main()
    assert exc.value.code not in (0, None)
    assert (run / "image_prompts.txt").read_text() == "001 | s | ORIGINAL prompt"


@pytest.fixture
def unrunnable_profile(monkeypatch):
    """One profile whose missing voice_id makes check_keys refuse before any paid call."""
    from types import SimpleNamespace
    monkeypatch.setattr(profile, "list_profiles", lambda: ["p"])
    monkeypatch.setattr(profile, "load_profile", lambda name: SimpleNamespace(name="p", voice={"voice_id": ""}))


def test_failed_runs_exit_nonzero(tmp_path, monkeypatch, unrunnable_profile):
    # shell scripts couldn't tell a failed run from a finished one: both exited 0
    script = tmp_path / "s.txt"
    script.write_text("TITLE: T\nbody")
    assert run_cli(tmp_path, monkeypatch, "script", str(script)) == 1
    assert run_cli(tmp_path, monkeypatch, "run", "some topic") == 1
