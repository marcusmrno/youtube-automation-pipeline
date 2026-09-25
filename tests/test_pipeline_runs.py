"""Run orchestration: stop handling, resume, empty model output. Every paid call is faked."""
import threading
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

import pipeline

PROFILE = SimpleNamespace(name="p", voice={"voice_id": "v"}, image_gen={}, image_style={})


@pytest.fixture
def run_env(monkeypatch, tmp_path):
    """Keys present, output in tmp_path, Anthropic client faked."""
    monkeypatch.setattr(pipeline, "OUTPUT_ROOT", tmp_path)
    monkeypatch.setattr(pipeline, "check_keys", lambda profile, log_fn: True)
    monkeypatch.setattr(pipeline.anthropic, "Anthropic", MagicMock())
    monkeypatch.setattr(pipeline, "_produce_from_script", lambda *a, **k: {"status": "complete"})
    return tmp_path


def test_stop_halts_the_voiceover(monkeypatch, tmp_path):
    stop, calls = threading.Event(), []

    def fake_chunk(text, *a, **k):
        calls.append(text)
        stop.set()                       # user hits Stop during chunk 1
        return b"ID3" + b"\x00" * 20

    monkeypatch.setattr(pipeline, "_tts_chunk", fake_chunk)
    monkeypatch.setattr(pipeline, "generate_all_images", lambda *a, **k: {})
    (tmp_path / "audio").mkdir()
    tts = "Every sentence here is billed. " * 600          # several ElevenLabs chunks
    assert len(pipeline._split_into_chunks(tts)) > 2
    r = pipeline._run_production("t", [], tts, PROFILE, tmp_path, lambda m: None, stop)
    assert r["status"] == "cancelled"
    assert len(calls) == 1


def test_stop_during_script_agent_skips_vet_and_approval(monkeypatch, run_env):
    stop, calls = threading.Event(), []

    def fake_agent(topic, profile, log_fn, approach_context=""):
        stop.set()                       # Stop pressed while the Opus agent writes
        return "TITLE: T\nbody", "notes"

    monkeypatch.setattr(pipeline, "VIDIQ_KEY", "key")
    monkeypatch.setattr(pipeline, "run_script_agent", fake_agent)
    monkeypatch.setattr(pipeline, "run_vet_agent", lambda *a: calls.append("vet") or "")
    r = pipeline.run_pipeline("t", PROFILE, stop_event=stop,
                              approval_callback=lambda s: calls.append("approval") or True)
    assert r["status"] == "cancelled"
    assert calls == []


def test_stop_during_research_skips_script_writing(monkeypatch, run_env):
    stop, calls = threading.Event(), []
    monkeypatch.setattr(pipeline, "VIDIQ_KEY", "")
    monkeypatch.setattr(pipeline, "research_topic", lambda *a: stop.set() or "facts")
    monkeypatch.setattr(pipeline, "generate_script", lambda *a, **k: calls.append("script") or "TITLE: T")
    r = pipeline.run_pipeline("t", PROFILE, stop_event=stop,
                              approval_callback=lambda s: calls.append("approval") or True)
    assert r["status"] == "cancelled"
    assert calls == []
