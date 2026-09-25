"""Run orchestration: stop handling, resume, empty model output. Every paid call is faked."""
import threading
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

import images
import pipeline
import voiceover
import writing

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

    monkeypatch.setattr(voiceover, "_tts_chunk", fake_chunk)
    monkeypatch.setattr(pipeline, "generate_all_images", lambda *a, **k: {})
    (tmp_path / "audio").mkdir()
    tts = "Every sentence here is billed. " * 600          # several ElevenLabs chunks
    assert len(voiceover._split_into_chunks(tts)) > 2
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


@pytest.fixture
def resumable(monkeypatch, run_env):
    """A run folder with script.txt; model calls recorded; production captured."""
    calls, produced = [], {}
    reply = SimpleNamespace(content=[SimpleNamespace(text="===TTS_SCRIPT===\nNEW TTS")])
    client = SimpleNamespace(messages=SimpleNamespace(create=lambda **k: calls.append("haiku tts") or reply))
    monkeypatch.setattr(pipeline.anthropic, "Anthropic", lambda **k: client)
    monkeypatch.setattr(writing, "_stream_text",
                        lambda *a, **k: calls.append("sonnet prompts") or "===IMAGE_PROMPTS===\n001 | s | none | scene")
    monkeypatch.setattr(writing, "_build_tts_prompt", lambda *a: "tts prompt")
    monkeypatch.setattr(writing, "_build_image_prompt_instructions", lambda *a: "prompt instructions")
    monkeypatch.setattr(pipeline, "_run_production",
                        lambda topic, prompts, tts, *a, **k: produced.update(tts=tts, prompts=prompts) or {"status": "complete"})
    run = run_env / "r1"
    run.mkdir()
    (run / "script.txt").write_text("TITLE: T\nbody")
    return SimpleNamespace(run=run, calls=calls, produced=produced,
                           profile=SimpleNamespace(**{**vars(PROFILE), "characters": [], "image_style": {"art_style_block": "STYLE"}}))


def test_resume_with_prompts_kept_skips_the_image_prompt_call(resumable):
    (resumable.run / "image_prompts.txt").write_text("001 | s | USER-EDITED PROMPT")
    pipeline.resume_pipeline("r1", resumable.profile)
    assert resumable.calls == ["haiku tts"]              # the 64k Sonnet call is not paid for
    assert (resumable.run / "tts_script.txt").read_text() == "NEW TTS"
    assert (resumable.run / "image_prompts.txt").read_text() == "001 | s | USER-EDITED PROMPT"


def test_resume_voices_the_saved_tts_script(resumable):
    (resumable.run / "tts_script.txt").write_text("SAVED TTS")
    pipeline.resume_pipeline("r1", resumable.profile)
    assert resumable.calls == ["sonnet prompts"]
    assert resumable.produced["tts"] == "SAVED TTS"      # audio must match tts_script.txt on disk


def test_empty_script_aborts_before_approval(monkeypatch, run_env):
    calls = []
    monkeypatch.setattr(pipeline, "VIDIQ_KEY", "")
    monkeypatch.setattr(pipeline, "research_topic", lambda *a: "facts")
    monkeypatch.setattr(pipeline, "generate_script", lambda *a, **k: "")   # reply had no ===SCRIPT===
    r = pipeline.run_pipeline("t", PROFILE, approval_callback=lambda s: calls.append(s) or True)
    assert r["status"] == "error" and calls == []


def test_untagged_model_replies_are_errors(monkeypatch):
    reply = SimpleNamespace(content=[SimpleNamespace(text="Here is the narration, no tag at all.")])
    client = SimpleNamespace(messages=SimpleNamespace(create=lambda **k: reply))
    monkeypatch.setattr(writing, "_build_tts_prompt", lambda *a: "p")
    monkeypatch.setattr(writing, "_build_image_prompt_instructions", lambda *a: "p")
    monkeypatch.setattr(writing, "_stream_text", lambda *a, **k: "001 | s | none | untagged")
    profile = SimpleNamespace(characters=[], image_style={"art_style_block": "S"})
    with pytest.raises(ValueError, match="TTS_SCRIPT"):
        writing._generate_tts("script", profile, client, lambda m: None)
    with pytest.raises(ValueError, match="IMAGE_PROMPTS"):
        writing._generate_image_prompts("script", profile, client, lambda m: None)


def test_resume_regenerates_an_empty_tts_script(resumable):
    (resumable.run / "image_prompts.txt").write_text("001 | s | prompt")
    (resumable.run / "tts_script.txt").write_text("")      # left by an untagged reply
    pipeline.resume_pipeline("r1", resumable.profile)
    assert resumable.calls == ["haiku tts"]
    assert resumable.produced["tts"] == "NEW TTS"


def _gemini_returning(data, monkeypatch):
    part = SimpleNamespace(inline_data=SimpleNamespace(mime_type="image/png", data=data))
    response = SimpleNamespace(candidates=[SimpleNamespace(content=SimpleNamespace(parts=[part]))])
    client = SimpleNamespace(models=SimpleNamespace(generate_content=lambda **k: response))
    monkeypatch.setattr(images, "_get_genai_client", lambda: client)
    monkeypatch.setattr(images.time, "sleep", lambda s: None)


def test_undecodable_image_never_replaces_a_good_one(monkeypatch, tmp_path):
    from PIL import Image
    good = tmp_path / "001.png"
    Image.new("RGB", (4, 4)).save(good)
    before = good.read_bytes()
    _gemini_returning(b"\x89PNG-garbage", monkeypatch)
    ok = images.generate_image_google("p", good, None, lambda m: None, model="m", anchor_parts=[], preamble="")
    assert ok is False
    assert good.read_bytes() == before                  # a failed regen keeps the old image
    fresh = tmp_path / "002.png"
    assert images.generate_image_google("p", fresh, None, lambda m: None, model="m",
                                          anchor_parts=[], preamble="") is False
    assert not fresh.exists()                          # nothing for resume to mistake as done


def test_empty_image_file_counts_as_missing(run_env):
    run = run_env / "r1"
    (run / "images").mkdir(parents=True)
    (run / "image_prompts.txt").write_text("001 | s | a\n002 | s | b")
    (run / "images" / "001.png").write_bytes(b"")      # a crash mid-write
    (run / "images" / "002.png").write_bytes(b"x")
    assert pipeline.run_status("r1")["images_on_disk"] == 1


def test_rerunning_a_topic_keeps_the_existing_run(monkeypatch, run_env):
    old = run_env / "vitamin-d"
    old.mkdir()
    (old / "script.txt").write_text("OLD FINISHED SCRIPT")
    (old / "profile.txt").write_text("other-profile")
    calls = []
    monkeypatch.setattr(pipeline, "VIDIQ_KEY", "")
    monkeypatch.setattr(pipeline, "research_topic", lambda *a: calls.append("research") or "facts")
    r = pipeline.run_pipeline("Vitamin D?", PROFILE, approval_callback=lambda s: False)
    assert r["status"] == "error" and calls == []                  # refused before any paid call
    assert (old / "script.txt").read_text() == "OLD FINISHED SCRIPT"
    assert (old / "profile.txt").read_text() == "other-profile"


def test_a_failed_voiceover_does_not_leave_the_old_one(monkeypatch, tmp_path):
    # a premade-script run overwrites its folder; the old audio belongs to a different script
    (tmp_path / "audio").mkdir()
    (tmp_path / "audio" / "voiceover.mp3").write_bytes(b"OLD VOICEOVER")
    monkeypatch.setattr(pipeline, "generate_voiceover", lambda *a, **k: None)   # ElevenLabs failed
    monkeypatch.setattr(pipeline, "generate_all_images", lambda *a, **k: {})
    r = pipeline._run_production("t", [], "new narration", PROFILE, tmp_path, lambda m: None)
    assert r["audio"] == "failed"
    assert not (tmp_path / "audio" / "voiceover.mp3").exists()      # run_status must not call it done


def test_regenerate_rejects_an_unknown_model_key(monkeypatch, run_env):
    # a typo silently regenerated with the default model
    calls = []
    monkeypatch.setattr(pipeline, "generate_image_google", lambda *a, **k: calls.append(k) or True)
    (run_env / "r1").mkdir()
    (run_env / "r1" / "image_prompts.txt").write_text("001 | s | p")
    profile = SimpleNamespace(image_gen={"default_model": "flash", "pro_model": "pro"}, image_style={})
    monkeypatch.setattr(pipeline, "_load_anchor_parts", lambda p: [])
    assert pipeline.regenerate_images("r1", ["1"], "3-pr0", profile)["status"] == "error"
    assert calls == []
    pipeline.regenerate_images("r1", ["1"], "3-pro", profile)
    assert calls[0]["model"] == "pro"
