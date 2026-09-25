import ast
import threading
from pathlib import Path

import pytest

import ui


@pytest.fixture(autouse=True)
def fresh_state(monkeypatch):
    # ui keeps one module-level job slot; give every test its own
    monkeypatch.setattr(ui, "_state", {"log_queue": None, "approval_queue": None, "stop_event": None})


@pytest.fixture
def run_dir(monkeypatch, tmp_path):
    """output/r1 with a script, TTS and profile.txt; profiles resolve to a stand-in."""
    monkeypatch.setattr(ui, "OUTPUT_ROOT", tmp_path)
    monkeypatch.setattr(ui, "load_profile", lambda name: f"<profile {name}>")
    monkeypatch.setattr(ui, "_load_ui_profile", lambda name: ("<profile>", None))
    run = tmp_path / "r1"
    run.mkdir()
    for f, text in [("script.txt", "TITLE: T"), ("tts_script.txt", "Hello."), ("profile.txt", "p")]:
        (run / f).write_text(text)
    return run


def test_ui_binds_localhost_only():
    tree = ast.parse(Path(ui.__file__).read_text())
    run = next(n for n in ast.walk(tree) if isinstance(n, ast.Call)
               and getattr(n.func, "attr", "") == "run" and getattr(n.func.value, "id", "") == "app")
    host = next(k.value.value for k in run.keywords if k.arg == "host")
    assert host == "127.0.0.1"


def _post_routes():
    return sorted({r.rule.replace("<path:run_slug>", "r1").replace("<run_name>", "r1")
                   for r in ui.app.url_map.iter_rules() if "POST" in r.methods})


def test_post_routes_refuse_non_json_bodies(monkeypatch):
    # a text/plain POST is a CORS "simple request": any web page could send it
    started = []
    for name in ("run_pipeline", "resume_pipeline", "run_from_script", "regenerate_images",
                 "revise_script", "generate_clarifying_questions", "generate_approach_pitches"):
        monkeypatch.setattr(ui, name, lambda *a, _n=name, **k: started.append(_n))
    monkeypatch.setattr(ui.threading, "Thread", lambda target, **k: type("T", (), {"start": lambda s: started.append("thread")})())
    client = ui.app.test_client()
    for rule in _post_routes():
        r = client.post(rule, data='{"topic": "x", "run_slug": "r1", "script": "s", "text": ""}',
                        content_type="text/plain")
        assert r.status_code == 415, (rule, r.status_code)
    assert started == []
    assert client.post("/stop", json={}).get_json() == {"ok": True}   # the page's own calls still work


def test_images_lists_each_number_once(monkeypatch, tmp_path):
    # a regen can leave 001.png next to 001.jpg; the gallery must not get two "001" cards
    img = tmp_path / "r1" / "images"
    img.mkdir(parents=True)
    for f in ("001.png", "001.jpg", "002.png"):
        (img / f).write_bytes(b"x")
    monkeypatch.setattr(ui, "OUTPUT_ROOT", tmp_path)
    assert ui.app.test_client().get("/images/r1").get_json() == ["001", "002"]


def test_pick_thumbnail_rejects_bool_index(monkeypatch):
    picked = []
    monkeypatch.setattr(ui._metadata_mod, "pick_thumbnail", lambda slug, i: picked.append(i) or {})
    r = ui.app.test_client().post("/metadata/r1/pick_thumbnail", json={"index": True})
    assert r.status_code == 400 and picked == []   # True would be saved as the chosen index


def test_stage_tracker_never_moves_back_from_images():
    # the real log lines between approval and the first image, in order
    lines = ["✅  Script approved — generating TTS and image prompts...",
             "🖼️  Generating image prompts...",
             "✅  Image prompts generated — 150 total",
             "📝  150 image prompts parsed",
             "🖼  Generating image 1/150 (001)"]
    stages = [s for s in map(ui.detect_stage, lines) if s]
    assert stages == sorted(stages, key=["research", "script", "images"].index), stages


def test_a_second_job_is_refused_while_a_run_waits_for_approval(monkeypatch, run_dir):
    answers, regens = [], []

    def fake_run(topic, profile, progress_callback, stop_event, approval_callback, approach_context):
        answers.append(approval_callback("SCRIPT"))          # blocks until /approve
        return {"status": "complete"}

    monkeypatch.setattr(ui, "run_pipeline", fake_run)
    monkeypatch.setattr(ui, "regenerate_images", lambda *a, **k: regens.append(a))
    client = ui.app.test_client()
    assert client.post("/run", json={"topic": "t"}).get_json() == {"ok": True}
    job = ui._state["thread"]

    r = client.post("/regenerate", json={"run_slug": "r1", "image_nums": ["001"]}).get_json()
    assert r["ok"] is False and "running" in r["error"]
    assert client.post("/approve", json={"script": "", "run_slug": ""}).get_json() == {"ok": True}
    job.join(2)
    assert answers == [True] and regens == []


def test_failed_audio_regen_leaves_the_live_job_alone(monkeypatch, run_dir):
    lq = ui._state["log_queue"] = object()
    (run_dir / "profile.txt").write_text("deleted-profile")
    monkeypatch.setattr(ui, "load_profile", lambda name: (_ for _ in ()).throw(ValueError("gone")))
    ui.app.test_client().post("/regenerate_audio", json={"run_slug": "r1"})
    assert ui._state["log_queue"] is lq


def test_metadata_generate_reports_a_busy_ui(run_dir):
    ui._state["thread"] = threading.Thread(target=threading.Event().wait, args=(1,), daemon=True)
    ui._state["thread"].start()
    r = ui.app.test_client().post("/metadata/r1/generate", json={"profile": "p"})
    assert r.status_code == 423 and "running" in r.get_json()["error"]
