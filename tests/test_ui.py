import ast
from pathlib import Path

import ui


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
