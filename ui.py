"""
YouTube Pipeline — Flask UI
Run: python ui.py
Opens at http://localhost:7860 (bound to 127.0.0.1 only)
"""

import anthropic
import json
import queue
import re
import threading

from flask import Flask, Response, jsonify, render_template, request, send_file
import pipeline
from pipeline import (OUTPUT_ROOT, run_pipeline, resume_pipeline, run_from_script,
                      regenerate_images, parse_image_prompts, revise_script,
                      run_status, ANTHROPIC_KEY, VIDIQ_KEY,
                      generate_clarifying_questions, generate_approach_pitches)
from profile import load_profile, list_profiles
from agents import run_vet_agent
import metadata as _metadata_mod

_noop_log = lambda _: None

_SLUG_RE = re.compile(r'^[a-z0-9][a-z0-9\-]*$')

def _safe_slug(run_slug: str) -> bool:
    return bool(_SLUG_RE.fullmatch(run_slug))


def _resolve_run_profile_name(run_slug: str, requested: str) -> str | None:
    """Requested profile name, else the run's saved profile.txt, else the first available."""
    if requested:
        return requested
    saved = OUTPUT_ROOT / run_slug / "profile.txt"
    if saved.exists():
        return saved.read_text().strip()
    available = list_profiles()
    return available[0] if available else None

app = Flask(__name__)


@app.before_request
def _require_json_posts():
    # A text/plain POST needs no CORS preflight, so any web page could start paid jobs here.
    if request.method == "POST" and not request.is_json:
        return jsonify({"ok": False, "error": "JSON body required"}), 415

_state: dict = {
    "log_queue":      None,
    "approval_queue": None,
    "stop_event":     None,
}

STAGE_KEYWORDS: dict[str, list[str]] = {
    "research": ["researching", "research complete", "starting pipeline",
                 "output directory", "style anchor", "api keys", "missing api"],
    "script":   ["writing script", "script written", "image prompts parsed",
                 "script ready", "📝", "vidiq script vet", "script vetted", "vet agent", "🔎"],
    "images":   ["generating image", "images generated", "sending request"],
    "voice":    ["generating voiceover", "voiceover generated", "voiceover failed", "🎙"],
    "done":     ["pipeline complete", "pipeline finished", "🎬", "╔", "╚"],
}


def detect_stage(msg: str) -> str | None:
    lower = msg.lower()
    for stage, keywords in STAGE_KEYWORDS.items():
        for kw in keywords:
            if kw.lower() in lower:
                return stage
    return None


def _load_ui_profile(profile_name: str):
    """(profile, error) — falls back to the only profile when exactly one exists."""
    available = list_profiles()
    if not available:
        return None, "No profiles found. Create profiles/<name>/profile.yaml first."
    if not profile_name:
        if len(available) > 1:
            return None, f"Select a profile. Available: {available}"
        profile_name = available[0]
    try:
        return load_profile(profile_name), None
    except ValueError as e:
        return None, str(e)


def _start_run(call):
    """Wire up the log queue + stop event, then run `call(progress_cb, stop_event)` on a thread."""
    lq = queue.Queue()
    se = threading.Event()

    _state["log_queue"]      = lq
    _state["stop_event"]     = se
    _state["approval_queue"] = None

    def progress_cb(msg: str):
        lq.put({"type": "log", "stage": detect_stage(msg), "msg": msg})

    def worker():
        try:
            call(progress_cb, se)
        except Exception as e:
            lq.put({"type": "log", "stage": None, "msg": f"❌  Error: {e}"})
        finally:
            lq.put({"type": "done"})

    threading.Thread(target=worker, daemon=True).start()
    return jsonify({"ok": True})


# ── Routes ────────────────────────────────────────────────────────────────────

@app.route("/profiles")
def get_profiles():
    profiles = []
    for name in list_profiles():
        try:
            p = load_profile(name)
            profiles.append({"name": name, "display": f"{p.channel['name']} — {p.channel['niche']}"})
        except Exception:
            profiles.append({"name": name, "display": name})
    return jsonify(profiles)


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/run", methods=["POST"])
def run():
    data               = request.get_json(force=True)
    topic              = (data.get("topic") or "").strip()
    profile_name       = (data.get("profile") or "").strip()
    approach_context   = (data.get("approach_context") or "").strip()

    if not topic:
        return jsonify({"ok": False, "error": "No topic provided"})

    profile, err = _load_ui_profile(profile_name)
    if err:
        return jsonify({"ok": False, "error": err})

    lq = queue.Queue()
    aq = queue.Queue()
    se = threading.Event()

    _state["log_queue"]      = lq
    _state["approval_queue"] = aq
    _state["stop_event"]     = se

    def progress_cb(msg: str):
        lq.put({"type": "log", "stage": detect_stage(msg), "msg": msg})

    def approval_cb(script_text: str) -> bool:
        lq.put({"type": "review", "script": script_text, "run_slug": pipeline.slugify(topic)})
        return aq.get()

    def worker():
        try:
            run_pipeline(topic, profile, progress_callback=progress_cb,
                         stop_event=se, approval_callback=approval_cb,
                         approach_context=approach_context)
        except Exception as e:
            lq.put({"type": "log", "stage": None, "msg": f"❌  Error: {e}"})
        finally:
            lq.put({"type": "done"})

    threading.Thread(target=worker, daemon=True).start()
    return jsonify({"ok": True})


@app.route("/stream")
def stream():
    def event_stream():
        lq = _state.get("log_queue")
        if not lq:
            yield f"data: {json.dumps({'type': 'error', 'msg': 'No pipeline running'})}\n\n"
            return
        while True:
            try:
                event = lq.get(timeout=1.0)
            except queue.Empty:
                yield f"data: {json.dumps({'type': 'ping'})}\n\n"
                continue
            yield f"data: {json.dumps(event)}\n\n"
            if event.get("type") in ("done", "regen_done"):
                break

    return Response(
        event_stream(),
        content_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )


@app.route("/approve", methods=["POST"])
def approve():
    data        = request.get_json(force=True)
    script_text = data.get("script", "")
    run_slug    = (data.get("run_slug") or "").strip()
    aq = _state.get("approval_queue")
    if aq:
        # Persist any edits the user made in the review panel
        if run_slug and _safe_slug(run_slug) and script_text.strip():
            (OUTPUT_ROOT / run_slug / "script.txt").write_text(script_text)
        aq.put(True)
        return jsonify({"ok": True})
    return jsonify({"ok": False, "error": "No pipeline waiting for approval"})


@app.route("/reject", methods=["POST"])
def reject():
    aq = _state.get("approval_queue")
    if aq:
        aq.put(False)
    se = _state.get("stop_event")
    if se:
        se.set()
    return jsonify({"ok": True})


@app.route("/resume", methods=["POST"])
def resume():
    data         = request.get_json(force=True)
    run_slug     = (data.get("run_slug") or "").strip()
    profile_name = (data.get("profile") or "").strip()

    if not run_slug:
        return jsonify({"ok": False, "error": "No run_slug provided"})
    if not _safe_slug(run_slug):
        return jsonify({"ok": False, "error": "invalid run_slug"})

    profile = None
    if profile_name:
        try:
            profile = load_profile(profile_name)
        except ValueError as e:
            return jsonify({"ok": False, "error": str(e)})

    return _start_run(lambda cb, se: resume_pipeline(
        run_slug, profile, progress_callback=cb, stop_event=se))


@app.route("/run_from_script", methods=["POST"])
def run_from_script_route():
    """Produce assets from a script the user already wrote — no research, writing, or approval."""
    data   = request.get_json(force=True)
    script = (data.get("script") or "").strip()
    topic  = (data.get("topic") or "").strip()

    if not script:
        return jsonify({"ok": False, "error": "No script provided"})

    profile, err = _load_ui_profile((data.get("profile") or "").strip())
    if err:
        return jsonify({"ok": False, "error": err})

    return _start_run(lambda cb, se: run_from_script(
        script, profile, topic, progress_callback=cb, stop_event=se))




@app.route("/stop", methods=["POST"])
def stop():
    se = _state.get("stop_event")
    if se:
        se.set()
    aq = _state.get("approval_queue")
    if aq:
        try:
            aq.put_nowait(False)
        except Exception:
            pass
    return jsonify({"ok": True})


# ── Data endpoints ────────────────────────────────────────────────────────────

def _list_runs(required_file: str | None = None):
    """Run slugs, newest name first, optionally filtered to those holding `required_file`."""
    if not OUTPUT_ROOT.exists():
        return jsonify([])
    return jsonify(sorted(
        (d.name for d in OUTPUT_ROOT.iterdir()
         if d.is_dir() and (required_file is None or (d / required_file).exists())),
        reverse=True,
    ))


@app.route("/runs")
def runs():
    return _list_runs()


@app.route("/runs_with_prompts")
def runs_with_prompts():
    return _list_runs("image_prompts.txt")


@app.route("/run_status/<path:run_slug>")
def run_status_route(run_slug):
    if not _safe_slug(run_slug):
        return jsonify({"error": "invalid run slug"}), 400
    return jsonify(run_status(run_slug))


@app.route("/metadata/<path:run_slug>", methods=["GET"])
def metadata_get(run_slug):
    if not _safe_slug(run_slug):
        return jsonify({"error": "invalid run slug"}), 400
    try:
        data = _metadata_mod.load_metadata(run_slug)
    except ValueError as e:
        return jsonify({"error": str(e)}), 400
    if data is None:
        return jsonify({"error": "not generated yet"}), 404
    return jsonify(data)


@app.route("/metadata/<path:run_slug>/generate", methods=["POST"])
def metadata_generate(run_slug):
    if not _safe_slug(run_slug):
        return jsonify({"error": "invalid run slug"}), 400
    body = request.get_json(force=True, silent=True) or {}
    regenerate = bool(body.get("regenerate", False))
    profile_name = (body.get("profile") or "").strip()
    if not profile_name:
        available = list_profiles()
        if len(available) == 1:
            profile_name = available[0]
        else:
            return jsonify({"error": "specify 'profile' in body"}), 400
    try:
        profile = load_profile(profile_name)
    except Exception as e:
        return jsonify({"error": str(e)}), 400

    if not regenerate:
        try:
            existing = _metadata_mod.load_metadata(run_slug)
        except ValueError as e:
            return jsonify({"error": str(e)}), 400
        if existing is not None:
            return jsonify({"error": "metadata already exists; pass regenerate=true to overwrite"}), 409

    # Stream progress on the same /stream SSE endpoint the main pipeline uses.
    # Structured events so the existing client log routing handles them.
    lq = queue.Queue()
    _state["log_queue"] = lq
    _state["approval_queue"] = None
    _state["stop_event"] = threading.Event()

    def runner():
        try:
            _metadata_mod.generate_metadata(
                run_slug, profile,
                log_fn=lambda m: lq.put({"type": "log", "stage": "metadata", "msg": m}),
                regenerate=regenerate,
            )
            lq.put({"type": "log", "stage": "metadata", "msg": "✅  Metadata generation complete"})
            lq.put({"type": "metadata_done", "run_slug": run_slug})
        except Exception as e:
            lq.put({"type": "log", "stage": "metadata", "msg": f"❌  Metadata generation failed: {e}"})
        finally:
            lq.put({"type": "done"})

    threading.Thread(target=runner, daemon=True).start()
    return jsonify({"status": "started"}), 202


@app.route("/metadata/<path:run_slug>/pick_thumbnail", methods=["POST"])
def metadata_pick_thumb(run_slug):
    if not _safe_slug(run_slug):
        return jsonify({"error": "invalid run slug"}), 400
    body = request.get_json(force=True, silent=True) or {}
    index = body.get("index")
    if not isinstance(index, int):
        return jsonify({"error": "body must include integer 'index'"}), 400
    try:
        return jsonify(_metadata_mod.pick_thumbnail(run_slug, index))
    except ValueError as e:
        return jsonify({"error": str(e)}), 400


@app.route("/metadata/<path:run_slug>/thumbnail/<int:index>", methods=["GET"])
def metadata_thumbnail_file(run_slug, index):
    if not _safe_slug(run_slug):
        return jsonify({"error": "invalid run slug"}), 400
    try:
        data = _metadata_mod.load_metadata(run_slug)
    except ValueError as e:
        return jsonify({"error": str(e)}), 400
    if data is None:
        return jsonify({"error": "not generated yet"}), 404
    try:
        slot = data["thumbnails"][index]
    except (KeyError, IndexError):
        return jsonify({"error": "index out of range"}), 404
    if "render_error" in slot:
        return jsonify({"error": "render failed for this slot"}), 404
    path = OUTPUT_ROOT / run_slug / slot["filename"]
    if not path.exists():
        return jsonify({"error": "file missing"}), 404
    mime = "image/jpeg" if path.suffix.lower() in (".jpg", ".jpeg") else "image/png"
    return send_file(path, mimetype=mime)


@app.route("/runs_with_scripts")
def runs_with_scripts():
    return _list_runs("script.txt")


@app.route("/prompts/<run_name>")
def get_prompts(run_name: str):
    if not _safe_slug(run_name):
        return jsonify({"error": "invalid run slug"}), 400
    path = OUTPUT_ROOT / run_name / "image_prompts.txt"
    if not path.exists():
        return jsonify({})
    return jsonify({
        p["num"]: {"source": p["source"], "prompt": p["prompt"]}
        for p in parse_image_prompts(path.read_text())
    })


@app.route("/runs_with_audio")
def runs_with_audio():
    return _list_runs("audio/voiceover.mp3")


@app.route("/audio/<run_name>")
def serve_audio(run_name: str):
    if not _safe_slug(run_name):
        return "", 400
    audio_path = OUTPUT_ROOT / run_name / "audio" / "voiceover.mp3"
    if not audio_path.exists():
        return "", 404
    return send_file(audio_path, mimetype="audio/mpeg")


@app.route("/run_files/<run_name>")
def run_files(run_name: str):
    if not _safe_slug(run_name):
        return jsonify({"error": "invalid run slug"}), 400
    run_dir = OUTPUT_ROOT / run_name
    if not run_dir.exists():
        return jsonify([])
    files = [str(p.relative_to(OUTPUT_ROOT))
             for p in sorted(run_dir.rglob("*")) if p.is_file()]
    return jsonify(files)


@app.route("/script/<run_name>")
def get_script(run_name: str):
    if not _safe_slug(run_name):
        return jsonify({"ok": False, "error": "invalid run slug"}), 400
    path = OUTPUT_ROOT / run_name / "script.txt"
    if not path.exists():
        return jsonify({"ok": False, "error": "script.txt not found"})
    return jsonify({"ok": True, "text": path.read_text()})


@app.route("/save_script/<run_name>", methods=["POST"])
def save_script(run_name: str):
    if not _safe_slug(run_name):
        return jsonify({"ok": False, "error": "invalid run slug"}), 400
    data = request.get_json(force=True)
    text = data.get("text", "")
    path = OUTPUT_ROOT / run_name / "script.txt"
    if not path.parent.exists():
        return jsonify({"ok": False, "error": "Run folder not found"})
    path.write_text(text)
    return jsonify({"ok": True})


# ── Script planning endpoints ─────────────────────────────────────────────────

@app.route("/clarifying_questions", methods=["POST"])
def get_clarifying_questions():
    data = request.get_json(force=True)
    topic = (data.get("topic") or "").strip()
    profile_name = (data.get("profile_name") or "").strip() or list_profiles()[0]

    if not topic:
        return jsonify({"ok": False, "error": "Topic required"})

    profile = load_profile(profile_name)
    client  = anthropic.Anthropic(api_key=ANTHROPIC_KEY)

    try:
        questions = generate_clarifying_questions(topic, profile, client, _noop_log)
        return jsonify({"ok": True, "questions": questions})
    except Exception as e:
        return jsonify({"ok": False, "error": str(e)})


@app.route("/approach_pitches", methods=["POST"])
def get_approach_pitches():
    data = request.get_json(force=True)
    topic = (data.get("topic") or "").strip()
    answers = (data.get("answers") or "").strip()
    profile_name = (data.get("profile_name") or "").strip() or list_profiles()[0]

    if not topic or not answers:
        return jsonify({"ok": False, "error": "Topic and answers required"})

    profile = load_profile(profile_name)
    client  = anthropic.Anthropic(api_key=ANTHROPIC_KEY)

    try:
        pitches = generate_approach_pitches(topic, answers, profile, client, _noop_log)
        return jsonify({"ok": True, "pitches": pitches})
    except Exception as e:
        return jsonify({"ok": False, "error": str(e)})


# ── Audio endpoints ───────────────────────────────────────────────────────────

@app.route("/regenerate_audio", methods=["POST"])
def regenerate_audio():
    from pipeline import generate_voiceover

    data     = request.get_json(force=True)
    run_slug = (data.get("run_slug") or "").strip()
    if not run_slug:
        return jsonify({"ok": False, "error": "No run_slug provided"})
    if not _safe_slug(run_slug):
        return jsonify({"ok": False, "error": "invalid run_slug"})

    run_dir  = OUTPUT_ROOT / run_slug
    tts_path = run_dir / "tts_script.txt"
    if not tts_path.exists():
        return jsonify({"ok": False, "error": "tts_script.txt not found for this run"})

    lq = queue.Queue()
    se = threading.Event()
    _state["log_queue"]      = lq
    _state["stop_event"]     = se
    _state["approval_queue"] = None

    def progress_cb(msg: str):
        lq.put({"type": "log", "stage": "voice", "msg": msg})

    profile_name = _resolve_run_profile_name(run_slug, (data.get("profile_name") or "").strip())
    if not profile_name:
        return jsonify({"ok": False, "error": "No profiles found."})
    profile = load_profile(profile_name)

    def worker():
        try:
            generate_voiceover(tts_path.read_text(), run_dir, profile, progress_cb)
        except Exception as e:
            lq.put({"type": "log", "stage": "voice", "msg": f"❌  Error: {e}"})
        finally:
            lq.put({"type": "audio_done", "run_slug": run_slug})

    threading.Thread(target=worker, daemon=True).start()
    return jsonify({"ok": True})


# ── Script revision ───────────────────────────────────────────────────────────

@app.route("/revise", methods=["POST"])
def revise():
    data     = request.get_json(force=True)
    script   = (data.get("script") or "").strip()
    feedback = (data.get("feedback") or "").strip()
    topic    = (data.get("topic") or "").strip()
    profile_name = (data.get("profile") or "").strip()
    if not script or not feedback:
        return jsonify({"ok": False, "error": "Missing script or feedback"})

    available = list_profiles()
    if not profile_name:
        profile_name = available[0] if available else None
    profile = load_profile(profile_name) if profile_name else None

    client  = anthropic.Anthropic(api_key=ANTHROPIC_KEY)
    revised = revise_script(script, feedback, topic, profile, client)

    if VIDIQ_KEY and revised and profile:
        vetted = run_vet_agent(topic or "video", revised, profile, _noop_log)
        if vetted:
            revised = vetted

    return jsonify({"ok": True, "script": revised})


# ── Image endpoints ───────────────────────────────────────────────────────────

@app.route("/images/<run_name>")
def list_images(run_name: str):
    if not _safe_slug(run_name):
        return jsonify({"error": "invalid run slug"}), 400
    img_dir = OUTPUT_ROOT / run_name / "images"
    if not img_dir.exists():
        return jsonify([])
    imgs = sorted(
        p.stem for p in img_dir.iterdir()
        if p.suffix.lower() in (".png", ".jpg", ".jpeg")
    )
    return jsonify(imgs)


@app.route("/image/<run_name>/<num>")
def serve_image(run_name: str, num: str):
    if not _safe_slug(run_name) or not num.isdigit():
        return "", 400
    img_dir = OUTPUT_ROOT / run_name / "images"
    for ext, mime in ((".png", "image/png"), (".jpg", "image/jpeg"), (".jpeg", "image/jpeg")):
        img_path = img_dir / f"{num}{ext}"
        if img_path.exists():
            return send_file(img_path, mimetype=mime)
    return "", 404


@app.route("/regenerate", methods=["POST"])
def regen():
    data         = request.get_json(force=True)
    run_slug     = (data.get("run_slug") or "").strip()
    image_nums   = data.get("image_nums") or []
    model_key    = (data.get("model_key") or "nano-banana-2").strip()
    profile_name = (data.get("profile") or "").strip()

    if not run_slug or not image_nums:
        return jsonify({"ok": False, "error": "Missing run_slug or image_nums"})
    if not _safe_slug(run_slug):
        return jsonify({"ok": False, "error": "invalid run_slug"})

    profile_name = _resolve_run_profile_name(run_slug, profile_name)
    if not profile_name:
        return jsonify({"ok": False, "error": "No profiles found."})

    try:
        profile = load_profile(profile_name)
    except ValueError as e:
        return jsonify({"ok": False, "error": str(e)})

    lq = queue.Queue()
    se = threading.Event()
    _state["log_queue"]      = lq
    _state["stop_event"]     = se
    _state["approval_queue"] = None

    def progress_cb(msg: str):
        lq.put({"type": "log", "stage": "images", "msg": msg})

    def worker():
        try:
            regenerate_images(run_slug, image_nums, model_key, profile, progress_callback=progress_cb)
        except Exception as e:
            lq.put({"type": "log", "stage": "images", "msg": f"❌  Error: {e}"})
        finally:
            lq.put({"type": "regen_done", "run_slug": run_slug, "nums": image_nums})

    threading.Thread(target=worker, daemon=True).start()
    return jsonify({"ok": True})


# ── Entry point ───────────────────────────────────────────────────────────────

if __name__ == "__main__":
    app.run(host="127.0.0.1", port=7860, debug=False, threaded=True)
