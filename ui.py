"""
YouTube Pipeline — Flask UI
Run: python ui.py
Opens at http://0.0.0.0:7860
"""

import anthropic
import json
import queue
import re
import threading
from pathlib import Path

from flask import Flask, Response, jsonify, render_template, request, send_file
import pipeline
from pipeline import (PROJECT_ROOT, run_pipeline, resume_pipeline, regenerate_images,
                      assemble_palmier_timeline, _palmier_available,
                      parse_image_prompts, get_audio_duration, GOOGLE_MODEL_OPTIONS,
                      run_status, ANTHROPIC_KEY, CLAUDE_MODEL, VIDIQ_KEY,
                      generate_clarifying_questions, generate_approach_pitches)
from profile import load_profile, list_profiles
from agents import run_vet_agent
from prompts import _build_agent_system_prompt, _extract

_noop_log = lambda _: None

app = Flask(__name__)
OUTPUT_ROOT = PROJECT_ROOT / "output"

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
    "images":   ["generating image", "images generated", "fal.ai", "sending request"],
    "voice":    ["generating voiceover", "voiceover generated", "voiceover failed", "🎙"],
    "timeline": ["pipeline complete", "pipeline finished", "🎬", "╔", "╚"],
}


def detect_stage(msg: str) -> str | None:
    lower = msg.lower()
    for stage, keywords in STAGE_KEYWORDS.items():
        for kw in keywords:
            if kw.lower() in lower:
                return stage
    return None


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

    available = list_profiles()
    if not available:
        return jsonify({"ok": False, "error": "No profiles found. Create profiles/<name>/profile.yaml first."})

    if not profile_name:
        if len(available) == 1:
            profile_name = available[0]
        else:
            return jsonify({"ok": False, "error": f"Select a profile. Available: {available}"})

    try:
        profile = load_profile(profile_name)
    except ValueError as e:
        return jsonify({"ok": False, "error": str(e)})

    lq = queue.Queue()
    aq = queue.Queue()
    se = threading.Event()

    _state["log_queue"]      = lq
    _state["approval_queue"] = aq
    _state["stop_event"]     = se

    def progress_cb(msg: str):
        lq.put({"type": "log", "stage": detect_stage(msg), "msg": msg})

    def approval_cb(script_text: str) -> bool:
        lq.put({"type": "review", "script": script_text})
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
    aq = _state.get("approval_queue")
    if aq:
        # Persist any edits the user made in the review panel
        runs = sorted(OUTPUT_ROOT.glob("*/script.txt"),
                      key=lambda p: p.stat().st_mtime, reverse=True)
        if runs and script_text.strip():
            runs[0].write_text(script_text)
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

    available = list_profiles()
    if not available:
        return jsonify({"ok": False, "error": "No profiles found. Create profiles/<name>/profile.yaml first."})

    if not profile_name:
        profile_name = available[0]

    try:
        profile = load_profile(profile_name)
    except ValueError as e:
        return jsonify({"ok": False, "error": str(e)})

    lq = queue.Queue()
    se = threading.Event()

    _state["log_queue"]  = lq
    _state["stop_event"] = se
    _state["approval_queue"] = None

    def progress_cb(msg: str):
        lq.put({"type": "log", "stage": detect_stage(msg), "msg": msg})

    def worker():
        try:
            resume_pipeline(run_slug, profile, progress_callback=progress_cb, stop_event=se)
        except Exception as e:
            lq.put({"type": "log", "stage": None, "msg": f"❌  Error: {e}"})
        finally:
            lq.put({"type": "done"})

    threading.Thread(target=worker, daemon=True).start()
    return jsonify({"ok": True})




@app.route("/palmier_url", methods=["GET"])
def get_palmier_url():
    return jsonify({"url": pipeline.PALMIER_MCP_URL})


@app.route("/palmier_url", methods=["POST"])
def set_palmier_url():
    data = request.get_json(force=True)
    url  = (data.get("url") or "").strip()
    if not url:
        return jsonify({"ok": False, "error": "Empty URL"})
    pipeline.PALMIER_MCP_URL = url
    # Persist to .env
    env_path = PROJECT_ROOT / ".env"
    lines = env_path.read_text().splitlines() if env_path.exists() else []
    updated = False
    for i, line in enumerate(lines):
        if line.startswith("PALMIER_MCP_URL="):
            lines[i] = f"PALMIER_MCP_URL={url}"
            updated = True
            break
    if not updated:
        lines.append(f"PALMIER_MCP_URL={url}")
    env_path.write_text("\n".join(lines) + "\n")
    return jsonify({"ok": True})


@app.route("/palmier_status")
def palmier_status():
    available = _palmier_available()
    return jsonify({"available": available})


@app.route("/send_to_palmier", methods=["POST"])
def send_to_palmier():
    data     = request.get_json(force=True)
    run_slug = (data.get("run_slug") or "").strip()
    if not run_slug:
        return jsonify({"ok": False, "error": "No run_slug provided"})

    profile_file = OUTPUT_ROOT / run_slug / "profile.txt"
    send_profile = None
    if profile_file.exists():
        try:
            send_profile = load_profile(profile_file.read_text().strip())
        except Exception:
            pass

    out_dir = OUTPUT_ROOT / run_slug
    if not out_dir.exists():
        return jsonify({"ok": False, "error": "Run folder not found"})

    prompts_file = out_dir / "image_prompts.txt"
    if not prompts_file.exists():
        return jsonify({"ok": False, "error": "image_prompts.txt not found — run must be complete"})

    from pipeline import _find_image
    prompts = parse_image_prompts(prompts_file.read_text())
    image_results = {
        p["num"]: _find_image(out_dir / "images", p["num"])
        for p in prompts
        if _find_image(out_dir / "images", p["num"])
    }
    if not image_results:
        return jsonify({"ok": False, "error": "No images found in this run"})

    audio_path = out_dir / "audio" / "voiceover.mp3"
    if not audio_path.exists():
        audio_path = None

    lq = queue.Queue()
    _state["log_queue"]      = lq
    _state["stop_event"]     = None
    _state["approval_queue"] = None

    def progress_cb(msg: str):
        lq.put({"type": "log", "stage": "timeline", "msg": msg})

    def worker():
        try:
            assemble_palmier_timeline(prompts, image_results, audio_path, out_dir, progress_cb, profile=send_profile)
        except Exception as e:
            lq.put({"type": "log", "stage": "timeline", "msg": f"❌  Error: {e}"})
        finally:
            lq.put({"type": "done"})

    threading.Thread(target=worker, daemon=True).start()
    return jsonify({"ok": True, "images": len(image_results), "has_audio": audio_path is not None})


@app.route("/runs_with_images")
def runs_with_images():
    if not OUTPUT_ROOT.exists():
        return jsonify([])
    result = []
    for d in OUTPUT_ROOT.iterdir():
        if not d.is_dir():
            continue
        img_dir = d / "images"
        if img_dir.exists() and any(
            p.suffix.lower() in (".png", ".jpg", ".jpeg")
            for p in img_dir.iterdir()
            if not p.stem.endswith("_bob")
        ):
            result.append(d.name)
    return jsonify(sorted(result, reverse=True))


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

@app.route("/runs")
def runs():
    if not OUTPUT_ROOT.exists():
        return jsonify([])
    return jsonify(sorted([d.name for d in OUTPUT_ROOT.iterdir() if d.is_dir()], reverse=True))


@app.route("/runs_with_prompts")
def runs_with_prompts():
    if not OUTPUT_ROOT.exists():
        return jsonify([])
    result = [d.name for d in OUTPUT_ROOT.iterdir()
              if d.is_dir() and (d / "image_prompts.txt").exists()]
    return jsonify(sorted(result, reverse=True))


@app.route("/run_status/<path:run_slug>")
def run_status_route(run_slug):
    return jsonify(run_status(run_slug))


@app.route("/runs_with_scripts")
def runs_with_scripts():
    if not OUTPUT_ROOT.exists():
        return jsonify([])
    result = [d.name for d in OUTPUT_ROOT.iterdir()
              if d.is_dir() and (d / "script.txt").exists()]
    return jsonify(sorted(result, reverse=True))


@app.route("/prompts/<run_name>")
def get_prompts(run_name: str):
    path = OUTPUT_ROOT / run_name / "image_prompts.txt"
    if not path.exists():
        return jsonify({})
    prompts = {}
    for line in path.read_text().splitlines():
        parts = line.split("|", 2)
        if len(parts) == 3:
            num    = parts[0].strip().zfill(3)
            source = parts[1].strip()
            txt    = parts[2].strip()
            prompts[num] = {"source": source, "prompt": txt}
    return jsonify(prompts)


@app.route("/runs_with_audio")
def runs_with_audio():
    if not OUTPUT_ROOT.exists():
        return jsonify([])
    result = [d.name for d in OUTPUT_ROOT.iterdir()
              if d.is_dir() and (d / "audio" / "voiceover.mp3").exists()]
    return jsonify(sorted(result, reverse=True))


@app.route("/audio/<run_name>")
def serve_audio(run_name: str):
    audio_path = OUTPUT_ROOT / run_name / "audio" / "voiceover.mp3"
    if not audio_path.exists():
        return "", 404
    return send_file(audio_path, mimetype="audio/mpeg")


@app.route("/run_files/<run_name>")
def run_files(run_name: str):
    run_dir = OUTPUT_ROOT / run_name
    if not run_dir.exists():
        return jsonify([])
    files = [str(p.relative_to(OUTPUT_ROOT))
             for p in sorted(run_dir.rglob("*")) if p.is_file()]
    return jsonify(files)


@app.route("/script/<run_name>")
def get_script(run_name: str):
    path = OUTPUT_ROOT / run_name / "script.txt"
    if not path.exists():
        return jsonify({"ok": False, "error": "script.txt not found"})
    return jsonify({"ok": True, "text": path.read_text()})


@app.route("/save_script/<run_name>", methods=["POST"])
def save_script(run_name: str):
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

    profile_name = (data.get("profile_name") or "").strip() or list_profiles()[0]
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

    client = anthropic.Anthropic(api_key=ANTHROPIC_KEY)
    system_ctx = _build_agent_system_prompt(topic or "video", profile) if profile else ""
    prompt = f"""{system_ctx}

---
You are revising a YouTube video script based on feedback. Apply the feedback precisely.
Keep everything that isn't mentioned in the feedback exactly as-is.
Return only the revised script — no preamble, no explanation.

FEEDBACK:
{feedback}

CURRENT SCRIPT:
{script}

===SCRIPT===
"""
    response = client.messages.create(
        model=CLAUDE_MODEL,
        max_tokens=8000,
        messages=[{"role": "user", "content": prompt}]
    )
    text    = response.content[0].text
    revised = _extract("SCRIPT", text) or text.strip()

    if VIDIQ_KEY and revised and profile:
        logs   = []
        vetted = run_vet_agent(topic or "video", revised, profile, lambda msg: logs.append(msg))
        if vetted:
            revised = vetted

    return jsonify({"ok": True, "script": revised})


# ── Image endpoints ───────────────────────────────────────────────────────────

@app.route("/images/<run_name>")
def list_images(run_name: str):
    img_dir = OUTPUT_ROOT / run_name / "images"
    if not img_dir.exists():
        return jsonify([])
    imgs = sorted(
        p.stem for p in img_dir.iterdir()
        if p.suffix.lower() in (".png", ".jpg", ".jpeg")
        and not p.stem.endswith("_bob")
    )
    return jsonify(imgs)


@app.route("/image/<run_name>/<num>")
def serve_image(run_name: str, num: str):
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

    available = list_profiles()
    if not available:
        return jsonify({"ok": False, "error": "No profiles found."})

    if not profile_name:
        profile_name = available[0]

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
    app.run(host="0.0.0.0", port=7860, debug=False, threaded=True)
