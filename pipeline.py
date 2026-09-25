"""
YouTube Pipeline — Core Orchestrator
Generates all video assets from a single topic string.
Usage: python pipeline.py "your topic here"
       or import run_pipeline() and call it from ui.py
"""
from __future__ import annotations

import json
import os
import re
import sys
import threading
from pathlib import Path
from typing import TYPE_CHECKING

import anthropic
from dotenv import load_dotenv

from agents import run_script_agent, run_vet_agent, VIDIQ_KEY
from images import (GOOGLE_KEY, _find_image, _load_anchor_parts, generate_all_images,
                    generate_flicker_frames, generate_image_google)
from voiceover import EL_KEY, generate_voiceover
from writing import (_generate_image_prompts, _generate_tts, _generate_tts_and_prompts, format_image_prompts,
                     generate_script, parse_image_prompts, research_topic)

if TYPE_CHECKING:
    from channel_profile import Profile

load_dotenv()

# ── Config ────────────────────────────────────────────────────────────────────

PROJECT_ROOT    = Path(__file__).parent
OUTPUT_ROOT     = PROJECT_ROOT / "output"

ANTHROPIC_KEY   = (os.getenv("ANTHROPIC_API_KEY") or "").strip()


def require_keys(*names: str) -> None:
    """Exit with a clear message if any required API key is unset."""
    missing = [n for n in names if not (os.getenv(n) or "").strip()]
    if missing:
        sys.exit(
            "Missing required env var(s): " + ", ".join(missing) +
            "\nCopy .env.example to .env and fill them in."
        )


# ── Helpers ───────────────────────────────────────────────────────────────────


def log(msg: str, progress_callback=None):
    print(msg)
    if progress_callback:
        progress_callback(msg)


def slugify(text: str) -> str:
    text = text.lower().strip()
    text = re.sub(r"[^\w\s-]", "", text)
    text = re.sub(r"[\s_]+", "-", text)
    return text[:60] or "untitled"   # '' would put the run straight into output/


def make_output_dir(slug: str) -> Path:
    out = OUTPUT_ROOT / slug
    (out / "images").mkdir(parents=True, exist_ok=True)
    (out / "audio").mkdir(parents=True, exist_ok=True)
    return out


def check_keys(profile: "Profile", log_fn) -> bool:
    missing = []
    if not ANTHROPIC_KEY:              missing.append("ANTHROPIC_API_KEY")
    if not GOOGLE_KEY:                 missing.append("GOOGLE_API_KEY")
    if not EL_KEY:                     missing.append("ELEVENLABS_API_KEY")
    if not profile.voice["voice_id"]:  missing.append("voice_id (in profile or ELEVENLABS_VOICE_ID env var)")
    if missing:
        log_fn(f"❌  Missing API keys in .env: {', '.join(missing)}")
        return False
    vidiq_status = "✅" if VIDIQ_KEY else "⚠️  not set (will use standard research)"
    log_fn(f"🔑  API keys loaded — Anthropic ✅  Google AI ✅  ElevenLabs ✅  vidIQ {vidiq_status}")
    return True


# ── Image regeneration ────────────────────────────────────────────────────────


def regenerate_images(run_slug: str, image_nums: list[str], model_key: str,
                      profile: "Profile", progress_callback=None) -> dict:
    """Manually regenerate specific images for a completed run."""
    def log_fn(msg):
        log(msg, progress_callback)

    slot  = {"3-pro": "pro_model", "nano-banana-2": "default_model"}.get(model_key)   # the UI's two choices
    model = profile.image_gen.get(slot) if slot else None
    if not model:
        log_fn(f"❌  Unknown model key '{model_key}' or no matching model configured on this profile")
        return {"status": "error", "reason": "unknown model key"}

    out_dir      = OUTPUT_ROOT / run_slug
    prompts_file = out_dir / "image_prompts.txt"
    if not prompts_file.exists():
        log_fn(f"❌  image_prompts.txt not found in {out_dir}")
        return {"status": "error", "reason": "image_prompts.txt not found"}

    all_prompts = {p["num"]: p for p in parse_image_prompts(prompts_file.read_text())}
    targets = [str(n).zfill(3) for n in image_nums]
    missing = [n for n in targets if n not in all_prompts]
    if missing:
        log_fn(f"⚠️  Image numbers not found in prompts: {missing}")

    anchor_parts = _load_anchor_parts(profile)
    results = {"regenerated": [], "failed": []}
    for num in targets:
        if num not in all_prompts:
            continue
        img_path = out_dir / "images" / f"{num}.png"
        log_fn(f"🔄  Regenerating {num} using {model_key}...")
        ok = generate_image_google(all_prompts[num]["prompt"], img_path, profile, log_fn, model=model, anchor_parts=anchor_parts)
        if ok:
            log_fn(f"  ✅  {num} regenerated")
            results["regenerated"].append(num)
            flicker_cfg = profile.image_gen.get("flicker", {})
            if flicker_cfg.get("enabled"):
                magnitude = flicker_cfg.get("magnitude", 0.004)
                found = _find_image(out_dir / "images", num) or img_path
                generate_flicker_frames(found, out_dir / "images", num, magnitude, log_fn)
        else:
            log_fn(f"  ❌  {num} failed")
            results["failed"].append(num)

    log_fn(f"\n✅  Done — {len(results['regenerated'])} regenerated, {len(results['failed'])} failed")
    return {"status": "complete", **results}


# ── Shared production phase ───────────────────────────────────────────────────

def _run_production(topic: str, prompts: list[dict], tts_script: str,
                    profile: "Profile", out_dir: Path,
                    log_fn, stop_event=None, skip_existing_images=False) -> dict:
    """Phases 2-4: images + voiceover."""
    audio_path = None

    def voiceover_thread():
        nonlocal audio_path
        existing_mp3 = out_dir / "audio" / "voiceover.mp3"
        if skip_existing_images and existing_mp3.exists():
            log_fn("🎙  Voiceover already exists — skipping")
            audio_path = existing_mp3
            return
        # a new production replaces the audio: a stale file would pass run_status if TTS fails
        existing_mp3.unlink(missing_ok=True)
        if not tts_script:
            log_fn("⚠️  No TTS script available — skipping voiceover")
            return
        audio_path = generate_voiceover(tts_script, out_dir, profile, log_fn, stop_event)

    vo_thread = threading.Thread(target=voiceover_thread, daemon=True)
    vo_thread.start()

    image_results = generate_all_images(
        prompts, out_dir, profile, log_fn, stop_event,
        skip_existing=skip_existing_images,
    )
    log_fn(f"✅  {len(image_results)}/{len(prompts)} images ready")

    vo_thread.join()

    if stop_event and stop_event.is_set():
        return {"status": "cancelled", "out_dir": str(out_dir)}

    log_fn("ℹ️  Review images and audio.")

    summary = {
        "status":  "complete",
        "topic":   topic,
        "out_dir": str(out_dir),
        "images":  len(image_results),
        "audio":   str(audio_path) if audio_path else "failed",
    }
    log_fn(f"✅  Pipeline complete — {summary['images']} images, "
           f"audio {'✅' if audio_path else '❌'} — {out_dir}")
    return summary


def _produce_from_script(script: str, topic: str, profile: "Profile", out_dir: Path,
                         client: anthropic.Anthropic, log_fn, stop_event=None) -> dict:
    """Phases 1b-4: a finished script -> TTS + image prompts -> images + voiceover."""
    tts_script, image_prompts_raw = _generate_tts_and_prompts(script, profile, client, log_fn)

    prompts = parse_image_prompts(image_prompts_raw)
    log_fn(f"📝  {len(prompts)} image prompts parsed")

    (out_dir / "tts_script.txt").write_text(tts_script)
    (out_dir / "image_prompts.txt").write_text(format_image_prompts(prompts))

    if stop_event and stop_event.is_set():
        return {"status": "cancelled", "out_dir": str(out_dir)}

    return _run_production(topic, prompts, tts_script, profile, out_dir, log_fn, stop_event)


# ── Run status + resume ───────────────────────────────────────────────────────

def run_status(run_slug: str) -> dict:
    """Return what assets exist and what is missing for a given run."""
    out_dir      = OUTPUT_ROOT / run_slug
    script_file  = out_dir / "script.txt"
    prompts_file = out_dir / "image_prompts.txt"
    tts_file     = out_dir / "tts_script.txt"
    audio_file   = out_dir / "audio" / "voiceover.mp3"
    images_dir   = out_dir / "images"

    has_script  = script_file.exists()
    has_prompts = prompts_file.exists()
    has_tts     = tts_file.exists()
    has_audio   = audio_file.exists()

    total_prompts  = 0
    images_on_disk = 0
    if has_prompts:
        prompts        = parse_image_prompts(prompts_file.read_text())
        total_prompts  = len(prompts)
        images_on_disk = sum(1 for p in prompts if _find_image(images_dir, p["num"]))

    missing = []
    if not has_script:  missing.append("script")
    if not has_prompts: missing.append("image_prompts")
    if not has_tts:     missing.append("tts_script")
    if not has_audio:   missing.append("voiceover")
    if total_prompts and images_on_disk < total_prompts:
        missing.append(f"{total_prompts - images_on_disk} images")

    return {
        "run_slug":       run_slug,
        "has_script":     has_script,
        "has_prompts":    has_prompts,
        "has_tts":        has_tts,
        "has_audio":      has_audio,
        "total_prompts":  total_prompts,
        "images_on_disk": images_on_disk,
        "missing":        missing,
    }


def resume_pipeline(run_slug: str, profile: "Profile | None" = None, progress_callback=None, stop_event=None) -> dict:
    """Resume a stopped pipeline run, regenerating only what is missing."""
    from channel_profile import load_profile as _load_profile

    def log_fn(msg):
        log(msg, progress_callback)

    out_dir = OUTPUT_ROOT / run_slug
    if not out_dir.exists():
        log_fn(f"❌  Output folder not found: {out_dir}")
        return {"status": "error", "reason": "run folder not found"}

    if profile is None:
        profile_file = out_dir / "profile.txt"
        if profile_file.exists():
            profile_name = profile_file.read_text().strip()
            log_fn(f"📋  Loading saved profile: {profile_name}")
            profile = _load_profile(profile_name)
        else:
            log_fn("❌  No profile.txt found in run folder and no profile passed — cannot resume")
            return {"status": "error", "reason": "profile unknown"}

    if not check_keys(profile, log_fn):
        return {"status": "error", "reason": "missing API keys"}

    script_file  = out_dir / "script.txt"
    prompts_file = out_dir / "image_prompts.txt"
    tts_file     = out_dir / "tts_script.txt"

    if not script_file.exists():
        log_fn("❌  script.txt missing — cannot resume without a script")
        return {"status": "error", "reason": "script.txt not found"}

    log_fn(f"\n▶️  Resuming pipeline: {run_slug}")
    client = anthropic.Anthropic(api_key=ANTHROPIC_KEY)
    script = script_file.read_text()

    # an empty file is what an untagged model reply used to leave behind
    needs_prompts = not prompts_file.exists() or not prompts_file.read_text().strip()
    needs_tts     = not tts_file.exists() or not tts_file.read_text().strip()

    if needs_prompts or needs_tts:
        log_fn("📝  Regenerating missing assets from script.txt:")
        if needs_prompts: log_fn("     • image_prompts.txt")
        if needs_tts:     log_fn("     • tts_script.txt")
        # only pay for what is missing; the voiceover must match the tts_script.txt on disk
        if needs_tts:
            tts_file.write_text(_generate_tts(script, profile, client, log_fn))
            log_fn("✅  tts_script.txt saved")
        if needs_prompts:
            prompts_file.write_text(format_image_prompts(parse_image_prompts(
                _generate_image_prompts(script, profile, client, log_fn))))
            log_fn("✅  image_prompts.txt saved")
    tts_script = tts_file.read_text()

    prompts = parse_image_prompts(prompts_file.read_text())
    log_fn(f"📝  {len(prompts)} image prompts loaded")

    images_on_disk = sum(1 for p in prompts if _find_image(out_dir / "images", p["num"]))
    log_fn(f"🖼   {images_on_disk}/{len(prompts)} images already on disk — skipping those")

    return _run_production(
        _topic_from_script(script), prompts, tts_script, profile, out_dir,
        log_fn, stop_event, skip_existing_images=True,
    )


# ── Main Orchestrator ─────────────────────────────────────────────────────────

def run_pipeline(topic: str, profile: "Profile", progress_callback=None,
                 stop_event=None, approval_callback=None, approach_context: str = "") -> dict:
    """
    Run the full pipeline for a given topic.

    progress_callback:  callable(str) — live log updates
    stop_event:         threading.Event — set to cancel mid-run
    approval_callback:  callable(script: str) -> bool
                        Called after script is written, before any paid API calls.
                        Return True to proceed, False to abort.
                        If None, CLI input() is used instead.
    approach_context:   str — user's clarifying answers about the video idea
    """
    def log_fn(msg):
        log(msg, progress_callback)

    log_fn(f"\n🚀  Starting pipeline for: {topic}\n")

    if not check_keys(profile, log_fn):
        return {"status": "error", "reason": "missing API keys"}

    client  = anthropic.Anthropic(api_key=ANTHROPIC_KEY)
    slug    = slugify(topic)
    # re-running a topic would overwrite that run's script before the approval gate
    if (OUTPUT_ROOT / slug / "script.txt").exists():
        log_fn(f"❌  output/{slug} already has a script — resume that run, or use a different topic")
        return {"status": "error", "reason": "run already exists"}
    out_dir = make_output_dir(slug)
    (out_dir / "profile.txt").write_text(profile.name)
    log_fn(f"📁  Output directory: {out_dir}")

    # Stop must also land between the paid writing steps, not just in the image loop
    stopped   = lambda: bool(stop_event and stop_event.is_set())
    cancelled = {"status": "cancelled", "out_dir": str(out_dir)}

    if VIDIQ_KEY:
        log_fn("🤖  Running vidIQ research + script agent...")
        script, research_notes = run_script_agent(topic, profile, log_fn, approach_context)
        if research_notes:
            (out_dir / "research.txt").write_text(research_notes)
    else:
        log_fn("🔬  No vidIQ key — running standard research and script phases...")
        research = research_topic(topic, client, log_fn)
        (out_dir / "research.txt").write_text(research)
        if stopped():
            return cancelled
        script = generate_script(topic, research, profile, client, log_fn, approach_context)

    if not script:   # the reply had no ===SCRIPT=== block
        log_fn("❌  No script was produced — aborting")
        return {"status": "error", "reason": "no script produced"}

    (out_dir / "script.txt").write_text(script)
    log_fn("📝  Script written")

    if stopped():
        return cancelled
    if VIDIQ_KEY:
        log_fn("🔎  Running vidIQ script vet...")
        vetted = run_vet_agent(topic, script, profile, log_fn)
        if vetted:
            script = vetted
            (out_dir / "script.txt").write_text(script)
            log_fn("✅  Script vetted and updated")
        else:
            log_fn("⚠️  Vet agent returned no output — proceeding with original script")
        if stopped():
            return cancelled

    log_fn("\n" + "─" * 50)
    log_fn("📋  SCRIPT READY FOR REVIEW")
    log_fn(f"     Read it at: {out_dir / 'script.txt'}")
    log_fn("─" * 50)

    if approval_callback is not None:
        approved = approval_callback(script)
    else:
        print("\n" + "=" * 60)
        print(script)
        print("=" * 60)
        answer = input("\n✅  Approve script and start production? [y/N]: ").strip().lower()
        approved = answer == "y"

    if not approved:
        log_fn("🚫  Script rejected — pipeline stopped.")
        return {"status": "rejected", "out_dir": str(out_dir), "script": str(out_dir / "script.txt")}

    log_fn("✅  Script approved — generating TTS and image prompts...")

    # re-read: the approval step may have persisted the user's edits
    return _produce_from_script((out_dir / "script.txt").read_text(), topic, profile,
                                out_dir, client, log_fn, stop_event)


def _topic_from_script(script: str) -> str:
    """Slug source for a premade script: its TITLE: line, else its first line."""
    for line in script.splitlines():
        line = line.strip()
        if not line.strip("`"):   # blank lines and markdown fences from a pasted chat reply
            continue
        m = re.match(r"[\W\d]*title\s*:\W*(.*)", line, re.I)   # TITLE:, **TITLE:**, 1. TITLE:, TITLE :
        if not m:
            return line
        if m[1].strip(" *"):
            return m[1].strip(" *")
        # an empty TITLE: line; the title is on a later line
    return "untitled"


def run_from_script(script: str, profile: "Profile", topic: str = "",
                    progress_callback=None, stop_event=None) -> dict:
    """Run the pipeline from an already-written script — no research, writing, or approval.

    Overwrites the run folder named after `topic` (or the script's TITLE line),
    same as re-running that topic would. Use resume_pipeline() to fill in only
    what a folder is missing.
    """
    def log_fn(msg):
        log(msg, progress_callback)

    script = script.strip()
    if not script:
        log_fn("❌  Empty script — nothing to produce")
        return {"status": "error", "reason": "empty script"}

    if not check_keys(profile, log_fn):
        return {"status": "error", "reason": "missing API keys"}

    topic   = topic.strip() or _topic_from_script(script)
    out_dir = make_output_dir(slugify(topic))
    (out_dir / "profile.txt").write_text(profile.name)
    (out_dir / "script.txt").write_text(script)

    log_fn(f"\n🚀  Starting pipeline from premade script: {topic}")
    log_fn(f"📁  Output directory: {out_dir}")

    return _produce_from_script(script, topic, profile, out_dir,
                                anthropic.Anthropic(api_key=ANTHROPIC_KEY), log_fn, stop_event)


# ── CLI entry point ───────────────────────────────────────────────────────────

if __name__ == "__main__":
    import argparse
    from channel_profile import load_profile, list_profiles

    parser = argparse.ArgumentParser(description="YouTube Pipeline")
    sub = parser.add_subparsers(dest="cmd")

    # Default command: run a topic (preserves existing behavior)
    run_p = sub.add_parser("run", help="Run the full pipeline for a topic")
    run_p.add_argument("topic", nargs="+")
    run_p.add_argument("--profile", default=None)

    # Run production from a script you already wrote
    sc = sub.add_parser("script", help="Run production from an existing script file ('-' for stdin)")
    sc.add_argument("path")
    sc.add_argument("--profile", default=None)
    sc.add_argument("--topic", default="", help="Overrides the slug taken from the script's TITLE line")

    # Finish a stopped run from what its folder already holds
    rs = sub.add_parser("resume", help="Finish a stopped run: generate only what its folder is missing")
    rs.add_argument("run_slug")
    rs.add_argument("--profile", default=None, help="Defaults to the run's saved profile.txt")

    # Metadata subcommand
    md = sub.add_parser("metadata", help="Generate or update metadata + thumbnails")
    md.add_argument("run_slug")
    md.add_argument("--profile", default=None)
    md.add_argument("--regenerate", action="store_true")
    md.add_argument("--pick-thumb", type=int, metavar="N", help="1-based index")
    md.add_argument("--show", action="store_true")

    # Back-compat: if first arg isn't a known subcommand, treat as run
    argv = sys.argv[1:]
    if not argv:
        parser.print_help()
        sys.exit(2)
    if argv[0] not in {"run", "metadata", "script", "resume", "-h", "--help"}:
        argv = ["run"] + argv

    args = parser.parse_args(argv)

    def _or_exit(fn, *a, **kw):
        """Bad input (a --profile typo, a missing run, --pick-thumb 0) -> a message, not a traceback."""
        try:
            return fn(*a, **kw)
        except ValueError as e:
            sys.exit(f"❌  {e}")

    def _resolve_profile(name: str | None):
        available = list_profiles()
        if not available:
            print("No profiles found. Create profiles/<name>/profile.yaml first.")
            sys.exit(1)
        if name:
            return _or_exit(load_profile, name)
        if len(available) == 1:
            print(f"Using profile: {available[0]}")
            return _or_exit(load_profile, available[0])
        print("Multiple profiles found. Specify one with --profile:")
        for p in available:
            print(f"  {p}")
        sys.exit(1)

    if args.cmd == "script":
        require_keys("ANTHROPIC_API_KEY", "GOOGLE_API_KEY", "ELEVENLABS_API_KEY")
        text = sys.stdin.read() if args.path == "-" else Path(args.path).read_text()
        result = run_from_script(text, _resolve_profile(args.profile), args.topic)
        sys.exit(0 if result.get("status") == "complete" else 1)   # scripts can tell a failed run

    if args.cmd == "resume":
        result = resume_pipeline(args.run_slug, _resolve_profile(args.profile) if args.profile else None)
        sys.exit(0 if result.get("status") == "complete" else 1)

    if args.cmd == "metadata":
        import metadata as md_mod

        log_fn = lambda m: print(m)

        if args.show:
            data = _or_exit(md_mod.load_metadata, args.run_slug)
            if data is None:
                print(f"No metadata for run '{args.run_slug}'")
                sys.exit(1)
            print(json.dumps(data, indent=2, ensure_ascii=False))
            sys.exit(0)

        if args.pick_thumb is not None:
            data = _or_exit(md_mod.pick_thumbnail, args.run_slug, args.pick_thumb - 1)
            print(f"✅ chosen_thumbnail_index = {data['chosen_thumbnail_index']}")
            sys.exit(0)

        # Full generate
        require_keys("ANTHROPIC_API_KEY", "GOOGLE_API_KEY")   # before asking anything
        if not args.regenerate:   # --regenerate must work even when metadata.json is malformed
            try:
                existing = md_mod.load_metadata(args.run_slug)
            except ValueError as e:
                sys.exit(f"❌  {e} — rerun with --regenerate to rebuild it")
            if existing is not None:
                answer = input("metadata exists, overwrite? (y/N): ").strip().lower()
                if answer != "y":
                    print("aborted")
                    sys.exit(0)
        saved = OUTPUT_ROOT / args.run_slug / "profile.txt"   # the profile the run was made with
        profile = _resolve_profile(args.profile or (saved.read_text().strip() if saved.exists() else None))
        _or_exit(md_mod.generate_metadata, args.run_slug, profile, log_fn, regenerate=True)
        sys.exit(0)

    # args.cmd == "run"
    require_keys("ANTHROPIC_API_KEY", "GOOGLE_API_KEY", "ELEVENLABS_API_KEY")
    topic = " ".join(args.topic)
    profile = _resolve_profile(args.profile)
    result = run_pipeline(topic, profile)
    sys.exit(0 if result.get("status") == "complete" else 1)
