"""
Metadata & Thumbnail Generator — standalone post-run step.

Produces YouTube titles, description, hashtags, and thumbnails for a
completed run. See docs/superpowers/specs/2026-06-29-metadata-thumbnail-generator-design.md.
"""
from __future__ import annotations

import asyncio
import datetime as _datetime
import json
import os
import re
import shutil
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from profile import Profile

import anthropic

from pipeline import ANTHROPIC_KEY, OUTPUT_ROOT, HAIKU_MODEL, generate_image_google, _load_anchor_parts
from agents import run_vidiq_agent, VIDIQ_KEY
from prompts import _extract, _build_metadata_titles_prompt, _build_metadata_desc_hashtags_prompt, _build_metadata_thumbnail_prompt


def _run_dir(run_slug: str) -> Path:
    return OUTPUT_ROOT / run_slug


def load_metadata(run_slug: str) -> dict | None:
    """Return parsed metadata.json for the run, or None if it doesn't exist.

    Raises ValueError if the file exists but is malformed.
    """
    path = _run_dir(run_slug) / "metadata.json"
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text())
    except json.JSONDecodeError as e:
        raise ValueError(f"metadata.json for run '{run_slug}' is malformed: {e}") from e


def _save_metadata(run_dir: Path, data: dict) -> None:
    """Atomically write metadata.json to the run directory."""
    final = run_dir / "metadata.json"
    tmp = run_dir / "metadata.json.tmp"
    tmp.write_text(json.dumps(data, indent=2, ensure_ascii=False))
    os.replace(tmp, final)


def _call_vidiq_agent(system_prompt: str, user_prompt: str, max_turns: int, log_fn) -> str:
    """Run a one-off vidIQ agent (Haiku) and return its full text output."""
    return asyncio.run(run_vidiq_agent(system_prompt, user_prompt, max_turns, log_fn, model=HAIKU_MODEL))


def _vidiq_keywords(topic: str, log_fn) -> list[dict]:
    """Use vidIQ MCP to fetch top keywords for the topic.

    Returns [] if VIDIQ_API_KEY is unset or any error occurs.
    """
    if not VIDIQ_KEY:
        log_fn("⚠️  vidIQ key not set — skipping keyword research")
        return []
    system = (
        "You are a YouTube SEO assistant. Use the vidiq_keyword_research tool to "
        "find the 15 strongest keywords for the topic (mix broad and long-tail). "
        "Return them in this exact format:\n"
        "===KEYWORDS===\n"
        "[json array of objects, each at minimum has a 'keyword' string]\n"
    )
    user = f"Topic: {topic}\n\nReturn 15 keywords as a JSON array, wrapped in ===KEYWORDS=== tags."
    try:
        raw = _call_vidiq_agent(system, user, max_turns=10, log_fn=log_fn)
        block = _extract("KEYWORDS", raw)
        if not block:
            log_fn("⚠️  vidIQ keyword research returned no KEYWORDS block")
            return []
        return json.loads(block)
    except Exception as e:
        log_fn(f"⚠️  vidIQ keyword research failed: {e}")
        return []


def _build_video_tags(topic: str, keywords: list[dict]) -> list[str]:
    """Build the YouTube tags-box list from topic + vidIQ keywords.

    Deduped case-insensitively, capped at YouTube's 500-char tags-field limit
    (tags are joined with ", " there, so that separator counts too).
    """
    seen: set[str] = set()
    tags: list[str] = []
    total = 0
    for raw in (topic, *(k.get("keyword", "") for k in keywords)):
        tag = raw.strip()
        key = tag.lower()
        if not tag or key in seen:
            continue
        total += len(tag) + (2 if tags else 0)
        if total > 500:
            break
        seen.add(key)
        tags.append(tag)
    return tags


def _parse_titles(raw: str) -> list[str]:
    """Extract numbered titles from a ===TITLES=== block."""
    block = _extract("TITLES", raw)
    if not block:
        return []
    titles: list[str] = []
    for line in block.splitlines():
        m = re.match(r"^\s*\d+[\.\)]\s+(.+?)\s*$", line)
        if not m:
            continue
        text = m.group(1).strip().strip('"').strip("'")
        if text:
            titles.append(text)
    return titles


def _generate_titles(topic, script, research, keywords, profile, client, log_fn) -> list[str]:
    log_fn("✍️  Generating 5 title candidates...")
    prompt = _build_metadata_titles_prompt(topic, script, research, keywords, profile)
    r = client.messages.create(
        model=HAIKU_MODEL,
        max_tokens=1000,
        messages=[{"role": "user", "content": prompt}],
    )
    titles = _parse_titles(r.content[0].text)
    if len(titles) < 5:
        log_fn(f"⚠️  Title generator returned {len(titles)} titles (fewer than 5) — proceeding")
    log_fn(f"✅  Got {len(titles)} titles")
    return titles


def _score_one_title(title: str, log_fn) -> dict:
    """Call vidIQ to score a single title. Returns {'score': int, 'breakdown': dict}.

    Raises on failure — caller (_score_titles) catches.
    """
    system = (
        "You are a YouTube SEO assistant. Use the vidiq_score_title tool to "
        "score the title. Return:\n"
        "===SCORE===\n"
        '{"score": <int 0-100>, "breakdown": <object from the tool>}\n'
    )
    user = f"Title: {title}\n\nScore this title. Return inside ===SCORE=== tags."
    raw = _call_vidiq_agent(system, user, max_turns=6, log_fn=log_fn)
    block = _extract("SCORE", raw)
    if not block:
        raise RuntimeError("vidIQ score response had no SCORE block")
    parsed = json.loads(block)
    return {"score": int(parsed["score"]), "breakdown": parsed.get("breakdown", {})}


def _score_titles(titles: list[str], log_fn) -> list[dict]:
    """Score every title via vidIQ in parallel. Returns one entry per input title in input order.

    Each entry: {"text": str, "score": int | None, "score_breakdown": dict}.
    """
    if not VIDIQ_KEY:
        log_fn("⚠️  vidIQ key not set — scores will be null")
        return [{"text": t, "score": None, "score_breakdown": {}} for t in titles]

    log_fn(f"📊  Scoring {len(titles)} titles via vidIQ in parallel...")
    results: list[dict | None] = [None] * len(titles)
    with ThreadPoolExecutor(max_workers=5) as ex:
        futures = {
            ex.submit(_score_one_title, t, log_fn): i
            for i, t in enumerate(titles)
        }
        for fut in futures:
            i = futures[fut]
            try:
                payload = fut.result()
                results[i] = {
                    "text": titles[i],
                    "score": payload["score"],
                    "score_breakdown": payload["breakdown"],
                }
            except Exception as e:
                log_fn(f"⚠️  Score failed for '{titles[i]}': {e}")
                results[i] = {"text": titles[i], "score": None, "score_breakdown": {}}
    log_fn("✅  Scoring complete")
    return [r for r in results if r is not None]  # type narrow


def _generate_description_hashtags(top_title, script, keywords, profile, client, log_fn) -> dict:
    log_fn("📝  Generating description and hashtags...")
    prompt = _build_metadata_desc_hashtags_prompt(top_title, script, keywords, profile)
    r = client.messages.create(
        model=HAIKU_MODEL,
        max_tokens=1500,
        messages=[{"role": "user", "content": prompt}],
    )
    text = r.content[0].text
    desc = _extract("DESCRIPTION", text)
    hashtags_raw = _extract("HASHTAGS", text)
    if not desc:
        raise ValueError("description generation: missing ===DESCRIPTION=== block")
    if not hashtags_raw:
        raise ValueError("description generation: missing ===HASHTAGS=== block")
    hashtags = [tok for tok in hashtags_raw.split() if tok.startswith("#")][:5]
    log_fn("✅  Description and hashtags ready")
    return {"description": desc.strip(), "hashtags": hashtags}


def _parse_thumbnail_prompts(raw: str) -> list[dict]:
    out = []
    for n in (1, 2, 3):
        block = _extract(f"THUMBNAIL_{n}", raw)
        if not block:
            return []
        lines = block.strip().splitlines()
        hook = ""
        if lines and lines[0].upper().startswith("HOOK:"):
            hook = lines[0].split(":", 1)[1].strip()
            body = "\n".join(lines[1:]).strip()
        else:
            body = block.strip()
        if not body:
            return []
        out.append({"prompt": body, "hook_text": hook})
    return out


def _generate_thumbnail_prompts(script, topic, profile, client, log_fn) -> list[dict]:
    log_fn("🎨  Generating 3 thumbnail prompts...")
    prompt = _build_metadata_thumbnail_prompt(script, topic, profile)
    for attempt in (1, 2):
        r = client.messages.create(
            model=HAIKU_MODEL,
            max_tokens=2500,
            messages=[{"role": "user", "content": prompt}],
        )
        parsed = _parse_thumbnail_prompts(r.content[0].text)
        if len(parsed) == 3:
            log_fn("✅  3 thumbnail prompts ready")
            return parsed
        log_fn(f"⚠️  Thumbnail prompts malformed on attempt {attempt}")
    raise ValueError("thumbnail prompt generation failed twice — aborting metadata run")


def _render_thumbnails(prompts: list[dict], run_dir: Path, profile, log_fn) -> list[dict]:
    """Render each thumbnail serially. Returns one entry per input with filename and any error."""
    thumb_dir = run_dir / "thumbnails"
    thumb_dir.mkdir(exist_ok=True)
    anchor_parts = _load_anchor_parts(profile)
    model = profile.image_gen.get("pro_model") or profile.image_gen["default_model"]

    out: list[dict] = []
    for i, p in enumerate(prompts, start=1):
        stem = f"thumb-{i:02d}"
        # Requested filename — generate_image_google may swap the extension to
        # .jpg when Google AI returns a JPEG mime, so we read the real path back
        # from disk after the call.
        requested_filename = f"thumbnails/{stem}.png"
        img_path = run_dir / requested_filename
        log_fn(f"🖼   Rendering thumbnail {i}/{len(prompts)}...")
        try:
            ok = generate_image_google(
                p["prompt"], img_path, profile, log_fn,
                model=model, anchor_parts=anchor_parts,
            )
        except Exception as e:
            log_fn(f"❌  Thumbnail {i} raised: {e}")
            out.append({**p, "filename": requested_filename, "render_error": str(e)})
            continue
        if not ok:
            out.append({**p, "filename": requested_filename, "render_error": "generator returned False"})
            continue
        actual = next(
            (q for q in (thumb_dir / f"{stem}.png", thumb_dir / f"{stem}.jpg", thumb_dir / f"{stem}.jpeg") if q.exists()),
            None,
        )
        if actual is None:
            out.append({**p, "filename": requested_filename, "render_error": "rendered file missing on disk"})
            continue
        out.append({**p, "filename": f"thumbnails/{actual.name}"})
    return out


def _read_run_inputs(run_slug: str) -> tuple[str, str, str]:
    """Return (script, research, topic). Raises if script missing."""
    run_dir = _run_dir(run_slug)
    script_path = run_dir / "script.txt"
    if not script_path.exists():
        raise ValueError(f"Run '{run_slug}' not found or incomplete (missing script.txt)")
    script = script_path.read_text()
    research_path = run_dir / "research.txt"
    research = research_path.read_text() if research_path.exists() else ""
    topic = run_slug.replace("-", " ").title()
    return script, research, topic


def _sort_titles_by_score(scored: list[dict]) -> list[dict]:
    return sorted(
        scored,
        key=lambda t: (t.get("score") is None, -(t.get("score") or 0)),
    )


def generate_metadata(run_slug: str, profile, log_fn, regenerate: bool = False) -> dict:
    """Generate full metadata + thumbnails for a completed run.

    Writes metadata.json, thumbnails/thumb-0N.png, and thumbnail.png.
    Returns the saved dict.
    """
    run_dir = _run_dir(run_slug)
    if not run_dir.exists():
        raise ValueError(f"Run '{run_slug}' does not exist at {run_dir}")
    if (run_dir / "metadata.json").exists() and not regenerate:
        raise ValueError(
            f"metadata.json already exists for '{run_slug}'. Pass regenerate=True to overwrite."
        )

    script, research, topic = _read_run_inputs(run_slug)
    log_fn(f"📦  Generating metadata for run: {run_slug}")

    client = anthropic.Anthropic(api_key=ANTHROPIC_KEY)

    keywords = _vidiq_keywords(topic, log_fn)
    tags = _build_video_tags(topic, keywords)
    titles_raw = _generate_titles(topic, script, research, keywords, profile, client, log_fn)
    scored = _score_titles(titles_raw, log_fn)
    titles_sorted = _sort_titles_by_score(scored)
    titles_with_idx = [
        {"index": i, "text": t["text"], "vidiq_score": t["score"], "score_breakdown": t["score_breakdown"]}
        for i, t in enumerate(titles_sorted)
    ]
    top_title = titles_with_idx[0]["text"] if titles_with_idx else topic

    desc_block = _generate_description_hashtags(top_title, script, keywords, profile, client, log_fn)
    thumb_prompts = _generate_thumbnail_prompts(script, topic, profile, client, log_fn)
    rendered = _render_thumbnails(thumb_prompts, run_dir, profile, log_fn)
    thumbs_with_idx = [
        {"index": i, "filename": r["filename"], "prompt": r["prompt"], "hook_text": r["hook_text"],
         **({"render_error": r["render_error"]} if "render_error" in r else {})}
        for i, r in enumerate(rendered)
    ]

    data = {
        "run_slug": run_slug,
        "generated_at": _datetime.datetime.now(_datetime.timezone.utc).isoformat(timespec="seconds").replace("+00:00", "Z"),
        "topic": topic,
        "vidiq_keywords": keywords,
        "tags": tags,
        "titles": titles_with_idx,
        "description": desc_block["description"],
        "hashtags": desc_block["hashtags"],
        "thumbnails": thumbs_with_idx,
        "chosen_thumbnail_index": 0,
    }

    # If chosen index 0 has render_error, find first non-errored
    for i, t in enumerate(thumbs_with_idx):
        if "render_error" not in t:
            data["chosen_thumbnail_index"] = i
            break

    _save_metadata(run_dir, data)

    # Copy chosen thumbnail to thumbnail.png
    chosen = thumbs_with_idx[data["chosen_thumbnail_index"]]
    if "render_error" not in chosen:
        shutil.copyfile(run_dir / chosen["filename"], run_dir / "thumbnail.png")

    log_fn(f"✅  Metadata written: {run_dir / 'metadata.json'}")
    return data


def pick_thumbnail(run_slug: str, index: int) -> dict:
    """Select a thumbnail by 0-based index. Rewrites metadata.json and copies file to thumbnail.png.

    Raises ValueError if metadata is missing, index is out of range, slot has render_error, or file is missing on disk.
    """
    data = load_metadata(run_slug)
    if data is None:
        raise ValueError(f"No metadata.json for run '{run_slug}' — generate first")
    n = len(data.get("thumbnails") or [])
    if not (0 <= index < n):
        raise ValueError(f"thumbnail index {index} out of range 0..{n - 1}")
    slot = data["thumbnails"][index]
    if "render_error" in slot:
        raise ValueError(f"thumbnail {index} is not available (render_error: {slot['render_error']})")
    run_dir = _run_dir(run_slug)
    src = run_dir / slot["filename"]
    if not src.exists():
        raise ValueError(f"thumbnail file missing on disk: {src}")
    shutil.copyfile(src, run_dir / "thumbnail.png")
    data["chosen_thumbnail_index"] = index
    _save_metadata(run_dir, data)
    return data
