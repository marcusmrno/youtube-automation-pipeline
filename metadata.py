"""
Metadata & Thumbnail Generator — standalone post-run step.

Produces YouTube titles, description, hashtags, and thumbnails for a
completed run. See docs/superpowers/specs/2026-06-29-metadata-thumbnail-generator-design.md.
"""
from __future__ import annotations

import asyncio
import json as _json
import json
import os
import re
import shutil
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from profile import Profile

from pipeline import OUTPUT_ROOT, generate_image_google, _load_anchor_parts
from prompts import _extract, _build_metadata_titles_prompt, _build_metadata_desc_hashtags_prompt, _build_metadata_thumbnail_prompt

_VIDIQ_KEY = (os.getenv("VIDIQ_API_KEY") or "").strip()
_VIDIQ_MCP_URL = "https://mcp.vidiq.com/mcp"


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
    """Run a one-off vidIQ agent and return its full text output.

    Mirrors agents._run_agent but isolated here so the vet/script agent
    code path stays untouched.
    """
    from claude_agent_sdk import query as agent_query, ClaudeAgentOptions
    from claude_agent_sdk.types import AssistantMessage, ResultMessage, TextBlock

    options = ClaudeAgentOptions(
        mcp_servers={
            "vidiq": {
                "type": "http",
                "url": _VIDIQ_MCP_URL,
                "headers": {"Authorization": f"Bearer {_VIDIQ_KEY}"},
            }
        },
        permission_mode="bypassPermissions",
        max_turns=max_turns,
        model="claude-haiku-4-5-20251001",
    )
    options.system_prompt = system_prompt

    full_text = ""

    async def _runner() -> str:
        nonlocal full_text
        async for message in agent_query(prompt=user_prompt, options=options):
            if isinstance(message, AssistantMessage):
                for block in message.content:
                    if isinstance(block, TextBlock) and block.text.strip():
                        log_fn(f"  {block.text[:120].strip()}")
                        full_text += block.text
            elif isinstance(message, ResultMessage):
                if message.result:
                    full_text += message.result
        return full_text

    return asyncio.run(_runner())


def _vidiq_keywords(topic: str, log_fn) -> list[dict]:
    """Use vidIQ MCP to fetch top keywords for the topic.

    Returns [] if VIDIQ_API_KEY is unset or any error occurs.
    """
    if not _VIDIQ_KEY:
        log_fn("⚠️  vidIQ key not set — skipping keyword research")
        return []
    system = (
        "You are a YouTube SEO assistant. Use the vidiq_keyword_research tool to "
        "find the 5 strongest keywords for the topic. Return them in this exact format:\n"
        "===KEYWORDS===\n"
        "[json array of objects, each at minimum has a 'keyword' string]\n"
    )
    user = f"Topic: {topic}\n\nReturn 5 keywords as a JSON array, wrapped in ===KEYWORDS=== tags."
    try:
        raw = _call_vidiq_agent(system, user, max_turns=10, log_fn=log_fn)
        block = _extract("KEYWORDS", raw)
        if not block:
            log_fn("⚠️  vidIQ keyword research returned no KEYWORDS block")
            return []
        return _json.loads(block)
    except Exception as e:
        log_fn(f"⚠️  vidIQ keyword research failed: {e}")
        return []


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
        model="claude-haiku-4-5-20251001",
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
    parsed = _json.loads(block)
    return {"score": int(parsed["score"]), "breakdown": parsed.get("breakdown", {})}


def _score_titles(titles: list[str], log_fn) -> list[dict]:
    """Score every title via vidIQ in parallel. Returns one entry per input title in input order.

    Each entry: {"text": str, "score": int | None, "score_breakdown": dict}.
    """
    if not _VIDIQ_KEY:
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
        model="claude-haiku-4-5-20251001",
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
            model="claude-haiku-4-5-20251001",
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
        filename = f"thumbnails/thumb-{i:02d}.png"
        img_path = run_dir / filename
        log_fn(f"🖼   Rendering thumbnail {i}/{len(prompts)}...")
        try:
            ok = generate_image_google(
                p["prompt"], img_path, profile, log_fn,
                model=model, anchor_parts=anchor_parts,
            )
        except Exception as e:
            log_fn(f"❌  Thumbnail {i} raised: {e}")
            out.append({**p, "filename": filename, "render_error": str(e)})
            continue
        if not ok:
            out.append({**p, "filename": filename, "render_error": "generator returned False"})
            continue
        out.append({**p, "filename": filename})
    return out


def pick_title(run_slug: str, index: int) -> dict:
    """Select a title by 0-based index. Rewrites metadata.json. Returns updated metadata dict.

    Raises ValueError if metadata is missing or index is out of range.
    """
    data = load_metadata(run_slug)
    if data is None:
        raise ValueError(f"No metadata.json for run '{run_slug}' — generate first")
    n = len(data.get("titles") or [])
    if not (0 <= index < n):
        raise ValueError(f"title index {index} out of range 0..{n - 1}")
    data["chosen_title_index"] = index
    _save_metadata(_run_dir(run_slug), data)
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
