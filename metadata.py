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
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from profile import Profile

from pipeline import OUTPUT_ROOT
from prompts import _extract, _build_metadata_titles_prompt, _build_metadata_desc_hashtags_prompt

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
