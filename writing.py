"""
Writing — the Claude text calls (research, planning, script, revision, TTS narration, image
prompts) and the image_prompts.txt line format they produce.
"""
from __future__ import annotations

import re
import time
from typing import TYPE_CHECKING

import anthropic

from prompts import (
    build_clarifying_questions_prompt,
    build_approach_pitch_prompt,
    build_research_prompt,
    build_script_prompt,
    build_tts_prompt,
    build_image_prompt_instructions,
    build_agent_system_prompt,
    extract,
)

if TYPE_CHECKING:
    from channel_profile import Profile


CLAUDE_MODEL  = "claude-sonnet-5"


# Sonnet 5 thinks by default and max_tokens caps thinking + text together,
# so every CLAUDE_MODEL call needs headroom for both (and must stream).
SONNET_MAX_TOKENS = 32000
SONNET_EFFORT     = "medium"
HAIKU_MODEL   = "claude-haiku-4-5-20251001"


def _text_of(response) -> str:
    """Concatenate text blocks. Thinking models put a thinking block first."""
    return "".join(b.text for b in response.content if b.type == "text")


def _stream_text(client: anthropic.Anthropic, attempts: int = 3, **kwargs) -> str:
    """Stream a Sonnet call and return its text.

    The SDK retries connection errors, but not once a stream has started — a
    mid-stream drop surfaces as APIConnectionError and loses the whole response.
    """
    for attempt in range(attempts):
        try:
            with client.messages.stream(**kwargs) as stream:
                return _text_of(stream.get_final_message())
        except anthropic.APIConnectionError:
            if attempt == attempts - 1:
                raise
            time.sleep(2 ** attempt)
    raise AssertionError("unreachable")


def research_topic(topic: str, client: anthropic.Anthropic, log_fn) -> str:
    log_fn("🔬  Researching topic and verifying facts...")
    response = client.messages.create(
        model=HAIKU_MODEL,
        max_tokens=3000,
        messages=[{"role": "user", "content": build_research_prompt(topic)}]
    )
    research = response.content[0].text
    log_fn("✅  Research complete")
    return research


def generate_clarifying_questions(topic: str, profile: "Profile",
                                  client: anthropic.Anthropic, log_fn) -> str:
    """Generate clarifying questions about the video topic."""
    prompt = build_clarifying_questions_prompt(topic, profile)
    log_fn("❓ Generating clarifying questions...")
    r = client.messages.create(
        model=HAIKU_MODEL,
        max_tokens=1000,
        messages=[{"role": "user", "content": prompt}]
    )
    questions = r.content[0].text.strip()
    log_fn("✅  Questions ready")
    return questions


def generate_approach_pitches(topic: str, answers: str, profile: "Profile",
                              client: anthropic.Anthropic, log_fn) -> str:
    """Generate 3 evidence-based approach pitches based on topic + user answers."""
    prompt = build_approach_pitch_prompt(topic, answers, profile)
    log_fn("💡 Pitching approaches...")
    r = client.messages.create(
        model=HAIKU_MODEL,
        max_tokens=2000,
        messages=[{"role": "user", "content": prompt}]
    )
    pitches = r.content[0].text.strip()
    log_fn("✅  Approaches pitched")
    return pitches


def generate_script(topic: str, research: str, profile: "Profile",
                    client: anthropic.Anthropic, log_fn, approach_context: str = "") -> str:
    prompt = build_script_prompt(topic, research, profile, approach_context)
    log_fn("✍️  Writing script...")
    script = extract("SCRIPT", _stream_text(
        client,
        model=CLAUDE_MODEL,
        max_tokens=SONNET_MAX_TOKENS,
        output_config={"effort": SONNET_EFFORT},
        messages=[{"role": "user", "content": prompt}],
    ))
    log_fn("✅  Script written")
    return script


def revise_script(script: str, feedback: str, topic: str, profile: "Profile | None",
                  client: anthropic.Anthropic) -> str:
    """Revise a script based on feedback via Sonnet. Returns the revised script."""
    system_ctx = build_agent_system_prompt(topic or "video", profile) if profile else ""
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
    text = _stream_text(
        client,
        model=CLAUDE_MODEL,
        max_tokens=SONNET_MAX_TOKENS,
        output_config={"effort": SONNET_EFFORT},
        messages=[{"role": "user", "content": prompt}],
    )
    return extract("SCRIPT", text) or text.strip()


def format_image_prompts(prompts: list[dict]) -> str:
    """Serialize parsed prompts back to image_prompts.txt form. Inverse of parse_image_prompts."""
    return "\n".join(f"{p['num']} | {p['source']} | {p['prompt']}" for p in prompts)


def _prompt_num(field: str) -> str | None:
    """'001', '- 002', '**003**', '012b' -> the zero-padded number; None for 'NNN' and other junk."""
    num = field.strip().lstrip("-*• ").strip("*").strip()
    return num.zfill(3) if re.fullmatch(r"\d+[a-z]?", num) else None


def parse_image_prompts(raw: str) -> list[dict]:
    """Parse NNN | source line | prompt lines into list of dicts, deduplicated by number."""
    seen: dict[str, dict] = {}
    for line in raw.strip().splitlines():
        line = line.strip()
        if not line:
            continue
        # Source lines must not contain '|' — the delimiter; narration sentences don't in practice
        parts = line.split("|", 2)
        num = _prompt_num(parts[0]) if len(parts) == 3 else None   # an echoed "NNN | ..." row isn't an image
        if num:
            source = parts[1].strip()
            prompt = parts[2].strip()
            seen[num] = {"num": num, "source": source, "prompt": prompt}
    return [seen[k] for k in sorted(seen)]


def _expand_short_prompts(raw: str, profile: "Profile") -> list[dict]:
    """Expand `NNN | source | character | scene` into full prompts.

    The model writes only the scene (~18% of the final text); the character
    description and art-style block are stitched in here from the profile.
    """
    descs = {c["name"].strip().lower(): c["description"].strip()
             for c in profile.characters}
    style = profile.image_style["art_style_block"].strip()

    seen: dict[str, dict] = {}
    for line in raw.strip().splitlines():
        parts = line.strip().split("|", 3)
        if len(parts) != 4:
            continue
        num, source, character, scene = (p.strip() for p in parts)
        num = _prompt_num(num)
        if not (num and scene):
            continue
        # "Orange Cat + White Cat" -> both descriptions, in the order named
        named = [descs.get(n.strip().lower(), "") for n in character.split("+")]
        desc = " and ".join(d for d in named if d)
        body = f"{desc} — {scene}" if desc else scene
        seen[num] = {
            "num": num,
            "source": source,
            "prompt": f"{body} — {style}",
        }
    return [seen[k] for k in sorted(seen)]


def generate_tts(script: str, profile: "Profile", client: anthropic.Anthropic, log_fn) -> str:
    """Haiku call: the finished script -> TTS narration with v3 audio tags."""
    log_fn("✍️  Extracting TTS narration from script...")
    tts_prompt = build_tts_prompt(script, profile)
    r1 = client.messages.create(
        model=HAIKU_MODEL,
        max_tokens=8000,
        messages=[{"role": "user", "content": tts_prompt}]
    )
    tts_script = extract("TTS_SCRIPT", r1.content[0].text)
    if not tts_script:
        raise ValueError("TTS reply had no ===TTS_SCRIPT=== block")
    log_fn("✅  TTS narration extracted")
    return tts_script


def generate_image_prompts(script: str, profile: "Profile", client: anthropic.Anthropic, log_fn) -> str:
    """Sonnet call: the finished script -> image_prompts.txt text."""
    log_fn("🖼️  Generating image prompts...")
    msg = (
        build_image_prompt_instructions(profile)
        + f"\nSCRIPT:\n{script}\n"
    )
    raw = _stream_text(
        client,
        model=CLAUDE_MODEL,
        max_tokens=64000,
        output_config={"effort": SONNET_EFFORT},
        messages=[{"role": "user", "content": msg}],
    )

    prompts = _expand_short_prompts(extract("IMAGE_PROMPTS", raw), profile)
    if not prompts:
        raise ValueError("Image-prompt reply had no usable ===IMAGE_PROMPTS=== lines")
    log_fn(f"✅  Image prompts generated — {len(prompts)} total")
    return format_image_prompts(prompts)


def generate_tts_and_prompts(script: str, profile: "Profile", client: anthropic.Anthropic, log_fn) -> tuple[str, str]:
    """Given a finished script, generate TTS narration and image prompts. Returns (tts_script, image_prompts_raw)."""
    return (generate_tts(script, profile, client, log_fn),
            generate_image_prompts(script, profile, client, log_fn))
