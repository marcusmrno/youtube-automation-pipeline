"""
YouTube Pipeline — Core Agent
Generates all video assets from a single topic string.
Usage: python pipeline.py "your topic here"
       or import run_pipeline() and call it from ui.py
"""
from __future__ import annotations

import os
import re
import json
import time
import asyncio
import requests
import threading
from pathlib import Path
from datetime import datetime
from PIL import Image
from dotenv import load_dotenv
import anthropic
from google import genai
from google.genai import types as genai_types
from claude_agent_sdk import query as agent_query, ClaudeAgentOptions
from claude_agent_sdk.types import AssistantMessage, ResultMessage, TextBlock

load_dotenv()

# ── Config ────────────────────────────────────────────────────────────────────

PROJECT_ROOT    = Path(__file__).parent
OUTPUT_ROOT     = PROJECT_ROOT / "output"

ANTHROPIC_KEY   = (os.getenv("ANTHROPIC_API_KEY") or "").strip()
EL_KEY          = (os.getenv("ELEVENLABS_API_KEY") or "").strip()
EL_VOICE_ID     = (os.getenv("ELEVENLABS_VOICE_ID") or "").strip()
VIDIQ_KEY       = (os.getenv("VIDIQ_API_KEY") or "").strip()
GOOGLE_KEY      = (os.getenv("GOOGLE_API_KEY") or "").strip()
PALMIER_MCP_URL = (os.getenv("PALMIER_MCP_URL") or "http://127.0.0.1:19789/mcp").strip()

CLAUDE_MODEL        = "claude-sonnet-4-6"
EL_MODEL            = "eleven_v3"
GOOGLE_FAST_MODEL   = "gemini-3.1-flash-image"  # initial bulk generation
GOOGLE_REGEN_MODEL  = "gemini-3.1-flash-image"
GOOGLE_PRO_MODEL    = "gemini-3-pro-image"               # highest quality, slowest

GOOGLE_MODEL_OPTIONS = {
    "2.5-flash":    GOOGLE_FAST_MODEL,
    "nano-banana-2": GOOGLE_REGEN_MODEL,
    "3-pro":        GOOGLE_PRO_MODEL,
}


# ── Helpers ───────────────────────────────────────────────────────────────────

def log(msg: str, progress_callback=None):
    print(msg)
    if progress_callback:
        progress_callback(msg)


def slugify(text: str) -> str:
    text = text.lower().strip()
    text = re.sub(r"[^\w\s-]", "", text)
    text = re.sub(r"[\s_]+", "-", text)
    return text[:60]


def make_output_dir(slug: str) -> Path:
    out = OUTPUT_ROOT / slug
    (out / "images").mkdir(parents=True, exist_ok=True)
    (out / "audio").mkdir(parents=True, exist_ok=True)
    return out

# ── Phase 0: Research & Fact-Check ───────────────────────────────────────────

def research_topic(topic: str, client: anthropic.Anthropic, log_fn) -> str:
    """
    Research the topic and return a verified fact sheet.
    The script writer receives this as its source of truth.
    """
    log_fn("🔬  Researching topic and verifying facts...")

    prompt = f"""
You are a research assistant preparing verified facts for a YouTube educational video script.

TOPIC: {topic}

Your job:
1. Identify the 10-15 most interesting, accurate, and surprising facts about this topic.
2. Flag any claims that are commonly misunderstood or frequently stated incorrectly online.
3. Note the current scientific or historical consensus on any contested points.
4. Highlight 2-3 counterintuitive angles that would make a strong video hook.

Return your response in this exact format:

===VERIFIED_FACTS===
[Numbered list of verified facts. Each fact on its own line. Be specific — include real numbers, dates, names, and sources where possible.]

===COMMON_MISCONCEPTIONS===
[Numbered list of myths or exaggerations to avoid. State what is wrong and what is actually true.]

===HOOK_ANGLES===
[2-3 bullet points — surprising or counterintuitive angles that would make a strong video opening]

===CONFIDENCE_NOTES===
[Any facts where certainty is lower, or where scientific consensus is still evolving. Flag these so the script writer can soften the language.]
"""

    response = client.messages.create(
        model=CLAUDE_MODEL,
        max_tokens=3000,
        messages=[{"role": "user", "content": prompt}]
    )

    research = response.content[0].text
    log_fn("✅  Research complete")
    return research


# ── Phase 1: Writing ──────────────────────────────────────────────────────────

def _extract(tag: str, text: str) -> str:
    m = re.search(rf"==={tag}===(.*?)(?====|\Z)", text, re.DOTALL)
    return m.group(1).strip() if m else ""


# ── Prompt builders (pure functions — no API calls) ───────────────────────────

def _build_script_prompt(topic: str, research: str, profile: "Profile") -> str:
    s = profile.script
    c = profile.channel
    target_words = s["target_mins"] * s["wpm"]
    min_words    = s["min_mins"] * s["wpm"]
    max_words    = s["max_mins"] * s["wpm"]
    hook_words   = round(s["hook_duration_s"] / 60 * s["wpm"])
    cta_words    = round(s["cta_duration_s"] / 60 * s["wpm"])
    section_min  = round(int(s["section_duration_s"].split("-")[0]) / 60 * s["wpm"])
    section_max  = round(int(s["section_duration_s"].split("-")[1]) / 60 * s["wpm"])

    template = f"""
You are a script writer for a faceless educational YouTube channel.

## Channel Identity
- Niche: {c["niche"]}
- Target audience: {c["audience"]}
- Tone: {c["tone"]}
- Reference channel: {c["reference_channel"]} — study the hook style and pacing
- Titles: {c["title_format"]}

## Script Structure
- Hook (0:00-0:{s["hook_duration_s"]:02d}): Provocative opening statement or surprising fact. No intro, no "welcome back".
- {s["section_count"]} content sections with clear [MM:SS-MM:SS] timestamps
- Each section {s["section_duration_s"]} seconds
- CTA close (last {s["cta_duration_s"]} seconds): Subscribe prompt only — no teasing or referencing a next video

## Word Count Rules
The voiceover is delivered at ~{s["wpm"]} words per minute.
- {s["target_mins"]}-minute target = ~{target_words} words of narration
- {s["min_mins"]}-minute minimum  = ~{min_words} words of narration
- {s["max_mins"]}-minute maximum = ~{max_words} words of narration
- Each {s["section_duration_s"]} second section needs {section_min}-{section_max} words of narration
- Hook ({s["hook_duration_s"]}s) = ~{hook_words} words. CTA close ({s["cta_duration_s"]}s) = ~{cta_words} words.
After writing, count your narration words. If under {round(min_words * 1.1)}, expand sections before returning.

---
TOPIC: {topic}

---
RESEARCH & VERIFIED FACTS:
{research}

IMPORTANT: Base the script only on the verified facts above.
- Do not invent statistics or claims not supported by the research.
- Avoid any misconceptions listed above.
- Where confidence is noted as lower, use softened language ("some researchers suggest", "evidence points to", etc.).
- Use the hook angles as inspiration for the opening {s["hook_duration_s"]} seconds.

Your task: Write a full narration-only script. Do not describe visuals, camera directions, or what should appear on screen — write only what the narrator speaks aloud. Structure with [MM:SS-MM:SS] section timestamps.

Return your response in this exact format — no other text:

===SCRIPT===
[full script here]
"""

    override = profile.dir / "overrides" / "script_prompt.txt"
    if override.exists():
        return override.read_text()
    return template


def _build_tts_prompt(script: str, profile: "Profile") -> str:
    template = f"""
You are preparing a TTS narration for ElevenLabs {profile.voice.get("model", "eleven_v3")} from a finished YouTube video script.

SCRIPT:
{script}

## Extraction rules
- Strip all timestamps, VISUAL lines, section headers, and stage directions.
- Keep only the words spoken aloud, in order, as naturally flowing prose.
- Do NOT use SSML tags — {profile.voice.get("model", "eleven_v3")} does not support them.
- Do NOT use tags that describe visuals or actions (e.g. [grinning], [pacing]) — only auditory tags.
- Do NOT change any words — only add/remove/reposition tags and adjust punctuation/capitalisation for emphasis.

## Emphasis techniques
- Use ellipses (...) for dramatic pauses and weight at key moments.
- Use ALL CAPS for a single word of genuine vocal stress — one per sentence max.
- Do not use both ALL CAPS and a tag on the same phrase — pick one.
- Short sentences = faster delivery. Long sentences = slower, more weight. Vary deliberately.
- Exclamation marks add energy; question marks invite the listener to lean in.

## Audio tags — place immediately before the segment they modify, or after a natural pause mid-sentence
Target density: 1–2 tags per 200 words (~10–16 tags for a full 12-minute script). Too few is flat; too many is performed.
Do not stack two tags back-to-back with no words between them.

Laughter (graduated — pick the right intensity):
  [chuckles]        mild irony, "of course this is how it works"
  [laughs]          a stat or fact is genuinely absurd
  [laughs harder]   escalating absurdity — rare
  [giggles]         lighter, more playful moments
  [snorts]          dry involuntary reaction to something ridiculous
  [wheezing]        extreme — use only for the single funniest moment in the whole script

Breathing & texture:
  [sighs]           tired of a myth; "and then obviously…" moments
  [exhales]         releasing tension after a heavy section
  [whispers]        sharing something counterintuitive that feels like a secret
  [swallows]        before delivering a hard truth
  [gulps]           before something shocking or uncomfortable

Emotions:
  [excited]         a genuinely surprising fact or big reveal
  [surprised]       when a fact defies common sense
  [curious]         posing a question the audience is already wondering
  [thoughtful]      before a nuanced or considered point
  [impressed]       acknowledging something remarkable
  [delighted]       a satisfying explanation clicking into place
  [sarcastic]       quoting conventional wisdom you're about to debunk
  [mischievously]   setting up a twist or gotcha
  [frustrated]      something preventable went wrong; systemic failure
  [angry]           genuine outrage — historical injustice, lives lost unnecessarily
  [annoyed]         milder frustration; "this again" energy
  [appalled]        moral shock at a behaviour or fact
  [sad]             acknowledging real human cost
  [sympathetic]     speaking to an audience who may have experienced this
  [sheepishly]      correcting a complication or admitting nuance
  [nervously]       building unease before a reveal
  [alarmed]         urgent warning; something worse than expected
  [panicking]       high-stakes escalation — use once max, near the climax
  [reassuring]      after a scary section; "here's what you can do"
  [warmly]          CTA close ONLY — one tag at the very start, then no more tags after it
  [professional]    delivering a crisp fact or instruction
  [questioning]     rhetorical question the audience is asking themselves
  [happy]           a good outcome; something working as intended

## Section-to-tag mapping
Hook first sentence          → [whispers] or [excited]
Hook absurd opening stat     → [laughs] or [surprised]
Setting up a myth to debunk  → [sarcastic] or [thoughtful]
The debunk itself            → [sighs] or [appalled]
Counterintuitive reveal      → [surprised] or [mischievously]
Gross or disturbing fact     → [appalled] or [gulps]
Historical injustice         → [angry] or [frustrated]
Human cost / empathy moment  → [sympathetic] or [sad]
Absurd statistic             → [chuckles] or [snorts]
"Here's what works" pivot    → [reassuring] or [exhales]
Tension before consequence   → [nervously] or [alarmed]
Rhetorical question          → [questioning] or [curious]
CTA close                    → [warmly] once at the very start, then no more tags

## Channel tone
{profile.voice["tone_description"]}

Return only this, no other text:

===TTS_SCRIPT===
[clean narration here]
"""

    override = profile.dir / "overrides" / "tts_prompt.txt"
    if override.exists():
        return override.read_text()
    return template


def _build_image_prompt_instructions(profile: "Profile") -> str:
    s = profile.script
    target_images = s["target_mins"] * 25  # ~25 cuts/minute

    char_block = profile.characters_block()
    behavior   = profile.character_behavior.strip()
    style      = profile.image_style["art_style_block"].strip()
    sky        = profile.image_style.get("sky_rotation", "")

    template = f"""
You are an image prompt writer for a flat 2D educational YouTube video pipeline targeting Google Gemini image generation.

Below is a segment of the script. Each VISUAL line shows the timestamp and narration that will be playing at that moment. Your job is to write one image prompt per VISUAL line — a scene that is a direct, literal translation of EXACTLY what the narrator says in that line.

---

## The Prime Directive

For each narration beat, ask: what is the single most concrete, specific thing being said right now? Build the entire image around showing that one thing as literally and directly as possible.

**Rules:**
- The image must be SPECIFIC to its narration line — it must be impossible to swap it with any other image in the video
- If the narration mentions a number, that number must appear large and prominent in the image
- If the narration names a specific thing (organ, vitamin, country, person, object), that thing must be the main visual element
- If the narration describes an action or process, a character must be physically performing or demonstrating it
- Never show a "mood" or "vibe" — show the exact fact being stated
- Never write a scene that could fit 3 different moments in the script

**Forbidden:**
- Characters looking surprised, confused, or reacting emotionally to narration
- Generic "character standing in environment" scenes with no specific prop
- Any prop or object not directly tied to the narration line
- Reusing the same scene composition for consecutive prompts
- Two consecutive prompts with the same sky color
- Style prefix at the start of a prompt (pipeline prepends it automatically)

---

## Character descriptions — embed verbatim in EVERY prompt

{char_block}

{behavior}

---

## Environment
- Always include a visible horizon line separating sky (top) from ground plane (bottom)
- Sky color rotation: {sky}
- Never the same sky twice in a row. No more than 1 in 3 prompts may use the first sky color.
- Ground: flat solid color plane filling bottom third. Always visible.
- Midground (optional): one flat silhouette layer. Solid fill only, no interior detail.
- Characters always stand on the ground plane — never floating.

## Text in image
Whenever narration states a fact, name, or stat: include it as exact bold handwritten uppercase marker text inside a grey rounded rectangle label box. Always write the exact words.

---

## Art style — end EVERY prompt with this exact block

{style}

---

## Format

Write one prompt per NARRATION BEAT. A {s["target_mins"]}-minute video needs ~{target_images} prompts total.

For each prompt, derive a tight timestamp from narration pacing (~3-4 seconds per image).
Format each line as:
NNN | MM:SS-MM:SS | [full prompt]

Number sequentially from wherever instructed — never restart from 001 mid-batch.
Do NOT include a style prefix. Write ALL prompts for this segment.

Return only:

===IMAGE_PROMPTS===
[prompts here]
"""

    override = profile.dir / "overrides" / "image_prompt_instructions.txt"
    if override.exists():
        return override.read_text()
    return template


def _build_agent_script_prompt(profile: "Profile") -> str:
    s = profile.script
    c = profile.channel
    target_words = s["target_mins"] * s["wpm"]
    min_words    = s["min_mins"] * s["wpm"]
    hook_words   = round(s["hook_duration_s"] / 60 * s["wpm"])
    cta_words    = round(s["cta_duration_s"] / 60 * s["wpm"])
    section_min  = round(int(s["section_duration_s"].split("-")[0]) / 60 * s["wpm"])
    section_max  = round(int(s["section_duration_s"].split("-")[1]) / 60 * s["wpm"])

    char_names = " / ".join(ch["name"] for ch in profile.characters)
    char_block = profile.characters_block()

    return f"""
You are writing a YouTube video script for a channel: {c["name"]}.
Follow this two-phase process exactly — never skip research to jump straight to writing.

## PHASE 1 — Research (run ALL of these in parallel)

- vidiq_keyword_research: search volume + competition for the topic
- vidiq_outliers: videos over-performing right now (reveals best angle/format)
- vidiq_youtube_search: what's already ranking (avoid duplicating it)
- vidiq_channel_analytics: channel avg views and best-performing topics
- vidiq_generate_titles: 5 title candidates using keyword data
- vidiq_score_title: score all 5 candidates — pick the highest scorer

Look for:
- High volume + low competition keywords → weave top 3-5 naturally into first 60s of narration
- Outlier videos → use their angle and hook structure, not their content
- Channel niche: {c["niche"]}
- Channel tone: {c["tone"]}
- Title format: {c["title_format"]}

## PHASE 2 — Write the script

Use the winning title + vidIQ keyword insights to write a complete script matching this exact format:

```
TITLE: [winning title]
KEYWORDS: [3-5 top keywords from vidIQ]

[00:00-00:{s["hook_duration_s"]:02d}] HOOK
[provocative opening statement or surprising fact — no intro, no "welcome back", no "in this video"]

[00:{s["hook_duration_s"]:02d}-02:00] SECTION 1 — [section title]
[narration prose]
VISUAL: [which character, what action, what prop — one line per scene beat]

... {s["section_count"]} sections total ...

[CTA CLOSE — last {s["cta_duration_s"]} seconds]
[subscribe prompt only — no teasing a next video]
```

### Script rules

- Target {s["target_mins"]}:00 total (never under {s["min_mins"]}:00, never over {s["max_mins"]}:00)
- The voiceover voice runs at ~{s["wpm"]} wpm:
  - {s["target_mins"]}-min target = ~{target_words} words of narration
  - {s["min_mins"]}-min minimum = ~{min_words} words
  - Each {s["section_duration_s"]}s section = {section_min}-{section_max} words of narration
  - Hook ({s["hook_duration_s"]}s) = ~{hook_words} words. CTA ({s["cta_duration_s"]}s) = ~{cta_words} words.
  - After writing, count narration words — if under {round(min_words * 1.1)}, expand sections before finishing
- Hook hard in the first 10 seconds — lead with the most surprising fact, not context
- Every narration beat gets a VISUAL line showing a character doing an action, not reacting
- VISUAL lines must be literal: "cat holds five flat gold trophies" not "cat looks amazed"
- Vary which character appears ({char_names}) — never the same character 3 beats in a row

## Characters

{char_block}

{profile.character_behavior.strip()}

### Forbidden
- Starting the hook with "In this video…", "Welcome back…", or "Today we're going to…"
- VISUAL lines describing character emotions
- Generic visuals that could fit any moment in any video
- CTA that teases a next video

## OUTPUT

Return the finished script in this exact format — nothing after it:

===SCRIPT===
[full script here with TITLE, KEYWORDS, timestamps, section headers, narration, and VISUAL lines]
"""


def _build_vet_prompt(profile: "Profile") -> str:
    s = profile.script
    target_words = s["target_mins"] * s["wpm"]
    min_words    = s["min_mins"] * s["wpm"]

    return f"""
You are vetting a YouTube video script for accuracy, SEO strength, and hook power.

STEP 1 — Pull live vidIQ data on the topic (run in parallel):
- vidiq_keyword_research: confirm the top keywords and their search volume
- vidiq_outliers: find the highest over-performing videos on this topic right now
- vidiq_youtube_search: see what is currently ranking and how it is framed
- vidiq_score_title: score the current script title and note the result

STEP 2 — Review the script against the data:
- FACTUAL ACCURACY: flag any claims that are outdated, exaggerated, or unsupported
- MISSING ANGLES: if the script misses the strongest outlier hook angle, note it
- KEYWORD GAPS: if the top keywords are absent from the first 60 seconds, flag them
- TITLE STRENGTH: if the vidIQ title score is below 70, propose a stronger alternative
- WORD COUNT: narration must be {round(min_words * 1.1)}-{target_words} words (voice runs at ~{s["wpm"]} wpm) — if short, expand thin sections

STEP 3 — Rewrite the script with all fixes applied:
Make only the changes the review identified. Do not restructure the whole script or change the channel tone.
Preserve all VISUAL lines, timestamps, and section headers exactly unless a section was expanded.

STEP 4 — Output:
Return the vetted script in this exact format — nothing after it:

===SCRIPT===
[full revised script here]
"""


def _build_agent_system_prompt(topic: str, profile: "Profile") -> str:
    c = profile.channel
    char_block = profile.characters_block()
    style      = profile.image_style["art_style_block"].strip()

    return f"""## Channel: {c["name"]}
Niche: {c["niche"]}
Audience: {c["audience"]}
Tone: {c["tone"]}

## Characters
{char_block}

{profile.character_behavior.strip()}

## Visual Style
{style}

---
TOPIC: {topic}
"""


def _generate_tts_and_prompts(script: str, profile: "Profile", client: anthropic.Anthropic, log_fn) -> tuple[str, str]:
    """Given a finished script, generate TTS narration and image prompts. Returns (tts_script, image_prompts_raw)."""

    # ── Call 1: TTS extraction ────────────────────────────────────────────────
    log_fn("✍️  Extracting TTS narration from script...")

    tts_prompt = _build_tts_prompt(script, profile)

    r1 = client.messages.create(
        model=CLAUDE_MODEL,
        max_tokens=8000,
        messages=[{"role": "user", "content": tts_prompt}]
    )
    tts_script = _extract("TTS_SCRIPT", r1.content[0].text)
    log_fn("✅  TTS narration extracted")

    # ── Call 2: Image prompts ─────────────────────────────────────────────────
    log_fn("🖼️  Generating image prompts...")

    base_instructions = _build_image_prompt_instructions(profile)
    prompts_prompt = base_instructions

    # Split script in half by section boundaries so each batch has full section context
    # but the model isn't overwhelmed with the entire script at once.
    script_lines  = script.strip().splitlines()
    section_starts = [i for i, l in enumerate(script_lines) if re.match(r'^\[[\d:–\-]+\]', l.strip())]

    if len(section_starts) >= 2:
        mid_section = section_starts[len(section_starts) // 2]
        half_a = "\n".join(script_lines[:mid_section])
        half_b = "\n".join(script_lines[mid_section:])
        batches = [(half_a, "first half"), (half_b, "second half")]
    else:
        batches = [(script, "full script")]

    all_prompt_lines: list[str] = []
    last_prompt_num  = 0

    for batch_idx, (batch_script, batch_label) in enumerate(batches):
        start_num = last_prompt_num + 1
        numbering = (
            "Start numbering from 001."
            if batch_idx == 0
            else f"Continue numbering from {start_num:03d}. Do NOT restart from 001 — the previous batch ended at {last_prompt_num:03d}."
        )
        batch_instruction = (
            prompts_prompt
            + f"\nSCRIPT SEGMENT ({batch_label}):\n{batch_script}\n\n"
            f"NUMBERING: {numbering}\n"
            f"TARGET: ~25 prompts per minute of content. Write one prompt per scene beat, not one per section."
        )
        log_fn(f"  Generating image prompts ({batch_label})...")
        with client.messages.stream(
            model=CLAUDE_MODEL,
            max_tokens=64000,
            messages=[{"role": "user", "content": batch_instruction}]
        ) as stream:
            batch_text = stream.get_final_text()

        batch_prompts = _extract("IMAGE_PROMPTS", batch_text)
        batch_lines   = [l for l in batch_prompts.splitlines() if l.strip()]
        all_prompt_lines.extend(batch_lines)

        for line in reversed(batch_lines):
            m = re.match(r'^(\d+)\s*\|', line.strip())
            if m:
                last_prompt_num = int(m.group(1))
                break

    # Deduplicate by prompt number — keep last occurrence so the second batch wins
    # where the model ignored the start-numbering instruction and restarted from 001.
    seen: dict[int, str] = {}
    for line in all_prompt_lines:
        m = re.match(r'^(\d+)\s*\|', line.strip())
        if m:
            seen[int(m.group(1))] = line
        else:
            # Non-numbered lines (blank separators etc.) — drop them
            pass
    deduped_lines = [seen[k] for k in sorted(seen)]
    last_prompt_num = max(seen) if seen else 0

    image_prompts = "\n".join(deduped_lines)
    log_fn(f"✅  Image prompts generated — {last_prompt_num} total")

    return tts_script, image_prompts


def generate_script(topic: str, research: str, profile: "Profile",
                    client: anthropic.Anthropic, log_fn) -> str:
    """Returns the script text."""
    prompt = _build_script_prompt(topic, research, profile)
    log_fn("✍️  Writing script...")
    r = client.messages.create(
        model=CLAUDE_MODEL,
        max_tokens=8000,
        messages=[{"role": "user", "content": prompt}]
    )
    script = _extract("SCRIPT", r.content[0].text)
    log_fn("✅  Script written")
    return script


def parse_image_prompts(raw: str) -> list[dict]:
    """Parse NNN | MM:SS-MM:SS | prompt lines into list of dicts, deduplicated by number."""
    seen: dict[str, dict] = {}
    for line in raw.strip().splitlines():
        line = line.strip()
        if not line:
            continue
        parts = line.split("|", 2)
        if len(parts) == 3:
            num    = parts[0].strip().zfill(3)
            ts     = parts[1].strip()
            prompt = parts[2].strip()
            seen[num] = {"num": num, "ts": ts, "prompt": prompt}
    return [seen[k] for k in sorted(seen)]

# ── Phase 2: Image Generation ─────────────────────────────────────────────────

def check_keys(profile: "Profile", log_fn) -> bool:
    """Verify all API keys are present. Returns False and logs if any are missing."""
    missing = []
    if not ANTHROPIC_KEY:              missing.append("ANTHROPIC_API_KEY")
    if not GOOGLE_KEY:                 missing.append("GOOGLE_API_KEY")
    if not EL_KEY:                     missing.append("ELEVENLABS_API_KEY")
    if not profile.voice["voice_id"]:  missing.append("voice_id (in profile or ELEVENLABS_VOICE_ID env var)")
    if not VIDIQ_KEY:                  missing.append("VIDIQ_API_KEY")
    if missing:
        log_fn(f"❌  Missing API keys in .env: {', '.join(missing)}")
        return False
    log_fn("🔑  API keys loaded — Anthropic ✅  Google AI ✅  ElevenLabs ✅")
    return True


def generate_image_google(prompt: str, output_path: Path, profile: "Profile", log_fn,
                          model: str = None) -> bool:
    """Generate image via Google AI. Passes style sheet + anchor images as references."""
    from profile import Profile  # local import avoids circular deps at module level

    if model is None:
        model = profile.image_gen["default_model"]

    # Load anchors from profile directory in priority order
    anchors_dir = profile.anchors_dir
    anchor_files = sorted(anchors_dir.glob("anchor-*.png")) + sorted(anchors_dir.glob("anchor-*.jpg"))
    anchor_map = {f.stem: f for f in anchor_files}

    reference_files = []
    for name in profile.image_style.get("anchor_priority", []):
        if name in anchor_map:
            reference_files.append(anchor_map[name])
    reference_files = reference_files[:profile.image_style.get("max_anchors", 14)]

    contents = [
        f"The FIRST reference image is the new multi-angle character reference sheet — match these two cat characters exactly in every scene. "
        f"The SECOND reference image is the original character reference sheet — use both together to lock in the character designs. "
        f"The remaining reference images define the art style to follow precisely.\n\n"
        f"STYLE CONSTRAINTS: Bold uneven black marker outlines. Color fills bleed outside lines with visible marker streaks. "
        f"Off-white warm paper background. No gradients. No drop shadows. No clean fonts. No photorealistic textures. No smooth digital lines.\n\n"
        f"{prompt}",
    ]
    for ref_path in reference_files:
        contents.append(
            genai_types.Part.from_bytes(
                data=ref_path.read_bytes(),
                mime_type=_image_mime(ref_path),
            )
        )

    client = genai.Client(api_key=GOOGLE_KEY)
    for attempt in range(3):
        try:
            log_fn(f"  ⏳  Sending request to Google AI [{model}] (attempt {attempt+1})...")
            response = client.models.generate_content(
                model=model,
                contents=contents,
                config=genai_types.GenerateContentConfig(
                    response_modalities=["IMAGE"],
                ),
            )
            for part in response.candidates[0].content.parts:
                if part.inline_data and part.inline_data.mime_type.startswith("image/"):
                    actual_mime = part.inline_data.mime_type  # e.g. image/jpeg or image/png
                    ext = ".jpg" if "jpeg" in actual_mime else ".png"
                    final_path = output_path.with_suffix(ext)
                    final_path.write_bytes(part.inline_data.data)
                    if final_path != output_path and output_path.exists():
                        output_path.unlink()
                    _standardize_image(final_path)
                    return True
            log_fn(f"  ⚠️  Google AI returned no image in response")
            time.sleep(3)
        except Exception as e:
            log_fn(f"  ⚠️  Google AI attempt {attempt+1} error: {e}")
            time.sleep(3)
    log_fn(f"  ❌  Image {output_path.name} failed after 3 attempts — skipping")
    return False


def _standardize_image(path: Path) -> None:
    """Center-crop to 16:9 and resize to 1920×1080 in place."""
    img = Image.open(path).convert("RGB")
    w, h = img.size
    target_w = min(w, int(h * 16 / 9))
    target_h = min(h, int(w * 9 / 16))
    left = (w - target_w) // 2
    top  = (h - target_h) // 2
    img  = img.crop((left, top, left + target_w, top + target_h))
    img  = img.resize((1920, 1080), Image.LANCZOS)
    img.save(path)


def _image_mime(path: Path) -> str:
    """Detect true image mime type from magic bytes, not file extension."""
    header = path.read_bytes()[:4]
    if header[:3] == b'\xff\xd8\xff':
        return "image/jpeg"
    return "image/png"


def _find_image(img_dir: Path, num: str) -> Path | None:
    """Return the path to an image file for the given number, regardless of extension."""
    for ext in (".png", ".jpg", ".jpeg"):
        p = img_dir / f"{num}{ext}"
        if p.exists():
            return p
    return None


def generate_all_images(prompts: list[dict], out_dir: Path,
                        profile: "Profile", log_fn, stop_event=None, skip_existing=False) -> dict:
    """Generate images sequentially. Returns {num: path} for successful images."""
    results = {}
    total = len(prompts)
    for i, p in enumerate(prompts):
        if stop_event and stop_event.is_set():
            break
        num      = p["num"]
        prompt   = p["prompt"]
        img_path = out_dir / "images" / f"{num}.png"  # default; may be renamed to .jpg by generator

        existing = _find_image(out_dir / "images", num)
        if skip_existing and existing:
            results[num] = existing
            continue

        log_fn(f"🖼  Generating image {i+1}/{total} ({num})")
        ok = generate_image_google(prompt, img_path, profile, log_fn)
        if ok:
            results[num] = _find_image(out_dir / "images", num) or img_path
        else:
            log_fn(f"  ❌ Image {num} failed after 3 attempts — flagged")
    return results

def regenerate_images(run_slug: str, image_nums: list[str], model_key: str,
                      profile: "Profile" = None, progress_callback=None) -> dict:
    """
    Manually regenerate specific images for a completed run.

    run_slug:    output folder name (e.g. "why-cats-sleep-so-much")
    image_nums:  list of zero-padded strings or ints, e.g. ["001", "012", "034"]
    model_key:   one of "2.5-flash", "nano-banana-2", "3-pro"
    """
    def log_fn(msg):
        log(msg, progress_callback)

    model = None
    if profile is not None:
        key_map = {"2.5-flash": "default_model", "nano-banana-2": "regen_model", "3-pro": "pro_model"}
        profile_key = key_map.get(model_key)
        if profile_key:
            model = profile.image_gen.get(profile_key)
    if model is None:
        model = GOOGLE_MODEL_OPTIONS.get(model_key)
    if not model:
        log_fn(f"❌  Unknown model key '{model_key}'. Choose from: {list(GOOGLE_MODEL_OPTIONS)}")
        return {"status": "error", "reason": "unknown model key"}

    out_dir = OUTPUT_ROOT / run_slug
    prompts_file = out_dir / "image_prompts.txt"
    if not prompts_file.exists():
        log_fn(f"❌  image_prompts.txt not found in {out_dir}")
        return {"status": "error", "reason": "image_prompts.txt not found"}

    all_prompts = {p["num"]: p for p in parse_image_prompts(prompts_file.read_text())}

    # Normalise nums to zero-padded strings
    targets = [str(n).zfill(3) for n in image_nums]
    missing = [n for n in targets if n not in all_prompts]
    if missing:
        log_fn(f"⚠️  Image numbers not found in prompts: {missing}")

    results = {"regenerated": [], "failed": []}
    for num in targets:
        if num not in all_prompts:
            continue
        prompt   = all_prompts[num]["prompt"]
        img_path = out_dir / "images" / f"{num}.png"  # default; may be renamed to .jpg by generator
        log_fn(f"🔄  Regenerating {num} using {model_key}...")
        ok = generate_image_google(prompt, img_path, profile, log_fn, model=model)
        if ok:
            log_fn(f"  ✅  {num} regenerated")
            results["regenerated"].append(num)
        else:
            log_fn(f"  ❌  {num} failed")
            results["failed"].append(num)

    log_fn(f"\n✅  Done — {len(results['regenerated'])} regenerated, {len(results['failed'])} failed")
    return {"status": "complete", **results}


# ── Phase 2b: Voiceover ────────────────────────────────────────────────────────

def _split_into_chunks(text: str, max_chars: int = 4500) -> list[str]:
    """Split text on sentence boundaries so each chunk is under max_chars."""
    chunks, current = [], []
    length = 0
    for sentence in re.split(r'(?<=[.!?])\s+', text.strip()):
        if length + len(sentence) + 1 > max_chars and current:
            chunks.append(" ".join(current))
            current, length = [], 0
        current.append(sentence)
        length += len(sentence) + 1
    if current:
        chunks.append(" ".join(current))
    return chunks


def _tts_chunk(text: str, headers: dict, voice_settings: dict, log_fn,
               voice_id: str = "", model: str = "",
               prev_text: str = "", next_text: str = "") -> bytes | None:
    """Send one chunk to ElevenLabs. Returns raw mp3 bytes or None on failure."""
    vid = voice_id or EL_VOICE_ID
    mdl = model or EL_MODEL
    url = f"https://api.elevenlabs.io/v1/text-to-speech/{vid}"
    params = {"output_format": "mp3_44100_192"}
    payload = {
        "text": text,
        "model_id": mdl,
        "voice_settings": voice_settings,
    }
    if mdl != "eleven_v3":
        if prev_text:
            payload["previous_text"] = prev_text
        if next_text:
            payload["next_text"] = next_text

    for attempt in range(3):
        try:
            resp = requests.post(url, headers=headers, params=params, json=payload, timeout=120)
            if not resp.ok:
                log_fn(f"  ⚠️  TTS attempt {attempt+1} failed: {resp.status_code} — {resp.text[:300]}")
                time.sleep(2)
                continue
            return resp.content
        except Exception as e:
            log_fn(f"  ⚠️  TTS attempt {attempt+1} error: {e}")
            time.sleep(2)
    return None


def generate_voiceover(tts_script: str, out_dir: Path, profile: "Profile", log_fn) -> Path | None:
    """Call ElevenLabs TTS API in chunks. Returns path to mp3 or None on failure."""
    log_fn("🎙  Generating voiceover...")

    voice_id = profile.voice["voice_id"]
    el_model = profile.voice.get("model", EL_MODEL)
    headers = {"xi-api-key": EL_KEY, "Content-Type": "application/json"}
    voice_settings = {
        "stability":        profile.voice.get("stability", 0.68),
        "similarity_boost": profile.voice.get("similarity_boost", 0.85),
        "style":            profile.voice.get("style", 0.0),
        "use_speaker_boost": profile.voice.get("use_speaker_boost", True),
    }

    chunks = _split_into_chunks(tts_script)
    log_fn(f"  📄  Script split into {len(chunks)} chunk(s)")

    parts = []
    for i, chunk in enumerate(chunks):
        log_fn(f"  ⏳  TTS chunk {i+1}/{len(chunks)}...")
        prev_text = chunks[i - 1] if i > 0 else ""
        next_text = chunks[i + 1] if i < len(chunks) - 1 else ""
        data = _tts_chunk(chunk, headers, voice_settings, log_fn,
                          voice_id=voice_id, model=el_model,
                          prev_text=prev_text, next_text=next_text)
        if data is None:
            log_fn(f"❌  Voiceover failed on chunk {i+1}")
            return None
        parts.append(data)

    def _strip_id3(data: bytes) -> bytes:
        """Remove ID3v2 tag from the start of an MP3 chunk."""
        if data[:3] == b'ID3':
            size = ((data[6] & 0x7f) << 21 | (data[7] & 0x7f) << 14 |
                    (data[8] & 0x7f) << 7  | (data[9] & 0x7f))
            return data[size + 10:]
        return data

    # Keep ID3 header from first chunk only; strip from the rest
    merged = parts[0] + b"".join(_strip_id3(p) for p in parts[1:])

    audio_path = out_dir / "audio" / "voiceover.mp3"
    audio_path.write_bytes(merged)
    log_fn("✅  Voiceover generated")
    return audio_path

# ── Phase 3: QA Review ────────────────────────────────────────────────────────

# ── Audio / Timeline helpers ──────────────────────────────────────────────────

def get_audio_duration(audio_path: Path) -> float:
    """Return audio duration in seconds by scanning MP3 frames.

    ElevenLabs chunks are concatenated after generation, which leaves the Xing
    VBR header (written for only the first chunk) with a stale frame count.
    We therefore scan the actual frame data rather than trusting the header.
    """
    if not audio_path or not audio_path.exists():
        return 720.0
    data = audio_path.read_bytes()
    # Skip ID3v2 tag if present
    offset = 0
    if data[:3] == b'ID3':
        size = ((data[6] & 0x7f) << 21 | (data[7] & 0x7f) << 14 |
                (data[8] & 0x7f) << 7  | (data[9] & 0x7f))
        offset = size + 10

    BITRATES    = [0,32,40,48,56,64,80,96,112,128,160,192,224,256,320]
    SAMPLERATES = [44100, 48000, 32000]

    total_samples = 0
    samplerate    = 44100
    i = offset
    n = len(data)
    while i < n - 3:
        if not (data[i] == 0xFF and (data[i+1] & 0xE0) == 0xE0):
            i += 1
            continue
        hdr    = int.from_bytes(data[i:i+4], 'big')
        br_idx = (hdr >> 12) & 0xF
        sr_idx = (hdr >> 10) & 0x3
        layer  = 4 - ((hdr >> 17) & 0x3)
        padding = (hdr >> 9) & 0x1
        if layer != 3 or br_idx in (0, 15) or sr_idx == 3:
            i += 1
            continue
        bitrate    = BITRATES[br_idx] * 1000
        samplerate = SAMPLERATES[sr_idx]
        frame_size = (144 * bitrate // samplerate) + padding
        if frame_size < 21:
            i += 1
            continue
        # Skip Xing/Info header frame — it contains no audio
        xing_off = i + (36 if ((hdr >> 6) & 0x3) != 3 else 21)
        if data[xing_off:xing_off+4] in (b'Xing', b'Info'):
            i += frame_size
            continue
        total_samples += 1152
        i += frame_size

    if total_samples > 0:
        return total_samples / samplerate
    # Last resort: file size at actual ElevenLabs output bitrate (192 kbps)
    return audio_path.stat().st_size / 24000.0


# ── Palmier MCP ───────────────────────────────────────────────────────────────

def _palmier_call(tool: str, arguments: dict) -> dict:
    """Call a Palmier MCP tool via JSON-RPC."""
    payload = {
        "jsonrpc": "2.0", "id": 1,
        "method": "tools/call",
        "params": {"name": tool, "arguments": arguments},
    }
    r = requests.post(PALMIER_MCP_URL, json=payload, timeout=60)
    r.raise_for_status()
    data = r.json()
    if "error" in data:
        raise RuntimeError(f"Palmier [{tool}]: {data['error']}")
    result = data.get("result", {})
    if result.get("isError"):
        # Extract the error message from the content array
        for item in result.get("content", []):
            if item.get("type") == "text":
                raise RuntimeError(f"Palmier [{tool}]: {item['text']}")
        raise RuntimeError(f"Palmier [{tool}]: unknown error")
    for item in result.get("content", []):
        if item.get("type") == "text":
            try:
                return json.loads(item["text"])
            except (json.JSONDecodeError, ValueError):
                return {"text": item["text"]}
    return result


def _extract_asset_id(result: dict, tool: str) -> str:
    """Extract asset ID from an import_media / generate_* response."""
    for key in ("assetId", "id", "asset_id", "mediaRef"):
        if key in result:
            return result[key]
    # Palmier returns plain-text confirmation — parse the UUID from it
    # e.g. "Imported '001' (id: 2B9E4677-..., type: image) from path."
    text = result.get("text", "")
    m = re.search(r"id:\s*([0-9A-Fa-f\-]{36})", text)
    if m:
        return m.group(1)
    raise RuntimeError(f"Palmier [{tool}]: could not find asset ID in response: {result}")


def _palmier_available() -> bool:
    try:
        requests.post(PALMIER_MCP_URL, json={"jsonrpc": "2.0", "id": 0, "method": "tools/list", "params": {}}, timeout=5)
        return True
    except Exception:
        return False


def assemble_palmier_timeline(
    prompts: list[dict],
    image_results: dict,
    audio_path: Path | None,
    out_dir: Path,
    log_fn,
) -> None:
    """Import all pipeline assets into Palmier and assemble the timeline."""
    log_fn("🎬  Assembling timeline in Palmier...")

    timeline_info = _palmier_call("get_timeline", {})
    project_fps   = timeline_info.get("fps", 30)

    valid_nums = [p["num"] for p in prompts if p["num"] in image_results]
    if not valid_nums:
        log_fn("⚠️  No images to place — skipping Palmier assembly")
        return

    total_duration   = get_audio_duration(audio_path) if audio_path else None
    base_dur_s       = (total_duration / len(valid_nums)) if total_duration else 2.5
    base_dur_frames  = max(1, round(base_dur_s * project_fps))
    scale            = _timestamp_scale(prompts, total_duration) if total_duration else 1.0

    log_fn(f"  ⏱  Audio duration: {total_duration:.1f}s — timestamp scale: {scale:.3f}")

    prompt_by_num = {p["num"]: p for p in prompts}

    # Import images
    log_fn(f"  Importing {len(valid_nums)} images into Palmier...")
    media_refs = {}
    for num in valid_nums:
        r = _palmier_call("import_media", {
            "source": {"path": str(image_results[num].resolve())},
            "name": num,
        })
        media_refs[num] = _extract_asset_id(r, "import_media")

    # Import audio
    audio_ref = None
    if audio_path and audio_path.exists():
        log_fn("  Importing voiceover...")
        r = _palmier_call("import_media", {
            "source": {"path": str(audio_path.resolve())},
            "name": "voiceover",
        })
        audio_ref = _extract_asset_id(r, "import_media")

    # Place image clips using per-prompt timestamps
    log_fn("  Placing clips on timeline...")
    entries = []
    start_frame = 0
    for num in valid_nums:
        clip_dur_s      = _prompt_duration(prompt_by_num[num], base_dur_s, scale)
        clip_dur_frames = max(1, round(clip_dur_s * project_fps))
        entries.append({
            "mediaRef":       media_refs[num],
            "startFrame":     start_frame,
            "durationFrames": clip_dur_frames,
        })
        start_frame += clip_dur_frames
    _palmier_call("add_clips", {"entries": entries})

    if audio_ref:
        log_fn("  Placing voiceover on timeline...")
        _palmier_call("add_clips", {"entries": [{"mediaRef": audio_ref, "startFrame": 0, "durationFrames": start_frame}]})

    log_fn("✅  Palmier timeline assembled")


def _ts_to_seconds(ts: str) -> float | None:
    """Parse MM:SS or HH:MM:SS timestamp string to seconds. Returns None if unparseable."""
    try:
        parts = [int(x) for x in ts.strip().split(":")]
        if len(parts) == 2:
            return parts[0] * 60 + parts[1]
        if len(parts) == 3:
            return parts[0] * 3600 + parts[1] * 60 + parts[2]
    except Exception:
        pass
    return None


def _prompt_duration(prompt: dict, fallback: float, scale: float = 1.0) -> float:
    """Return clip duration in seconds from a prompt's ts field scaled to actual audio, or fallback."""
    ts = prompt.get("ts", "")
    if "-" in ts:
        start_str, end_str = ts.split("-", 1)
        start = _ts_to_seconds(start_str)
        end   = _ts_to_seconds(end_str)
        if start is not None and end is not None and end > start:
            return (end - start) * scale
    return fallback


def _timestamp_scale(prompts: list[dict], actual_duration: float) -> float:
    """Compute scale factor = actual audio duration / last timestamp end in prompts."""
    last_end = 0.0
    for p in prompts:
        ts = p.get("ts", "")
        if "-" in ts:
            _, end_str = ts.split("-", 1)
            end = _ts_to_seconds(end_str)
            if end is not None and end > last_end:
                last_end = end
    if last_end > 0 and actual_duration > 0:
        return actual_duration / last_end
    return 1.0


# ── Agent: vidIQ Research + Script Writing ────────────────────────────────────

VIDIQ_MCP_URL = "https://mcp.vidiq.com/mcp"

async def _run_script_agent(topic: str, profile: "Profile", log_fn) -> str:
    """Run Claude agent with vidIQ MCP to research topic and write script. Returns script text."""
    system_prompt = _build_agent_system_prompt(topic, profile)
    agent_script_prompt = _build_agent_script_prompt(profile)

    options = ClaudeAgentOptions(
        system_prompt=system_prompt,
        mcp_servers={
            "vidiq": {
                "type": "http",
                "url": VIDIQ_MCP_URL,
                "headers": {"Authorization": f"Bearer {VIDIQ_KEY}"},
            }
        },
        permission_mode="bypassPermissions",
        max_turns=30,
    )

    full_text = ""
    async for message in agent_query(
        prompt=f"Topic: {topic}\n\n{agent_script_prompt}",
        options=options,
    ):
        if isinstance(message, AssistantMessage):
            for block in message.content:
                if isinstance(block, TextBlock) and block.text.strip():
                    log_fn(f"  {block.text[:120].strip()}")
                    full_text += block.text
        elif isinstance(message, ResultMessage):
            if message.result:
                full_text += message.result

    return _extract("SCRIPT", full_text)


def run_script_agent(topic: str, profile: "Profile", log_fn) -> str:
    """Sync wrapper around the async agent. Runs in a new event loop."""
    return asyncio.run(_run_script_agent(topic, profile, log_fn))


async def _run_vet_agent(topic: str, script: str, profile: "Profile", log_fn) -> str:
    """Run a vidIQ-backed vetting pass on a generated script. Returns revised script text."""
    system_prompt = _build_agent_system_prompt(topic, profile) + f"\n\nCURRENT SCRIPT TO VET:\n{script}"
    vet_prompt = _build_vet_prompt(profile)

    options = ClaudeAgentOptions(
        system_prompt=system_prompt,
        mcp_servers={
            "vidiq": {
                "type": "http",
                "url": VIDIQ_MCP_URL,
                "headers": {"Authorization": f"Bearer {VIDIQ_KEY}"},
            }
        },
        permission_mode="bypassPermissions",
        max_turns=20,
    )

    full_text = ""
    async for message in agent_query(
        prompt=f"Topic: {topic}\n\n{vet_prompt}",
        options=options,
    ):
        if isinstance(message, AssistantMessage):
            for block in message.content:
                if isinstance(block, TextBlock) and block.text.strip():
                    log_fn(f"  {block.text[:120].strip()}")
                    full_text += block.text
        elif isinstance(message, ResultMessage):
            if message.result:
                full_text += message.result

    return _extract("SCRIPT", full_text)


def run_vet_agent(topic: str, script: str, profile: "Profile", log_fn) -> str:
    """Sync wrapper around the async vet agent."""
    return asyncio.run(_run_vet_agent(topic, script, profile, log_fn))


# ── Shared production phase ───────────────────────────────────────────────────

def _run_production(topic: str, prompts: list[dict], tts_script: str,
                    profile: "Profile", out_dir: Path, client: anthropic.Anthropic,
                    log_fn, stop_event=None, skip_existing_images=False) -> dict:
    """
    Phases 2-4: images + voiceover + QA + FCPXML.
    Called by both run_pipeline and resume_pipeline.
    """
    audio_path = None

    def voiceover_thread():
        nonlocal audio_path
        existing_mp3 = out_dir / "audio" / "voiceover.mp3"
        if existing_mp3.exists():
            log_fn("🎙  Voiceover already exists — skipping")
            audio_path = existing_mp3
            return
        if not tts_script:
            log_fn("⚠️  No TTS script available — skipping voiceover")
            return
        audio_path = generate_voiceover(tts_script, out_dir, profile, log_fn)

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

    log_fn("ℹ️  Review images and audio, then use ⬡ Palmier in the UI to send to Palmier Pro.")

    summary = {
        "status":  "complete",
        "topic":   topic,
        "out_dir": str(out_dir),
        "images":  len(image_results),
        "audio":   str(audio_path) if audio_path else "failed",
    }

    log_fn(f"""
╔══════════════════════════════════════════╗
║           Pipeline Complete ✅           ║
╠══════════════════════════════════════════╣
║  Images generated : {summary['images']:<21}║
║  Audio            : {'✅' if audio_path else '❌':<21}║
║  Output folder    :                      ║
║  {str(out_dir)[-40:]:<40}  ║
╚══════════════════════════════════════════╝
""")
    return summary


# ── Resume ────────────────────────────────────────────────────────────────────

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
        prompts = parse_image_prompts(prompts_file.read_text())
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


def resume_pipeline(run_slug: str, profile: "Profile", progress_callback=None, stop_event=None) -> dict:
    """
    Resume a stopped pipeline run. Detects what is missing and regenerates only
    what is needed: image_prompts, tts_script, images, voiceover.
    """
    def log_fn(msg):
        log(msg, progress_callback)

    if not check_keys(profile, log_fn):
        return {"status": "error", "reason": "missing API keys"}

    out_dir = OUTPUT_ROOT / run_slug
    if not out_dir.exists():
        log_fn(f"❌  Output folder not found: {out_dir}")
        return {"status": "error", "reason": "run folder not found"}

    script_file  = out_dir / "script.txt"
    prompts_file = out_dir / "image_prompts.txt"
    tts_file     = out_dir / "tts_script.txt"

    if not script_file.exists():
        log_fn("❌  script.txt missing — cannot resume without a script")
        return {"status": "error", "reason": "script.txt not found"}

    log_fn(f"\n▶️  Resuming pipeline: {run_slug}")

    client = anthropic.Anthropic(api_key=ANTHROPIC_KEY)
    script = script_file.read_text()

    # Regenerate image_prompts + tts_script together if either is missing
    needs_prompts = not prompts_file.exists()
    needs_tts     = not tts_file.exists()

    if needs_prompts or needs_tts:
        log_fn("📝  Regenerating missing assets from script.txt:")
        if needs_prompts: log_fn("     • image_prompts.txt")
        if needs_tts:     log_fn("     • tts_script.txt")
        tts_script, image_prompts_raw = _generate_tts_and_prompts(script, profile, client, log_fn)
        if needs_tts:
            tts_file.write_text(tts_script)
            log_fn("✅  tts_script.txt saved")
        if needs_prompts:
            prompts_file.write_text(image_prompts_raw)
            log_fn("✅  image_prompts.txt saved")
    else:
        tts_script = tts_file.read_text()

    prompts = parse_image_prompts(prompts_file.read_text())
    log_fn(f"📝  {len(prompts)} image prompts loaded")

    images_on_disk = sum(1 for p in prompts if _find_image(out_dir / "images", p["num"]))
    log_fn(f"🖼   {images_on_disk}/{len(prompts)} images already on disk — skipping those")

    topic = script[:80]

    return _run_production(
        topic, prompts, tts_script, profile, out_dir, client,
        log_fn, stop_event, skip_existing_images=True,
    )


# ── Main Orchestrator ─────────────────────────────────────────────────────────

def run_pipeline(topic: str, profile: "Profile", progress_callback=None,
                 stop_event=None, approval_callback=None) -> dict:
    """
    Run the full pipeline for a given topic.

    progress_callback:  callable(str) — live log updates
    stop_event:         threading.Event — set to cancel mid-run
    approval_callback:  callable(script: str) -> bool
                        Called after script is written, before any paid API calls.
                        Return True to proceed, False to abort.
                        If None, CLI input() is used instead.
    """
    def log_fn(msg):
        log(msg, progress_callback)

    log_fn(f"\n🚀  Starting pipeline for: {topic}\n")

    if not check_keys(profile, log_fn):
        return {"status": "error", "reason": "missing API keys"}

    client  = anthropic.Anthropic(api_key=ANTHROPIC_KEY)
    slug    = slugify(topic)
    out_dir = make_output_dir(slug)
    log_fn(f"📁  Output directory: {out_dir}")

    if VIDIQ_KEY:
        log_fn("🤖  Running vidIQ research + script agent...")
        script = run_script_agent(topic, profile, log_fn)
        (out_dir / "research.txt").write_text("(handled by agent — see script.txt)")
        if not script:
            log_fn("❌  Agent did not produce a script — aborting")
            return {"status": "error", "reason": "agent produced no script"}
    else:
        log_fn("🔬  No vidIQ key — running standard research and script phases...")
        research = research_topic(topic, client, log_fn)
        (out_dir / "research.txt").write_text(research)
        script = generate_script(topic, research, profile, client, log_fn)

    (out_dir / "script.txt").write_text(script)
    log_fn("📝  Script written")

    if VIDIQ_KEY:
        log_fn("🔎  Running vidIQ script vet...")
        vetted = run_vet_agent(topic, script, profile, log_fn)
        if vetted:
            script = vetted
            (out_dir / "script.txt").write_text(script)
            log_fn("✅  Script vetted and updated")
        else:
            log_fn("⚠️  Vet agent returned no output — proceeding with original script")

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

    script = (out_dir / "script.txt").read_text()
    tts_script, image_prompts_raw = _generate_tts_and_prompts(script, profile, client, log_fn)

    prompts = parse_image_prompts(image_prompts_raw)
    log_fn(f"📝  {len(prompts)} image prompts parsed")

    (out_dir / "tts_script.txt").write_text(tts_script)
    (out_dir / "image_prompts.txt").write_text(
        "\n".join(f"{p['num']} | {p['ts']} | {p['prompt']}" for p in prompts)
    )

    if stop_event and stop_event.is_set():
        return {"status": "cancelled", "out_dir": str(out_dir)}

    return _run_production(topic, prompts, tts_script, profile, out_dir, client, log_fn, stop_event)


# ── CLI entry point ───────────────────────────────────────────────────────────

if __name__ == "__main__":
    import sys
    if len(sys.argv) < 2:
        print("Usage: python pipeline.py \"your topic here\"")
        sys.exit(1)
    topic = " ".join(sys.argv[1:])
    run_pipeline(topic)
