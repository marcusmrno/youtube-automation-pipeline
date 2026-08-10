"""
YouTube Pipeline — Core Orchestrator
Generates all video assets from a single topic string.
Usage: python pipeline.py "your topic here"
       or import run_pipeline() and call it from ui.py
"""
from __future__ import annotations

import functools
import json
import mimetypes
import os
import re
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import anthropic
import requests
from dotenv import load_dotenv
from google import genai
from google.genai import types as genai_types
from PIL import Image, ImageOps

from agents import run_script_agent, run_vet_agent
from prompts import (
    _build_clarifying_questions_prompt,
    _build_approach_pitch_prompt,
    _build_script_prompt,
    _build_tts_prompt,
    _build_image_prompt_instructions,
    _build_image_prompt_vet_instructions,
    _build_image_prompt_rewrite_instructions,
    _build_agent_system_prompt,
    _extract,
)

load_dotenv()

# ── Config ────────────────────────────────────────────────────────────────────

PROJECT_ROOT    = Path(__file__).parent
OUTPUT_ROOT     = PROJECT_ROOT / "output"

ANTHROPIC_KEY   = (os.getenv("ANTHROPIC_API_KEY") or "").strip()
EL_KEY          = (os.getenv("ELEVENLABS_API_KEY") or "").strip()
VIDIQ_KEY       = (os.getenv("VIDIQ_API_KEY") or "").strip()
GOOGLE_KEY      = (os.getenv("GOOGLE_API_KEY") or "").strip()

CLAUDE_MODEL  = "claude-sonnet-4-6"
HAIKU_MODEL   = "claude-haiku-4-5-20251001"
EL_MODEL      = "eleven_v3"
# Image models come from the profile (image_gen.default_model / pro_model).



# ── Helpers ───────────────────────────────────────────────────────────────────

@functools.cache
def _get_genai_client() -> "genai.Client":
    return genai.Client(api_key=GOOGLE_KEY)


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


# ── Phase 0: Research & Fact-Check ───────────────────────────────────────────

def research_topic(topic: str, client: anthropic.Anthropic, log_fn) -> str:
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
        model=HAIKU_MODEL,
        max_tokens=3000,
        messages=[{"role": "user", "content": prompt}]
    )
    research = response.content[0].text
    log_fn("✅  Research complete")
    return research


# ── Phase 1: Writing ──────────────────────────────────────────────────────────

def generate_clarifying_questions(topic: str, profile: "Profile",
                                  client: anthropic.Anthropic, log_fn) -> str:
    """Generate clarifying questions about the video topic."""
    prompt = _build_clarifying_questions_prompt(topic, profile)
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
    prompt = _build_approach_pitch_prompt(topic, answers, profile)
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
    prompt = _build_script_prompt(topic, research, profile, approach_context)
    log_fn("✍️  Writing script...")
    r = client.messages.create(
        model=CLAUDE_MODEL,
        max_tokens=8000,
        messages=[{"role": "user", "content": prompt}]
    )
    script = _extract("SCRIPT", r.content[0].text)
    log_fn("✅  Script written")
    return script


def revise_script(script: str, feedback: str, topic: str, profile: "Profile | None",
                  client: anthropic.Anthropic) -> str:
    """Revise a script based on feedback via Sonnet. Returns the revised script."""
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
    text = response.content[0].text
    return _extract("SCRIPT", text) or text.strip()


def parse_image_prompts(raw: str) -> list[dict]:
    """Parse NNN | source line | prompt lines into list of dicts, deduplicated by number."""
    seen: dict[str, dict] = {}
    for line in raw.strip().splitlines():
        line = line.strip()
        if not line:
            continue
        # Source lines must not contain '|' — the delimiter; narration sentences don't in practice
        parts = line.split("|", 2)
        if len(parts) == 3:
            num    = parts[0].strip().zfill(3)
            source = parts[1].strip()
            prompt = parts[2].strip()
            seen[num] = {"num": num, "source": source, "prompt": prompt}
    return [seen[k] for k in sorted(seen)]


def _generate_tts_and_prompts(script: str, profile: "Profile", client: anthropic.Anthropic, log_fn) -> tuple[str, str]:
    """Given a finished script, generate TTS narration and image prompts. Returns (tts_script, image_prompts_raw)."""

    log_fn("✍️  Extracting TTS narration from script...")
    tts_prompt = _build_tts_prompt(script, profile)
    r1 = client.messages.create(
        model=HAIKU_MODEL,
        max_tokens=8000,
        messages=[{"role": "user", "content": tts_prompt}]
    )
    tts_script = _extract("TTS_SCRIPT", r1.content[0].text)
    log_fn("✅  TTS narration extracted")

    log_fn("🖼️  Generating image prompts...")
    base_instructions = _build_image_prompt_instructions(profile)

    # Split script in half by section boundaries so each batch has full section context
    script_lines   = script.strip().splitlines()
    section_starts = [i for i, l in enumerate(script_lines) if re.match(r'^\[[\d:–\-]+\]', l.strip())]

    if len(section_starts) >= 2:
        mid = section_starts[len(section_starts) // 2]
        batches = [
            ("\n".join(script_lines[:mid]),  "first half"),
            ("\n".join(script_lines[mid:]),  "second half"),
        ]
    else:
        batches = [(script, "full script")]

    all_prompt_lines: list[str] = []
    last_prompt_num  = 0

    for batch_idx, (batch_script, batch_label) in enumerate(batches):
        start_num  = last_prompt_num + 1
        numbering  = (
            "Start numbering from 001."
            if batch_idx == 0
            else f"Continue numbering from {start_num:03d}. Do NOT restart from 001 — the previous batch ended at {last_prompt_num:03d}."
        )
        batch_msg = (
            base_instructions
            + f"\nSCRIPT SEGMENT ({batch_label}):\n{batch_script}\n\n"
            f"NUMBERING: {numbering}\n"
            f"Timestamps are 3–4 seconds each, never more than 5. Verify timestamps are contiguous — end of prompt N = start of prompt N+1, no gaps."
        )
        log_fn(f"  Generating image prompts ({batch_label})...")
        with client.messages.stream(
            model=CLAUDE_MODEL,
            max_tokens=64000,
            messages=[{"role": "user", "content": batch_msg}]
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
    deduped_lines   = [seen[k] for k in sorted(seen)]
    last_prompt_num = max(seen) if seen else 0

    image_prompts = "\n".join(deduped_lines)
    log_fn(f"✅  Image prompts generated — {last_prompt_num} total")
    return tts_script, image_prompts


def _vet_image_prompts(
    prompts: list[dict],
    profile: "Profile",
    client: anthropic.Anthropic,
    log_fn,
) -> list[dict]:
    """Two-stage vetting: Haiku detects problems, Sonnet rewrites flagged prompts."""
    prompt_lines = "\n".join(
        f"{p['num']} | {p['source']} | {p['prompt']}" for p in prompts
    )

    # Stage 1 — detection (Haiku)
    vet_instructions = _build_image_prompt_vet_instructions(profile)
    log_fn("🔎  Vetting image prompts (Stage 1: detection)...")
    try:
        r1 = client.messages.create(
            model=HAIKU_MODEL,
            max_tokens=4096,
            messages=[{"role": "user", "content": vet_instructions + "\n\n" + prompt_lines}]
        )
        raw_flags = r1.content[0].text.strip()
        flagged: list[dict] = json.loads(raw_flags)
    except Exception as e:
        log_fn(f"⚠️  Vetting Stage 1 failed ({e}) — using original prompts")
        return prompts

    if not flagged:
        log_fn(f"✅  All {len(prompts)} image prompts passed vetting")
        return prompts

    log_fn(f"  ✏️  {len(flagged)} prompts flagged — rewriting (Stage 2: Sonnet)...")

    # Build flagged subset for Stage 2
    prompt_by_num = {p["num"]: p for p in prompts}
    flagged_lines = []
    for f in flagged:
        num = str(f.get("num", "")).zfill(3)
        if not num.strip("0") or num not in prompt_by_num:
            continue
        reason = f.get("reason", "")
        detail = f.get("detail", reason)
        p = prompt_by_num[num]
        flagged_lines.append(
            f"{p['num']} | {p['source']} | {p['prompt']} | REASON: {reason} — {detail}"
        )

    if not flagged_lines:
        return prompts

    # Stage 2 — rewrite (Sonnet)
    rewrite_instructions = _build_image_prompt_rewrite_instructions(profile)
    try:
        r2 = client.messages.create(
            model=CLAUDE_MODEL,
            max_tokens=8192,
            messages=[{"role": "user", "content": rewrite_instructions + "\n\n" + "\n".join(flagged_lines)}]
        )
        rewritten_raw = r2.content[0].text.strip()
        rewritten = parse_image_prompts(rewritten_raw)
    except Exception as e:
        log_fn(f"⚠️  Vetting Stage 2 failed ({e}) — using original prompts")
        return prompts

    if not rewritten:
        log_fn("⚠️  Vetting Stage 2 returned no prompts — using original prompts")
        return prompts

    # Merge rewrites back into original list
    rewritten_by_num = {p["num"]: p for p in rewritten}
    result = []
    for p in prompts:
        result.append(rewritten_by_num.get(p["num"], p))

    log_fn(f"✅  {len(rewritten)} prompts rewritten by vetting")
    return result


# ── Phase 2: Image Generation ─────────────────────────────────────────────────

def _standardize_image(path: Path, size: tuple[int, int] = (1920, 1080)) -> None:
    """Center-crop to `size`'s aspect ratio and resize to `size` in place."""
    ImageOps.fit(Image.open(path).convert("RGB"), size, Image.LANCZOS).save(path)


def _stretch_horizontal(img: Image.Image, pct: float) -> Image.Image:
    w, h = img.size
    new_w = int(w * (1 + pct))
    stretched = img.resize((new_w, h), Image.LANCZOS)
    left = (new_w - w) // 2
    return stretched.crop((left, 0, left + w, h))


def _stretch_vertical(img: Image.Image, pct: float) -> Image.Image:
    w, h = img.size
    new_h = int(h * (1 + pct))
    stretched = img.resize((w, new_h), Image.LANCZOS)
    top = (new_h - h) // 2
    return stretched.crop((0, top, w, top + h))


def generate_flicker_frames(source_path: Path, out_dir: Path, num: str, magnitude: float, log_fn) -> None:
    """Generate b (horiz stretch) and c (vert stretch) flicker frames in out_dir/flicker/."""
    flicker_dir = out_dir / "flicker"
    flicker_dir.mkdir(exist_ok=True)
    img = Image.open(source_path).convert("RGB")
    b_path = flicker_dir / f"{num}b.png"
    c_path = flicker_dir / f"{num}c.png"
    _stretch_horizontal(img, magnitude).save(b_path)
    _stretch_vertical(img, magnitude).save(c_path)
    log_fn(f"  🎞️  Flicker frames saved: flicker/{b_path.name}, flicker/{c_path.name}")


def _find_image(img_dir: Path, num: str) -> Path | None:
    for ext in (".png", ".jpg", ".jpeg"):
        p = img_dir / f"{num}{ext}"
        if p.exists():
            return p
    return None


IMAGE_PREAMBLE = (
    "Reference images are provided in this order:\n"
    "  anchor-01: Full cast reference sheet — all characters side by side, exact proportions and scale.\n"
    "  anchor-02: Multi-angle reference sheet — each character shown front, 3/4, side, and back. Match these character designs exactly in every scene.\n"
    "  anchor-03: Example content scene — shows background, props, and art style in a real frame.\n"
    "  anchor-04 onward: Additional style and scene references — follow the art style shown precisely.\n\n"
    "STYLE CONSTRAINTS: Bold uneven black marker outlines. Color fills bleed outside lines with visible marker streaks. "
    "Off-white warm paper background. No gradients. No drop shadows. No clean fonts. No photorealistic textures. No smooth digital lines.\n\n"
)

ANCHOR_PREAMBLE = (
    "Reference images define the art style and characters — match them precisely.\n\n"
    "STYLE CONSTRAINTS: Bold uneven black marker outlines. Color fills bleed outside lines "
    "with visible marker streaks. Off-white warm paper background. No gradients. No drop shadows. "
    "No clean fonts. No photorealistic textures. No smooth digital lines.\n\n"
)


def _load_anchors_from_dir(anchors_dir: Path, max_anchors: int) -> list:
    """Pre-load anchor images from a directory as genai Parts."""
    anchor_files = sorted(anchors_dir.glob("anchor-*.png")) + sorted(anchors_dir.glob("anchor-*.jpg"))
    seen = set()
    reference_files = []
    for f in sorted(anchor_files, key=lambda p: p.stem):
        if f.stem not in seen:
            seen.add(f.stem)
            reference_files.append(f)
    reference_files = reference_files[:max_anchors]
    return [
        genai_types.Part.from_bytes(data=ref.read_bytes(), mime_type=mimetypes.guess_type(ref.name)[0] or "image/png")
        for ref in reference_files
    ]


def _load_anchor_parts(profile: "Profile") -> list:
    """Pre-load anchor images as genai Parts. Call once per run, not per image."""
    return _load_anchors_from_dir(profile.anchors_dir, profile.image_style.get("max_anchors", 14))


def generate_image_google(prompt: str, output_path: Path, profile: "Profile", log_fn,
                          model: str = None, anchor_parts: list | None = None,
                          preamble: str = IMAGE_PREAMBLE,
                          standardize_size: tuple[int, int] = (1920, 1080)) -> bool:
    """Generate image via Google AI with style anchors as references."""
    if model is None:
        model = profile.image_gen["default_model"]

    if anchor_parts is None:
        anchor_parts = _load_anchor_parts(profile)

    contents = [f"{preamble}{prompt}"]
    contents.extend(anchor_parts)

    client = _get_genai_client()
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
                    ext        = ".jpg" if "jpeg" in part.inline_data.mime_type else ".png"
                    final_path = output_path.with_suffix(ext)
                    final_path.write_bytes(part.inline_data.data)
                    if final_path != output_path and output_path.exists():
                        output_path.unlink()
                    _standardize_image(final_path, standardize_size)
                    return True
            log_fn(f"  ⚠️  Google AI returned no image in response")
            time.sleep(3)
        except Exception as e:
            log_fn(f"  ⚠️  Google AI attempt {attempt+1} error: {e}")
            time.sleep(3)
    log_fn(f"  ❌  Image {output_path.name} failed after 3 attempts — skipping")
    return False


def generate_all_images(prompts: list[dict], out_dir: Path,
                        profile: "Profile", log_fn, stop_event=None, skip_existing=False,
                        max_workers: int = 4) -> dict:
    """Generate images concurrently (rate-limited by max_workers). Returns {num: path} for successful images."""
    anchor_parts = _load_anchor_parts(profile)
    results = {}
    total = len(prompts)

    def _one(i: int, p: dict):
        num      = p["num"]
        img_path = out_dir / "images" / f"{num}.png"
        log_fn(f"🖼  Generating image {i+1}/{total} ({num})")
        ok = generate_image_google(p["prompt"], img_path, profile, log_fn, anchor_parts=anchor_parts)
        if not ok:
            log_fn(f"  ❌ Image {num} failed after 3 attempts — flagged")
            return num, None
        found = _find_image(out_dir / "images", num) or img_path
        flicker_cfg = profile.image_gen.get("flicker", {})
        if flicker_cfg.get("enabled"):
            magnitude = flicker_cfg.get("magnitude", 0.004)
            generate_flicker_frames(found, out_dir / "images", num, magnitude, log_fn)
        return num, found

    with ThreadPoolExecutor(max_workers=max_workers) as ex:
        futures = {}
        for i, p in enumerate(prompts):
            if stop_event and stop_event.is_set():
                break
            num = p["num"]
            existing = _find_image(out_dir / "images", num)
            if skip_existing and existing:
                results[num] = existing
                continue
            futures[ex.submit(_one, i, p)] = num

        for fut in as_completed(futures):
            num, found = fut.result()
            if found:
                results[num] = found

    return results


def regenerate_images(run_slug: str, image_nums: list[str], model_key: str,
                      profile: "Profile", progress_callback=None) -> dict:
    """Manually regenerate specific images for a completed run."""
    def log_fn(msg):
        log(msg, progress_callback)

    model = profile.image_gen.get("pro_model" if model_key == "3-pro" else "default_model")
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


# ── Phase 2b: Voiceover ────────────────────────────────────────────────────────

def _split_into_chunks(text: str, max_chars: int = 4500) -> list[str]:
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
               voice_id: str, model: str,
               prev_text: str = "", next_text: str = "") -> bytes | None:
    """Send one chunk to ElevenLabs. Returns raw mp3 bytes or None on failure."""
    url = f"https://api.elevenlabs.io/v1/text-to-speech/{voice_id}"
    payload = {
        "text": text,
        "model_id": model,
        "voice_settings": voice_settings,
    }
    if model != "eleven_v3":
        if prev_text:
            payload["previous_text"] = prev_text
        if next_text:
            payload["next_text"] = next_text

    for attempt in range(3):
        try:
            resp = requests.post(url, headers=headers, params={"output_format": "mp3_44100_192"},
                                 json=payload, timeout=120)
            if not resp.ok:
                log_fn(f"  ⚠️  TTS attempt {attempt+1} failed: {resp.status_code} — {resp.text[:300]}")
                time.sleep(2)
                continue
            return resp.content
        except Exception as e:
            log_fn(f"  ⚠️  TTS attempt {attempt+1} error: {e}")
            time.sleep(2)
    return None


def _id3_skip_offset(data) -> int:
    """Return byte offset of the first MP3 frame, skipping the ID3v2 header if present."""
    if data[:3] == b'ID3':
        return ((data[6] & 0x7f) << 21 | (data[7] & 0x7f) << 14 |
                (data[8] & 0x7f) << 7  | (data[9] & 0x7f)) + 10
    return 0


def _strip_id3(data: bytes) -> bytes:
    """Strip the ID3v2 header from the start of MP3 data."""
    offset = _id3_skip_offset(data)
    return data[offset:] if offset else data


def _fix_mp3_duration(path: Path, log_fn) -> None:
    """Null out the Xing/Info VBR header so players use file-size/bitrate for duration."""
    data   = bytearray(path.read_bytes())
    offset = _id3_skip_offset(bytes(data))
    for marker in [b'Xing', b'Info']:
        pos = data[offset:offset + 4096].find(marker)
        if pos != -1:
            data[offset + pos:offset + pos + 4] = b'    '
            log_fn(f"  🔧  Fixed MP3 duration header ({marker.decode()} marker removed)")
    path.write_bytes(data)


def generate_voiceover(tts_script: str, out_dir: Path, profile: "Profile", log_fn) -> Path | None:
    """Call ElevenLabs TTS API in chunks. Returns path to mp3 or None on failure."""
    log_fn("🎙  Generating voiceover...")

    voice_id = profile.voice["voice_id"]
    el_model = profile.voice.get("model", EL_MODEL)
    headers  = {"xi-api-key": EL_KEY, "Content-Type": "application/json"}
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
        data = _tts_chunk(chunk, headers, voice_settings, log_fn,
                          voice_id=voice_id, model=el_model,
                          prev_text=chunks[i - 1] if i > 0 else "",
                          next_text=chunks[i + 1] if i < len(chunks) - 1 else "")
        if data is None:
            log_fn(f"❌  Voiceover failed on chunk {i+1}")
            return None
        parts.append(data)

    # Keep ID3 header from first chunk only; strip from the rest
    merged = parts[0] + b"".join(_strip_id3(p) for p in parts[1:])

    audio_path = out_dir / "audio" / "voiceover.mp3"
    audio_path.write_bytes(merged)
    _fix_mp3_duration(audio_path, log_fn)
    log_fn("✅  Voiceover generated")
    return audio_path


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
    from profile import load_profile as _load_profile

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
            fresh_prompts = parse_image_prompts(image_prompts_raw)
            fresh_prompts = _vet_image_prompts(fresh_prompts, profile, client, log_fn)
            prompts_file.write_text(
                "\n".join(f"{p['num']} | {p['source']} | {p['prompt']}" for p in fresh_prompts)
            )
            log_fn("✅  image_prompts.txt saved")
    else:
        tts_script = tts_file.read_text()

    prompts = parse_image_prompts(prompts_file.read_text())
    log_fn(f"📝  {len(prompts)} image prompts loaded")

    images_on_disk = sum(1 for p in prompts if _find_image(out_dir / "images", p["num"]))
    log_fn(f"🖼   {images_on_disk}/{len(prompts)} images already on disk — skipping those")

    topic = script[:80]
    return _run_production(
        topic, prompts, tts_script, profile, out_dir,
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
    out_dir = make_output_dir(slug)
    (out_dir / "profile.txt").write_text(profile.name)
    log_fn(f"📁  Output directory: {out_dir}")

    if VIDIQ_KEY:
        log_fn("🤖  Running vidIQ research + script agent...")
        script, research_notes = run_script_agent(topic, profile, log_fn, approach_context)
        if research_notes:
            (out_dir / "research.txt").write_text(research_notes)
        if not script:
            log_fn("❌  Agent did not produce a script — aborting")
            return {"status": "error", "reason": "agent produced no script"}
    else:
        log_fn("🔬  No vidIQ key — running standard research and script phases...")
        research = research_topic(topic, client, log_fn)
        (out_dir / "research.txt").write_text(research)
        script = generate_script(topic, research, profile, client, log_fn, approach_context)

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

    prompts = _vet_image_prompts(prompts, profile, client, log_fn)

    (out_dir / "tts_script.txt").write_text(tts_script)
    (out_dir / "image_prompts.txt").write_text(
        "\n".join(f"{p['num']} | {p['source']} | {p['prompt']}" for p in prompts)
    )

    if stop_event and stop_event.is_set():
        return {"status": "cancelled", "out_dir": str(out_dir)}

    return _run_production(topic, prompts, tts_script, profile, out_dir, log_fn, stop_event)


# ── CLI entry point ───────────────────────────────────────────────────────────

if __name__ == "__main__":
    import sys
    import argparse
    from profile import load_profile, list_profiles

    parser = argparse.ArgumentParser(description="YouTube Pipeline")
    sub = parser.add_subparsers(dest="cmd")

    # Default command: run a topic (preserves existing behavior)
    run_p = sub.add_parser("run", help="Run the full pipeline for a topic")
    run_p.add_argument("topic", nargs="+")
    run_p.add_argument("--profile", default=None)

    # Metadata subcommand
    md = sub.add_parser("metadata", help="Generate or update metadata + thumbnails")
    md.add_argument("run_slug")
    md.add_argument("--profile", default=None)
    md.add_argument("--regenerate", action="store_true")
    md.add_argument("--pick-thumb", type=int, metavar="N", help="1-based index")
    md.add_argument("--show", action="store_true")

    # Back-compat: if first arg isn't a known subcommand, treat as run
    argv = sys.argv[1:]
    if argv and argv[0] not in {"run", "metadata"}:
        argv = ["run"] + argv

    args = parser.parse_args(argv)

    def _resolve_profile(name: str | None):
        available = list_profiles()
        if not available:
            print("No profiles found. Create profiles/<name>/profile.yaml first.")
            sys.exit(1)
        if name:
            return load_profile(name)
        if len(available) == 1:
            print(f"Using profile: {available[0]}")
            return load_profile(available[0])
        print("Multiple profiles found. Specify one with --profile:")
        for p in available:
            print(f"  {p}")
        sys.exit(1)

    if args.cmd == "metadata":
        import metadata as md_mod

        log_fn = lambda m: print(m)

        if args.show:
            data = md_mod.load_metadata(args.run_slug)
            if data is None:
                print(f"No metadata for run '{args.run_slug}'")
                sys.exit(1)
            print(json.dumps(data, indent=2, ensure_ascii=False))
            sys.exit(0)

        if args.pick_thumb is not None:
            data = md_mod.pick_thumbnail(args.run_slug, args.pick_thumb - 1)
            print(f"✅ chosen_thumbnail_index = {data['chosen_thumbnail_index']}")
            sys.exit(0)

        # Full generate
        existing = md_mod.load_metadata(args.run_slug)
        if existing is not None and not args.regenerate:
            answer = input("metadata exists, overwrite? (y/N): ").strip().lower()
            if answer != "y":
                print("aborted")
                sys.exit(0)
        profile = _resolve_profile(args.profile)
        md_mod.generate_metadata(args.run_slug, profile, log_fn, regenerate=True)
        sys.exit(0)

    # args.cmd == "run"
    topic = " ".join(args.topic)
    profile = _resolve_profile(args.profile)
    run_pipeline(topic, profile)
