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
import threading
import time
from pathlib import Path

import anthropic
import requests
from dotenv import load_dotenv
from google import genai
from google.genai import types as genai_types
from PIL import Image

from agents import run_script_agent, run_vet_agent
from prompts import (
    _build_agent_system_prompt,   # re-exported for bot.py
    _build_script_prompt,
    _build_tts_prompt,
    _build_image_prompt_instructions,
    _extract,
)

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

CLAUDE_MODEL  = "claude-sonnet-4-6"
EL_MODEL      = "eleven_v3"
GOOGLE_MODEL  = "gemini-3.1-flash-image"   # bulk generation + regen
GOOGLE_PRO_MODEL = "gemini-3-pro-image"    # highest quality, slowest

GOOGLE_MODEL_OPTIONS = {
    "2.5-flash":     GOOGLE_MODEL,
    "nano-banana-2": GOOGLE_MODEL,
    "3-pro":         GOOGLE_PRO_MODEL,
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
        model=CLAUDE_MODEL,
        max_tokens=3000,
        messages=[{"role": "user", "content": prompt}]
    )
    research = response.content[0].text
    log_fn("✅  Research complete")
    return research


# ── Phase 1: Writing ──────────────────────────────────────────────────────────

def generate_script(topic: str, research: str, profile: "Profile",
                    client: anthropic.Anthropic, log_fn) -> str:
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


def _generate_tts_and_prompts(script: str, profile: "Profile", client: anthropic.Anthropic, log_fn) -> tuple[str, str]:
    """Given a finished script, generate TTS narration and image prompts. Returns (tts_script, image_prompts_raw)."""

    log_fn("✍️  Extracting TTS narration from script...")
    tts_prompt = _build_tts_prompt(script, profile)
    r1 = client.messages.create(
        model=CLAUDE_MODEL,
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
            f"TARGET DENSITY: one image every 3–4 seconds — roughly 15+ prompts per minute of narration. A 14-minute script needs ~210+ prompts. If you finish a batch with fewer than 15 prompts per minute, you are combining too many sentences — go back and split them. "
            f"Every sentence gets its own image, including transition sentences. Never combine two sentences into one prompt. A sentence listing multiple items (A, B, and C) must be split into one prompt per item. "
            f"Transition phrases like 'Now the opposite kind.', 'The team continues.', 'So back to that opening promise.', 'Remember X' each get their own prompt. "
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


# ── Phase 2: Image Generation ─────────────────────────────────────────────────

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


def _image_mime(path: Path) -> str:
    """Detect image mime type from magic bytes."""
    header = path.read_bytes()[:4]
    if header[:3] == b'\xff\xd8\xff':
        return "image/jpeg"
    return "image/png"


def _find_image(img_dir: Path, num: str) -> Path | None:
    for ext in (".png", ".jpg", ".jpeg"):
        p = img_dir / f"{num}{ext}"
        if p.exists():
            return p
    return None


def generate_image_google(prompt: str, output_path: Path, profile: "Profile", log_fn,
                          model: str = None) -> bool:
    """Generate image via Google AI with style anchors as references."""
    from profile import Profile  # local import avoids circular deps at module level

    if model is None:
        model = profile.image_gen["default_model"]

    anchors_dir = profile.anchors_dir
    anchor_files = sorted(anchors_dir.glob("anchor-*.png")) + sorted(anchors_dir.glob("anchor-*.jpg"))
    seen = set()
    reference_files = []
    for f in sorted(anchor_files, key=lambda p: p.stem):
        if f.stem not in seen:
            seen.add(f.stem)
            reference_files.append(f)
    reference_files = reference_files[:profile.image_style.get("max_anchors", 14)]

    contents = [
        f"Reference images are provided in this order:\n"
        f"  anchor-01: Full cast reference sheet — all characters side by side, exact proportions and scale.\n"
        f"  anchor-02: Multi-angle reference sheet — each character shown front, 3/4, side, and back. Match these character designs exactly in every scene.\n"
        f"  anchor-03: Example content scene — shows background, props, and art style in a real frame.\n"
        f"  anchor-04 onward: Additional style and scene references — follow the art style shown precisely.\n\n"
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
                    ext        = ".jpg" if "jpeg" in part.inline_data.mime_type else ".png"
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


def generate_all_images(prompts: list[dict], out_dir: Path,
                        profile: "Profile", log_fn, stop_event=None, skip_existing=False) -> dict:
    """Generate images sequentially. Returns {num: path} for successful images."""
    results = {}
    total = len(prompts)
    for i, p in enumerate(prompts):
        if stop_event and stop_event.is_set():
            break
        num      = p["num"]
        img_path = out_dir / "images" / f"{num}.png"

        existing = _find_image(out_dir / "images", num)
        if skip_existing and existing:
            results[num] = existing
            continue

        log_fn(f"🖼  Generating image {i+1}/{total} ({num})")
        ok = generate_image_google(p["prompt"], img_path, profile, log_fn)
        if ok:
            found = _find_image(out_dir / "images", num) or img_path
            results[num] = found
            flicker_cfg = profile.image_gen.get("flicker", {})
            if flicker_cfg.get("enabled"):
                magnitude = flicker_cfg.get("magnitude", 0.004)
                generate_flicker_frames(found, out_dir / "images", num, magnitude, log_fn)
        else:
            log_fn(f"  ❌ Image {num} failed after 3 attempts — flagged")
    return results


def regenerate_images(run_slug: str, image_nums: list[str], model_key: str,
                      profile: "Profile" = None, progress_callback=None) -> dict:
    """Manually regenerate specific images for a completed run."""
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

    results = {"regenerated": [], "failed": []}
    for num in targets:
        if num not in all_prompts:
            continue
        img_path = out_dir / "images" / f"{num}.png"
        log_fn(f"🔄  Regenerating {num} using {model_key}...")
        ok = generate_image_google(all_prompts[num]["prompt"], img_path, profile, log_fn, model=model)
        if ok:
            log_fn(f"  ✅  {num} regenerated")
            results["regenerated"].append(num)
            flicker_cfg = profile.image_gen.get("flicker", {}) if profile else {}
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
               voice_id: str = "", model: str = "",
               prev_text: str = "", next_text: str = "") -> bytes | None:
    """Send one chunk to ElevenLabs. Returns raw mp3 bytes or None on failure."""
    vid = voice_id or EL_VOICE_ID
    mdl = model or EL_MODEL
    url = f"https://api.elevenlabs.io/v1/text-to-speech/{vid}"
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


def _fix_mp3_duration(path: Path, log_fn) -> None:
    """Null out the Xing/Info VBR header so players use file-size/bitrate for duration."""
    data = bytearray(path.read_bytes())
    offset = 0
    if data[:3] == b'ID3':
        offset = ((data[6] & 0x7f) << 21 | (data[7] & 0x7f) << 14 |
                  (data[8] & 0x7f) << 7  | (data[9] & 0x7f)) + 10
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

    def _strip_id3(data: bytes) -> bytes:
        if data[:3] == b'ID3':
            size = ((data[6] & 0x7f) << 21 | (data[7] & 0x7f) << 14 |
                    (data[8] & 0x7f) << 7  | (data[9] & 0x7f))
            return data[size + 10:]
        return data

    # Keep ID3 header from first chunk only; strip from the rest
    merged = parts[0] + b"".join(_strip_id3(p) for p in parts[1:])

    audio_path = out_dir / "audio" / "voiceover.mp3"
    audio_path.write_bytes(merged)
    _fix_mp3_duration(audio_path, log_fn)
    log_fn("✅  Voiceover generated")
    return audio_path


# ── Audio duration ─────────────────────────────────────────────────────────────

def get_audio_duration(audio_path: Path) -> float:
    """Return audio duration in seconds by scanning MP3 frames.

    ElevenLabs chunks are concatenated after generation, which leaves the Xing
    VBR header (written for only the first chunk) with a stale frame count.
    We therefore scan the actual frame data rather than trusting the header.
    """
    if not audio_path or not audio_path.exists():
        return 720.0
    data = audio_path.read_bytes()
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
        hdr     = int.from_bytes(data[i:i+4], 'big')
        br_idx  = (hdr >> 12) & 0xF
        sr_idx  = (hdr >> 10) & 0x3
        layer   = 4 - ((hdr >> 17) & 0x3)
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
    for key in ("assetId", "id", "asset_id", "mediaRef"):
        if key in result:
            return result[key]
    # Palmier returns plain-text confirmation — parse the UUID from it
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


def _ts_to_seconds(ts: str) -> float | None:
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
    ts = prompt.get("ts", "")
    if "-" in ts:
        start_str, end_str = ts.split("-", 1)
        start = _ts_to_seconds(start_str)
        end   = _ts_to_seconds(end_str)
        if start is not None and end is not None and end > start:
            return (end - start) * scale
    return fallback


def _timestamp_scale(prompts: list[dict], actual_duration: float) -> float:
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


def assemble_palmier_timeline(
    prompts: list[dict],
    image_results: dict,
    audio_path: Path | None,
    out_dir: Path,
    log_fn,
    profile: "Profile | None" = None,
) -> None:
    """Import all pipeline assets into Palmier and assemble the timeline."""
    log_fn("🎬  Assembling timeline in Palmier...")

    timeline_info = _palmier_call("get_timeline", {})
    project_fps   = timeline_info.get("fps", 30)

    valid_nums = [p["num"] for p in prompts if p["num"] in image_results]
    if not valid_nums:
        log_fn("⚠️  No images to place — skipping Palmier assembly")
        return

    total_duration  = get_audio_duration(audio_path) if audio_path else None
    base_dur_s      = (total_duration / len(valid_nums)) if total_duration else 2.5
    base_dur_frames = max(1, round(base_dur_s * project_fps))
    scale           = _timestamp_scale(prompts, total_duration) if total_duration else 1.0

    log_fn(f"  ⏱  Audio duration: {total_duration:.1f}s — timestamp scale: {scale:.3f}")

    prompt_by_num = {p["num"]: p for p in prompts}

    images_dir  = out_dir / "images"
    flicker_dir = images_dir / "flicker"
    flicker_cfg = (profile.image_gen.get("flicker", {}) if profile else {})
    use_flicker   = flicker_cfg.get("enabled", False)

    log_fn(f"  Importing {len(valid_nums)} images into Palmier{'  (flicker enabled)' if use_flicker else ''}...")
    media_refs = {}   # num -> {"a": id, "b": id, "c": id}
    for num in valid_nums:
        r = _palmier_call("import_media", {
            "source": {"path": str(image_results[num].resolve())},
            "name": num,
        })
        refs = {"a": _extract_asset_id(r, "import_media")}
        if use_flicker:
            for variant in ("b", "c"):
                vpath = _find_image(flicker_dir, f"{num}{variant}")
                if vpath:
                    rv = _palmier_call("import_media", {
                        "source": {"path": str(vpath.resolve())},
                        "name": f"{num}{variant}",
                    })
                    refs[variant] = _extract_asset_id(rv, "import_media")
        media_refs[num] = refs

    audio_ref = None
    if audio_path and audio_path.exists():
        log_fn("  Importing voiceover...")
        r = _palmier_call("import_media", {
            "source": {"path": str(audio_path.resolve())},
            "name": "voiceover",
        })
        audio_ref = _extract_asset_id(r, "import_media")

    log_fn("  Placing clips on timeline...")
    entries     = []
    start_frame = 0

    # Flicker cycle: b(1) c(1) alternating
    FLICKER_CYCLE = [("b", 1), ("c", 1)]

    for num in valid_nums:
        clip_dur_s      = _prompt_duration(prompt_by_num[num], base_dur_s, scale)
        clip_dur_frames = max(1, round(clip_dur_s * project_fps))
        refs            = media_refs[num]

        if use_flicker and "b" in refs and "c" in refs:
            cycle_len  = sum(f for _, f in FLICKER_CYCLE)
            remaining  = clip_dur_frames
            frame_pos  = start_frame
            while remaining > 0:
                for variant, frames in FLICKER_CYCLE:
                    take = min(frames, remaining)
                    if take <= 0:
                        break
                    entries.append({
                        "mediaRef":       refs[variant],
                        "startFrame":     frame_pos,
                        "durationFrames": take,
                    })
                    frame_pos += take
                    remaining -= take
        else:
            entries.append({
                "mediaRef":       refs["a"],
                "startFrame":     start_frame,
                "durationFrames": clip_dur_frames,
            })

        start_frame += clip_dur_frames

    _palmier_call("add_clips", {"entries": entries})

    if audio_ref:
        log_fn("  Placing voiceover on timeline...")
        _palmier_call("add_clips", {"entries": [{"mediaRef": audio_ref, "startFrame": 0, "durationFrames": start_frame}]})

    log_fn("✅  Palmier timeline assembled")


# ── Shared production phase ───────────────────────────────────────────────────

def _run_production(topic: str, prompts: list[dict], tts_script: str,
                    profile: "Profile", out_dir: Path, client: anthropic.Anthropic,
                    log_fn, stop_event=None, skip_existing_images=False) -> dict:
    """Phases 2-4: images + voiceover + Palmier."""
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
    (out_dir / "profile.txt").write_text(profile.name)
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
    import argparse
    from profile import load_profile, list_profiles

    parser = argparse.ArgumentParser(description="YouTube Pipeline")
    parser.add_argument("topic", nargs="+", help="Video topic")
    parser.add_argument("--profile", default=None, help="Profile name (folder under profiles/)")
    args = parser.parse_args()

    topic = " ".join(args.topic)

    available = list_profiles()
    if not available:
        print("No profiles found. Create profiles/<name>/profile.yaml first.")
        sys.exit(1)

    if args.profile:
        profile_name = args.profile
    elif len(available) == 1:
        profile_name = available[0]
        print(f"Using profile: {profile_name}")
    else:
        print("Multiple profiles found. Specify one with --profile:")
        for p in available:
            print(f"  {p}")
        sys.exit(1)

    profile = load_profile(profile_name)
    run_pipeline(topic, profile)
