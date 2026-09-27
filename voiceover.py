"""
Voiceover — the TTS script sent to ElevenLabs in chunks, spliced into audio/voiceover.mp3.
"""
from __future__ import annotations

import os
import re
import textwrap
import time
from pathlib import Path
from typing import TYPE_CHECKING

import requests
from dotenv import load_dotenv

if TYPE_CHECKING:
    from channel_profile import Profile

load_dotenv()

EL_KEY = (os.getenv("ELEVENLABS_API_KEY") or "").strip()


EL_MODEL      = "eleven_v3"


def _split_into_chunks(text: str, max_chars: int = 4500) -> list[str]:
    if not text.strip():
        return []
    chunks, current = [], []
    length = 0
    for sentence in re.split(r'(?<=[.!?])\s+', text.strip()):
        # text with no terminal punctuation (or sentences ending in a quote) splits into one long "sentence"
        for piece in textwrap.wrap(sentence, max_chars) if len(sentence) > max_chars else [sentence]:
            if length + len(piece) + 1 > max_chars and current:
                chunks.append(" ".join(current))
                current, length = [], 0
            current.append(piece)
            length += len(piece) + 1
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


def generate_voiceover(tts_script: str, out_dir: Path, profile: "Profile", log_fn,
                      stop_event=None) -> Path | None:
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
    if not chunks:
        log_fn("⚠️  TTS script is empty — skipping voiceover")
        return None
    log_fn(f"  📄  Script split into {len(chunks)} chunk(s)")

    parts = []
    for i, chunk in enumerate(chunks):
        if stop_event and stop_event.is_set():
            log_fn("🛑  Voiceover stopped")
            return None
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
