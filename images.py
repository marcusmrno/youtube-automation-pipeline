"""
Image generation — Gemini calls with a profile's anchor references, and the image files on disk.

Everything here is profile-driven: the art style, the anchors sent as references and the models
come from the profile. The run orchestration that decides *which* images to make lives in pipeline.py.
"""
from __future__ import annotations

import functools
import io
import mimetypes
import os
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import TYPE_CHECKING

import yaml
from dotenv import load_dotenv
from google import genai
from google.genai import types as genai_types
from PIL import Image, ImageOps

if TYPE_CHECKING:
    from channel_profile import Profile

load_dotenv()

GOOGLE_KEY = (os.getenv("GOOGLE_API_KEY") or "").strip()


# Image models come from the profile (image_gen.default_model / pro_model).
# Anchors passed as references per image call, when a profile doesn't set image_style.max_anchors.
DEFAULT_MAX_ANCHORS = 14


@functools.cache
def _get_genai_client() -> "genai.Client":
    return genai.Client(api_key=GOOGLE_KEY)


def standardize_image(path: Path, size: tuple[int, int] = (1920, 1080)) -> None:
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


def find_image(img_dir: Path, num: str) -> Path | None:
    for ext in (".png", ".jpg", ".jpeg"):
        p = img_dir / f"{num}{ext}"
        if p.exists() and p.stat().st_size:   # a 0-byte file is a crashed write, not an image
            return p
    return None


GENERIC_ANCHOR_REFS = "Reference images define the art style and characters — match them precisely."


def _anchor_files(anchors_dir: Path, max_anchors: int) -> list[Path]:
    """The anchor images sent with every image, in send order: by label, one per label, capped."""
    anchor_files = sorted(anchors_dir.glob("anchor-*.png")) + sorted(anchors_dir.glob("anchor-*.jpg"))
    seen = set()
    reference_files = []
    for f in sorted(anchor_files, key=lambda p: p.stem):
        if f.stem not in seen:
            seen.add(f.stem)
            reference_files.append(f)
    return reference_files[:max_anchors]


def _anchor_manifest(anchors_dir: Path, max_anchors: int = DEFAULT_MAX_ANCHORS) -> str:
    """Per-anchor descriptions from anchors/manifest.yaml, written at profile creation.

    Empty when absent — GENERIC_ANCHOR_REFS covers it. Never describe anchor slots
    inline here: the layout depends on the profile's roster size. Only the files actually
    sent are described, in send order, so a seed (anchor-00) or a failed anchor can't
    shift every description by one.
    """
    f = anchors_dir / "manifest.yaml"
    if not f.exists():
        return ""
    purposes = {s["label"]: s["purpose"] for s in (yaml.safe_load(f.read_text()) or []) if s.get("label")}
    lines = "\n".join(f"  {p.stem}: {purposes[p.stem]}"
                      for p in _anchor_files(anchors_dir, max_anchors) if p.stem in purposes)
    return f"Reference images are provided in this order:\n{lines}" if lines else ""


def build_preamble(image_style: dict, anchors_dir: Path | None = None) -> str:
    """Prefix sent ahead of every image prompt — entirely profile-driven.

    `style_constraints` is the short hard-rule form; falls back to the full
    art_style_block when a profile doesn't define one.
    """
    constraints = (image_style.get("style_constraints") or image_style["art_style_block"]).strip()
    max_anchors = image_style.get("max_anchors", DEFAULT_MAX_ANCHORS)
    refs = (_anchor_manifest(anchors_dir, max_anchors) if anchors_dir else "") or GENERIC_ANCHOR_REFS
    return f"{refs}\n\nSTYLE CONSTRAINTS: {constraints}\n\n"


def load_anchors_from_dir(anchors_dir: Path, max_anchors: int) -> list:
    """Pre-load anchor images from a directory as genai Parts."""
    reference_files = _anchor_files(anchors_dir, max_anchors)
    return [
        genai_types.Part.from_bytes(data=ref.read_bytes(), mime_type=mimetypes.guess_type(ref.name)[0] or "image/png")
        for ref in reference_files
    ]


def load_anchor_parts(profile: "Profile") -> list:
    """Pre-load anchor images as genai Parts. Call once per run, not per image."""
    return load_anchors_from_dir(profile.anchors_dir, profile.image_style.get("max_anchors", DEFAULT_MAX_ANCHORS))


def generate_image_google(prompt: str, output_path: Path, profile: "Profile", log_fn,
                          model: str = None, anchor_parts: list | None = None,
                          preamble: str | None = None,
                          standardize_size: tuple[int, int] = (1920, 1080)) -> bool:
    """Generate image via Google AI with style anchors as references."""
    if model is None:
        model = profile.image_gen["default_model"]

    if anchor_parts is None:
        anchor_parts = load_anchor_parts(profile)

    if preamble is None:
        preamble = build_preamble(profile.image_style, profile.anchors_dir)

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
                    # decode first: undecodable bytes must not replace a good image or pass as done
                    Image.open(io.BytesIO(part.inline_data.data)).load()
                    final_path.write_bytes(part.inline_data.data)
                    if final_path != output_path and output_path.exists():
                        output_path.unlink()
                    standardize_image(final_path, standardize_size)
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
    anchor_parts = load_anchor_parts(profile)
    results = {}
    total = len(prompts)

    def _one(i: int, p: dict):
        num      = p["num"]
        # Every prompt is submitted up front, so a stop that lands after submission
        # must be caught here — otherwise ~150 queued images keep billing.
        if stop_event and stop_event.is_set():
            return num, None
        img_path = out_dir / "images" / f"{num}.png"
        log_fn(f"🖼  Generating image {i+1}/{total} ({num})")
        ok = generate_image_google(p["prompt"], img_path, profile, log_fn, anchor_parts=anchor_parts)
        if not ok:
            log_fn(f"  ❌ Image {num} failed after 3 attempts — flagged")
            return num, None
        found = find_image(out_dir / "images", num) or img_path
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
            existing = find_image(out_dir / "images", num)
            if skip_existing and existing:
                results[num] = existing
                continue
            futures[ex.submit(_one, i, p)] = num

        try:
            for fut in as_completed(futures):
                num, found = fut.result()
                if found:
                    results[num] = found
        except BaseException:
            # Ctrl-C (the CLI's only stop): drop the queue, or the executor's exit bills every image
            for f in futures:
                f.cancel()
            raise

    return results
