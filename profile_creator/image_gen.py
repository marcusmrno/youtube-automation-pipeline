"""Shared image generation helper for profile creator."""
from __future__ import annotations

import os
from pathlib import Path

from PIL import Image
from google import genai
from google.genai import types as genai_types

GOOGLE_KEY = (os.getenv("GOOGLE_API_KEY") or "").strip()


def _image_mime(path: Path) -> str:
    return "image/jpeg" if path.suffix.lower() in (".jpg", ".jpeg") else "image/png"


def _standardize_image(path: Path) -> None:
    """Convert to RGB PNG at 1280×720."""
    try:
        img = Image.open(path).convert("RGB")
        img = img.resize((1280, 720), Image.LANCZOS)
        png_path = path.with_suffix(".png")
        img.save(png_path, "PNG")
        if png_path != path:
            path.unlink(missing_ok=True)
    except Exception:
        pass


def generate_anchor(
    prompt: str,
    output_path: Path,
    profile_yaml: dict,
    anchors_dir: Path,
) -> bool:
    """Generate a single anchor image. Returns True on success, False on failure."""
    model = profile_yaml["image_gen"]["default_model"]
    max_anchors = profile_yaml["image_style"].get("max_anchors", 14)

    # Load existing anchors in numerical order (lower numbers = higher priority)
    anchor_files = sorted(anchors_dir.glob("anchor-*.png")) + sorted(anchors_dir.glob("anchor-*.jpg"))
    seen = set()
    unique = []
    for f in sorted(anchor_files, key=lambda p: p.stem):
        if f.stem not in seen:
            seen.add(f.stem)
            unique.append(f)
    reference_files = unique[:max_anchors]

    preamble = (
        "Reference images define the art style and characters — match them precisely.\n\n"
        "STYLE CONSTRAINTS: Bold uneven black marker outlines. Color fills bleed outside lines "
        "with visible marker streaks. Off-white warm paper background. No gradients. No drop shadows. "
        "No clean fonts. No photorealistic textures. No smooth digital lines.\n\n"
        f"{prompt}"
    )

    contents: list = [preamble]
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
            print(f"  ⏳  Generating {output_path.name} (attempt {attempt + 1})...")
            response = client.models.generate_content(
                model=model,
                contents=contents,
                config=genai_types.GenerateContentConfig(
                    response_modalities=["IMAGE"],
                ),
            )
            for part in response.candidates[0].content.parts:
                if part.inline_data and part.inline_data.mime_type.startswith("image/"):
                    actual_mime = part.inline_data.mime_type
                    ext = ".jpg" if "jpeg" in actual_mime else ".png"
                    final_path = output_path.with_suffix(ext)
                    final_path.write_bytes(part.inline_data.data)
                    if final_path != output_path and output_path.exists():
                        output_path.unlink()
                    _standardize_image(final_path)
                    return True
            print(f"  ⚠️  No image in response")
        except Exception as e:
            print(f"  ⚠️  Attempt {attempt + 1} error: {e}")
    return False
