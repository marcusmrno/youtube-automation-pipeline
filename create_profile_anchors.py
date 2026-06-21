"""Anchor image orchestration for profile creator."""
from __future__ import annotations

import re
import anthropic
from pathlib import Path

from create_profile_image_gen import generate_anchor

ANCHOR_SYSTEM = """You write image generation prompts for YouTube channel anchor images.
Each prompt must:
- Be 150-250 words
- Fully describe the scene, characters, environment, and art style (do not rely on prior context)
- Include the character's full physical description inline
- Follow the provided style sheet exactly
- Not repeat the same sky color or scene type consecutively

Output ONLY a numbered list: one prompt per line, starting with the anchor number.
Format: anchor-01: [full prompt text]
"""


def generate_anchor_prompts(
    client: anthropic.Anthropic,
    profile_yaml: dict,
    style_sheet: str,
    n: int,
) -> list[str]:
    """Ask Claude to write n anchor prompts for this profile. Returns list of prompt strings."""
    characters = profile_yaml["characters"]["roster"]
    char_block = "\n".join(
        f"- {c['name']}: {c['description']}" for c in characters
    )
    sky_rotation = profile_yaml["image_style"].get("sky_rotation", "")
    art_style    = profile_yaml["image_style"].get("art_style_block", "")

    user_msg = f"""Write {n} anchor image prompts for this channel profile.

## Characters
{char_block}

## Art Style
{art_style}

## Sky Rotation
{sky_rotation}

## Style Sheet
{style_sheet}

## Special requirements
- anchor-01: character reference sheet — all roster characters side by side, plain off-white background, full body, no scene context. Pure character design reference.
- anchor-02: scene test — one character in a typical content scene for this channel. Representative environment. No text labels.
- anchor-03 onward: varied content scenes, cycling through sky_rotation colors, mixing roster characters.

Output {n} prompts in the format:
anchor-01: [full prompt]
anchor-02: [full prompt]
...
"""

    response = client.messages.create(
        model="claude-sonnet-4-6",
        max_tokens=8192,
        system=ANCHOR_SYSTEM,
        messages=[{"role": "user", "content": user_msg}],
    )
    raw = response.content[0].text

    prompts = []
    for line in raw.splitlines():
        m = re.match(r"anchor-\d+:\s*(.+)", line.strip())
        if m:
            prompts.append(m.group(1).strip())
    return prompts


def run_verification_anchors(
    profile_yaml: dict,
    anchors_dir: Path,
    prompts: list[str],
) -> bool:
    """Generate anchor-01 and anchor-02. Pause for user approval. Returns True if approved."""
    anchors_dir.mkdir(parents=True, exist_ok=True)
    failed = []

    for i, label in enumerate(["anchor-01", "anchor-02"]):
        prompt = prompts[i] if i < len(prompts) else f"flat 2D hand-drawn scene for {label}"
        out_path = anchors_dir / f"{label}.png"
        ok = generate_anchor(prompt, out_path, profile_yaml, anchors_dir)
        if not ok:
            failed.append(label)

    print("\n" + "─" * 60)
    if failed:
        print(f"⚠️  Failed to generate: {', '.join(failed)}")
    print("✓  Verification images generated:")
    for f in sorted(anchors_dir.glob("anchor-0[12].*")):
        print(f"   {f}")
    print()
    print("Open them and check the style looks right.")
    input("Press Enter to generate remaining anchors, or Ctrl+C to abort: ")
    return True


def run_full_anchors(
    profile_yaml: dict,
    anchors_dir: Path,
    prompts: list[str],
    start_from: int = 3,
) -> dict:
    """Generate anchors from start_from onward, skipping existing files."""
    anchors_dir.mkdir(parents=True, exist_ok=True)
    ok_list: list[str] = []
    failed_list: list[str] = []

    for i, prompt in enumerate(prompts[start_from - 1:], start=start_from):
        label    = f"anchor-{i:02d}"
        out_path = anchors_dir / f"{label}.png"

        # Skip if any variant already exists
        existing = list(anchors_dir.glob(f"{label}.*"))
        if existing:
            print(f"  ⏭  Skipping {label} (already exists)")
            ok_list.append(label)
            continue

        ok = generate_anchor(prompt, out_path, profile_yaml, anchors_dir)
        if ok:
            ok_list.append(label)
        else:
            failed_list.append(label)

    return {"ok": ok_list, "failed": failed_list}
