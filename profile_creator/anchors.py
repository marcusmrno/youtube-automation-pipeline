"""Anchor image orchestration for profile creator."""
from __future__ import annotations

import re
import anthropic
import yaml
from pathlib import Path

from pipeline import (
    generate_image_google, _load_anchors_from_dir, build_preamble, DEFAULT_MAX_ANCHORS,
)
from .claude_helpers import MODEL


def _roster(profile_yaml: dict) -> list[dict]:
    """Characters are optional — mirror profile.load_profile's tolerance for a missing block."""
    return (profile_yaml.get("characters") or {}).get("roster") or []


def _generate_anchor(prompt: str, output_path: Path, profile_yaml: dict, anchors_dir: Path) -> bool:
    image_style = profile_yaml["image_style"]
    anchor_parts = _load_anchors_from_dir(anchors_dir, image_style.get("max_anchors", DEFAULT_MAX_ANCHORS))
    return generate_image_google(
        prompt, output_path, profile=None, log_fn=print,
        model=profile_yaml["image_gen"]["default_model"],
        anchor_parts=anchor_parts,
        # No manifest yet — these anchors are what it will describe.
        preamble=build_preamble(image_style),
        standardize_size=(1280, 720),
    )


def write_manifest(anchors_dir: Path, plan: list[dict]) -> None:
    """Persist what each anchor slot is, so image generation can describe the references."""
    anchors_dir.mkdir(parents=True, exist_ok=True)
    (anchors_dir / "manifest.yaml").write_text(yaml.safe_dump(
        [{"label": s["label"], "purpose": s["purpose"]} for s in plan],
        sort_keys=False, default_flow_style=False, allow_unicode=True,
    ))

ANCHOR_SYSTEM = """You write image generation prompts for YouTube channel anchor images.
Each prompt must:
- Be 150-250 words
- Fully describe the scene, characters, environment, and art style inline (no references to prior context)
- Include every character's full physical description whenever they appear
- Follow the provided style sheet exactly

Output ONLY the prompts, one per line, in the format:
anchor-01: [full prompt text]
anchor-02: [full prompt text]
...
"""


def build_anchor_plan(profile_yaml: dict) -> list[dict]:
    """
    Build a structured anchor plan from the profile.
    Returns a list of slot dicts: {label, purpose, tier}
    Tier: "verification" (shown to user before proceeding) or "full"
    """
    characters = _roster(profile_yaml)
    plan = []

    # ── Tier: verification — character reference sheets ───────────────────────

    if len(characters) == 0:
        pass  # no character sheets needed

    elif len(characters) == 1:
        # Single character: one combined sheet (front + angles)
        char = characters[0]
        plan.append({
            "label": "anchor-01",
            "purpose": (
                f"Character reference sheet for {char['name']} — left side shows full body "
                f"front view on a plain neutral background; right side shows front, 3/4, side, "
                f"and back views arranged in a row. No scene context, no props. "
                f"Pure character design reference from all angles."
            ),
            "tier": "verification",
        })

    else:
        # 2+ characters: cast sheet first, then pair them across multi-angle sheets
        char_names = ", ".join(c["name"] for c in characters)
        plan.append({
            "label": "anchor-01",
            "purpose": (
                f"Full cast reference sheet — {char_names} standing side by side on a plain "
                f"neutral background. Full body, no scene context, no props. "
                f"Shows exact proportions and scale relationship between all characters."
            ),
            "tier": "verification",
        })

        # Pair characters across multi-angle sheets (2 per sheet)
        pairs = [characters[i:i + 2] for i in range(0, len(characters), 2)]
        for pair in pairs:
            n = len(plan) + 1
            pair_names = " and ".join(c["name"] for c in pair)
            if len(pair) == 2:
                purpose = (
                    f"Multi-angle reference sheet for {pair_names} — each character shown "
                    f"in four views side by side: full front, 3/4 front, side profile, and "
                    f"3/4 back. Plain neutral background, no scene context. "
                    f"Locks in both characters' designs from every angle."
                )
            else:
                purpose = (
                    f"Multi-angle reference sheet for {pair[0]['name']} — four views arranged "
                    f"on a plain neutral background: full front, 3/4 front, side profile, and "
                    f"3/4 back. No scene context. Locks in design from every angle."
                )
            plan.append({
                "label": f"anchor-{n:02d}",
                "purpose": purpose,
                "tier": "verification",
            })

    # Example scene — always verification, shows background/props/style in context
    n = len(plan) + 1
    primary = characters[0]["name"] if characters else None
    if primary:
        scene_desc = (
            f"{primary} in a fully dressed content scene typical for this channel — "
            f"relevant background, props, and objects all visible. "
            f"Shows what a real video frame looks like: art style, color palette, and prop design together."
        )
    else:
        scene_desc = (
            f"A fully dressed content scene typical for this channel — "
            f"relevant background, props, and objects all visible, no characters. "
            f"Shows what a real video frame looks like: art style, color palette, and prop design together."
        )
    plan.append({
        "label": f"anchor-{n:02d}",
        "purpose": scene_desc,
        "tier": "verification",
    })

    # ── Tier: full ────────────────────────────────────────────────────────────

    # Character-free profiles must not get an invented figure: these anchors go with every image
    subject = "one character" if characters else "the channel's main subject (no characters)"

    # Character interaction (only if 2+ characters)
    if len(characters) >= 2:
        n = len(plan) + 1
        char_names = " and ".join(c["name"] for c in characters[:2])
        plan.append({
            "label": f"anchor-{n:02d}",
            "purpose": (
                f"{char_names} together in a typical content scene for this channel. "
                f"Establishes their scale relationship and how they look side by side in context. "
                f"Use a neutral background color."
            ),
            "tier": "full",
        })

    # Props/objects scene — no characters, tests art style on objects
    n = len(plan) + 1
    plan.append({
        "label": f"anchor-{n:02d}",
        "purpose": (
            f"A flat lay of many props and objects relevant to the channel niche, "
            f"no characters present. Tests how the art style renders objects — "
            f"this is where style drift usually first appears. "
            f"Objects must match the channel's art style exactly."
        ),
        "tier": "full",
    })

    # Text/labels scene
    n = len(plan) + 1
    plan.append({
        "label": f"anchor-{n:02d}",
        "purpose": (
            (f"One character pointing at or presenting a diagram, chart, or annotation with "
             if characters else
             f"A diagram, chart, or annotation about the channel's subject, no characters, with ") +
            f"handwritten-style text labels clearly visible. Locks in how text, labels, "
            f"and annotations should look in the art style. Use a neutral background."
        ),
        "tier": "full",
    })

    # Wide shot — characters small in a larger environment
    n = len(plan) + 1
    plan.append({
        "label": f"anchor-{n:02d}",
        "purpose": (
            f"Wide establishing shot — {subject} shown small within a larger detailed "
            f"environment relevant to the channel niche. Tests whether the art style holds "
            f"when characters are not the primary focus. Neutral background."
        ),
        "tier": "full",
    })

    # Close-up — character face/upper body (nothing to portray without a roster)
    if characters:
        n = len(plan) + 1
        plan.append({
            "label": f"anchor-{n:02d}",
            "purpose": (
                f"Close-up portrait — one character from the waist up, facing slightly toward "
                f"camera. Tests character detail consistency at close range. Plain background."
            ),
            "tier": "full",
        })

    # Background/landscape environments — lock in how different settings look in the art style
    background_slots = [
        (
            f"Indoor scene — {subject} in a fully dressed interior environment relevant to the channel "
            "(desk, shelves, walls, floor all visible). Locks in how indoor backgrounds render in the art style."
        ),
        (
            f"Outdoor scene — {subject} in a detailed exterior environment relevant to the channel "
            "(ground, horizon line, sky, and background objects all visible). Locks in how outdoor "
            "settings look in the art style."
        ),
        (
            f"Abstract or diagrammatic background — {subject} in front of a flat graphic background "
            "(grid, chart, map, or pattern). Locks in how non-realistic backgrounds work with the characters."
        ),
    ]
    for desc in background_slots:
        n = len(plan) + 1
        plan.append({
            "label": f"anchor-{n:02d}",
            "purpose": desc,
            "tier": "full",
        })

    return plan


def generate_anchor_prompts(
    client: anthropic.Anthropic,
    profile_yaml: dict,
    style_sheet: str,
    plan: list[dict],
) -> tuple[dict[str, str], list[dict]]:
    """Ask Claude to write one targeted prompt per anchor slot. Returns ({label: prompt}, trimmed plan).

    Respects image_style.max_anchors — verification slots fill the budget first,
    then full-tier slots fill the remainder in order.
    """
    max_anchors = profile_yaml.get("image_style", {}).get("max_anchors", DEFAULT_MAX_ANCHORS)
    verification = [s for s in plan if s["tier"] == "verification"]
    full         = [s for s in plan if s["tier"] == "full"]
    plan = (verification + full)[:max_anchors]

    char_block = "\n".join(
        f"- {c['name']}: {c['description']}" for c in _roster(profile_yaml)
    ) or "(no recurring characters — these anchors define style only)"
    art_style = profile_yaml["image_style"].get("art_style_block", "")

    slot_block = "\n".join(
        f"{slot['label']}: {slot['purpose']}" for slot in plan
    )

    user_msg = f"""Write one image generation prompt for each anchor slot below.

## Characters
{char_block}

## Art Style
{art_style}

## Style Sheet
{style_sheet}

## Anchor slots — write exactly one prompt per slot, in order
{slot_block}

Output {len(plan)} prompts in the format:
anchor-01: [full prompt]
anchor-02: [full prompt]
...
"""

    response = client.messages.create(
        model=MODEL,
        max_tokens=8192,
        system=ANCHOR_SYSTEM,
        messages=[{"role": "user", "content": user_msg}],
    )
    raw = response.content[0].text

    prompts = {}
    for line in raw.splitlines():
        m = re.match(r"(anchor-\d+):\s*(.+)", line.strip())
        if m:
            prompts[m.group(1)] = m.group(2).strip()
    missing = [slot["label"] for slot in plan if slot["label"] not in prompts]
    if missing:
        print(f"  ⚠️  Model didn't return prompts for: {', '.join(missing)} — will use generic fallback")
    return prompts, plan


def run_verification_anchors(
    profile_yaml: dict,
    anchors_dir: Path,
    prompts: dict[str, str],
    plan: list[dict],
) -> None:
    """Generate all verification-tier anchors. Pause for user approval before continuing."""
    anchors_dir.mkdir(parents=True, exist_ok=True)
    verification_slots = [s for s in plan if s["tier"] == "verification"]
    failed = []

    for slot in verification_slots:
        prompt = prompts.get(slot["label"]) or slot["purpose"]
        out_path = anchors_dir / f"{slot['label']}.png"
        print(f"  ⏳  Generating {slot['label']} ({slot['purpose'][:60]}...)")
        ok = _generate_anchor(prompt, out_path, profile_yaml, anchors_dir)
        if not ok:
            failed.append(slot["label"])

    print("\n" + "─" * 60)
    if failed:
        print(f"⚠️  Failed to generate: {', '.join(failed)}")
    print("✓  Verification anchors generated:")
    for slot in verification_slots:
        for f in sorted(anchors_dir.glob(f"{slot['label']}.*")):
            print(f"   {f}  —  {slot['purpose'][:70]}")
    print()
    print("Open them and check the character designs and style look right.")
    input("Press Enter to generate remaining anchors, or Ctrl+C to abort: ")


def run_full_anchors(
    profile_yaml: dict,
    anchors_dir: Path,
    prompts: dict[str, str],
    plan: list[dict],
) -> dict:
    """Generate all full-tier anchors, skipping any that already exist."""
    anchors_dir.mkdir(parents=True, exist_ok=True)
    full_slots = [s for s in plan if s["tier"] == "full"]
    ok_list: list[str] = []
    failed_list: list[str] = []

    for slot in full_slots:
        label = slot["label"]

        existing = list(anchors_dir.glob(f"{label}.*"))
        if existing:
            print(f"  ⏭  Skipping {label} (already exists)")
            ok_list.append(label)
            continue

        prompt   = prompts.get(label) or slot["purpose"]
        out_path = anchors_dir / f"{label}.png"
        print(f"  ⏳  Generating {label} — {slot['purpose'][:60]}...")
        ok = _generate_anchor(prompt, out_path, profile_yaml, anchors_dir)
        if ok:
            ok_list.append(label)
        else:
            failed_list.append(label)

    return {"ok": ok_list, "failed": failed_list}
