"""New profile creation flow."""
from __future__ import annotations

import re
import shutil
import sys
import yaml
from pathlib import Path

import anthropic

from .claude_helpers import SYSTEM_PROMPT_NEW, clarification_loop, generate_profile_content
from .anchors import build_anchor_plan, generate_anchor_prompts, run_verification_anchors, run_full_anchors, write_manifest
from pipeline import ANTHROPIC_KEY
from channel_profile import PROFILES_ROOT, load_profile


def _read_brain_dump() -> str:
    print("\n" + "─" * 60)
    print("Paste your channel brain dump below.")
    print("When done, enter '---' on its own line and press Enter.")
    print("─" * 60 + "\n")
    lines = []
    while True:
        line = input()
        if line.strip() == "---":
            break
        lines.append(line)
    return "\n".join(lines).strip()


def _choose_profile_name() -> str:
    while True:
        name = input("\nProfile folder name (e.g. dark-history): ").strip()
        # sanitize to slug, then check: '!!!' becomes '', and profiles/'' is the profiles folder itself
        name = re.sub(r"[^\w-]", "-", name.lower()).strip("-")
        if not name:
            continue
        dest = PROFILES_ROOT / name
        if dest.exists():
            overwrite = input(f"Profile '{name}' already exists. Overwrite? [y/N]: ").strip().lower()
            if overwrite != "y":
                continue
            # old anchors would be kept ("already exists") and used as references for the new ones
            shutil.rmtree(dest)
        return name


def run_create(seed_image: str | None = None) -> None:
    client = anthropic.Anthropic(api_key=ANTHROPIC_KEY)

    brain_dump = _read_brain_dump()
    if not brain_dump:
        print("No input provided. Exiting.")
        sys.exit(0)

    profile_name = _choose_profile_name()
    profile_dir  = PROFILES_ROOT / profile_name
    anchors_dir  = profile_dir / "anchors"

    if seed_image:   # create_profile.py has already checked that it exists
        seed_path = Path(seed_image)
        anchors_dir.mkdir(parents=True, exist_ok=True)
        from images import _standardize_image
        # always .png: only anchor-*.png/.jpg are sent, and _standardize_image re-encodes by extension
        dest = anchors_dir / "anchor-00.png"
        shutil.copy2(seed_path, dest)
        _standardize_image(dest, (1280, 720))
        print(f"✓  Seed image installed as {dest.name}")

    # Clarification loop
    messages = [{"role": "user", "content": brain_dump}]
    messages = clarification_loop(client, SYSTEM_PROMPT_NEW, messages)

    # Generate profile files
    print("\n⏳  Generating profile...")
    yaml_content, style_content = generate_profile_content(client, SYSTEM_PROMPT_NEW, messages)

    # Validate YAML parses
    try:
        profile_yaml = yaml.safe_load(yaml_content)
    except yaml.YAMLError as e:
        print(f"✗ Claude produced invalid YAML: {e}")
        sys.exit(1)

    # Write files
    profile_dir.mkdir(parents=True, exist_ok=True)
    (profile_dir / "profile.yaml").write_text(yaml_content)
    (profile_dir / "style-sheet.md").write_text(style_content)
    print(f"✓  Written: {profile_dir / 'profile.yaml'}")
    print(f"✓  Written: {profile_dir / 'style-sheet.md'}")
    try:   # the pipeline must be able to load it — check before paying for anchors
        load_profile(profile_name, PROFILES_ROOT)
    except ValueError as e:
        sys.exit(f"✗ {e}. Fix {profile_dir / 'profile.yaml'} and run --revise, or start again.")

    # Build anchor plan and generate prompts
    plan = build_anchor_plan(profile_yaml)
    print(f"\n⏳  Generating {len(plan)} anchor prompts ({sum(1 for s in plan if s['tier'] == 'verification')} verification, {sum(1 for s in plan if s['tier'] == 'full')} full)...")
    prompts, plan = generate_anchor_prompts(client, profile_yaml, style_content, plan)
    seed_slot = [{"label": "anchor-00", "purpose": "User-supplied seed image — the style target"}]
    write_manifest(anchors_dir, (seed_slot if seed_image else []) + plan)

    # Verification anchors (character sheets — shown to user before continuing)
    run_verification_anchors(profile_yaml, anchors_dir, prompts, plan)

    # Full anchor set
    print(f"\n⏳  Generating remaining anchors...")
    result = run_full_anchors(profile_yaml, anchors_dir, prompts, plan)

    print("\n" + "─" * 60)
    print(f"✓  Profile '{profile_name}' created at {profile_dir}")
    print(f"   Anchors: {len(result['ok'])} ok, {len(result['failed'])} failed")
    if result["failed"]:
        print(f"   Failed: {', '.join(result['failed'])}")
    print("─" * 60)
