"""Profile revision flow — creates a versioned copy of an existing profile."""
from __future__ import annotations

import re
import shutil
import sys
import yaml

import anthropic

from .claude_helpers import (
    SYSTEM_PROMPT_REVISE,
    clarification_loop,
    generate_profile_content,
)
from .anchors import build_anchor_plan, generate_anchor_prompts, run_verification_anchors, run_full_anchors, write_manifest
from pipeline import ANTHROPIC_KEY
from profile import PROFILES_ROOT, load_profile

STYLE_SENSITIVE_KEYS = {"art_style_block", "style_constraints"}


def next_version_name(base: str) -> str:
    """Given 'my-channel' return 'my-channel-v2'; given 'my-channel-v2' return 'my-channel-v3'."""
    m = re.match(r"^(.+)-v(\d+)$", base)
    if m:
        return f"{m.group(1)}-v{int(m.group(2)) + 1}"
    return f"{base}-v2"


def _style_changed(old_yaml: dict, new_yaml: dict) -> bool:
    """Return True if characters or style-sensitive image_style fields changed."""
    if old_yaml.get("characters") != new_yaml.get("characters"):
        return True
    old_style = old_yaml.get("image_style", {})
    new_style = new_yaml.get("image_style", {})
    for key in STYLE_SENSITIVE_KEYS:
        if old_style.get(key) != new_style.get(key):
            return True
    return False


def run_revise(profile_name: str) -> None:
    # Sanitize profile_name to prevent path traversal regardless of call site
    profile_name = re.sub(r"[^\w-]", "-", profile_name.lower()).strip("-")
    profile_dir = PROFILES_ROOT / profile_name
    if not profile_dir.exists() or not (profile_dir / "profile.yaml").exists():
        print(f"x Profile '{profile_name}' not found at {profile_dir}")
        sys.exit(1)

    old_yaml_text  = (profile_dir / "profile.yaml").read_text()
    old_style_text = (profile_dir / "style-sheet.md").read_text() if (profile_dir / "style-sheet.md").exists() else ""
    old_yaml       = yaml.safe_load(old_yaml_text)

    # the next free version: revising twice makes -v3, never overwrites (or merges into) -v2
    v2_name = next_version_name(profile_name)
    while (PROFILES_ROOT / v2_name).exists():
        v2_name = next_version_name(v2_name)
    v2_dir  = PROFILES_ROOT / v2_name

    client = anthropic.Anthropic(api_key=ANTHROPIC_KEY)

    print(f"\n  Revising '{profile_name}' -> '{v2_name}'")
    print("\nDescribe what you want to change (then press Enter):")
    change_desc = input("You: ").strip()

    context_msg = (
        f"Here is the existing profile:\n\n"
        f"## profile.yaml\n```yaml\n{old_yaml_text}\n```\n\n"
        f"## style-sheet.md\n```markdown\n{old_style_text}\n```\n\n"
        f"The user wants to change: {change_desc}"
    )
    messages = [{"role": "user", "content": context_msg}]
    messages = clarification_loop(client, SYSTEM_PROMPT_REVISE, messages)

    print("\n  Generating updated profile...")
    new_yaml_text, new_style_text = generate_profile_content(client, SYSTEM_PROMPT_REVISE, messages)

    try:
        new_yaml = yaml.safe_load(new_yaml_text)
    except yaml.YAMLError as e:
        print(f"x Claude produced invalid YAML: {e}")
        sys.exit(1)

    # Determine if anchors should be regenerated
    # Claude signals with REGENERATE_ANCHORS: true/false in the raw reply
    last_reply = messages[-1]["content"] if messages else ""
    regen_anchors = False
    m = re.search(r"REGENERATE_ANCHORS:\s*(true|false)", last_reply, re.IGNORECASE)
    if m:
        regen_anchors = m.group(1).lower() == "true"
    else:
        # Fallback: detect style changes ourselves
        regen_anchors = _style_changed(old_yaml, new_yaml)
        if regen_anchors:
            print("\n  Style-relevant fields changed. Anchors will be regenerated.")
        else:
            ans = input("\nRegenerate anchors? [y/N]: ").strip().lower()
            regen_anchors = ans == "y"

    # Write versioned profile
    v2_dir.mkdir(parents=True, exist_ok=True)
    (v2_dir / "profile.yaml").write_text(new_yaml_text)
    (v2_dir / "style-sheet.md").write_text(new_style_text)
    print(f"  Written: {v2_dir / 'profile.yaml'}")
    print(f"  Written: {v2_dir / 'style-sheet.md'}")
    try:   # the pipeline must be able to load it — check before paying for anchors
        load_profile(v2_name, PROFILES_ROOT)
    except ValueError as e:
        sys.exit(f"x {e}. Fix {v2_dir / 'profile.yaml'} by hand, or revise again.")

    anchors_dir = v2_dir / "anchors"

    if regen_anchors:
        plan = build_anchor_plan(new_yaml)
        print(f"\n  Generating {len(plan)} anchor prompts...")
        prompts, plan = generate_anchor_prompts(client, new_yaml, new_style_text, plan)
        write_manifest(anchors_dir, plan)
        run_verification_anchors(new_yaml, anchors_dir, prompts, plan)
        print("\n  Generating remaining anchors...")
        result = run_full_anchors(new_yaml, anchors_dir, prompts, plan)
    else:
        # Copy anchors from original
        src_anchors = profile_dir / "anchors"
        if src_anchors.exists():
            shutil.copytree(src_anchors, anchors_dir, dirs_exist_ok=True)
            print(f"  Copied anchors from '{profile_name}'")
        result = {"ok": [], "failed": []}

    print("\n" + "-" * 60)
    print(f"  Profile '{v2_name}' created at {v2_dir}")
    if regen_anchors:
        print(f"   Anchors: {len(result['ok'])} ok, {len(result['failed'])} failed")
        if result["failed"]:
            print(f"   Failed: {', '.join(result['failed'])}")
    else:
        print(f"   Anchors: copied from '{profile_name}'")
    print("-" * 60)
