#!/usr/bin/env python3
"""
Preview script → image_prompts generation without generating any images.

Usage:
    python preview_prompts.py <script.txt> [--profile <profile_name>] [--out <output_dir>]

Examples:
    python preview_prompts.py output/my-run/script.txt --out /tmp/preview
    python preview_prompts.py my_script.txt --profile example
    python preview_prompts.py my_script.txt --out /tmp/test_run
"""

import argparse
import sys
from pathlib import Path

import anthropic
from pipeline import _generate_tts_and_prompts, parse_image_prompts, ANTHROPIC_KEY
from profile import load_profile, list_profiles


def main():
    parser = argparse.ArgumentParser(description="Test script → image prompt generation")
    parser.add_argument("script", help="Path to script.txt")
    parser.add_argument("--profile", default=None, help="Profile name (default: the only one, if there is only one)")
    parser.add_argument("--out", default=None, help="Directory to write outputs (default: same dir as script)")
    args = parser.parse_args()

    script_path = Path(args.script).resolve()
    if not script_path.exists():
        print(f"❌  Script not found: {script_path}")
        sys.exit(1)

    out_dir = Path(args.out).resolve() if args.out else script_path.parent
    if not args.out and (out_dir / "image_prompts.txt").exists():
        # a run's prompts are what made its images; a preview must not replace them
        sys.exit(f"❌  {out_dir} already has image_prompts.txt — pass --out <dir> to preview elsewhere")
    out_dir.mkdir(parents=True, exist_ok=True)

    available = list_profiles()
    if not args.profile:
        if len(available) != 1:
            print(f"❌  Specify --profile. Available: {available}")
            sys.exit(1)
        args.profile = available[0]

    try:
        profile = load_profile(args.profile)   # raises with the available list
    except ValueError as e:
        print(f"❌  {e}")
        sys.exit(1)
    script  = script_path.read_text()
    client  = anthropic.Anthropic(api_key=ANTHROPIC_KEY)

    def log(msg):
        print(msg)

    print(f"\n📄  Script:  {script_path}")
    print(f"🎨  Profile: {args.profile}")
    print(f"📁  Output:  {out_dir}\n")

    tts_script, image_prompts_raw = _generate_tts_and_prompts(script, profile, client, log)

    tts_out    = out_dir / "tts_script.txt"
    prompts_out = out_dir / "image_prompts.txt"

    tts_out.write_text(tts_script)
    prompts_out.write_text(image_prompts_raw)

    prompts = parse_image_prompts(image_prompts_raw)
    print(f"\n✅  Done")
    print(f"   tts_script.txt  → {tts_out}")
    print(f"   image_prompts.txt → {prompts_out}  ({len(prompts)} prompts)")


if __name__ == "__main__":
    main()
