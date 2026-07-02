#!/usr/bin/env python3
"""
Preview script → image_prompts generation without generating any images.

Usage:
    python preview_prompts.py <script.txt> [--profile <profile_name>] [--out <output_dir>]

Examples:
    python preview_prompts.py ../old-outputs/every-essential-vitamin-explained/script.txt
    python preview_prompts.py my_script.txt --profile cat-educational-v2
    python preview_prompts.py my_script.txt --out /tmp/test_run
"""

import argparse
import sys
from pathlib import Path

# Add the pipeline directory to path
sys.path.insert(0, str(Path(__file__).parent))

import anthropic
from pipeline import _generate_tts_and_prompts, parse_image_prompts, ANTHROPIC_KEY
from profile import load_profile

DEFAULT_PROFILE = "cat-educational-v2"


def main():
    parser = argparse.ArgumentParser(description="Test script → image prompt generation")
    parser.add_argument("script", help="Path to script.txt")
    parser.add_argument("--profile", default=DEFAULT_PROFILE, help=f"Profile name (default: {DEFAULT_PROFILE})")
    parser.add_argument("--out", default=None, help="Directory to write outputs (default: same dir as script)")
    args = parser.parse_args()

    script_path = Path(args.script).resolve()
    if not script_path.exists():
        print(f"❌  Script not found: {script_path}")
        sys.exit(1)

    out_dir = Path(args.out).resolve() if args.out else script_path.parent
    out_dir.mkdir(parents=True, exist_ok=True)

    profiles_dir = Path(__file__).parent / "profiles"
    if not (profiles_dir / args.profile).exists():
        print(f"❌  Profile not found: {args.profile}")
        print(f"   Available: {[p.name for p in profiles_dir.iterdir() if p.is_dir()]}")
        sys.exit(1)

    profile = load_profile(args.profile)
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
