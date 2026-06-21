"""
Profile Creator — generate or revise a YouTube channel profile.
Usage:
  python create_profile.py                     # interactive: create new
  python create_profile.py --revise <name>     # revise existing profile
"""
from __future__ import annotations

import os
import re
import sys
import argparse
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

PROJECT_ROOT  = Path(__file__).parent
PROFILES_ROOT = PROJECT_ROOT / "profiles"

ANTHROPIC_KEY = (os.getenv("ANTHROPIC_API_KEY") or "").strip()
GOOGLE_KEY    = (os.getenv("GOOGLE_API_KEY") or "").strip()

CLAUDE_MODEL = "claude-sonnet-4-6"


def check_env() -> None:
    missing = []
    if not ANTHROPIC_KEY:
        missing.append("ANTHROPIC_API_KEY")
    if not GOOGLE_KEY:
        missing.append("GOOGLE_API_KEY")
    if missing:
        print(f"✗ Missing required env vars: {', '.join(missing)}")
        print("  Check your .env file.")
        sys.exit(1)


def next_version_name(base: str) -> str:
    """Given 'my-channel' return 'my-channel-v2'; given 'my-channel-v2' return 'my-channel-v3'."""
    m = re.match(r"^(.+)-v(\d+)$", base)
    if m:
        return f"{m.group(1)}-v{int(m.group(2)) + 1}"
    return f"{base}-v2"


def main() -> None:
    parser = argparse.ArgumentParser(description="YouTube channel profile creator")
    parser.add_argument("--revise", metavar="PROFILE_NAME",
                        help="Name of an existing profile to revise")
    args = parser.parse_args()

    check_env()

    if args.revise:
        from create_profile_revise import run_revise
        run_revise(args.revise)
    else:
        # Interactive: ask create-new or revise existing
        from profile import list_profiles
        existing = list_profiles(PROFILES_ROOT)

        if existing:
            print("\nExisting profiles:")
            for i, name in enumerate(existing, 1):
                print(f"  {i}. {name}")
            print()
            choice = input("Create [n]ew profile or [r]evise existing? [n/r]: ").strip().lower()
            if choice == "r":
                idx = input(f"Enter profile number (1-{len(existing)}): ").strip()
                try:
                    name = existing[int(idx) - 1]
                except (ValueError, IndexError):
                    print("Invalid choice.")
                    sys.exit(1)
                from create_profile_revise import run_revise
                run_revise(name)
                return

        from create_profile_new import run_create
        run_create()


if __name__ == "__main__":
    main()
