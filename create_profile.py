"""
Profile Creator — generate or revise a YouTube channel profile.
Usage:
  python create_profile.py                     # interactive: create new
  python create_profile.py --revise <name>     # revise existing profile
"""
from __future__ import annotations

import re
import sys
import argparse

from pipeline import require_keys
from profile import PROFILES_ROOT


def main() -> None:
    parser = argparse.ArgumentParser(description="YouTube channel profile creator")
    parser.add_argument("--revise", metavar="PROFILE_NAME",
                        help="Name of an existing profile to revise")
    parser.add_argument("--seed", metavar="IMAGE_PATH",
                        help="Path to a seed image that anchors the visual style for all generated anchors")
    args = parser.parse_args()

    require_keys("ANTHROPIC_API_KEY", "GOOGLE_API_KEY")

    if args.revise:
        from profile_creator.revise import run_revise
        revise_name = re.sub(r"[^\w-]", "-", args.revise.lower()).strip("-")
        run_revise(revise_name)
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
                from profile_creator.revise import run_revise
                run_revise(name)
                return

        from profile_creator.new import run_create
        run_create(seed_image=args.seed)


if __name__ == "__main__":
    main()
