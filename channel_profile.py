from __future__ import annotations
import os
import re
from dataclasses import dataclass
from pathlib import Path
import yaml

from prompts import word_budget


PROFILES_ROOT = Path(__file__).parent / "profiles"


@dataclass
class Profile:
    name: str
    dir: Path
    anchors_dir: Path
    channel: dict
    script: dict
    characters: list[dict]
    character_behavior: str
    image_style: dict
    voice: dict
    image_gen: dict

    def characters_block(self) -> str:
        """Returns character descriptions for embedding in image prompts."""
        return "\n\n".join(
            f"**{c['name']}:** {c['description']}"
            for c in self.characters
        )


def load_profile(name: str, profiles_root: Path = PROFILES_ROOT) -> Profile:
    # names come from UI request bodies too; a path here would load any directory on disk
    if not re.fullmatch(r"[\w-]+", name):
        raise ValueError(f"Invalid profile name '{name}'")
    profile_dir = profiles_root / name
    yaml_path = profile_dir / "profile.yaml"

    if not profile_dir.exists() or not yaml_path.exists():
        available = list_profiles(profiles_root)
        raise ValueError(
            f"Profile '{name}' not found. Available: {available}"
        )

    data = yaml.safe_load(yaml_path.read_text())
    _validate(name, data)

    # Resolve ${ENV_VAR} references in voice_id
    voice_id = data["voice"]["voice_id"]
    m = re.match(r'^\$\{(.+)\}$', str(voice_id))
    if m:
        voice_id = os.getenv(m.group(1), "")
    data["voice"]["voice_id"] = voice_id

    chars = data.get("characters") or {}
    return Profile(
        name=name,
        dir=profile_dir,
        anchors_dir=profile_dir / "anchors",
        channel=data["channel"],
        script=data["script"],
        characters=chars.get("roster") or [],
        character_behavior=chars.get("behavior") or "",
        image_style=data["image_style"],
        voice=data["voice"],
        image_gen=data["image_gen"],
    )


def _validate(name: str, data) -> None:
    """ValueError naming what's wrong, instead of a bare KeyError/TypeError deep in a run."""
    data = data if isinstance(data, dict) else {}
    missing = [k for k in ("channel", "script", "image_style", "voice", "image_gen")
               if not isinstance(data.get(k), dict)]
    missing += [key for section, key in (("image_style", "art_style_block"), ("voice", "voice_id"),
                                         ("image_gen", "default_model"))
                if isinstance(data.get(section), dict) and key not in data[section]]
    if missing:
        raise ValueError(f"Profile '{name}' is missing: {', '.join(missing)}")
    try:
        word_budget(data["script"])   # every script prompt is built from these numbers
    except (KeyError, TypeError, ValueError, AttributeError) as e:
        raise ValueError(f"Profile '{name}' has an invalid script section: {e!r}") from e


def list_profiles(profiles_root: Path = PROFILES_ROOT) -> list[str]:
    if not profiles_root.exists():
        return []
    return sorted(
        d.name for d in profiles_root.iterdir()
        if d.is_dir() and (d / "profile.yaml").exists()
    )
