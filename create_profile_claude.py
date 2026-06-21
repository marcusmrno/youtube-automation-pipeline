"""Claude conversation helpers for profile creator."""
from __future__ import annotations

import re
from pathlib import Path

import anthropic

# Load the cat-educational profile as a concrete example for the system prompt
_EXAMPLE_PROFILE_PATH = Path(__file__).parent / "profiles" / "cat-educational" / "profile.yaml"
_EXAMPLE_STYLE_PATH   = Path(__file__).parent / "profiles" / "cat-educational" / "style-sheet.md"

SYSTEM_PROMPT_NEW = f"""You are a YouTube channel profile designer. Your job is to take a user's channel concept and produce a complete channel profile.

A profile has two files:
1. profile.yaml — channel config (schema shown below)
2. style-sheet.md — visual rules doc for image generation

## Profile YAML Schema
```yaml
channel:
  name: "..."
  niche: "..."
  audience: "..."
  tone: "..."
  reference_channel: "..."
  title_format: "..."

script:
  target_mins: 12
  min_mins: 9
  max_mins: 15
  wpm: 160
  hook_duration_s: 35
  cta_duration_s: 30
  section_count: "10-14"
  section_duration_s: "60-90"

characters:
  roster:
    - name: "..."
      description: "..."   # highly specific physical description for image gen
  behavior: |
    ...

image_style:
  art_style_block: |
    ...
  sky_rotation: "color1 → color2 → ..."
  anchor_priority:
    - anchor-01
    - anchor-02
  max_anchors: 14

voice:
  voice_id: "${{ELEVENLABS_VOICE_ID}}"
  model: "eleven_v3"
  stability: 0.68
  similarity_boost: 0.85
  style: 0.0
  use_speaker_boost: true
  tone_description: "..."

image_gen:
  default_model: "gemini-3.1-flash-image"
  regen_model: "gemini-3.1-flash-image"
  pro_model: "gemini-3-pro-image"
```

## Example profile.yaml (cat-educational)
```yaml
{_EXAMPLE_PROFILE_PATH.read_text() if _EXAMPLE_PROFILE_PATH.exists() else "# not found"}
```

## Example style-sheet.md
{_EXAMPLE_STYLE_PATH.read_text() if _EXAMPLE_STYLE_PATH.exists() else "# not found"}

## Character description rules
- Physical descriptions must be precise enough for an AI image model to reproduce the character consistently
- Describe: head shape, ear shape, eye style, whiskers, mouth, body shape, distinguishing marks, clothing/accessories, hands, feet
- Example: "large orange tabby cat, round head slightly lopsided, small uneven triangle ears, three short whisker lines on each side of face, dot eyes at slightly different heights, simple curved mouth, solid orange body asymmetric and lumpy, small black bowtie at neck slightly crooked, small filled circle hands at end of arms, small flat oval feet"

## Your process
1. Read the user's brain dump
2. Ask ONE clarifying question at a time if you need more info (channel niche, characters, art style direction, tone). Ask at most 3 questions total.
3. When you have enough to generate, output EXACTLY the token PROFILE_READY on its own line, then stop.
4. When asked to generate, output:
   - A ```yaml block containing the complete profile.yaml
   - A ```markdown block containing the complete style-sheet.md

Always fill every field. Never leave TBD or placeholders. voice.voice_id is always ${{ELEVENLABS_VOICE_ID}}.
"""

SYSTEM_PROMPT_REVISE = """You are a YouTube channel profile designer helping revise an existing profile.

The user will describe what they want to change. Ask ONE clarifying question at a time if needed (at most 2 questions).

When you have enough, output EXACTLY the token PROFILE_READY on its own line, then stop.

When asked to generate, output:
- A ```yaml block containing the COMPLETE updated profile.yaml (all fields, not just changed ones)
- A ```markdown block containing the COMPLETE updated style-sheet.md
- A line: REGENERATE_ANCHORS: true  (if characters, art_style_block, or sky_rotation changed meaningfully)
  OR:     REGENERATE_ANCHORS: false (if only channel/script/voice fields changed)

Never leave TBD or placeholders. voice.voice_id is always ${ELEVENLABS_VOICE_ID}.
"""


def extract_fenced_block(text: str, lang: str) -> str | None:
    """Extract content of the first ```lang ... ``` block. Returns None if not found."""
    pattern = rf"```{lang}\n(.*?)```"
    m = re.search(pattern, text, re.DOTALL)
    if m:
        return m.group(1).strip()
    return None


def clarification_loop(
    client: anthropic.Anthropic,
    system: str,
    messages: list[dict],
) -> list[dict]:
    """Run multi-turn clarification until Claude outputs PROFILE_READY."""
    while True:
        response = client.messages.create(
            model="claude-sonnet-4-6",
            max_tokens=1024,
            system=system,
            messages=messages,
        )
        reply = response.content[0].text
        messages.append({"role": "assistant", "content": reply})

        if "PROFILE_READY" in reply:
            return messages

        question = reply.replace("PROFILE_READY", "").strip()
        if question:
            print(f"\nClaude: {question}\n")
        user_input = input("You: ").strip()
        if not user_input:
            user_input = "(no answer)"
        messages.append({"role": "user", "content": user_input})


def generate_profile_content(
    client: anthropic.Anthropic,
    system: str,
    messages: list[dict],
) -> tuple[str, str]:
    """Request profile generation. Returns (yaml_content, markdown_content)."""
    messages.append({"role": "user", "content": "Generate the profile now."})
    response = client.messages.create(
        model="claude-sonnet-4-6",
        max_tokens=8192,
        system=system,
        messages=messages,
    )
    reply = response.content[0].text
    messages.append({"role": "assistant", "content": reply})

    yaml_content = extract_fenced_block(reply, "yaml")
    md_content   = extract_fenced_block(reply, "markdown")

    if not yaml_content:
        raise ValueError("Claude did not output a ```yaml block. Raw reply:\n" + reply[:500])
    if not md_content:
        raise ValueError("Claude did not output a ```markdown block. Raw reply:\n" + reply[:500])

    return yaml_content, md_content
