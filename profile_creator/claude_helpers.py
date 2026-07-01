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

# OPTIONAL — omit entirely for styles with no recurring characters
characters:
  roster:
    - name: "..."
      description: "..."   # highly specific physical description for image gen
  behavior: |
    ...

image_style:
  art_style_block: |
    # The exact sentence appended to EVERY image prompt. Defines the rendering style.
    # This is what tells the image model HOW to draw — medium, line quality, color treatment, texture.
    # Be highly specific. Bad: "flat illustration". Good: "flat 2D hand-drawn illustration on
    # slightly off-white warm paper, bold black outlines with heavy uneven stroke weight..."
    ...

  scene_rules: |
    # Freeform markdown injected into the image prompt instructions as the scene composition guide.
    # This is the most important section — it defines HOW scenes are composed for this style.
    # Include: background/environment rules, lighting, layout, what to vary between scenes,
    # forbidden compositions, text/label rules, character placement rules (if any).
    # Write it as direct instructions to the prompt writer, not as a description of the style.
    # There are no required sections — write whatever rules this style needs.
    ...

  max_anchors: 6

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
  pro_model: "gemini-3-pro-image"
  flicker:
    enabled: false    # set true to generate b/c flicker frames per image (0.4% stretch)
    magnitude: 0.004
```

## The two most important fields

### `art_style_block`
This sentence is appended verbatim to every image prompt. It must describe the visual medium and rendering qualities precisely enough that an AI image model reproduces the same look in every scene. Include: medium (illustration, photo, 3D, etc.), line quality, color treatment, texture, paper/background, and any anti-rules (no gradients, no photorealism, etc.).

### `scene_rules`
This is injected as the scene composition section of the prompt writer's instructions. It fully replaces any hardcoded pipeline defaults — the pipeline has no opinions about composition, environment, or layout. Everything lives here. Write it as direct instructions:
- What backgrounds/environments to use and how to vary them
- What to do with sky, ground, horizon (or whether to have any)
- Whether and how to use text labels in images
- Forbidden compositions or elements
- Character placement rules (if characters exist)
- Anything else that makes this style's scenes recognizable and consistent

**There is no required structure.** A minimalist diagram style might have 3 bullet points. A rich illustrated style might have 10+ rules covering landscape types, sky rotation, foreground elements, text labels, and forbidden clichés.

## Example profile.yaml (cat-educational)
```yaml
{_EXAMPLE_PROFILE_PATH.read_text() if _EXAMPLE_PROFILE_PATH.exists() else "# not found"}
```

## Example style-sheet.md
{_EXAMPLE_STYLE_PATH.read_text() if _EXAMPLE_STYLE_PATH.exists() else "# not found"}

## Character description rules (only if using characters)
- Physical descriptions must be precise enough for an AI image model to reproduce the character consistently across hundreds of images
- Describe: head shape, ear shape, eye style, body shape, distinguishing marks, clothing/accessories, hands, feet
- Every detail should be specific and visual — "slightly lopsided" not "quirky"
- If there are no recurring characters, omit the `characters` block entirely

## Your process
1. Read the user's channel concept
2. Ask ONE clarifying question at a time if you need more info (niche, art direction, characters, tone). Ask at most 3 questions total — prioritise the ones that would most change the output.
3. When you have enough to generate, output EXACTLY the token PROFILE_READY on its own line, then stop.
4. When asked to generate, output:
   - A ```yaml block containing the complete profile.yaml
   - A ```markdown block containing the complete style-sheet.md

Always fill every field. Never leave TBD or placeholders. voice.voice_id is always ${{ELEVENLABS_VOICE_ID}}. Write `scene_rules` as thorough, specific instructions — it is the primary driver of image quality for this style.
"""

SYSTEM_PROMPT_REVISE = """You are a YouTube channel profile designer helping revise an existing profile.

The user will describe what they want to change. Ask ONE clarifying question at a time if needed (at most 2 questions).

When you have enough, output EXACTLY the token PROFILE_READY on its own line, then stop.

When asked to generate, output:
- A ```yaml block containing the COMPLETE updated profile.yaml (all fields, not just changed ones)
- A ```markdown block containing the COMPLETE updated style-sheet.md
- A line: REGENERATE_ANCHORS: true  (if characters, art_style_block, or scene_rules changed meaningfully)
  OR:     REGENERATE_ANCHORS: false (if only channel/script/voice fields changed)

Key fields to get right:
- `art_style_block`: the exact sentence appended to every image prompt — changes here affect every generated image
- `scene_rules`: freeform instructions for how scenes are composed — this is where environment, layout, text labels, forbidden elements, and background variety rules live. Be thorough and specific.
- `characters`: fully optional — omit the block entirely for character-free styles

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
    """Run multi-turn clarification until Claude outputs PROFILE_READY and user confirms summary."""
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
            # Ask Claude to summarise its understanding before generating
            messages.append({
                "role": "user",
                "content": (
                    "Before generating, summarise back to me what you understand about this channel: "
                    "the niche, tone, characters (names, appearance, behavior), art style, and any "
                    "other key decisions you've made. Be concise — bullet points are fine."
                ),
            })
            summary_response = client.messages.create(
                model="claude-sonnet-4-6",
                max_tokens=1024,
                system=system,
                messages=messages,
            )
            summary = summary_response.content[0].text
            messages.append({"role": "assistant", "content": summary})

            print("\n" + "─" * 60)
            print("Claude's understanding:\n")
            print(summary)
            print("\n" + "─" * 60)
            confirm = input("Looks good? [y] to generate, or type corrections: ").strip()

            if confirm.lower() in ("y", "yes", ""):
                return messages

            # User has corrections — feed them back and continue the loop
            messages.append({"role": "user", "content": confirm})
            continue

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
    """Request profile generation. Returns (yaml_content, markdown_content).

    NOTE: mutates the messages list in place by appending the "Generate the profile now."
    user turn and Claude's assistant reply. Callers that read messages[-1]["content"] after
    this call are relying on that final assistant message being present.
    """
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
