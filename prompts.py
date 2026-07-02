"""
Prompt builder functions — pure functions that return strings, no API calls.
"""
from __future__ import annotations

import re
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from profile import Profile


def _extract(tag: str, text: str) -> str:
    m = re.search(rf"==={tag}===(.*?)(?====|\Z)", text, re.DOTALL)
    return m.group(1).strip() if m else ""


def _build_clarifying_questions_prompt(topic: str, profile: "Profile") -> str:
    c = profile.channel
    template = f"""
You are a content strategist for a {c["niche"]} educational YouTube channel targeting {c["audience"]}.

A creator wants to make a video about: {topic}

Generate 4-5 clarifying questions that will help nail the angle of this video. For each question:
- Ask something specific and useful (angle, outcome, misconceptions, target audience, emphasis)
- Immediately follow it with a "Default:" line: a short, evidence-backed suggested answer that reflects the strongest/most interesting direction for this topic on YouTube

Make questions punchy and answerable in 1-2 sentences. Defaults should be opinionated and grounded in what actually performs well for educational content on this subject.

Format EXACTLY like this (no deviations):

1. [Question text]
   Default: [Suggested evidence-backed answer]

2. [Question text]
   Default: [Suggested evidence-backed answer]

(continue for all questions)

Return only the numbered questions and defaults, no preamble or closing text."""
    return template


def _build_approach_pitch_prompt(topic: str, answers: str, profile: "Profile") -> str:
    c = profile.channel
    template = f"""
You are a content strategist for a {c["niche"]} educational YouTube channel targeting {c["audience"]}.

## Video Topic
{topic}

## Creator's Context & Answers
{answers}

Based on this, pitch 3 distinct, evidence-based approaches to this video. For each approach:
- Give it a short name (3-5 words)
- Explain the core angle in one sentence
- List 2-3 reasons why this angle works for the audience
- Note what misconceptions or ideas it directly addresses

Make pitches concrete and opinionated. The creator will pick one to move forward with.

Format as:

### Approach 1: [Name]
**Angle:** [One sentence core idea]
**Why it works:**
- [Reason 1]
- [Reason 2]
- [Reason 3]
**Addresses:** [Key misconceptions or ideas]

[Repeat for approaches 2 and 3]"""
    return template


def _word_budget(s: dict) -> tuple[int, int, int, int, int, int, int]:
    """wpm-derived word counts for a profile's script config.

    Returns (target_words, min_words, max_words, hook_words, cta_words, section_min, section_max).
    """
    section_lo, section_hi = (int(x) for x in s["section_duration_s"].split("-"))
    return (
        s["target_mins"] * s["wpm"],
        s["min_mins"] * s["wpm"],
        s["max_mins"] * s["wpm"],
        round(s["hook_duration_s"] / 60 * s["wpm"]),
        round(s["cta_duration_s"] / 60 * s["wpm"]),
        round(section_lo / 60 * s["wpm"]),
        round(section_hi / 60 * s["wpm"]),
    )


def _build_script_prompt(topic: str, research: str, profile: "Profile", approach_context: str = "") -> str:
    s = profile.script
    c = profile.channel
    target_words, min_words, max_words, hook_words, cta_words, section_min, section_max = _word_budget(s)

    approach_section = ""
    if approach_context.strip():
        approach_section = f"""
## Creator's Video Idea & Angle
Based on the creator's clarifying answers below, tailor your script to match their intended focus:
{approach_context}

Use this context to guide your angle, emphasis, and which aspects of the research to highlight.
---
"""

    template = f"""
You are a script writer for a faceless educational YouTube channel.

## Channel Identity
- Niche: {c["niche"]}
- Target audience: {c["audience"]}
- Tone: {c["tone"]}
- Reference channel: {c["reference_channel"]} — study the hook style and pacing
- Titles: {c["title_format"]}

## Script Structure
- Hook (0:00-0:{s["hook_duration_s"]:02d}): Provocative opening statement or surprising fact. No intro, no "welcome back".
- {s["section_count"]} content sections with clear [MM:SS-MM:SS] timestamps
- Each section {s["section_duration_s"]} seconds
- CTA close (last {s["cta_duration_s"]} seconds): Subscribe prompt only — no teasing or referencing a next video

## Word Count Rules
The voiceover is delivered at ~{s["wpm"]} words per minute.
- {s["target_mins"]}-minute target = ~{target_words} words of narration
- {s["min_mins"]}-minute minimum  = ~{min_words} words of narration
- {s["max_mins"]}-minute maximum = ~{max_words} words of narration
- Each {s["section_duration_s"]} second section needs {section_min}-{section_max} words of narration
- Hook ({s["hook_duration_s"]}s) = ~{hook_words} words. CTA close ({s["cta_duration_s"]}s) = ~{cta_words} words.
After writing, count your narration words. If under {round(target_words * 0.9)}, expand sections before returning.

---
TOPIC: {topic}

{approach_section}---
RESEARCH & VERIFIED FACTS:
{research}

IMPORTANT: Base the script only on the verified facts above.
- Do not invent statistics or claims not supported by the research.
- Avoid any misconceptions listed above.
- Where confidence is noted as lower, use softened language ("some researchers suggest", "evidence points to", etc.).
- Use the hook angles as inspiration for the opening {s["hook_duration_s"]} seconds.
- Each core idea may be introduced ONCE. Callbacks are only allowed if they add new information — never re-explain a mechanism already stated. If you find yourself repeating the same concept in different words, cut it.

Your task: Write a full narration-only script. Do not describe visuals, camera directions, or what should appear on screen — write only what the narrator speaks aloud. Structure with [MM:SS-MM:SS] section timestamps.

Return your response in this exact format — no other text:

===SCRIPT===
[full script here]
"""
    return template


def _build_tts_prompt(script: str, profile: "Profile") -> str:
    template = f"""
You are preparing a TTS narration for ElevenLabs {profile.voice.get("model", "eleven_v3")} from a finished YouTube video script.

SCRIPT:
{script}

## Extraction rules
- Strip all timestamps, VISUAL lines, section headers, and stage directions.
- Keep only the words spoken aloud, in order, as naturally flowing prose.
- Do NOT use SSML tags — {profile.voice.get("model", "eleven_v3")} does not support them.
- Do NOT use tags that describe visuals or actions (e.g. [grinning], [pacing]) — only auditory tags.
- Do NOT change any words — only add/remove/reposition tags and adjust punctuation/capitalisation for emphasis.

## Emphasis techniques
- Use ellipses (...) for dramatic pauses and weight at key moments.
- Use ALL CAPS for a single word of genuine vocal stress — one per sentence max.
- Do not use both ALL CAPS and a tag on the same phrase — pick one.
- Short sentences = faster delivery. Long sentences = slower, more weight. Vary deliberately.
- Exclamation marks add energy; question marks invite the listener to lean in.

## Audio tags — place immediately before the segment they modify, or after a natural pause mid-sentence
Target density: 1–2 tags per 200 words (~10–16 tags for a full 12-minute script). Too few is flat; too many is performed.
Do not stack two tags back-to-back with no words between them.

Laughter (graduated — pick the right intensity):
  [chuckles]        mild irony, "of course this is how it works"
  [laughs]          a stat or fact is genuinely absurd
  [laughs harder]   escalating absurdity — rare
  [giggles]         lighter, more playful moments
  [snorts]          dry involuntary reaction to something ridiculous
  [wheezing]        extreme — use only for the single funniest moment in the whole script

Breathing & texture:
  [sighs]           tired of a myth; "and then obviously…" moments
  [exhales]         releasing tension after a heavy section
  [whispers]        sharing something counterintuitive that feels like a secret — use at most once per script, only if no other tag fits
  [swallows]        before delivering a hard truth
  [gulps]           before something shocking or uncomfortable

Emotions:
  [excited]         a genuinely surprising fact or big reveal
  [surprised]       when a fact defies common sense
  [curious]         posing a question the audience is already wondering
  [thoughtful]      before a nuanced or considered point
  [impressed]       acknowledging something remarkable
  [delighted]       a satisfying explanation clicking into place
  [sarcastic]       quoting conventional wisdom you're about to debunk
  [mischievously]   setting up a twist or gotcha
  [frustrated]      something preventable went wrong; systemic failure
  [angry]           genuine outrage — historical injustice, lives lost unnecessarily
  [annoyed]         milder frustration; "this again" energy
  [appalled]        moral shock at a behaviour or fact
  [sad]             acknowledging real human cost
  [sympathetic]     speaking to an audience who may have experienced this
  [sheepishly]      correcting a complication or admitting nuance
  [nervously]       building unease before a reveal
  [alarmed]         urgent warning; something worse than expected
  [panicking]       high-stakes escalation — use once max, near the climax
  [reassuring]      after a scary section; "here's what you can do"
  [warmly]          CTA close ONLY — one tag at the very start, then no more tags after it
  [professional]    delivering a crisp fact or instruction
  [questioning]     rhetorical question the audience is asking themselves
  [happy]           a good outcome; something working as intended

## Section-to-tag mapping
Hook first sentence          → [excited] or [curious]
Hook absurd opening stat     → [laughs] or [surprised]
Setting up a myth to debunk  → [sarcastic] or [thoughtful]
The debunk itself            → [sighs] or [appalled]
Counterintuitive reveal      → [surprised] or [mischievously]
Gross or disturbing fact     → [appalled] or [gulps]
Historical injustice         → [angry] or [frustrated]
Human cost / empathy moment  → [sympathetic] or [sad]
Absurd statistic             → [chuckles] or [snorts]
"Here's what works" pivot    → [reassuring] or [exhales]
Tension before consequence   → [nervously] or [alarmed]
Rhetorical question          → [questioning] or [curious]
CTA close                    → [warmly] once at the very start, then no more tags

## Channel tone
{profile.voice["tone_description"]}

Return only this, no other text:

===TTS_SCRIPT===
[clean narration here]
"""
    return template


def _build_image_prompt_instructions(profile: "Profile") -> str:
    s = profile.script

    style       = profile.image_style["art_style_block"].strip()
    scene_rules = (profile.image_style.get("scene_rules") or "").strip()

    char_block = profile.characters_block()
    behavior   = (profile.character_behavior or "").strip()
    characters_section = f"""## Characters — embed description verbatim in EVERY prompt

{char_block}

{behavior}

---
""" if char_block else ""

    template = f"""
You are an image prompt writer for a YouTube video pipeline targeting Google Gemini image generation.

Below is a segment of the script. Each VISUAL line shows the timestamp and narration that will be playing at that moment. Your job is to write one image prompt per VISUAL line — a scene that is a direct, literal translation of EXACTLY what the narrator says in that line.

---

## The Prime Directive

For each narration beat, ask: what is the single most concrete, specific thing being said right now? Build the entire image around showing that one thing as literally and directly as possible.

**Rules:**
- The image must be SPECIFIC to its narration line — it must be impossible to swap it with any other image in the video
- If the narration mentions a number, that number must appear large and prominent in the image
- If the narration names a specific thing (organ, vitamin, country, person, object), that thing must be the main visual element
- If the narration describes an action or process, that action must be physically shown
- Never show a "mood" or "vibe" — show the exact fact being stated
- Never write a scene that could fit 3 different moments in the script
- Style prefix at the start of a prompt is forbidden (pipeline prepends it automatically)
- **Every word of narration must be covered by an image. Zero gaps.** Timestamps must span the full audio with no uncovered narration.

**Transition sentences are not skippable.** Short pivot phrases like "Now the opposite kind.", "The team continues.", "So back to that opening promise.", "Remember the fat-soluble ones" are their own image beats. Never merge them silently into the next content beat.

How to visualize transitions by type:
- **Section pivot** ("Now…", "Next up…", "Moving on…"): title-card showing the name/letter of the incoming topic large center frame — cat pointing at it
- **Callback/recall** ("Back to…", "Remember…"): the key prop from the earlier beat re-shown, with a bold RECALL label or arrow pointing back to it
- **Summary pivot** ("Here's where X really matters", "The team continues"): a visual summary of what is about to be elaborated — the relevant diagram or object group already on screen
- **Consequence setup** ("Without X…", "Run low for long enough…"): show the consequence visually with a red X or fading/cracking element, even before the full sentence lands

---

{characters_section}{scene_rules}

---

## Art style — end EVERY prompt with this exact block

{style}

---

## Format

Target density: **one image every 3–4 seconds**. A 14-minute video should produce ~210–280 prompts. If you are writing fewer than 15 prompts per minute of narration, you are combining too many sentences — stop and split them.

Every sentence gets its own image. Every distinct idea, fact, or statement is a separate visual frame — do not combine two sentences into one image. If a sentence contains two distinct claims, split it into two images. A sentence with a list (e.g. "It does A, B, and C") must be split into one image per item if each item is meaningfully different.

**Before writing, read every sentence in the segment.** For each sentence, decide its image. Transition sentences ("Now…", "Next up…", "The team continues.", "Remember…", "So back to…", "Here's where…") must each get their own prompt — they are not merging candidates.

After writing, do a final gap check: confirm every narration sentence has a prompt.

For each prompt, copy the exact narration sentence you are illustrating as the source field.
Format each line as:
NNN | [exact source sentence] | [full prompt]

Number sequentially from wherever instructed — never restart from 001 mid-batch.
Do NOT include a style prefix. Write ALL prompts for this segment.

Return only:

===IMAGE_PROMPTS===
[prompts here]
"""
    return template


def _build_image_prompt_vet_instructions(profile: "Profile") -> str:
    char_block = profile.characters_block()
    style      = profile.image_style["art_style_block"].strip()

    return f"""You are vetting image prompts for a YouTube video pipeline.

## Channel context
{char_block}

## Art style
{style}

---

You will receive a list of image prompts in this format:
NNN | [source line] | [prompt]

The source line is the exact narration sentence the prompt was generated from.

## Your job

Read all prompts and flag any that have one or more of these problems:

1. **RELEVANCE** — the prompt does not illustrate what the source line says. The visual content is unrelated, too vague, or shows something from a different part of the script.

2. **ADJACENT_SIMILARITY** — this prompt and the one immediately before or after it describe nearly the same scene (same character position, same object, same setting with no meaningful visual difference).

3. **SCENE_OVERUSE** — the same scene archetype appears more than 3 times across the full list. Count occurrences of recurring archetypes (e.g. "cat at a whiteboard", "cat floating in space", "cat pointing at a chart") and flag the less important occurrences beyond the third.

## Output format

Return a JSON array. If no prompts need fixing, return an empty array `[]`.

For each flagged prompt:
{{
  "num": "NNN",
  "reason": "RELEVANCE | ADJACENT_SIMILARITY | SCENE_OVERUSE",
  "detail": "one sentence explaining the specific problem"
}}

Return ONLY the JSON array. No explanation, no markdown fences.
"""


def _build_image_prompt_rewrite_instructions(profile: "Profile") -> str:
    char_block = profile.characters_block()
    style      = profile.image_style["art_style_block"].strip()
    scene_rules = (profile.image_style.get("scene_rules") or "").strip()

    characters_section = f"""## Characters — embed description verbatim in EVERY prompt

{char_block}

---
""" if char_block else ""

    return f"""You are rewriting flagged image prompts for a YouTube video pipeline.

{characters_section}## Art style — end EVERY prompt with this exact block

{style}

{scene_rules}

---

You will receive a list of flagged prompts in this format:
NNN | [source line] | [prompt] | REASON: [reason and detail]

## Your job

Rewrite each prompt to fix the stated problem:

- **RELEVANCE**: rewrite the prompt so it directly and literally illustrates the source line
- **ADJACENT_SIMILARITY**: change the scene so it is visually distinct from its neighbors (different angle, object placement, background element, or subject action)
- **SCENE_OVERUSE**: vary the scene — use a different setting, activity, or framing that still fits the source line

Rules:
- Keep the same prompt number
- Keep the same source line verbatim
- Do NOT include a style prefix — the art style block goes at the end only
- The rewritten prompt must still match the source line content

## Output format

Return each rewritten prompt as:
NNN | [source line unchanged] | [rewritten prompt]

One per line. Return ONLY the prompt lines. No explanation.
"""


def _build_agent_script_prompt(profile: "Profile", approach_context: str = "") -> str:
    s = profile.script
    c = profile.channel
    target_words, min_words, _, hook_words, cta_words, section_min, section_max = _word_budget(s)

    char_names = " / ".join(ch["name"] for ch in profile.characters)
    char_block = profile.characters_block()

    approach_section = ""
    if approach_context.strip():
        approach_section = f"""
## Creator's Angle & Context
{approach_context}

Use this context alongside the vidIQ research to shape the angle, emphasis, and which aspects to highlight.

---
"""

    return f"""
You are writing a YouTube video script for a channel: {c["name"]}.
Follow this two-phase process exactly — never skip research to jump straight to writing.

## PHASE 1 — Research (run ALL of these in parallel)

- vidiq_keyword_research: search volume + competition for the topic
- vidiq_outliers: videos over-performing right now (reveals best angle/format)
- vidiq_youtube_search: what's already ranking (avoid duplicating it)
- vidiq_channel_analytics: channel avg views and best-performing topics
- vidiq_generate_titles: 5 title candidates using keyword data
- vidiq_score_title: score all 5 candidates — pick the highest scorer

Look for:
- High volume + low competition keywords → weave top 3-5 naturally into first 60s of narration
- Outlier videos → use their angle and hook structure, not their content
- Channel niche: {c["niche"]}
- Channel tone: {c["tone"]}
- Title format: {c["title_format"]}

## PHASE 2 — Write the script
{approach_section}
Use the winning title + vidIQ keyword insights to write a complete script matching this exact format:

```
TITLE: [winning title]
KEYWORDS: [3-5 top keywords from vidIQ]

[00:00-00:{s["hook_duration_s"]:02d}] HOOK
[provocative opening statement or surprising fact — no intro, no "welcome back", no "in this video"]

[00:{s["hook_duration_s"]:02d}-02:00] SECTION 1 — [section title]
[narration prose]
VISUAL: [which character, what action, what prop — one line per scene beat]

... {s["section_count"]} sections total ...

[CTA CLOSE — last {s["cta_duration_s"]} seconds]
[subscribe prompt only — no teasing a next video]
```

### Script rules

- Target {s["target_mins"]}:00 total (never under {s["min_mins"]}:00, never over {s["max_mins"]}:00)
- The voiceover voice runs at ~{s["wpm"]} wpm:
  - {s["target_mins"]}-min target = ~{target_words} words of narration
  - {s["min_mins"]}-min minimum = ~{min_words} words
  - Each {s["section_duration_s"]}s section = {section_min}-{section_max} words of narration
  - Hook ({s["hook_duration_s"]}s) = ~{hook_words} words. CTA ({s["cta_duration_s"]}s) = ~{cta_words} words.
  - After writing, count narration words — if under {round(min_words * 1.1)}, expand sections before finishing
- Hook hard in the first 10 seconds — lead with the most surprising fact, not context
- Every narration beat gets a VISUAL line showing a character doing an action, not reacting
- VISUAL lines must be literal: "cat holds five flat gold trophies" not "cat looks amazed"
- Vary which character appears ({char_names}) — never the same character 3 beats in a row

## Characters

{char_block}

{profile.character_behavior.strip()}

### Forbidden
- Starting the hook with "In this video…", "Welcome back…", or "Today we're going to…"
- VISUAL lines describing character emotions
- Generic visuals that could fit any moment in any video
- CTA that teases a next video
- Re-explaining a concept already introduced — each core idea appears once. Callbacks only if they add new information.

## OUTPUT

Return the finished script in this exact format — nothing after it:

===SCRIPT===
[full script here with TITLE, KEYWORDS, timestamps, section headers, narration, and VISUAL lines]
"""


def _build_vet_prompt(profile: "Profile") -> str:
    s = profile.script
    target_words, min_words, *_ = _word_budget(s)

    return f"""
You are vetting a YouTube video script for accuracy, SEO strength, and hook power.

STEP 1 — Pull live vidIQ data on the topic (run in parallel):
- vidiq_keyword_research: confirm the top keywords and their search volume
- vidiq_outliers: find the highest over-performing videos on this topic right now
- vidiq_youtube_search: see what is currently ranking and how it is framed
- vidiq_score_title: score the current script title and note the result

STEP 2 — Review the script against the data:
- FACTUAL ACCURACY: flag any claims that are outdated, exaggerated, or unsupported
- MISSING ANGLES: if the script misses the strongest outlier hook angle, note it
- KEYWORD GAPS: if the top keywords are absent from the first 60 seconds, flag them
- TITLE STRENGTH: if the vidIQ title score is below 70, propose a stronger alternative
- WORD COUNT: narration must be {round(min_words * 1.1)}-{target_words} words (voice runs at ~{s["wpm"]} wpm) — if short, expand thin sections
- IDEA REPETITION: flag any core idea explained more than once. Each mechanism or concept must appear only once — callbacks are only allowed if they add new information. Remove or rewrite any section that re-explains something already stated.

STEP 3 — Rewrite the script with all fixes applied:
Make only the changes the review identified. Do not restructure the whole script or change the channel tone.
Preserve all VISUAL lines, timestamps, and section headers exactly unless a section was expanded.

STEP 4 — Output:
Return the vetted script in this exact format — nothing after it:

===SCRIPT===
[full revised script here]
"""


def _build_agent_system_prompt(topic: str, profile: "Profile") -> str:
    c = profile.channel
    char_block = profile.characters_block()
    style      = profile.image_style["art_style_block"].strip()

    return f"""## Channel: {c["name"]}
Niche: {c["niche"]}
Audience: {c["audience"]}
Tone: {c["tone"]}

## Characters
{char_block}

{profile.character_behavior.strip()}

## Visual Style
{style}

---
TOPIC: {topic}
"""


def _build_metadata_titles_prompt(
    topic: str,
    script: str,
    research: str,
    keywords: list[dict],
    profile: "Profile",
) -> str:
    c = profile.channel
    kw_lines = "\n".join(f"- {k.get('keyword', '')}" for k in keywords if k.get("keyword"))
    kw_section = f"\nTop vidIQ keywords for this topic (work the best 1-2 into at least 3 of the titles):\n{kw_lines}\n" if kw_lines else ""

    template = f"""
You are a YouTube title strategist for a {c["niche"]} channel targeting {c["audience"]}.

Channel title format reference: {c["title_format"]}
Tone: {c["tone"]}

TOPIC: {topic}
{kw_section}
RESEARCH NOTES:
{research[:2000]}

SCRIPT HOOK (first 600 chars — use this to ground the curiosity gap):
{script[:600]}

Write 5 candidate YouTube titles for this video. Rules:
- 50-65 characters each (Google clips longer titles in search).
- Curiosity gap or counterintuitive framing — never a flat description.
- No clickbait that the script can't actually deliver.
- Title-case or sentence-case, whatever reads better — no all-caps.
- No emoji. No quote marks around the title.
- Make all 5 visibly different — different framings, not 5 paraphrases of one.

Return ONLY this format:

===TITLES===
1. [title 1]
2. [title 2]
3. [title 3]
4. [title 4]
5. [title 5]
===END===
"""
    return template


def _build_metadata_desc_hashtags_prompt(top_title, script, keywords, profile) -> str:
    c = profile.channel
    kw_line = ", ".join(k.get("keyword", "") for k in keywords if k.get("keyword"))
    kw_section = f"\nTop keywords to weave in naturally: {kw_line}\n" if kw_line else ""

    template = f"""
You are a YouTube metadata writer for a {c["niche"]} channel.

TITLE (already chosen — do NOT change it):
{top_title}

SCRIPT:
{script}
{kw_section}
Write a YouTube description AND a hashtag set.

DESCRIPTION rules:
- Start with a 1-2 sentence hook that previews the video without spoiling the payoff.
- Add a blank line.
- Then chapter timestamps derived from the script. Format each as "MM:SS Section title" on its own line. If the script has explicit timestamps or [SECTION] markers, use those; otherwise pick natural beats.
- Close with one line inviting the viewer to subscribe (tone: {c["tone"]}).
- Plain text only — no markdown, no emoji.

HASHTAGS rules:
- 5 hashtags, all lowercase, no spaces, no punctuation beyond `#`.
- Single line, space-separated.
- Mix one broad topic tag, two specific subject tags, one audience tag, one channel-vibe tag.

Return ONLY this format:

===DESCRIPTION===
[description body here]

===HASHTAGS===
[#a #b #c #d #e]
===END===
"""
    return template


def _build_metadata_thumbnail_prompt(script, topic, profile) -> str:
    c = profile.channel
    chars = profile.characters_block()
    style = profile.image_style["art_style_block"]
    template = f"""
You are writing image prompts for 3 YouTube thumbnail variants for a {c["niche"]} video.

TOPIC: {topic}

CHARACTERS (embed the full description of whichever cat appears in each prompt):
{chars}

ART STYLE (append this block to every thumbnail prompt verbatim — do not paraphrase):
{style}

SCRIPT (use the opening to extract a 1-3 word visual hook):
{script[:1500]}

THUMBNAIL RULES (apply to all 3):
1. 1920×1080 aspect ratio, composition optimized for visibility at small sizes.
2. A clear visual hook (a single bold object, action, or contrast) occupying ~60% of the frame.
3. One cat character active (pointing, holding, reacting to the hook object) — never idle, never staring at the viewer.
4. A 1-3 word marker handwriting text overlay, max one phrase per image, all uppercase, baked into the image as if hand-drawn.
5. Do NOT reference the title text — only the hook idea.

Return EXACTLY this format. Each variant needs a HOOK line (the 1-3 word overlay) and a prompt body of 150-250 words.

===THUMBNAIL_1===
HOOK: [1-3 word overlay text in caps]
[prompt body for variant 1]

===THUMBNAIL_2===
HOOK: [different 1-3 word overlay]
[prompt body for variant 2]

===THUMBNAIL_3===
HOOK: [different 1-3 word overlay]
[prompt body for variant 3]
===END===
"""
    return template
