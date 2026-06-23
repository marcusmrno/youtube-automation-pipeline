# YouTube Pipeline

An end-to-end AI video production system for a faceless educational YouTube channel. Input a topic — the system researches it, writes a structured script, generates ~300 images, records a voiceover, and assembles a timeline ready for export. Every stage is fully automated, resumable, and runs on a multi-model AI stack.

---

## Tech Stack

| Layer | Technology |
|-------|-----------|
| **Orchestration** | Python — async pipeline with checkpoint-based resumability |
| **LLM** | Anthropic Claude (Sonnet + Haiku) — research, script writing, script planning, TTS enhancement, revision |
| **AI Agents** | Claude-powered agents via vidIQ API — script generation, SEO vetting, keyword research |
| **Image generation** | Google Gemini (`gemini-3.1-flash-image` / `gemini-3-pro-image`) — ~300 images per video with style anchor references |
| **Text-to-speech** | ElevenLabs v3 — chunked MP3 generation with custom audio tags for emotion and pacing |
| **Web UI** | Flask + vanilla JS — real-time SSE log streaming, image gallery, lightbox, script review modal |
| **Bot interface** | python-telegram-bot — full pipeline control via Telegram with inline keyboard interactions |
| **Timeline assembly** | Palmier Pro via MCP (Model Context Protocol) — programmatic clip placement and keyframe animation |

---

## What it does

```
topic → clarifying questions → approach selection → research → script
      → [approval gate] → TTS narration → image prompts → images + voiceover → Palmier timeline
```

| Stage | Detail |
|-------|--------|
| **Script planning** | Claude generates targeted clarifying questions with evidence-backed defaults, then pitches 3 distinct video angles — creator picks one before anything is written |
| **Research** | vidIQ agent pulls keyword data, outlier analysis, and title scoring to ground the script in what actually performs |
| **Script** | Claude Sonnet writes a full structured script (~12 min) shaped by the chosen approach, with timestamps and section markers |
| **Vet** | Second agent pass fact-checks, closes SEO gaps, and enforces word count targets |
| **Approval gate** | Pipeline pauses for human review — script can be edited, revised with feedback, or rejected before any paid generation |
| **TTS narration** | Claude Haiku strips stage directions and adds ElevenLabs v3 audio tags (emotion, pacing, texture) to the clean narration |
| **Image prompts** | One detailed prompt per ~3–4 seconds of narration (`NNN \| MM:SS-MM:SS \| [description]`) with embedded character and style rules |
| **Images** | Gemini generates each image against a character style sheet and persistent anchor references for visual consistency |
| **Voiceover** | ElevenLabs generates chunked audio in parallel with image generation |
| **Timeline** | All assets are imported into Palmier Pro and placed programmatically via MCP |

---

## Architecture highlights

- **Checkpoint-based resumability** — every stage writes its output to disk; the pipeline detects what's missing and resumes from that point without re-running earlier work
- **Multi-model routing** — Sonnet for creative/long-form tasks, Haiku for extraction and formatting; model selection is per-task not global
- **Real-time SSE streaming** — the web UI receives live log events from the pipeline thread via Server-Sent Events, with per-stage progress tracking
- **Approach context propagation** — the angle chosen during planning is injected into the script prompt, so the final script reflects the creator's intent end-to-end
- **Dual interface parity** — the full planning → script → production flow works identically in the web UI and Telegram bot; no features are UI-only
- **Channel profiles** — a YAML-driven profile system defines channel identity, character descriptions, voice settings, and image style rules; multiple profiles can be maintained and switched per run

---

## Setup

```bash
pip install -r requirements.txt
```

Copy `.env.example` to `.env`:

```env
ANTHROPIC_API_KEY=sk-ant-...
GOOGLE_API_KEY=...
ELEVENLABS_API_KEY=...
ELEVENLABS_VOICE_ID=...
VIDIQ_API_KEY=...           # optional — enables agent research + vetting; falls back to standard Claude research
TELEGRAM_BOT_TOKEN=...      # optional — for bot.py
TELEGRAM_USER_ID=...        # optional — your Telegram user ID
PALMIER_MCP_URL=http://127.0.0.1:19789/mcp   # optional — for timeline assembly
```

---

## Running the pipeline

### Web UI

```bash
python ui.py
# Opens at http://localhost:7860
```

**Starting a run:**
1. Enter your topic and click **Run Pipeline**
2. Answer 4-5 clarifying questions (each has a suggested default — click **↓ Use All Defaults** to fill them all at once)
3. Pick one of 3 pitched approaches
4. Pipeline runs, then pauses at the script approval gate before any paid generation
5. Edit, revise with feedback, or approve — production starts on approval

**Image gallery:**
- Browse all generated images at consistent size with scroll
- Lightbox shows the matching script section for each image's timestamp; image prompt available as a collapsible dropdown
- Mark images for regeneration while browsing, then regen all flagged at once

### Telegram bot

```bash
./run_bot.sh
```

| Command | Description |
|---------|-------------|
| `/run <topic>` | Starts the planning flow — clarifying questions → approach pitches → pipeline |
| `/resume [slug]` | Resume an incomplete run |
| `/runs` | List recent runs with status |
| `/download [slug]` | Download run assets as a zip (splits at 50 MB) |
| `/stop` | Cancel the current run |

**Bot planning flow:**
1. `/run why do we dream` → bot sends clarifying questions with suggested defaults
2. Reply with numbered answers, or send `default` to use all suggestions
3. Bot pitches 3 approaches as inline buttons — tap to select
4. Script is sent for review before production begins

### CLI

```bash
python pipeline.py "why humans sleep"
```

---

## Output structure

```
output/why-humans-sleep/
├── research.txt          # verified facts and hook angles
├── script.txt            # full script with timestamps and VISUAL lines
├── tts_script.txt        # clean narration with ElevenLabs audio tags
├── image_prompts.txt     # NNN | MM:SS-MM:SS | [prompt] per image
├── images/
│   ├── 001.jpg
│   └── ...
└── audio/
    └── voiceover.mp3
```

---

## Channel profiles

A profile is the complete identity spec for a channel — it drives the script tone, character descriptions, image style rules, voice settings, and the anchor reference images used for visual consistency across every video.

```
profiles/my-channel/
├── profile.yaml          # channel identity, characters, script settings, voice, image style
├── style-sheet.md        # extended visual rules injected into every image prompt
└── anchors/
    ├── anchor-01.png     # character reference sheet — always sent first to the image model
    ├── anchor-02.png     # scene style reference
    └── ...               # additional style anchors (environments, props, lighting)
```

### What's inside profile.yaml

```yaml
channel:
  name: "My Channel"
  niche: "educational science"
  audience: "curious adults 25-40"
  tone: "warm, precise, slightly dry"
  reference_channel: "Kurzgesagt"
  title_format: "Why [surprising claim]"

script:
  target_mins: 12
  wpm: 160
  hook_duration_s: 35
  section_count: "10-14"
  section_duration_s: "60-90"

characters:
  roster:
    - name: "Orange Cat"
      description: "Large orange tabby, slightly lopsided round head..."
  behavior: |
    Characters react to information with curiosity, never alarm.

image_style:
  art_style_block: |
    Flat 2D illustration on off-white paper texture, bold black outlines...
```

Every field flows directly into LLM prompts — the script writer, TTS enhancer, and image prompt builder all read from the active profile.

---

## Profile creator

The profile creator is a Claude-powered interactive CLI that takes your channel concept and produces a complete, ready-to-use profile — including all anchor images — from a free-form brain dump.

```bash
python create_profile.py
```

### How it works

**1. Brain dump**
Paste a free-form description of your channel concept — niche, vibe, characters, visual style, reference channels. No specific format required.

**2. Clarification loop**
Claude asks follow-up questions to pin down anything ambiguous — character appearance details, art style specifics, audience tone. You answer conversationally until it has enough to work with.

**3. Profile generation**
Claude produces a validated `profile.yaml` and a `style-sheet.md` with detailed visual rules for the image prompt builder.

**4. Anchor image generation (tiered)**
The creator builds a structured anchor plan from your profile, then generates images in two passes:

- **Verification tier** — character reference sheets (full body, multiple angles) generated first and shown to you before proceeding. These establish the "ground truth" appearance for every character.
- **Full tier** — environment anchors, scene style references, prop sheets, and sky/lighting variants generated after you approve the characters.

All anchors are generated with Gemini and saved to `anchors/`. The image pipeline always sends `anchor-01` first (the cast sheet), followed by the rest in order, so the model has consistent visual context for every image in a run.

**Seed image** — if you already have a visual reference, pass it as a seed:
```bash
python create_profile.py --seed path/to/reference.png
```
It's installed as `anchor-00` and sent before all other anchors, giving the model a concrete style target to match.

---

### Revising a profile

```bash
python create_profile.py --revise my-channel
# or interactively: python create_profile.py → choose [r]evise
```

Revision creates a versioned copy (`my-channel-v2`) rather than overwriting the original, so you can always roll back.

The revise flow:
1. Describe what you want to change
2. Claude asks clarifying questions if needed
3. Updated `profile.yaml` and `style-sheet.md` are written to the versioned folder
4. **Smart anchor detection** — if characters or core style fields changed, anchors are automatically regenerated. If only metadata changed (title format, script settings, etc.), you're asked whether to regenerate or copy the originals.

Versioning follows `my-channel` → `my-channel-v2` → `my-channel-v3` automatically.

---

## Image generation models

| Alias | Model | Notes |
|-------|-------|-------|
| `nano-banana-2` | `gemini-3.1-flash-image` | Default — fast and cost-effective for full runs |
| `3-pro` | `gemini-3-pro-image` | Highest quality — use for hero images or anchors |

---

## Project structure

```
youtube-pipeline/
├── pipeline.py           # core orchestrator — all stage logic and resumability
├── prompts.py            # LLM prompt builders — pure functions, no API calls
├── agents.py             # vidIQ Claude agent runners (script + vet)
├── profile.py            # Profile dataclass + YAML loader
├── ui.py                 # Flask web UI + SSE streaming endpoints
├── bot.py                # Telegram bot interface
├── create_profile.py     # interactive profile creator CLI
├── profile_creator/      # profile creation subpackage
├── profiles/example/     # fully annotated profile schema — copy to get started
├── templates/index.html  # web UI (vanilla JS, SSE, gallery, lightbox)
└── output/               # generated assets per run (gitignored)
```
