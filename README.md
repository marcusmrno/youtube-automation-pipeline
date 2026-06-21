# YouTube Pipeline

Automated end-to-end video production for a faceless educational YouTube channel. Give it a topic — it researches, writes a script, generates ~300 images, records a voiceover, and assembles a timeline in Palmier Pro.

---

## What it does

```
topic → research + script → TTS narration → image prompts → images → voiceover → Palmier timeline
```

1. **Research** — vidIQ keyword research + outlier analysis + title scoring
2. **Script** — full structured script with timestamps, sections, and VISUAL lines (~12 min target)
3. **Vet** — second vidIQ pass checks facts, SEO gaps, and word count
4. **Script approval** — pauses for review before any paid generation
5. **TTS narration** — ElevenLabs v3 with audio tags (emotion, pacing, texture)
6. **Image prompts** — one prompt per ~3-4 seconds of narration
7. **Image generation** — Google Gemini (`gemini-3.1-flash-image`) with character reference sheets + style anchors
8. **Voiceover** — ElevenLabs chunked MP3, runs in parallel with image gen
9. **Palmier assembly** — imports all assets and places clips on the timeline

---

## Setup

```bash
pip install -r requirements.txt
```

Copy `.env.example` to `.env` and fill in your keys:

```env
ANTHROPIC_API_KEY=sk-ant-...
GOOGLE_API_KEY=...
ELEVENLABS_API_KEY=...
ELEVENLABS_VOICE_ID=...
VIDIQ_API_KEY=...                  # optional — enables agent-based research
TELEGRAM_BOT_TOKEN=...             # optional — for bot.py
TELEGRAM_USER_ID=...               # optional — your Telegram user ID
PALMIER_MCP_URL=http://127.0.0.1:19789/mcp   # optional — for timeline assembly
```

---

## Running

**Web UI (recommended)**
```bash
python ui.py
# Opens at http://localhost:7860
```

**CLI**
```bash
python pipeline.py "why humans sleep"
```

**Telegram bot**
```bash
./run_bot.sh        # macOS / Linux
start_bot.bat       # Windows
```

### Telegram bot commands

| Command | Description |
|---------|-------------|
| `/run <topic>` | Start a new pipeline run for the given topic |
| `/runs` | List the 15 most recent runs with image count and status |
| `/status` | Show current run progress, or summary of the last completed run |
| `/resume [slug]` | Resume an incomplete run (omit slug to resume the most recent incomplete) |
| `/download [slug]` | Download the full run as zip(s) — splits into multiple parts if over 50 MB; omit slug for latest |
| `/stop` | Cancel the current run |
| `/help` | Show command list |

---

## Output

Everything saves to `output/{slug}/`:

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

## Project structure

```
youtube-pipeline/
├── pipeline.py           # core orchestrator — research, script, images, audio, Palmier
├── prompts.py            # all LLM prompt builders (pure functions, no API calls)
├── agents.py             # vidIQ Claude agent runners (script + vet)
├── profile.py            # Profile dataclass + loader
├── ui.py                 # Flask web UI
├── bot.py                # Telegram bot interface
├── create_profile.py     # CLI entry point for profile creator
├── profile_creator/      # profile creation subpackage
│   ├── new.py            # new profile creation flow
│   ├── revise.py         # profile revision flow (versioned copies)
│   ├── claude_helpers.py # Claude conversation helpers
│   ├── anchors.py        # anchor image generation orchestration
│   └── image_gen.py      # shared Google image gen helper
├── profiles/             # one subfolder per channel profile
│   └── cat-educational/
│       ├── profile.yaml
│       ├── style-sheet.md
│       └── anchors/      # reference images passed to image gen
├── templates/
│   └── index.html        # web UI template
├── tests/
│   ├── test_pipeline_prompts.py
│   ├── test_profile.py
│   └── test_create_profile.py
└── output/               # generated assets (gitignored)
```

---

## Channel profiles

A profile defines everything about a channel — identity, characters, voice settings, image style, and anchor reference images. Profiles live under `profiles/<name>/`.

**Create a new profile**
```bash
python create_profile.py
```

**Revise an existing profile**
```bash
python create_profile.py --revise cat-educational
```

The profile creator runs an interactive Claude conversation, generates anchor images for style approval, and writes a `profile.yaml` ready to use with the pipeline.

---

## Characters

Two flat 2D cat mascots appear in every video as visual helpers — they hold objects, point at diagrams, and stand in as example figures. They do not narrate or react emotionally.

- **Orange Cat** — large orange tabby, black bowtie, explainer role
- **White Cat** — small white fluffy cat, pink collar, grey eye patch, reactor role

Character reference sheets (`anchor-01.png`) are always passed first to the image model to lock in the designs.

---

## Resuming a run

If a run is interrupted, resume it without re-running research or script:

```python
from pipeline import resume_pipeline
resume_pipeline("why-humans-sleep")
```

Or use the Resume tab in the web UI.
