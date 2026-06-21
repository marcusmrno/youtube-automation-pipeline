# YouTube Pipeline

Automated end-to-end video production for a faceless educational YouTube channel. Give it a topic — it researches, writes a script, generates ~300 images, records a voiceover, and assembles a timeline in Palmier Pro.

---

## How it works

```
topic → research → script → [approval gate] → TTS narration → image prompts → images + voiceover → Palmier timeline
```

| Step | What happens |
|------|-------------|
| **Research** | vidIQ keyword research + outlier analysis + title scoring |
| **Script** | Full structured script with timestamps, sections, and VISUAL lines (~12 min target) |
| **Vet** | Second vidIQ pass checks facts, SEO gaps, and word count |
| **Approval gate** | Pipeline pauses — you review and approve the script before any paid generation |
| **TTS narration** | ElevenLabs v3 script with audio tags (emotion, pacing, texture) |
| **Image prompts** | One prompt per ~3–4 seconds of narration, format: `NNN \| MM:SS-MM:SS \| [description]` |
| **Images** | Google Gemini (`gemini-3.1-flash-image`) with character reference sheets + style anchors |
| **Voiceover** | ElevenLabs chunked MP3, runs in parallel with image gen |
| **Palmier** | Imports all assets and places clips on the timeline |

Every step saves its output to `output/{slug}/` as a checkpoint, so any step can be resumed or rerun independently.

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
VIDIQ_API_KEY=...                  # optional — enables agent-based research and vetting; falls back to standard Claude research without it
TELEGRAM_BOT_TOKEN=...             # optional — for bot.py
TELEGRAM_USER_ID=...               # optional — your Telegram user ID
PALMIER_MCP_URL=http://127.0.0.1:19789/mcp   # optional — for timeline assembly
```

---

## Running the pipeline

### Web UI (recommended)

```bash
python ui.py
# Opens at http://localhost:7860
```

The UI has three tabs:

- **Run** — start a new pipeline from a topic. The script approval gate appears as a modal before generation begins.
- **Resume** — pick up any interrupted run from where it left off.
- **Regen** — regenerate individual images by number without rerunning the full pipeline.

### CLI

```bash
python pipeline.py "why humans sleep"
```

The script prints to stdout and prompts you to approve before generation starts.

### Telegram bot

```bash
./run_bot.sh        # macOS / Linux
start_bot.bat       # Windows
```

| Command | Description |
|---------|-------------|
| `/run <topic>` | Start a new pipeline run for the given topic |
| `/runs` | List the 15 most recent runs with image count and status |
| `/status` | Show current run progress, or summary of the last completed run |
| `/resume [slug]` | Resume an incomplete run (omit slug to resume the most recent incomplete) |
| `/download [slug]` | Download the full run as a zip — splits into multiple parts if over 50 MB |
| `/stop` | Cancel the current run |
| `/help` | Show command list |

The bot sends the script via message for approval before generating anything.

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

Each file is a checkpoint. Resume picks up from where it left off without re-running research or script generation.

---

## Resuming a run

**UI:** use the Resume tab and select the run slug.

**CLI:**
```python
from pipeline import resume_pipeline
resume_pipeline("why-humans-sleep")
```

**Bot:** `/resume why-humans-sleep` (or just `/resume` for the most recent incomplete run).

Resume detects which files are missing (`tts_script.txt`, `image_prompts.txt`, images) and regenerates only what's needed.

---

## Regenerating individual images

If specific images look wrong, you don't need to rerun the whole pipeline:

- **UI Regen tab:** enter the run slug and a comma-separated list of image numbers (e.g. `12, 45, 103`).
- **CLI:** `regen_images("why-humans-sleep", [12, 45, 103])`

You can also edit a prompt in `image_prompts.txt` directly and regen that image number to apply the change.

---

## Channel profiles

A profile defines everything about a channel — identity, characters, voice settings, image style, and anchor reference images. Profiles live under `profiles/<name>/`.

```
profiles/my-channel/
├── profile.yaml          # identity, characters, voice settings, image style rules
├── style-sheet.md        # extended style notes used by the image prompt builder
└── anchors/
    ├── anchor-01.png     # primary style reference (always sent first to the model)
    ├── anchor-02.png
    └── ...
```

See `profiles/example/profile.yaml` for a fully annotated schema with every supported field.

### Creating a profile

```bash
python create_profile.py
```

Runs an interactive Claude conversation, generates anchor images for style approval, and writes a `profile.yaml` ready to use.

### Revising an existing profile

```bash
python create_profile.py --revise my-channel
```

Saves a versioned copy of the old profile before making changes so you can roll back.

### Switching profiles

```bash
python pipeline.py "why humans sleep" --profile my-channel
```

Or select the profile in the web UI dropdown.

---

## Image generation models

| Alias | Model | Notes |
|-------|-------|-------|
| `nano-banana-2` | `gemini-3.1-flash-image` | Default — fast and cost-effective for full runs |
| `2.5-flash` | `gemini-3.1-flash-image` | Same as above |
| `3-pro` | `gemini-3-pro-image` | Highest quality, slowest — use for hero images or anchors |

---

## Project structure

```
youtube-pipeline/
├── pipeline.py           # core orchestrator — research, script, images, audio, Palmier
├── prompts.py            # all LLM prompt builders (pure functions, no API calls)
├── agents.py             # vidIQ Claude agent runners (script + vet)
├── profile.py            # Profile dataclass + loader
├── ui.py                 # Gradio web UI
├── bot.py                # Telegram bot interface
├── create_profile.py     # CLI entry point for profile creator
├── profile_creator/      # profile creation subpackage
│   ├── new.py            # new profile creation flow
│   ├── revise.py         # profile revision flow (versioned copies)
│   ├── claude_helpers.py # Claude conversation helpers
│   ├── anchors.py        # anchor image generation orchestration
│   └── image_gen.py      # shared Google image gen helper
├── profiles/             # gitignored — your profiles live here
│   └── example/          # committed reference — copy and rename to get started
│       ├── profile.yaml  # full annotated schema with every supported field
│       ├── style-sheet.md
│       └── anchors/      # drop anchor-01.png, anchor-02.png … here
├── templates/
│   └── index.html        # web UI template
├── tests/
│   ├── test_pipeline_prompts.py
│   ├── test_profile.py
│   └── test_create_profile.py
└── output/               # generated assets (gitignored)
```
