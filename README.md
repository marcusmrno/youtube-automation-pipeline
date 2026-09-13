# YouTube Pipeline

[![tests](https://github.com/marcusmrno/youtube-automation-pipeline/actions/workflows/tests.yml/badge.svg)](https://github.com/marcusmrno/youtube-automation-pipeline/actions/workflows/tests.yml)
![python](https://img.shields.io/badge/python-3.11%2B-blue)
![license](https://img.shields.io/badge/license-MIT-green)

An asset pipeline for a faceless educational YouTube channel. Input a topic — the system researches it, writes a structured script, generates ~150 style-consistent images, and records a voiceover, leaving a complete asset set to drop into an editor. Every stage is automated, resumable, and runs on a multi-model AI stack.

**Scope:** the pipeline produces images, audio, script, and SEO metadata — it does not render a video file. Final assembly happens in a video editor. `apply_flicker.py` can add a flicker overlay track to a Palmier project once the clips are placed.

![Sample run — all 28 frames from a single pipeline run](assets/sample-run.gif)

*Every frame from one run, in order, generated with the bundled `example` profile. Nothing was
hand-picked or retouched — the style consistency and the in-image text come from the profile's
`art_style_block` and `scene_rules`, not from post-processing. The run also produced a 1:47
narration track from the same script.*

---

## Tech Stack

| Layer | Technology |
|-------|-----------|
| **Orchestration** | Python — threaded pipeline with checkpoint-based resumability |
| **LLM** | Anthropic Claude (Sonnet + Haiku) — research, script writing, script planning, TTS enhancement, revision |
| **AI Agents** | Claude Agent SDK, connected to vidIQ's MCP server as a tool source — script generation, SEO vetting, keyword research |
| **Image generation** | Google Gemini (`gemini-3.1-flash-image` / `gemini-3-pro-image`) — ~130-150 images per 12-minute video with style anchor references, generated in parallel |
| **Text-to-speech** | ElevenLabs v3 — chunked MP3 generation with custom audio tags for emotion and pacing |
| **Web UI** | Flask + vanilla JS — real-time SSE log streaming, image gallery, lightbox, script review modal |
| **Bot interface** | python-telegram-bot — full pipeline control via Telegram with inline keyboard interactions |

---

## What it does

```
topic → clarifying questions → approach selection → research → script
      → [approval gate] → TTS narration → image prompts → images + voiceover
      → (optional) flicker pass, metadata + thumbnails
```

| Stage | Detail |
|-------|--------|
| **Script planning** | Claude generates targeted clarifying questions with evidence-backed defaults, then pitches 3 distinct video angles — creator picks one before anything is written |
| **Research** | vidIQ agent pulls keyword data, outlier analysis, and title scoring to ground the script in what actually performs |
| **Script** | Claude Sonnet writes a full structured script (~12 min) shaped by the chosen approach, with timestamps and section markers |
| **Vet** | Second agent pass fact-checks, closes SEO gaps, and enforces word count targets |
| **Approval gate** | Pipeline pauses for human review — script can be edited, revised with feedback, or rejected before any paid generation |
| **TTS narration** | Claude Haiku strips stage directions and adds ElevenLabs v3 audio tags (emotion, pacing, texture) to the clean narration |
| **Image prompts** | One prompt per ~3–4 seconds of narration. Sonnet writes only the scene (`NNN \| source sentence \| character \| scene`); the pipeline stitches in the profile's character descriptions and art-style block, and stores the expanded form as `NNN \| source \| prompt` |
| **Images** | Gemini generates each image against a character style sheet and persistent anchor references for visual consistency |
| **Voiceover** | ElevenLabs generates chunked audio in parallel with image generation |
| **Flicker pass (optional, manual)** | If the profile enables `image_gen.flicker`, each image gets stretched b/c variants generated alongside it; `apply_flicker.py` then layers alternating b/c segments over the placed clips in the live Palmier project for a hand-drawn flicker effect |
| **Metadata & thumbnails (optional, manual)** | `python pipeline.py metadata <slug>` (or the UI/bot equivalent) generates titles, description, hashtags, and thumbnail options after a run completes |

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

Requires Python 3.11+.

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
```

A profile's `voice.voice_id` can also reference an arbitrary env var with `${VAR_NAME}` syntax, resolved at load time — not limited to the fixed list above.

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

**Starting from a script you already wrote:**
- Switch the input card to **Script**, paste it, and click **Produce From Script** — no research, writing, or approval gate, straight to TTS, image prompts, images, and voiceover
- The run is named after the script's `TITLE:` line unless you fill in **Run Name**; a run folder of that name is overwritten

**Image gallery:**
- Browse all generated images at consistent size with scroll
- Lightbox shows the matching script section for each image's timestamp; image prompt available as a collapsible dropdown
- Mark images for regeneration while browsing, then regen all flagged at once

**Metadata:**
- "Metadata & Thumbnail" section (or `POST /metadata/<slug>/generate`) generates titles, description, hashtags, and thumbnails; pick the chosen thumbnail from the UI

### Telegram bot

```bash
./run_bot.sh
```

| Command | Description |
|---------|-------------|
| `/run <topic>` | Starts the planning flow — clarifying questions → approach pitches → pipeline |
| `/script` | Produce from a script you already wrote — then paste it, or upload it as a `.txt` |
| `/resume [slug]` | Resume an incomplete run |
| `/runs` | List recent runs with status |
| `/status` | Show status of the current/last run |
| `/download [slug]` | Download run assets as a zip (splits at 50 MB) |
| `/metadata [slug] [regenerate]` | Generate or show SEO metadata + thumbnails |
| `/profile` | List/switch channel profiles |
| `/stop` | Cancel the current run |
| `/help` | Show command help (alias of `/start`) |

**Bot planning flow:**
1. `/run why do we dream` → bot sends clarifying questions with suggested defaults
2. Reply with numbered answers, or send `default` to use all suggestions
3. Bot pitches 3 approaches as inline buttons — tap to select
4. Script is sent for review before production begins

### CLI

```bash
python pipeline.py "why humans sleep"
```

Already have a script? Skip research, writing, and the approval gate — go straight to
TTS, image prompts, images, and voiceover. The run folder is named after the script's
`TITLE:` line unless `--topic` says otherwise.

```bash
python pipeline.py script my-script.txt           # --profile optional when only one profile exists
python pipeline.py script - --profile <name>      # read the script from stdin
```

### Metadata & Thumbnail

After a run completes, generate SEO metadata and thumbnails:

```bash
python pipeline.py metadata <run-slug>               # generate 5 titles, description, hashtags, 3 thumbnails
python pipeline.py metadata <run-slug> --regenerate  # overwrite existing metadata.json
python pipeline.py metadata <run-slug> --pick-thumb N    # 1-based; pick which thumbnail is "chosen"
python pipeline.py metadata <run-slug> --show         # print current metadata.json
```

Also available as `/metadata` in the Telegram bot, and as a "Metadata & Thumbnail" section in the Flask web UI.

### Flicker effect

There are two independent flicker mechanisms:

- **Built into the pipeline** — if a profile sets `image_gen.flicker.enabled: true`, `pipeline.py` generates horizontally/vertically stretched `NNNb.png`/`NNNc.png` variants alongside every image.
- **`apply_flicker.py`** — a standalone script, unrelated to `pipeline.py`'s internal logic, that adds a flicker overlay track to whatever is *already open* in a live Palmier project by reusing `NNN`/`NNNb`/`NNNc` media already imported there:

```bash
python apply_flicker.py --interval 6           # add alternating b/c flicker layer
python apply_flicker.py --interval 6 --dry-run # preview without writing
python apply_flicker.py --undo                 # remove the flicker layer (uses .flicker_snapshot.json)
```

---

## Output structure

```
output/why-humans-sleep/
├── profile.txt            # name of the profile used for this run
├── research.txt           # verified facts and hook angles (or the agent's vidIQ findings)
├── script.txt             # full script with timestamps and narration
├── tts_script.txt         # clean narration with ElevenLabs audio tags
├── image_prompts.txt      # NNN | [source narration sentence] | [full prompt] per image
├── metadata.json          # titles, description, hashtags, thumbnail choice (after `metadata` step)
├── thumbnail.png          # copy of the chosen thumbnail
├── images/
│   ├── 001.png
│   ├── flicker/           # NNNb.png / NNNc.png stretch variants — only if profile.image_gen.flicker is enabled
│   └── ...
├── audio/
│   └── voiceover.mp3
└── thumbnails/
    ├── thumb-01.png        # rendered thumbnail candidates (after `metadata` step)
    ├── thumb-02.png
    └── thumb-03.png
```

---

## Channel profiles

A profile is the complete identity spec for a channel — it drives the script tone, character descriptions, image style rules, voice settings, and the anchor reference images used for visual consistency across every video.

```
profiles/my-channel/
├── profile.yaml          # channel identity, characters, script settings, voice, image style
├── style-sheet.md        # extended visual rules injected into every image prompt
└── anchors/
    ├── manifest.yaml     # what each anchor slot is — read back to describe the references
    ├── anchor-01.png     # first reference slot (layout depends on the roster size)
    ├── anchor-02.png
    └── ...               # additional style anchors (environments, props, lighting)
```

The anchor layout is not fixed: `build_anchor_plan` derives it from the roster, so a two-character
profile starts with a cast sheet while a character-free one starts with an example scene.
`manifest.yaml` records what each slot actually is. Without it the pipeline falls back to a generic
"match these references" line rather than guessing.

### What's inside profile.yaml

```yaml
channel:
  name: "Deep Field"
  niche: "astronomy and space science"
  audience: "curious adults who want the real physics"
  tone: "precise, unhurried, quietly astonished"
  reference_channel: "PBS Space Time"
  title_format: "What [Object] Actually [Does] — And Why It Matters"

script:
  target_mins: 12
  wpm: 150
  hook_duration_s: 30
  section_count: 6
  section_duration_s: "90-120"

characters:
  roster:
    - name: "The Surveyor"
      description: "Small figure in a rounded retro spacesuit, blank visor..."
  behavior: |
    The Surveyor observes and measures — it never emotes.

image_style:
  art_style_block: |
    Two-color risograph screenprint. Exactly two inks: fluorescent orange...
  style_constraints: >
    Two inks only. Visible halftone dots. No black, no gradients, no photorealism.
  scene_rules: |
    ## Composition
    - Every scene states its scale explicitly...
```

Every field flows directly into LLM prompts — the script writer, TTS enhancer, and image prompt builder all read from the active profile. The pipeline itself holds no style of its own:

- `art_style_block` is appended to every image prompt
- `style_constraints` is sent *ahead* of every image prompt and every anchor generation, as hard rules (falls back to `art_style_block` if omitted)
- `scene_rules` is injected as the composition guide for the image prompt writer
- `characters` is fully optional — omit the block for a style with no recurring characters, and every character-specific instruction drops out of the prompts automatically

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

Model IDs are not hardcoded — they come from the active profile's `image_gen` block. The
regenerate selector in the UI picks between the two slots by alias:

| Alias | Profile field | Notes |
|-------|---------------|-------|
| `nano-banana-2` | `image_gen.default_model` | Default — fast and cost-effective for full runs |
| `3-pro` | `image_gen.pro_model` | Highest quality — used for hero images, anchors, and thumbnails |

`profiles/example` ships with `gemini-3.1-flash-image` and `gemini-3-pro-image` in those slots.

---

## Tests

```bash
python -m pytest tests/ test_pipeline_helpers.py test_bot_script_mode.py -q
```

`tests/` is the pytest suite. The top-level `test_*.py` files are assert-based checks that
pytest collects but that also run on their own — `python test_pipeline_helpers.py`.

---

## Project structure

```
youtube-pipeline/
├── pipeline.py           # core orchestrator — all stage logic and resumability
├── prompts.py            # LLM prompt builders — pure functions, no API calls
├── agents.py             # Claude Agent SDK runners connected to vidIQ's MCP server (script + vet)
├── metadata.py           # titles/description/hashtags/thumbnail generation, metadata.json
├── profile.py            # Profile dataclass + YAML loader
├── apply_flicker.py      # standalone CLI — adds/undoes a flicker overlay in a live Palmier project
├── ui.py                 # Flask web UI + SSE streaming endpoints (gallery, metadata)
├── bot.py                # Telegram bot interface
├── create_profile.py     # interactive profile creator CLI
├── preview_prompts.py    # script → tts_script.txt + image_prompts.txt, no images generated
├── profile_creator/      # profile creation subpackage
├── profiles/example/     # fully annotated profile schema — copy to get started
├── templates/index.html  # web UI (vanilla JS, SSE, gallery, lightbox)
├── tests/                # pytest suite (pipeline parsing, metadata, profile, create_profile)
├── test_*.py             # assert-based checks that also run standalone: `python test_<name>.py`
├── docs/                 # design docs for past features (gitignored)
└── output/               # generated assets per run (gitignored)
```

---

## License

MIT — see [LICENSE](LICENSE).
