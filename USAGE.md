# Usage reference

The full reference for the pipeline. For the overview, see the [README](README.md).

- [Pipeline stages](#pipeline-stages)
- [Running the pipeline](#running-the-pipeline)
- [Output structure](#output-structure)
- [Channel profiles](#channel-profiles)
- [Profile creator](#profile-creator)
- [Image generation models](#image-generation-models)
- [Environment notes](#environment-notes)

---

## Pipeline stages

| Stage | Detail |
|-------|--------|
| **Script planning** | Claude generates targeted clarifying questions with evidence-backed defaults, then pitches 3 distinct video angles — creator picks one before anything is written (web UI and bot; the CLI starts at research) |
| **Research** | vidIQ agent pulls keyword data, outlier analysis, and title scoring to ground the script in what actually performs (without `VIDIQ_API_KEY`, Claude Haiku researches the topic instead) |
| **Script** | Claude writes a full structured script (the profile's target length; 12 min in the example) shaped by the chosen approach, with timestamps and section markers — Opus via the vidIQ agent when `VIDIQ_API_KEY` is set, Sonnet otherwise |
| **Vet** | With `VIDIQ_API_KEY`, a second agent pass (Haiku + vidIQ) fact-checks, closes SEO gaps, and enforces word count targets |
| **Approval gate** | Pipeline pauses for human review before any paid generation — the script can be edited (web UI), revised with feedback (web UI and bot), approved or rejected (all three; the CLI asks y/N) |
| **TTS narration** | Claude Haiku strips stage directions and adds ElevenLabs v3 audio tags (emotion, pacing, texture) to the clean narration |
| **Image prompts** | Sonnet is asked for one prompt per ~3–4 seconds of narration (real runs land around 130–150 for 12 minutes). It writes only the scene (`NNN \| source sentence \| character \| scene`); the pipeline stitches in the profile's character descriptions and art-style block, and stores the expanded form as `NNN \| source \| prompt` |
| **Images** | Gemini generates each image against a character style sheet and persistent anchor references for visual consistency |
| **Voiceover** | ElevenLabs generates chunked audio in parallel with image generation |
| **Metadata & thumbnails (optional, manual)** | `python pipeline.py metadata <slug>` (or the UI/bot equivalent) generates titles, description, hashtags, and thumbnail options after a run completes |

---

## Running the pipeline

### Web UI

```bash
python ui.py
# Opens at http://localhost:7860
```

**Starting a run:**
1. Pick a channel profile (required when there are several), enter your topic and click **Run Pipeline**
2. Answer 4-5 clarifying questions (each has a suggested default — click **↓ Use All Defaults** to fill them all at once)
3. Pick one of 3 pitched approaches
4. Pipeline runs, then pauses at the script approval gate before any paid generation
5. Edit, revise with feedback, or approve — production starts on approval

**Starting from a script you already wrote:**
- Switch the input card to **Script**, paste it, and click **Produce From Script** — no research, writing, or approval gate, straight to TTS, image prompts, images, and voiceover
- The run is named after the script's `TITLE:` line unless you fill in **Run Name**; a run folder of that name is overwritten

**Image gallery:**
- Browse all generated images at consistent size with scroll
- Lightbox shows the matching script section for each image's source sentence; image prompt available as a collapsible dropdown
- Mark images for regeneration while browsing, then regen all flagged at once

**Metadata:**
- The **Metadata** mode (or `POST /metadata/<slug>/generate` with a JSON body) generates titles, description, hashtags, and thumbnails; pick the chosen thumbnail from the UI

### Telegram bot

```bash
python bot.py
```

| Command | Description |
|---------|-------------|
| `/run <topic>` | Starts the planning flow — clarifying questions → approach pitches → pipeline |
| `/script [run name]` | Produce from a script you already wrote — then paste it, or upload it as a `.txt`; the run name overrides its `TITLE:` line |
| `/resume [slug]` | Resume an incomplete run |
| `/runs` | List recent runs with status |
| `/status` | Show status of the current/last run |
| `/download [slug]` | Download run assets as a zip (split into parts under Telegram's 50 MB limit) |
| `/metadata [slug] [regenerate]` | Generate or show SEO metadata + thumbnails |
| `/profile` | List/switch channel profiles (pick one before /run or /script when several exist) |
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
python pipeline.py run "script writing tips"      # a topic starting with a command word needs "run"
```

The CLI skips the clarifying questions and approach pitches. At the approval gate it prints the
script and asks y/N — edit `script.txt` before answering to change it.

Already have a script? Skip research, writing, and the approval gate — go straight to
TTS, image prompts, images, and voiceover. The run folder is named after the script's
`TITLE:` line unless `--topic` says otherwise.

```bash
python pipeline.py script my-script.txt           # --profile optional when only one profile exists
python pipeline.py script - --profile <name>      # read the script from stdin
python pipeline.py resume <run-slug>              # finish a stopped run (uses its profile.txt)
```

Both exit 0 only when the run completes (1 on an error, a rejected script or a cancel).

### Metadata & Thumbnail

After a run completes, generate SEO metadata and thumbnails:

```bash
python pipeline.py metadata <run-slug>               # generate 5 titles, description, hashtags, 3 thumbnails
python pipeline.py metadata <run-slug> --regenerate  # overwrite existing metadata.json
python pipeline.py metadata <run-slug> --pick-thumb N    # 1-based; pick which thumbnail is "chosen"
python pipeline.py metadata <run-slug> --show         # print current metadata.json
```

Also available as `/metadata` in the Telegram bot, and as the **Metadata** mode in the web UI. All three use the profile the run was made with (its `profile.txt`); `--profile` overrides it on the CLI.

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
├── style-sheet.md        # human-readable style reference; the profile creator writes anchor prompts from it (the pipeline reads profile.yaml)
└── anchors/
    ├── manifest.yaml     # what each anchor slot is — read back to describe the references
    ├── anchor-01.png     # first reference slot (layout depends on the roster size)
    ├── anchor-02.png
    └── ...               # additional style anchors (props, text labels, environments)
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
Claude produces a `profile.yaml` (checked to load before any anchor is generated) and a `style-sheet.md` with detailed visual rules for the anchor prompts.

**4. Anchor image generation (tiered)**
The creator builds a structured anchor plan from your profile, then generates images in two passes:

- **Verification tier** — character reference sheets (full body, multiple angles) generated first and shown to you before proceeding. These establish the "ground truth" appearance for every character.
- **Full tier** — a prop sheet, a text-and-labels scene, a wide shot, a close-up (with characters), and indoor, outdoor and abstract background anchors, generated after you approve the characters.

All anchors are generated with Gemini and saved to `anchors/`. The image pipeline sends them in label order, up to `image_style.max_anchors` (`anchor-00`, the seed, first if you gave one), and describes each from `manifest.yaml`, so the model has consistent visual context for every image in a run.

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

Revision creates a versioned copy (`my-channel-v2`, then `-v3`, …) rather than overwriting the original or an earlier revision, so you can always roll back.

The revise flow:
1. Describe what you want to change
2. Claude asks clarifying questions if needed
3. Updated `profile.yaml` and `style-sheet.md` are written to the versioned folder
4. **Smart anchor detection** — Claude says whether anchors need regenerating (a `REGENERATE_ANCHORS` line). If it doesn't say, anchors are regenerated when characters or core style fields changed, and otherwise you're asked whether to regenerate or copy the originals.

Versioning follows `my-channel` → `my-channel-v2` → `my-channel-v3` automatically.

---

## Image generation models

Model IDs are not hardcoded — they come from the active profile's `image_gen` block. The
regenerate selector in the UI picks between the two slots by alias:

| Alias | Profile field | Notes |
|-------|---------------|-------|
| `nano-banana-2` | `image_gen.default_model` | Default — fast and cost-effective for full runs |
| `3-pro` | `image_gen.pro_model` | Highest quality — used for thumbnails and for manual regeneration with `3-pro` (anchors use `default_model`) |

`profiles/example` ships with `gemini-3.1-flash-image` and `gemini-3-pro-image` in those slots.

---

## Environment notes

Keep comments on their own line in `.env` — python-dotenv reads `KEY=   # note` as the value `# note`.

A profile's `voice.voice_id` can also reference an arbitrary env var with `${VAR_NAME}` syntax, resolved at load time — not limited to the fixed list in `.env.example`.
