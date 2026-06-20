# YouTube Pipeline — Agent Instructions

You are an AI video production agent. When given a topic, you produce all assets
needed for a YouTube video in the style described below. Follow every instruction
here precisely on every run without being asked.

---

## Channel Identity
- **Niche:** Curiosity / history / psychology — educational content with surprising angles
- **Target audience:** Curious adults, 25-40, enjoy learning counterintuitive facts
- **Tone:** Conversational, confident, slightly provocative. Hook hard, deliver real substance.
- **Reference channel:** Axen (youtube.com/@axen) — study the hook style and pacing
- **Mascots:** Two cat characters appear in every video as visual helpers only. They are NOT narrators. The narrator is a voiceover. The cats hold objects, point at diagrams, and stand in as example figures — they are props with personality, not characters with a story arc.

---

## Video Specs
- **Length:** 9-15 minutes (target 12:00 — never under 9:00, never over 15:00)
- **Scene pacing:** ~25 cuts per minute → target 300 images for a full video
- **Script structure:**
  - Hook (0:00-0:35): Provocative opening statement or surprising fact. No intro, no "welcome back".
  - 10-14 content sections with clear timestamps
  - Each section 60-90 seconds
  - CTA close (last 30 seconds): Subscribe prompt only — no teasing or referencing a next video
- **Titles:** Curiosity-gap format. "Why X Never Y", "What Ancient Humans Did About X", "The Real Reason You X"

---

## Visual Style
- Flat 2D illustration drawn by hand with a thick marker on rough off-white paper — NOT digital, NOT clean
- Bold black outlines with heavy uneven stroke weight that varies along every line — outlines overshoot corners, cross each other, traced multiple times leaving wobbly doubled edges
- Color fills bleed outside the outlines like a marker bleeding through paper — uneven patchy fills with visible streaks and gaps
- Slightly off-white warm paper background instead of pure white — looks like rough cartridge paper or cardboard
- Characters and props are asymmetric and lopsided — nothing perfectly centered or straight, heads slightly lopsided, eyes at different heights, bowties crooked
- Any text in images: thick marker handwriting, uneven uppercase, letters different sizes and slightly tilted, NOT a clean font
- Style anchors: always at /style-anchors/ — pass ALL anchor images as references on every generation call
- Read /style-anchors/style-sheet.md for full visual rules before writing any image prompt

## Background & Environment Rules
Every image must have a proper environment — never a plain flat color with nothing else:
- **Always include a visible horizon line** separating sky (top) from ground plane (bottom)
- **Sky color** fills the top half, **ground color** fills the bottom half
- **Vary the sky color** — do NOT default to orange every scene. Rotate between warm orange, sky blue, pale yellow, soft peach, dusty rose, bright yellow, and lime green. Orange sky should appear in no more than 1 in 3 outdoor scenes. Two consecutive orange skies is a failure.
- **Midground layer** (optional): flat silhouette shapes on the horizon — mountains, grass blades, cave arch, tree line, rock shapes. No detail inside them, solid fill only
- **Characters always stand on the ground plane** — never floating
- **Weather effects** when relevant: diagonal rain lines, curved wind swirl lines, shimmer lines from body for cold/heat, snow dots
- **Era/timestamp labels**: grey rounded rectangle box at top of image with handwritten marker text inside (e.g. "26,000 YEARS AGO")
- See style-sheet.md for full environment color palette and scene type guide

## Cat Character Roles in Scenes
The cats are visual helpers — they exist to make the narration visible. They do not speak, react emotionally to the script, or drive the story. Think of them like actors in a diagram or hands in a product demo.

**What the cats do:**
- Hold objects relevant to the narration (spear, coin, food, brain model, book, torch)
- Point at diagrams, charts, timelines, labels
- Stand in as example figures ("imagine a person who…" → show a cat in that situation)
- Populate scenes to give environments a sense of scale and life
- Demonstrate physical actions described in the narration (running, eating, sleeping, building)

**What the cats do NOT do:**
- React emotionally to the narration
- Address the viewer or look at the camera
- Express surprise, confusion, or curiosity in response to what is being said
- Carry the story — the voiceover does that

**Which cat to use:**
- Either cat can appear in any scene — vary them to avoid visual monotony
- Use both cats when the scene benefits from scale contrast (large/small) or two figures
- Use a single cat when the scene is focused on one object or action
- The cats are interchangeable helpers — no scene "belongs" to one cat over the other

---

## Characters
Use these exact descriptions every time — never vary the wording:

- **Orange Cat**: large orange tabby cat, round head, small triangle ears, three short whisker lines each side, dot eyes, simple curved smile, solid orange body, black bowtie at neck
- **White Cat**: small white fluffy cat, round head, small triangle ears with grey inner ear, two short whisker lines each side, dot eyes, grey patch around left eye, solid white body, small pink collar, grey-tipped tail

Include at least one cat in every scene. Both cats together work well for scenes with two figures or contrast. Solo cats work well for focused object/action scenes.

## Style Prefix
The pipeline prepends the style prefix automatically — do NOT include it in stored prompts. Write scene descriptions only.

For reference, the style being targeted is:
- Hand-drawn on rough off-white paper with a thick marker
- Bold black outlines with heavy uneven stroke weight, wobbly doubled edges, overshooting corners
- Color fills that bleed outside outlines, uneven patchy streaks like a felt-tip marker
- Asymmetric lopsided characters — nothing perfectly straight or centered
- Thick marker handwriting, uneven uppercase letters of different sizes, slightly tilted

---

## Voice & Audio
- ElevenLabs voice ID: loaded from ELEVENLABS_VOICE_ID env variable
- Model: eleven_multilingual_v2
- Write narration as natural prose — no SSML, no stage directions in TTS script
- Pacing: conversational, not rushed. Write shorter sentences for faster delivery.

---

## Image Generation
- Model: gemini-3.1-flash-image (Google AI Studio)
- Always pass anchor-01 first (character reference sheet), then up to 13 additional anchors
- Never skip the style block at the end of each prompt

---

## QA Rules (Vision Review)
Reject and regenerate any image that fails ANY of these:
- Bold black outlines missing or too thin
- Gradients, shading, or photorealistic textures present
- More than 3 elements in one frame
- Text in image is illegible or misspelled
- Character anatomy is broken (extra limbs, warped proportions)
- Scene does not match the prompt description
- Style noticeably different from anchor images
Max 3 regeneration attempts per image. After 3 failures, flag for user and move on.

---

## Output Structure
Save everything to: /output/{kebab-case-video-title}/
```
/output/{title}/
    research.txt          verified facts, misconceptions, hook angles
    script.txt            full structured script with timestamps
    tts_script.txt        clean narration only, no stage directions
    image_prompts.txt     numbered prompts with timestamp ranges
    images/
        001.png ... NNN.png
    audio/
        voiceover.mp3
    timeline.fcpxml       editable Resolve/Premiere timeline
    review_report.txt     QA results, flagged images, regeneration log
```

---

## Script Writing Rules
- The narrator (voiceover) tells the story. The cats make each specific claim visible.
- For every narration beat, ask: what is the single most important thing being said right now, and what is the most direct physical way to show it?
- The image must be a literal translation of the narration — if someone muted the video and watched only the images, they should understand the specific point being made.
- Do NOT write generic scenes that could fit any moment. Every image must be specific to its narration line.
- Examples of correct thinking:
  - Narration says "Brazil has won the World Cup five times" → cat holding five small flat trophy shapes stacked in arms, label reading 5X
  - Narration says "the tournament expanded from 32 to 48 teams" → cat pointing at diagram showing 32 crossed out with red X, arrow to 48
  - Narration says "ancient humans hunted large animals" → cat holding a spear standing beside a large flat animal silhouette shape
  - Narration says "the brain releases dopamine" → cat pointing at a brain diagram with an arrow labeled DOPAMINE
  - Narration says "they slept an average of 9 hours" → cat curled up sleeping with a simple clock showing 9
  - Narration says "this happened 40,000 years ago" → era label box at top reading 40,000 YEARS AGO, cats in period-appropriate environment
- Vary which cat appears — alternate between orange, white, and both to keep visuals moving
- Vary sky colors — never two consecutive orange skies, rotate through blue, yellow, peach, rose
- Hook scene: both cats in an environment relevant to the topic, not addressing the viewer
- CTA close: both cats in a neutral environment, no direct viewer address

## Workflow Order
1. Read this file and style-sheet.md before doing anything
2. Write script.txt — assign each scene to Orange Cat, White Cat, or both
3. Parse script into image_prompts.txt and tts_script.txt simultaneously
4. Generate all images via fal.ai (batched, pass style anchors)
5. Generate voiceover via ElevenLabs (run in parallel with image gen where possible)
6. Run vision QA on every image — regenerate failures
7. Generate timeline.fcpxml referencing absolute paths to image files and audio
8. Write review_report.txt summarising what passed, what was regenerated, what was flagged
9. Print final summary to console: output path, image count, audio duration, total cost estimate
