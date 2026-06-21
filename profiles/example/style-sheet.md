# Curious Critters — Visual Style Sheet

This file is a human-readable reference for the channel's visual identity.
The pipeline reads `profile.yaml` for generation — update both if you change the style.

---

## Art style

Flat 2D marker illustration. Think a children's science book drawn with thick Copic
markers on warm off-white paper.

- **Outlines:** Bold, uneven black marker lines. Slightly wobbly — hand-drawn feel,
  not vector-perfect.
- **Color fills:** Bleed slightly outside the outlines. Visible marker streaks and
  texture — no flat digital fills.
- **Background:** Off-white warm paper (#F5F0E8). Never pure white, never grey.
- **No gradients, no drop shadows, no 3D effects, no photorealistic textures.**
- **Text:** Bold handwritten uppercase in black marker. Numbers are always large and
  prominent when narration states a statistic.

---

## Composition (16:9)

- Visible horizon line divides sky (top) from ground plane (bottom)
- Sky: flat solid color — rotates through pale yellow, soft coral, sky blue, mint
  green, lavender. Never the same color twice in a row.
- Ground: flat solid color filling the bottom third. Always visible.
- Midground (optional): one flat silhouette layer — solid fill, no interior detail.
- Characters always stand on the ground plane. Never floating.

---

## Characters

### Orange Cat
Large, round orange tabby. Black bowtie. Stands upright. Slightly pudgy.
Role: explainer — holds diagrams, points at labels, demonstrates processes.

### White Cat
Small, fluffy white cat. Pink collar with bell. Grey eye patch. Stands upright,
noticeably shorter than Orange Cat.
Role: reactor / second perspective — used when comparing two things or showing scale.

**Both characters are always flat 2D — no shading, no 3D volume.**

---

## Anchor images

Place reference images in `anchors/` as `anchor-01.png`, `anchor-02.png`, etc.
The pipeline passes these to the image model in priority order (set in `profile.yaml`)
to lock in the art style and character designs across every generated image.

Recommended anchor set:
- `anchor-01` — multi-angle character sheet (both cats, front/side/back)
- `anchor-02` — original character reference sheet
- `anchor-03` — environment example (horizon, ground plane, sky)
- Additional anchors — style samples showing text labels, props, compositions
