#!/usr/bin/env python3
"""
Apply flicker effect to all image clips in the active Palmier Pro project.

Non-destructive: leaves originals in place and adds a new video track on top
containing the alternating NNNb / NNNc segments. Higher tracks render above
lower ones, so the flicker layer visually replaces the originals during
playback. Undo deletes the layer in a single call.

Usage:
    python apply_flicker.py --interval 6
    python apply_flicker.py --interval 6 --dry-run
    python apply_flicker.py --undo
"""

import argparse
import json
import re
import sys
from itertools import count
from pathlib import Path

import requests

PALMIER_URL = "http://127.0.0.1:19789/mcp"
ADD_BATCH_SIZE = 50
SNAPSHOT_PATH = Path(__file__).parent / ".flicker_snapshot.json"
_id = count(1)

ORIGINAL_RE = re.compile(r'^(\d+)$')
FLICKER_B_RE = re.compile(r'^(\d+)b$', re.IGNORECASE)
FLICKER_C_RE = re.compile(r'^(\d+)c$', re.IGNORECASE)


def rpc(method: str, params: dict, notify: bool = False):
    payload = {"jsonrpc": "2.0", "method": method, "params": params}
    if not notify:
        payload["id"] = next(_id)
    r = requests.post(PALMIER_URL, json=payload, timeout=180)
    r.raise_for_status()
    if notify:
        return None
    data = r.json()
    if "error" in data:
        raise RuntimeError(f"RPC error on {method}: {data['error']}")
    return data.get("result", {})


def call_tool(name: str, arguments: dict) -> dict:
    result = rpc("tools/call", {"name": name, "arguments": arguments})
    if result.get("isError"):
        content = result.get("content", [])
        msg = content[0]["text"] if content else "unknown error"
        raise RuntimeError(f"{name} failed: {msg}")
    content = result.get("content", [])
    if content and content[0].get("type") == "text":
        text = content[0]["text"].strip()
        if text:
            try:
                return json.loads(text)
            except json.JSONDecodeError:
                return {"text": text}
    return result


def initialize():
    rpc("initialize", {
        "protocolVersion": "2024-11-05",
        "capabilities": {},
        "clientInfo": {"name": "apply_flicker", "version": "2.0"},
    })
    rpc("notifications/initialized", {}, notify=True)


def classify_asset(name: str):
    if m := ORIGINAL_RE.match(name):
        return m.group(1), None
    if m := FLICKER_B_RE.match(name):
        return m.group(1), 'b'
    if m := FLICKER_C_RE.match(name):
        return m.group(1), 'c'
    return None, None


def build_segments(start_frame, duration_frames, interval, id_b, id_c, start_with_b=True):
    """Build alternating b/c segments. Returns (segments, next_start_with_b)
    so the caller can keep the alternation continuous across clips."""
    segments = []
    frame = start_frame
    end = start_frame + duration_frames
    use_b = start_with_b
    while frame < end:
        chunk = min(interval, end - frame)
        segments.append({
            "mediaRef": id_b if use_b else id_c,
            "startFrame": frame,
            "durationFrames": chunk,
        })
        frame += chunk
        use_b = not use_b
    return segments, use_b  # use_b is already toggled = what the NEXT segment would be


def find_track_by_label(timeline: dict, label: str):
    for i, track in enumerate(timeline.get("tracks", [])):
        if track.get("label") == label:
            return i
    return None


def apply_flicker(interval_frames: int, dry_run: bool = False) -> None:
    print(f"Connecting to Palmier Pro at {PALMIER_URL} ...")
    initialize()

    timeline = call_tool("get_timeline", {})
    media_data = call_tool("get_media", {})

    fps = timeline.get("fps", 30)
    print(f"FPS: {fps}  |  interval: {interval_frames} frames ({interval_frames / fps:.3f}s per segment)")

    assets = media_data.get("entries", [])
    lookup: dict = {}
    for asset in assets:
        base, variant = classify_asset(asset.get("name", ""))
        if base is None:
            continue
        entry = lookup.setdefault(base, {})
        if variant == 'b':
            entry["b"] = asset["id"]
        elif variant == 'c':
            entry["c"] = asset["id"]

    asset_by_id = {a["id"]: a for a in assets}
    segments = []
    skipped = []
    originals_count = 0
    next_start_with_b = True  # carries alternation state across clips

    for track in timeline.get("tracks", []):
        if track.get("type") != "video":
            continue
        # Walk clips in timeline order so the rhythm flows left-to-right
        for clip in sorted(track.get("clips", []), key=lambda c: c.get("startFrame", 0)):
            asset = asset_by_id.get(clip.get("mediaRef", ""))
            if not asset:
                continue
            base, variant = classify_asset(asset.get("name", ""))
            if base is None or variant is not None:
                continue  # not an original image clip
            entry = lookup.get(base, {})
            id_b, id_c = entry.get("b"), entry.get("c")
            if not id_b or not id_c:
                skipped.append(asset["name"])
                continue
            originals_count += 1
            new_segments, next_start_with_b = build_segments(
                clip["startFrame"], clip["durationFrames"],
                interval_frames, id_b, id_c,
                start_with_b=next_start_with_b,
            )
            segments.extend(new_segments)

    print(f"\nOriginal clips covered:     {originals_count}")
    print(f"Flicker segments to add:    {len(segments)}")
    if skipped:
        print(f"Skipped (missing b/c):      {skipped}")

    if dry_run:
        print("\n[DRY RUN] No changes applied.")
        return

    if not segments:
        print("Nothing to add.")
        return

    pre_labels = {t.get("label") for t in timeline.get("tracks", [])}

    print("\nAdding flicker layer ...")

    # First batch: omit trackIndex so Palmier auto-creates a new video track.
    first_batch = segments[:ADD_BATCH_SIZE]
    call_tool("add_clips", {"entries": first_batch})
    print(f"  added {len(first_batch)}/{len(segments)}")

    # Detect the new track and pin all subsequent batches to it.
    post = call_tool("get_timeline", {})
    new_labels = [t.get("label") for t in post.get("tracks", [])
                  if t.get("label") not in pre_labels and t.get("type") == "video"]
    if not new_labels:
        print("\nWarning: could not detect the new flicker track. Aborting before more batches.")
        return

    flicker_label = new_labels[0]
    flicker_idx = find_track_by_label(post, flicker_label)

    # Write the snapshot as soon as the track is known so a mid-loop failure
    # (network error, Palmier timeout) still leaves --undo able to clean up.
    # ponytail: the narrower window before this point (add succeeds but track
    # detection above fails) still orphans the first batch with no snapshot —
    # add a pre-add snapshot or a clip-id-based cleanup if that gets hit in practice.
    SNAPSHOT_PATH.write_text(json.dumps({
        "flicker_track_label": flicker_label,
        "interval": interval_frames,
        "segments": len(segments),
    }, indent=2))

    for i in range(ADD_BATCH_SIZE, len(segments), ADD_BATCH_SIZE):
        batch = segments[i: i + ADD_BATCH_SIZE]
        for entry in batch:
            entry["trackIndex"] = flicker_idx
        call_tool("add_clips", {"entries": batch})
        print(f"  added {min(i + ADD_BATCH_SIZE, len(segments))}/{len(segments)}")

    print(f"\nFlicker layer created on track '{flicker_label}'.")
    print(f"Run 'python apply_flicker.py --undo' to remove it.")


def undo() -> None:
    if not SNAPSHOT_PATH.exists():
        print(f"No snapshot at {SNAPSHOT_PATH}. Nothing to undo.")
        sys.exit(1)

    snap = json.loads(SNAPSHOT_PATH.read_text())
    label = snap.get("flicker_track_label")
    if not label:
        print("Snapshot is missing the flicker track label.")
        sys.exit(1)

    print(f"Connecting to Palmier Pro at {PALMIER_URL} ...")
    initialize()

    timeline = call_tool("get_timeline", {})
    idx = find_track_by_label(timeline, label)
    if idx is None:
        print(f"Track '{label}' not found in current timeline. Already removed?")
        SNAPSHOT_PATH.unlink()
        return

    # The snapshot only has a generic label like "V2": in another project, or after the layer
    # was deleted by hand, that's the user's own track. Remove it only if it's all b/c frames.
    names = {a["id"]: a.get("name", "") for a in call_tool("get_media", {}).get("entries", [])}
    clips = timeline["tracks"][idx].get("clips", [])
    if not clips or any(classify_asset(names.get(c.get("mediaRef"), ""))[1] is None for c in clips):
        print(f"Track '{label}' holds clips that aren't flicker frames — not removing it.")
        print(f"If the flicker layer is already gone, delete {SNAPSHOT_PATH.name}.")
        sys.exit(1)

    print(f"Removing track '{label}' (index {idx}) ...")
    call_tool("remove_tracks", {"trackIndexes": [idx]})
    SNAPSHOT_PATH.unlink()
    print("Undo complete.")


def main():
    parser = argparse.ArgumentParser(
        description="Apply (or undo) a non-destructive flicker layer in Palmier Pro."
    )
    parser.add_argument("--interval", type=int,
                        help="Frames per flicker segment (e.g. 3 = 0.05s at 60fps)")
    parser.add_argument("--dry-run", action="store_true",
                        help="Show what would change without applying anything")
    parser.add_argument("--undo", action="store_true",
                        help="Remove the flicker track created by the most recent apply")
    args = parser.parse_args()

    if args.undo:
        undo()
        return

    if args.interval is None:
        parser.error("--interval is required (unless using --undo)")
    if args.interval < 1:
        print("Error: --interval must be at least 1")
        sys.exit(1)

    apply_flicker(args.interval, args.dry_run)


if __name__ == "__main__":
    main()
