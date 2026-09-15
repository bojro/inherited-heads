"""Text analogue of the seed paper's PANEL task, from their own captions.

For the VLM inheritance test: are the "visual grounding" heads they report the
same heads that track segments in plain text? That is the H7 analysis moved into
their modality, and it needs a text version of their task.

Their comic dataset ships caption_1..caption_6 alongside the panels, so the text
form is free and content-matched: identical semantic material, same six-way
structure, no images. Same shape as testbed/text_strips (built from LibriSpeech
transcripts for the audio arms) so run_text_discovery.py drives it unchanged.

    python testbed/build_caption_strips.py --n-items 150
"""
import argparse, json
from pathlib import Path

ap = argparse.ArgumentParser()
ap.add_argument("--comics", type=Path, default=Path("testbed/comics"))
ap.add_argument("--out", type=Path, default=Path("testbed/caption_strips"))
ap.add_argument("--n-items", type=int, default=150)
a = ap.parse_args()

import pandas as pd

meta = pd.read_parquet(a.comics / "metadata.parquet")
a.out.mkdir(parents=True, exist_ok=True)
rows = []
for item_id, g in meta.groupby("item_id"):
    if item_id >= a.n_items:
        break
    g = g.sort_values("panel_idx")
    segs = [c.strip() for c in g.caption]
    # mirrors build_text_strips.py exactly: six numbered segments, blank-line separated
    body = "\n\n".join(f"Segment {i+1}: {t}" for i, t in enumerate(segs))
    rows.append(dict(item_id=int(item_id), body=body, segments=segs,
                     speaker_ids=list(range(6))))
(a.out / "text_strips.jsonl").write_text("\n".join(json.dumps(r) for r in rows))
print(f"wrote {len(rows)} caption strips to {a.out}")
print("--- example (first 2 segments) ---")
print("\n\n".join(rows[0]["body"].split("\n\n")[:2])[:300])
