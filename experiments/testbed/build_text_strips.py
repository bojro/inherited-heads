"""Text analogue of the audio comic strips — for the inheritance test.

Ultravox's backbone (Llama-3.1-8B-Instruct) was never trained on audio, yet a
top-100 head set found by the gaze score steers it to a chosen audio segment at
98%. Our own hypothesis (from reading the Ultravox model card,
2026-08-14) was that such heads would be "a pre-existing text-LM head
repurposed at inference". This builds the control that tests it: the SAME task
in pure text, on the SAME backbone, with no audio path at all.

Six LibriSpeech transcripts — the very passages the audio strips are read
from — concatenated as numbered text segments, then "what does segment k talk
about?". If the heads that track text segments are the heads that track audio
segments, the tracking is inherited, not learned from the audio adapter.

    python testbed/build_text_strips.py --strips testbed/strips --out testbed/text_strips
"""

import argparse
import json
from pathlib import Path

ap = argparse.ArgumentParser()
ap.add_argument("--strips", type=Path, default=Path("testbed/strips"))
ap.add_argument("--out", type=Path, default=Path("testbed/text_strips"))
ap.add_argument("--n-items", type=int, default=50)
a = ap.parse_args()

import pandas as pd

meta = pd.read_parquet(a.strips / "metadata.parquet")
a.out.mkdir(parents=True, exist_ok=True)
rows = []
for item, g in meta.groupby("item_id"):
    if item >= a.n_items:
        break
    g = g.sort_values("seg_idx")
    segs = [t.strip().capitalize() + "." for t in g.transcript]
    # mirrors the audio strip: six passages, distinct speakers, in order,
    # with an explicit separator standing in for the silence
    body = "\n\n".join(f"Segment {i+1}: {t}" for i, t in enumerate(segs))
    rows.append(dict(item_id=int(item), body=body, segments=segs,
                     speaker_ids=[int(x) for x in g.speaker_id]))
(a.out / "text_strips.jsonl").write_text("\n".join(json.dumps(r) for r in rows))
print(f"wrote {len(rows)} text strips to {a.out}")
print("--- example (first 2 segments) ---")
print("\n\n".join(rows[0]["body"].split("\n\n")[:2])[:400])
