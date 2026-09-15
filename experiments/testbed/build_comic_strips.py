"""Comic-strip testbed — experiment V2, the seed paper's OWN modality.

Extracts baulab/openai-comic-strips into the same on-disk shape as the audio
testbed, so the judge, the item-offset split and the bootstrap all carry over
unchanged:

    testbed/comics/images/strip_00000/panel_{0..5}.png
    testbed/comics/metadata.parquet   item_id, panel_idx, caption, style, protagonist

`caption` is the analogue of `transcript`: judge.attribute() scores an answer by
IDF-weighted content-word overlap against the six captions, exactly as it does
against six transcripts.

Panels are saved square and downscaled from 1024; the per-panel TOKEN count is
controlled at inference time by the processor's min_pixels/max_pixels, not here,
so this size only needs to be generous enough not to be the bottleneck.

    python testbed/build_comic_strips.py --n-items 150 --size 448
"""
import argparse, glob, io
from pathlib import Path

ap = argparse.ArgumentParser()
ap.add_argument("--src", default=str(Path.home() / ".cache/huggingface/hub/datasets--baulab--openai-comic-strips"))
ap.add_argument("--out", type=Path, default=Path("testbed/comics"))
ap.add_argument("--n-items", type=int, default=150)
ap.add_argument("--size", type=int, default=448)
a = ap.parse_args()

import pandas as pd
import pyarrow.parquet as pq
from PIL import Image

files = sorted(glob.glob(f"{a.src}/snapshots/*/data/*.parquet"))
if not files:
    raise SystemExit(f"no parquet under {a.src} — is the dataset downloaded?")

(a.out / "images").mkdir(parents=True, exist_ok=True)
rows, item = [], 0
for f in files:
    if item >= a.n_items:
        break
    for r in pq.read_table(f).to_pylist():
        if item >= a.n_items:
            break
        d = a.out / "images" / f"strip_{item:05d}"
        d.mkdir(exist_ok=True)
        for k in range(6):
            blob = r[f"panel_{k+1}"]
            blob = blob["bytes"] if isinstance(blob, dict) else blob
            im = Image.open(io.BytesIO(blob)).convert("RGB")
            if im.size != (a.size, a.size):
                im = im.resize((a.size, a.size), Image.LANCZOS)
            im.save(d / f"panel_{k}.png")
            rows.append(dict(item_id=item, panel_idx=k,
                             caption=r[f"caption_{k+1}"],
                             style=r["style"], protagonist=r["protagonist"],
                             comic_id=int(r["comic_id"])))
        item += 1

meta = pd.DataFrame(rows)
meta.to_parquet(a.out / "metadata.parquet", index=False)
print(f"wrote {item} strips ({len(meta)} panels) to {a.out} at {a.size}x{a.size}")
print(meta.head(3).to_string())
