"""Extra seeds for an existing layer-histogram-matched random control.

Why this exists. The K=100 layer-matched control was run at five seeds after a
single seed proved unusable (range .147-.663). The controls at the SMALLER sizes
-- the ones the interference rows in ledger section 19 are read against -- were
never given the same treatment and are still one draw each. Ledger section 21
then measured three draws of 29 heads from Ultravox's top-100 and got .250 /
.417 / .667, which is the same order of spread. A difference quoted against a
single draw from a population that wide is not a measurement.

Same construction as `make_shared_split.matched_control`: identical per-layer
counts to the subset being controlled, drawn from outside the discovered top-K,
so a difference cannot be depth and cannot be a redraw of the subset itself.

    python make_matched_seeds.py --e1 results/ultravox/scores.npz \
        --subset results/sets/ultravox_audioonly29.npz --k 100 \
        --out results/sets/mseeds --tag ultravox_randmatched29 --seeds 1 2 3 4
"""
import argparse, json
from collections import defaultdict
from pathlib import Path

import numpy as np

ap = argparse.ArgumentParser()
ap.add_argument("--e1", type=Path, required=True)
ap.add_argument("--subset", type=Path, required=True)
ap.add_argument("--k", type=int, default=100)
ap.add_argument("--seeds", type=int, nargs="+", required=True)
ap.add_argument("--out", type=Path, required=True)
ap.add_argument("--tag", required=True)
a = ap.parse_args()

g = np.load(a.e1)["scores"]
L, H = g.shape
excluded = set(np.argsort(g.ravel())[::-1][: a.k].tolist())
subset = sorted(np.flatnonzero(np.load(a.subset)["scores"].ravel() > 0).tolist())
want = defaultdict(int)
for i in subset:
    want[i // H] += 1

a.out.mkdir(parents=True, exist_ok=True)
meta = {"e1": str(a.e1), "subset": str(a.subset), "k": a.k, "shape": [L, H],
        "size": len(subset), "layer_counts": {str(k): v for k, v in sorted(want.items())},
        "draws": {}}

for sd in a.seeds:
    rng = np.random.default_rng(sd)
    picked = []
    for lay, n in sorted(want.items()):
        pool = [lay * H + h for h in range(H) if lay * H + h not in excluded]
        if len(pool) < n:
            raise SystemExit(f"layer {lay}: need {n}, strict pool has {len(pool)}")
        picked += rng.choice(pool, n, replace=False).tolist()
    picked = sorted(picked)
    assert not (set(picked) & excluded), "drew a top-K head"
    arr = np.zeros(L * H, dtype=np.float32)
    arr[picked] = 1.0
    name = f"{a.tag}_s{sd}"
    np.savez(a.out / f"{name}.npz", scores=arr.reshape(L, H))
    meta["draws"][name] = {"seed": sd, "size": len(picked),
                           "layer_mean": float(np.mean([i // H for i in picked]))}
    print(f"  {name:36s} n={len(picked):3d}  mean layer "
          f"{np.mean([i // H for i in picked]):.1f}")

(a.out / f"{a.tag}_seeds.json").write_text(json.dumps(meta, indent=2))
