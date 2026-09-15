"""Split a modality head set by whether each head is ALSO in its own text-only
backbone's text head set, and build depth-matched random controls for each half.

This is the causal-sufficiency test. Overlap between two head lists is
correlational: a published objection (OpenReview QVcsBMonuz) is that a
computation constrains only the aggregate routed contribution across heads, so
two models can share head identities without sharing per-head roles. The answer
is not a bigger overlap number, it is to intervene on each half separately.

Each control is drawn with the SAME PER-LAYER COUNTS as the subset it controls,
and excludes the whole modality top-K, so a difference cannot be depth and
cannot be "we happened to redraw the same heads".

Emits scores.npz files that are 1.0 on the chosen heads and 0 elsewhere, so
`run_steer.py --condition top --k <size>` selects exactly them.

    python make_shared_split.py --modality results/qwen2_audio_v2/scores.npz \
        --text results/text/qwen15_7b_chat/scores.npz --k 100 \
        --out results/sets/qwen2_audio --tag qwen2_audio
"""
import argparse, json
from collections import defaultdict
from pathlib import Path

import numpy as np

ap = argparse.ArgumentParser()
ap.add_argument("--modality", type=Path, required=True)
ap.add_argument("--text", type=Path, required=True)
ap.add_argument("--k", type=int, default=100)
ap.add_argument("--seed", type=int, default=0)
ap.add_argument("--out", type=Path, required=True)
ap.add_argument("--tag", required=True, help="filename prefix, e.g. qwen2_audio")
a = ap.parse_args()

g = np.load(a.modality)["scores"]
t = np.load(a.text)["scores"]
assert g.shape == t.shape, (g.shape, t.shape)
L, H = g.shape

gi = np.argsort(g.ravel())[::-1][: a.k]
ti = np.argsort(t.ravel())[::-1][: a.k]
shared = sorted(set(gi.tolist()) & set(ti.tolist()))
only = sorted(set(gi.tolist()) - set(ti.tolist()))
assert len(shared) + len(only) == a.k

rng = np.random.default_rng(a.seed)
excluded = set(gi.tolist())


RELAXED = []


def matched_control(subset):
    """Same per-layer counts as `subset`, drawn from heads outside the top-K.

    Where a layer is so saturated by the top-K that the strict pool is too
    small, fall back for THAT LAYER to excluding only the subset being
    controlled. The fallback is recorded in the split json rather than applied
    silently: it weakens the control from "no top-K head" to "no head from this
    subset", and any arm that used it must say so.
    """
    want = defaultdict(int)
    for i in subset:
        want[i // H] += 1
    sub = set(subset)
    picked = []
    for lay, n in sorted(want.items()):
        pool = [lay * H + h for h in range(H) if lay * H + h not in excluded]
        if len(pool) < n:
            pool = [lay * H + h for h in range(H) if lay * H + h not in sub]
            RELAXED.append({"layer": lay, "needed": n, "strict_pool": 0,
                            "relaxed_pool": len(pool)})
            if len(pool) < n:
                raise SystemExit(f"layer {lay}: need {n}, only {len(pool)} even relaxed")
        picked += rng.choice(pool, n, replace=False).tolist()
    return sorted(picked)


sets = {f"{a.tag}_shared{len(shared)}": shared,
        f"{a.tag}_only{len(only)}": only,
        f"{a.tag}_randmatched{len(shared)}": matched_control(shared),
        f"{a.tag}_randmatched{len(only)}": matched_control(only)}

a.out.mkdir(parents=True, exist_ok=True)
meta = {"k": a.k, "shape": [L, H], "seed": a.seed,
        "relaxed_layers": RELAXED,
        "modality": str(a.modality), "text": str(a.text),
        "n_shared": len(shared), "n_only": len(only)}
for name, idx in sets.items():
    arr = np.zeros(L * H, dtype=np.float32)
    arr[idx] = 1.0
    np.savez(a.out / f"{name}.npz", scores=arr.reshape(L, H))
    lays = [i // H for i in idx]
    meta[name] = {"size": len(idx), "layer_mean": float(np.mean(lays)),
                  "layer_min": int(min(lays)), "layer_max": int(max(lays))}
(a.out / f"{a.tag}_split.json").write_text(json.dumps(meta, indent=2))
if RELAXED:
    print(f"  NOTE: control relaxed on {len(RELAXED)} layer(s) — see relaxed_layers "
          f"in the split json; this arm's control excludes only its own subset there.")
print(json.dumps({k: v for k, v in meta.items() if k in ("n_shared", "n_only")}, indent=2))
for name, idx in sets.items():
    print(f"  {name:34s} n={len(idx):3d}  mean layer {np.mean([i // H for i in idx]):.1f}")
