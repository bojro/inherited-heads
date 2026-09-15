"""Bootstrap CIs for steering accuracy, resampling ITEMS (not trials).

Each strip contributes 6 correlated trials (one per target segment), so a
binomial interval over 120 trials overstates precision. Resample the 20 strips
with replacement, recompute accuracy over all their trials.

    python steer_ci.py results/e2/qwen2_audio_top100 [more dirs...]
    python steer_ci.py --diff results/e2/qwen2_audio_top100 results/e2/qwen2_audio_masstop100
"""
import json, sys
from pathlib import Path

import numpy as np


def per_item(d):
    """-> dict item -> (n_correct, n_trials)"""
    acc = {}
    for line in open(Path(d) / "answers.jsonl"):
        r = json.loads(line)
        c, n = acc.get(r["item"], (0, 0))
        acc[r["item"]] = (c + int(r["judged"] == r["target"]), n + 1)
    return acc


def boot(d, n_boot=10000, seed=0):
    a = per_item(d)
    items = np.array(sorted(a))
    C = np.array([a[i][0] for i in items], float)
    N = np.array([a[i][1] for i in items], float)
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(items), (n_boot, len(items)))
    vals = C[idx].sum(1) / N[idx].sum(1)
    return C.sum() / N.sum(), vals


if __name__ == "__main__":
    args = sys.argv[1:]
    if args and args[0] == "--diff":
        a, b = args[1], args[2]
        pa, va = boot(a); pb, vb = boot(b, seed=1)
        d = va - vb
        print(f"{Path(a).name}  {pa:.3f}")
        print(f"{Path(b).name}  {pb:.3f}")
        print(f"difference {pa-pb:+.3f}  95% CI [{np.quantile(d,.025):+.3f}, {np.quantile(d,.975):+.3f}]"
              f"  P(diff>0) = {(d>0).mean():.3f}")
    else:
        print(f"{'run':34} {'acc':>6} {'95% CI':>16}")
        for d in args:
            p, v = boot(d)
            print(f"{Path(d).name:34} {p:6.3f}  [{np.quantile(v,.025):.3f}, {np.quantile(v,.975):.3f}]")
