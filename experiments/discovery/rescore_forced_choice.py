"""Re-score finished steering runs under the SEED PAPER'S judging contract.

Their judge picks which of 6 panels an answer best describes (chance = 16.7%).
Ours is lexical and returns -1 when no segment has evidence, and those
trials are counted as WRONG. On Qwen2-VL that is 27.5% of trials and on
Bunny-grid 40%, so our accuracy is systematically pessimistic relative to
theirs — which matters only when comparing against their PUBLISHED numbers.

This does not touch the model. Every answer already stores its per-segment
lexical scores, so forcing a choice is an offline re-read: take the argmax, and
where no segment has any evidence, pick uniformly at random (seeded) — exactly
the chance-level contribution a forced-choice judge makes on an uninformative
answer.

**The lexical judge remains the primary endpoint.** Every audio confirmatory
number was computed with it and none of them is restated here. This is an
additional, clearly-labelled view for like-for-like comparison with Table 2, and
the fraction of trials decided by a coin flip is reported alongside so the
reader can see how much of it is manufactured.

    python rescore_forced_choice.py results/vlm/e2/*/
"""
import json
import sys
from pathlib import Path

import numpy as np


def rescore(d, seed=0):
    rng = np.random.default_rng(seed)
    per_item, forced, n = {}, 0, 0
    for line in open(Path(d) / "answers.jsonl"):
        r = json.loads(line)
        sc = np.asarray(r.get("scores", []), dtype=float)
        if sc.size == 0 or not np.isfinite(sc).any() or sc.max() <= 0:
            pick = int(rng.integers(len(sc) if sc.size else 6))
            forced += 1
        else:
            tied = np.flatnonzero(sc == sc.max())
            pick = int(tied[0] if tied.size == 1 else rng.choice(tied))
        c, t = per_item.get(r["item"], (0, 0))
        per_item[r["item"]] = (c + int(pick == r["target"]), t + 1)
        n += 1
    items = sorted(per_item)
    corr = np.array([per_item[i][0] for i in items], float)
    tot = np.array([per_item[i][1] for i in items], float)
    acc = corr.sum() / tot.sum()
    idx = rng.integers(0, len(items), size=(10000, len(items)))
    boot = corr[idx].sum(1) / tot[idx].sum(1)
    return acc, np.percentile(boot, [2.5, 97.5]), forced / n, n


def main():
    dirs = [d for d in sys.argv[1:] if (Path(d) / "answers.jsonl").exists()]
    if not dirs:
        raise SystemExit("usage: rescore_forced_choice.py <run dirs...>")
    print(f"{'run':34s} {'lexical':>8s} {'forced':>8s}  {'95% CI':>18s}  coin-flipped")
    for d in sorted(dirs):
        rep = Path(d) / "steer_report.json"
        lex = json.load(open(rep))["accuracy"] if rep.exists() else float("nan")
        acc, ci, frac, n = rescore(d)
        print(f"{Path(d).name:34s} {lex:8.3f} {acc:8.3f}  [{ci[0]:.3f}, {ci[1]:.3f}]  "
              f"{frac:6.1%} of {n}")


if __name__ == "__main__":
    main()
