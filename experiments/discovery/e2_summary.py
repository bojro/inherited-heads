"""Collate every E2 run into one table + one figure.

    python e2_summary.py --results results --out results/E2_SUMMARY

Reads results/e2/<arm>_<condition><K>[_sN]/steer_report.json (+ answers.jsonl
for fluency) and results/mass/<arm>/scores.npz, writes E2_table.md,
E2_curves.png and e2_table.json. Accuracy alone is not enough — a degenerate
generation that repeats target words scores as correct — so every row carries
distinct-2 and the empty rate beside it.
"""
import argparse, json, re
from collections import defaultdict
from pathlib import Path

import numpy as np

from fluency import metrics  # noqa

ARMS = [("qwen2_audio", "Qwen2-Audio-7B", "joint"),
        ("salmonn", "SALMONN-13B", "all frozen + LoRA"),
        ("ultravox", "Ultravox-8B", "frozen backbone")]
CHANCE = 1 / 6


def load(res: Path):
    out = defaultdict(dict)
    for d in sorted((res / "e2").glob("*/steer_report.json")):
        r = json.load(open(d))
        name = d.parent.name
        arm = next(a for a, _, _ in ARMS if name.startswith(a))
        rest = name[len(arm) + 1:]
        m = re.match(r"(masstop|top|random|all|none|gaze_not_mass|mass_not_gaze)(\d*)(?:_s(\d+))?$", rest)
        if not m:
            continue
        cond, k, seed = m.group(1), m.group(2), m.group(3)
        ans = [json.loads(l)["answer"] for l in open(d.parent / "answers.jsonl")]
        fl = metrics(ans)
        out[arm][(cond, int(k or r["k"] or 0), int(seed or 0))] = dict(
            acc=r["accuracy"], unattr=r["unattributed_rate"],
            distinct2=fl["distinct2"], empty=fl["empty_rate"], n=fl["n"])
    return out


def mean_over_seeds(d, cond, k):
    vals = [v for (c, kk, s), v in d.items() if c == cond and kk == k]
    if not vals:
        return None
    return dict(acc=float(np.mean([v["acc"] for v in vals])),
                distinct2=float(np.mean([v["distinct2"] for v in vals])),
                n_seeds=len(vals))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", type=Path, default=Path("results"))
    ap.add_argument("--out", type=Path, default=Path("results/E2_SUMMARY"))
    a = ap.parse_args()
    data = load(a.results)
    a.out.mkdir(parents=True, exist_ok=True)

    KS = [5, 10, 25, 50, 100, 250, 500]
    lines = ["# E2 steering — all conditions", "",
             f"20 strips x 6 target segments = 120 generations per cell; chance = {CHANCE:.3f}.",
             "`d2` = distinct-2 (1.0 = no repetition; < 0.6 is visibly degenerate).", ""]
    table = {}
    for arm, label, regime in ARMS:
        d = data.get(arm)
        if not d:
            continue
        lines += [f"## {label} ({regime})", "",
                  "| condition | " + " | ".join(f"K={k}" for k in KS) + " |",
                  "|---|" + "---|" * len(KS)]
        for cond, pretty in [("top", "gaze top-K"), ("masstop", "audio-mass top-K"), ("random", "random-K (same band)")]:
            cells = []
            for k in KS:
                v = mean_over_seeds(d, cond, k)
                cells.append(f"**{v['acc']:.3f}** (d2 {v['distinct2']:.2f})" if v else "—")
            lines.append(f"| {pretty} | " + " | ".join(cells) + " |")
        for cond, pretty in [("none", "unsteered"), ("all", "all heads")]:
            v = [x for (c, _, _), x in d.items() if c == cond]
            if v:
                lines.append(f"| {pretty} | " + f"{v[0]['acc']:.3f} (d2 {v[0]['distinct2']:.2f}, empty {v[0]['empty']:.2f})"
                             + " | " * 0 + " |" + " |" * (len(KS) - 1))
        for cond, pretty in [("gaze_not_mass", "gaze-only heads (not audio-attentive)"),
                             ("mass_not_gaze", "mass-only heads (not gaze-ranked)")]:
            v = [(kk, x) for (c, kk, _), x in d.items() if c == cond]
            if v:
                k, x = v[0]
                lines.append(f"| {pretty} (K={k}) | {x['acc']:.3f} (d2 {x['distinct2']:.2f})" + " |" * len(KS))
        lines.append("")
        table[arm] = {f"{c}{k}_s{s}": v for (c, k, s), v in d.items()}
    (a.out / "E2_table.md").write_text("\n".join(lines))
    (a.out / "e2_table.json").write_text(json.dumps(table, indent=1))

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, 3, figsize=(13, 4.2), sharey=True)
    for ax, (arm, label, regime) in zip(axes, ARMS):
        d = data.get(arm, {})
        for cond, lab, col, mk in [("top", "gaze top-K", "#c0392b", "o"),
                                   ("masstop", "audio-mass top-K", "#8e44ad", "s"),
                                   ("random", "random-K", "#7f8c8d", "^")]:
            xs, ys = [], []
            for k in KS:
                v = mean_over_seeds(d, cond, k)
                if v:
                    xs.append(k); ys.append(v["acc"])
            if xs:
                ax.plot(xs, ys, marker=mk, color=col, label=lab, lw=1.8, ms=5)
        for cond, lab, col, ls in [("none", "unsteered", "#2c3e50", ":"), ("all", "all heads", "#16a085", "--")]:
            v = [x for (c, _, _), x in d.items() if c == cond]
            if v:
                ax.axhline(v[0]["acc"], color=col, ls=ls, lw=1.2, label=lab)
        ax.axhline(CHANCE, color="k", lw=0.9, alpha=.5)
        ax.set_xscale("log"); ax.set_xticks(KS); ax.set_xticklabels(KS)
        ax.set(xlabel="K (heads biased)", title=f"{label}\n({regime})")
        ax.grid(alpha=.25)
    axes[0].set_ylabel("steering accuracy (chance = 1/6)")
    axes[0].legend(fontsize=7.5, loc="upper left")
    fig.tight_layout(); fig.savefig(a.out / "E2_curves.png", dpi=150)
    print("\n".join(lines[:40]))
    print(f"\nwrote {a.out}/E2_table.md, E2_curves.png, e2_table.json")


if __name__ == "__main__":
    main()
