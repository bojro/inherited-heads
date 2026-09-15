"""Apply the PREREGISTRATION.md decision rules to the confirmatory results.

Written BEFORE the confirmatory data existed, so each hypothesis is scored by a
criterion that cannot be renegotiated after seeing the numbers. Run it as often
as you like while the queue is still going: cells whose runs have not landed are
reported PENDING and never silently skipped.

    python confirm_report.py --results results/confirm --out results/CONFIRM_REPORT.md

Two estimator notes:
  * Differences use a PAIRED item bootstrap — the same resampled item indices
    for both conditions. steer_ci.py --diff resamples the two arms with
    different seeds, which ignores the positive correlation between conditions
    on the same item and so returns a WIDER (conservative) interval.
  * random-K pools its two seeds by summing per-item counts, so an item that
    appears in both contributes 12 trials rather than 6.
"""
import argparse, json
from pathlib import Path

import numpy as np

ARMS = [("qwen2_audio", "Qwen2-Audio-7B"), ("salmonn", "SALMONN-13B"),
        ("ultravox", "Ultravox-8B")]
CHANCE = 1 / 6
NBOOT = 10000
ALPHA = 0.05


def per_item(d, require_complete=True):
    """Per-item (correct, total) counts for a run.

    require_complete: a run is only read once its report file exists. Without
    this, a run that is still being written is pooled in at partial weight and
    silently skews any mid-flight figure — which happened on 2026-08-17 when a
    half-written random-250 seed produced a wrong H2 number. Final results were
    never affected (every run completes), but no interim figure was trustworthy.
    """
    d = Path(d)
    if not (d / "answers.jsonl").exists():
        return None
    if require_complete and not ((d / "steer_report.json").exists()
                                 or (d / "report.json").exists()):
        return None
    acc = {}
    for line in open(d / "answers.jsonl"):
        r = json.loads(line)
        c, n = acc.get(r["item"], (0, 0))
        acc[r["item"]] = (c + int(r["judged"] == r["target"]), n + 1)
    return acc or None


def pooled(dirs):
    """Sum per-item counts across runs (used to pool random-K seeds)."""
    out = {}
    got = False
    for d in dirs:
        a = per_item(d)
        if a is None:
            continue
        got = True
        for k, (c, n) in a.items():
            pc, pn = out.get(k, (0, 0))
            out[k] = (pc + c, pn + n)
    return out if got else None


def arrays(a, items=None):
    items = sorted(a) if items is None else items
    C = np.array([a[i][0] for i in items], float)
    N = np.array([a[i][1] for i in items], float)
    return np.array(items), C, N


def ci(a, seed=0):
    if a is None:
        return None
    _, C, N = arrays(a)
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(C), (NBOOT, len(C)))
    v = C[idx].sum(1) / N[idx].sum(1)
    return dict(acc=C.sum() / N.sum(), lo=float(np.quantile(v, .025)),
                hi=float(np.quantile(v, .975)), n_items=len(C))


def diff(a, b, seed=0):
    """PAIRED bootstrap over the items both runs share."""
    if a is None or b is None:
        return None
    items = sorted(set(a) & set(b))
    if not items:
        return None
    _, CA, NA = arrays(a, items)
    _, CB, NB = arrays(b, items)
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(items), (NBOOT, len(items)))
    d = CA[idx].sum(1) / NA[idx].sum(1) - CB[idx].sum(1) / NB[idx].sum(1)
    p = 2 * min((d <= 0).mean(), (d >= 0).mean())
    return dict(d=CA.sum() / NA.sum() - CB.sum() / NB.sum(),
                lo=float(np.quantile(d, .025)), hi=float(np.quantile(d, .975)),
                p=float(max(p, 1.0 / NBOOT)), n_items=len(items))


def holm(pvals, alpha=ALPHA):
    m = len(pvals)
    order = np.argsort(pvals)
    rej = np.zeros(m, bool)
    for rank, i in enumerate(order):
        if pvals[i] <= alpha / (m - rank):
            rej[i] = True
        else:
            break
    return rej


def fmt_ci(c):
    return "PENDING" if c is None else f"{c['acc']:.3f} [{c['lo']:.3f}, {c['hi']:.3f}]"


def fmt_diff(d):
    return "PENDING" if d is None else f"{d['d']:+.3f} [{d['lo']:+.3f}, {d['hi']:+.3f}] p={d['p']:.4f}"


def excludes_zero(d):
    return d is not None and (d["lo"] > 0 or d["hi"] < 0)


def verdict(ok, ready):
    if not ready:
        return "PENDING"
    return "PASS" if ok else "**FAIL**"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", type=Path, default=Path("discovery/results/confirm"))
    ap.add_argument("--text", type=Path, default=Path("discovery/results/text/llama31_8b"),
                    help="default assumes you run from experiments/")
    ap.add_argument("--ultravox-gaze", type=Path,
                    default=Path("discovery/results/ultravox/scores.npz"))
    ap.add_argument("--out", type=Path, default=Path("discovery/results/CONFIRM_REPORT.md"))
    a = ap.parse_args()
    R = a.results

    def cond(arm, tag):
        return per_item(R / f"{arm}_{tag}")

    def rnd(arm, k):
        return pooled([R / f"{arm}_random{k}", R / f"{arm}_random{k}_s1"])

    # label the report by the items actually present, so a run pointed at the
    # exploratory tree can never be mistaken for a confirmatory one
    seen = set()
    for d in sorted(R.glob("*")):
        pi = per_item(d)
        if pi:
            seen |= set(pi)
    if not seen:
        rng_txt = "NO RUNS FOUND under " + str(R)
    elif min(seen) >= 100:
        rng_txt = f"Held-out items {min(seen)}-{max(seen)} (confirmatory)."
    else:
        rng_txt = (f"**items {min(seen)}-{max(seen)} — NOT the held-out set; "
                   f"this is EXPLORATORY data and must not be reported as confirmed.**")

    L = ["# Confirmatory report — pre-registered criteria applied mechanically",
         "",
         f"{rng_txt} Primary endpoint: steering accuracy, chance = {CHANCE:.3f}.",
         f"Item bootstrap, {NBOOT} resamples, 95% percentile CIs; differences PAIRED.",
         "Criteria are quoted from PREREGISTRATION.md and applied without adjustment.",
         "PENDING = that run has not landed yet; it is never treated as a pass.",
         ""]

    # ---------------- H1 ----------------
    L += ["## H1 (replication) — top-100 CI lower bound > 1/6 in all three arms", "",
          "| arm | unsteered | gaze top-100 | lower > 1/6 |", "|---|---|---|---|"]
    h1_ok, h1_ready = True, True
    for arm, name in ARMS:
        c, u = ci(cond(arm, "top100")), ci(cond(arm, "none0"))
        ok = c is not None and c["lo"] > CHANCE
        h1_ready &= c is not None
        h1_ok &= ok
        L.append(f"| {name} | {fmt_ci(u)} | {fmt_ci(c)} | {'yes' if ok else ('PENDING' if c is None else 'NO')} |")
    L += ["", f"**H1: {verdict(h1_ok, h1_ready)}**", ""]

    # ---------------- H2 ----------------
    L += ["## H2 (specificity) — gaze top-K > random-K, >= 8 of 9 cells with CI excluding 0", "",
          "| arm | K | gaze top-K | random-K (2 seeds) | paired difference | excl. 0 |",
          "|---|---|---|---|---|---|"]
    cells, pv = [], []
    for arm, name in ARMS:
        for k in (50, 100, 250):
            g, r = cond(arm, f"top{k}"), rnd(arm, k)
            d = diff(g, r)
            cells.append((name, k, d))
            L.append(f"| {name} | {k} | {fmt_ci(ci(g))} | {fmt_ci(ci(r))} | {fmt_diff(d)} | "
                     f"{'PENDING' if d is None else ('yes' if excludes_zero(d) else 'no')} |")
    ready = all(d is not None for _, _, d in cells)
    n_excl = sum(excludes_zero(d) and d["d"] > 0 for _, _, d in cells)
    L += ["", f"Cells with CI excluding 0 in the predicted direction: **{n_excl} of 9** (criterion: >= 8)."]
    if ready:
        pv = [d["p"] for _, _, d in cells]
        rej = holm(pv)
        L.append(f"Holm-Bonferroni across the 9 cells at alpha {ALPHA}: **{int(rej.sum())} of 9** survive.")
        for (name, k, d), rj in zip(cells, rej):
            if not rj:
                L.append(f"  - not surviving Holm: {name} K={k} (p={d['p']:.4f})")
    L += ["", f"**H2: {verdict(n_excl >= 8, ready)}**", ""]

    # ---------------- H3 ----------------
    L += ["## H3 (gaze is not audio mass) — gaze top-100 > mass top-100 in Qwen and Ultravox, NOT in SALMONN", "",
          "| arm | gaze top-100 | mass top-100 | paired difference | predicted | met |",
          "|---|---|---|---|---|---|"]
    want = {"qwen2_audio": True, "ultravox": True, "salmonn": False}
    h3_ok, h3_ready = True, True
    for arm, name in ARMS:
        d = diff(cond(arm, "top100"), cond(arm, "masstop100"))
        h3_ready &= d is not None
        pred = "excludes 0, positive" if want[arm] else "includes 0"
        met = (d is not None and ((excludes_zero(d) and d["d"] > 0) if want[arm] else not excludes_zero(d)))
        h3_ok &= met
        L.append(f"| {name} | {fmt_ci(ci(cond(arm,'top100')))} | {fmt_ci(ci(cond(arm,'masstop100')))} | "
                 f"{fmt_diff(d)} | {pred} | {'PENDING' if d is None else ('yes' if met else 'NO')} |")
    L += ["", f"**H3: {verdict(h3_ok, h3_ready)}**", ""]

    # ---------------- H4 ----------------
    L += ["## H4 (mass and depth controlled) — Qwen2-Audio", "",
          "Clause 1: gaze_highmass > mass_bandmatched (CI excludes 0).",
          "Clause 2 was REPLACED after being contradicted on exploratory data",
          "(Deviations 2026-08-17): re-test of gaze_lowmass > gaze_highmass.",
          "Depth-matched pair is a disclosed addition (Deviations 2026-08-17b): layer",
          "histogram identical by construction, gaze matched, mass differs ~2x.", "",
          "| arm | comparison | paired difference | criterion | met |", "|---|---|---|---|---|"]
    h4_ok, h4_ready = True, True
    for arm, name in ARMS:
        for (x, y, crit, need_pos, primary) in [
                ("gaze_highmass", "mass_bandmatched", "excludes 0, positive", True, arm == "qwen2_audio"),
                ("gaze_lowmass", "gaze_highmass", "excludes 0, positive (re-test)", True, arm == "qwen2_audio"),
                ("gaze_lowmass_dm", "gaze_highmass_dm", "depth-matched, disclosed addition", True, False)]:
            d = diff(cond(arm, x), cond(arm, y))
            met = d is not None and excludes_zero(d) and (d["d"] > 0) == need_pos
            if primary:
                h4_ready &= d is not None
                h4_ok &= met
            L.append(f"| {name} | {x} - {y} | {fmt_diff(d)} | {crit} | "
                     f"{'PENDING' if d is None else ('yes' if met else 'NO')}{' (primary)' if primary else ''} |")
    L += ["", f"**H4 (Qwen2-Audio clauses only): {verdict(h4_ok, h4_ready)}**", ""]

    # ---------------- H5 ----------------
    L += ["## H5 (concentration does not predict steerability) — Ultravox not lowest at K=100", "",
          "| arm | E1 separation ratio | gaze top-100 accuracy |", "|---|---|---|"]
    accs = {}
    for arm, name in ARMS:
        c = ci(cond(arm, "top100"))
        accs[arm] = None if c is None else c["acc"]
        sep = "see results/<arm>/report.json"
        L.append(f"| {name} | {sep} | {fmt_ci(c)} |")
    h5_ready = all(v is not None for v in accs.values())
    h5_ok = h5_ready and min(accs, key=lambda k: accs[k]) != "ultravox"
    L += ["", f"**H5: {verdict(h5_ok, h5_ready)}** "
          f"({'Ultravox is lowest' if h5_ready and not h5_ok else 'Ultravox is not lowest' if h5_ready else 'pending'})", ""]

    # ---------------- H6 ----------------
    L += ["## H6 (censoring signature) — random-K rises with K in Ultravox, falls in Qwen and SALMONN", "",
          "| arm | random-50 | random-100 | random-250 | 250 - 50 | predicted | met |",
          "|---|---|---|---|---|---|---|"]
    rise = {"ultravox": True, "qwen2_audio": False, "salmonn": False}
    h6_ok, h6_ready = True, True
    for arm, name in ARMS:
        cs = {k: ci(rnd(arm, k)) for k in (50, 100, 250)}
        d = diff(rnd(arm, 250), rnd(arm, 50))
        h6_ready &= d is not None
        met = d is not None and ((d["d"] > 0) == rise[arm])
        h6_ok &= met
        L.append(f"| {name} | {fmt_ci(cs[50])} | {fmt_ci(cs[100])} | {fmt_ci(cs[250])} | {fmt_diff(d)} | "
                 f"{'increase' if rise[arm] else 'decrease'} | {'PENDING' if d is None else ('yes' if met else 'NO')} |")
    L += ["", f"**H6: {verdict(h6_ok, h6_ready)}**", ""]

    # ---------------- H7 ----------------
    L += ["## H7 (inheritance) — the one genuine prediction", "",
          "Ultravox audio gaze top-100 vs bare Llama-3.1-8B-Instruct TEXT gaze top-100.",
          "Audio ranking is frozen from exploratory items 0-49 by design; the text run",
          "uses held-out passages 100-149, so a shared-passage confound cannot inflate",
          "the overlap. Pre-committed threshold: >= 25 of 100 (chance = 100^2/1024 = 9.77).", ""]
    tp = a.text / "scores.npz"
    if tp.exists() and a.ultravox_gaze.exists():
        gt = np.load(a.ultravox_gaze)["scores"]
        tt = np.load(tp)["scores"]
        if gt.shape != tt.shape:
            L.append(f"**shape mismatch** audio {gt.shape} vs text {tt.shape} — cannot compare.")
            h7_ok = h7_ready = False
        else:
            n = gt.size
            ga = set(np.argsort(gt.ravel())[::-1][:100].tolist())
            ta = set(np.argsort(tt.ravel())[::-1][:100].tolist())
            ov = len(ga & ta)
            exp = 100 * 100 / n
            # exact tail probability under the hypergeometric null
            from math import comb
            p = sum(comb(100, i) * comb(n - 100, 100 - i) for i in range(ov, 101)) / comb(n, 100)
            L += [f"Overlap: **{ov} of 100** heads (chance {exp:.2f}, hypergeometric one-sided p = {p:.3g}).", ""]
            h7_ready = True
            h7_ok = ov >= 25
    else:
        L.append("Text run has not landed (`results/text/llama31_8b/scores.npz` missing).")
        h7_ok = h7_ready = False
    L += ["", f"**H7: {verdict(h7_ok, h7_ready)}**", ""]

    # ---------------- consequences ----------------
    L += ["## What the failures mean (from PREREGISTRATION.md, not written after the fact)", "",
          "- H3 or H4 fails -> withdraw \"the gaze score is not a proxy for audio",
          "  attentiveness\"; the result reduces to a causal replication of the seed",
          "  paper in a new modality.",
          "- H5 fails -> withdraw the methodological claim that concentration criteria",
          "  have false negatives.",
          "- H7 fails -> the frozen backbone's tracking is not inherited text machinery",
          "  and needs another explanation.",
          "- None of these is to be reframed as a success.", ""]

    a.out.parent.mkdir(parents=True, exist_ok=True)
    a.out.write_text("\n".join(L))
    print("\n".join(L))
    print(f"\nwrote {a.out}")


if __name__ == "__main__":
    main()
