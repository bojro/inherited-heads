#!/usr/bin/env python3
"""Generate paper/tex/figs/*.pdf -- the six figures -- from the result files.

Same discipline as discovery/make_ledger.py: the figures are
GENERATED, never hand-drawn from transcribed numbers.

  * Every plotted value is read from experiments/discovery/results/<run>/ where
    the run file carries it, and otherwise parsed out of RESULTS_LEDGER.md
    (modality mass is the only value that lives solely in the ledger).
  * Every value is then checked back against the ledger section the paper cites
    it from -- see cite() -- so a figure cannot silently drift from the ledger.
  * A value that is in neither a result file nor the ledger is NOT drawn. The
    element is omitted and a WARNING goes to stderr; nothing is invented and
    nothing is interpolated across a missing cell.
  * No model is run and no GPU is touched: json, npz and numpy only.

Usage:
  python3 make_figures.py                    # all six, into ../tex/figs
  python3 make_figures.py f2 f5              # a subset
  python3 make_figures.py --png-dir /tmp/qa  # extra PNG copies for screen QA
                                             # (PNGs are a QA aid, not artefacts)

LAYOUT. Two-column ACL. Each figure declares the width it is built for and the
PDF canvas is exactly that wide, so

    \\includegraphics[width=\\columnwidth]{...}   (F4, F5   -- 3.03in)
    \\includegraphics[width=\\textwidth]{...}     (F1,F2,F3,F6 -- 6.3in, figure*)

places it at 1:1 and the point sizes set in RC below are the point sizes that
print. Do not scale these files in LaTeX; change WIDTH_COL/WIDTH_FULL here.

COLOUR. Okabe-Ito, colourblind-safe. Nothing is encoded by colour alone: every
series also differs in hatch (bars) or marker+linestyle (lines).
"""
import argparse
import json
import os
import re
import sys

import numpy as np

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import FancyBboxPatch, Rectangle

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.abspath(os.path.join(HERE, "..", ".."))
RESULTS = os.path.join(REPO, "experiments", "discovery", "results")
LEDGER = os.path.join(REPO, "experiments", "RESULTS_LEDGER.md")
OUTDIR = os.path.join(REPO, "paper", "tex", "figs")

# Measured from acl.sty ([preprint], a4paper, margin=2.5cm) with
# \typeout{\the\textwidth \the\columnwidth}: 455.244pt and 219.086pt at
# 72.27pt/in. Re-measure if the style options change.
WIDTH_COL = 3.03   # ACL \columnwidth, inches
WIDTH_FULL = 6.30  # ACL \textwidth, inches

WARNINGS = []


def warn(msg):
    """Record and print an omission. Never silently drop a figure element."""
    WARNINGS.append(msg)
    print(f"WARNING: {msg}", file=sys.stderr)


# --------------------------------------------------------------- result files
def j(path, *keys, default=None):
    """Read a value out of a result json, or default if absent."""
    p = os.path.join(RESULTS, path)
    if not os.path.exists(p):
        return default
    d = json.load(open(p))
    for k in keys:
        if not isinstance(d, dict) or k not in d:
            return default
        d = d[k]
    return d


def steer_acc(run):
    """Steering accuracy of one run directory, or None if the run is absent."""
    return j(f"{run}/steer_report.json", "accuracy")


def per_target(run):
    return j(f"{run}/steer_report.json", "per_target_accuracy")


def unattr(run):
    """Fraction of trials the judge could not attribute to ANY segment.

    Added 2026-08-19. Steering accuracy conflates two failures -- describing the
    wrong speaker, and describing nothing placeable -- and after ten draws of the
    layer-matched control it is the SECOND that separates the raw-mass head set
    from a random one. See ledger section 11.
    """
    return j(f"{run}/steer_report.json", "unattributed_rate")


def _topset(x, k):
    return set(np.argsort(np.asarray(x).ravel())[::-1][:k].tolist())


def overlap(pa, pb, k=100):
    """Top-k head-set overlap between two scores.npz, or None if either is absent."""
    pa, pb = os.path.join(RESULTS, pa), os.path.join(RESULTS, pb)
    if not (os.path.exists(pa) and os.path.exists(pb)):
        return None
    return len(_topset(np.load(pa)["scores"], k) & _topset(np.load(pb)["scores"], k))


def layer_matched_null(pa, pb, k=100, B=5000, seed=0):
    """Null overlap distribution: resample k heads with the SAME layer histogram.

    Identical procedure and seed to discovery/make_ledger.py, so the numbers it
    returns are the ledger's numbers (cite() verifies that, it is not assumed).
    """
    pa, pb = os.path.join(RESULTS, pa), os.path.join(RESULTS, pb)
    if not (os.path.exists(pa) and os.path.exists(pb)):
        return None, None
    rng = np.random.default_rng(seed)
    a, b = np.load(pa)["scores"], np.load(pb)["scores"]
    L, H = a.shape
    atop = np.argsort(a.ravel())[::-1][:k]
    btop = _topset(b, k)
    hist = np.bincount(atop // H, minlength=L)
    null = np.empty(B, dtype=int)
    for i in range(B):
        pick = []
        for layer, n in enumerate(hist):
            if n:
                pick.extend(layer * H + rng.choice(H, size=n, replace=False))
        null[i] = len(set(pick) & btop)
    return float(null.mean()), float(np.percentile(null, 95))


# -------------------------------------------------------------- ledger access
_LED = None


def ledger():
    global _LED
    if _LED is None:
        _LED = open(LEDGER, encoding="utf-8").read()
    return _LED


def ledger_section(n):
    """The text of '## n. ...' up to the next '## '."""
    m = re.search(rf"^## {n}\..*?$(.*?)(?=^## |\Z)", ledger(), re.S | re.M)
    if not m:
        warn(f"ledger section {n} not found in {LEDGER}")
        return ""
    return m.group(1)


def _clean(cell):
    return cell.replace("**", "").replace("`", "").strip()


def ledger_tables(n):
    """Every markdown table in a ledger section, as (header, [rows])."""
    tables, header, rows = [], None, []
    lines = ledger_section(n).splitlines()
    for i, line in enumerate(lines):
        s = line.strip()
        if not (s.startswith("|") and s.endswith("|")):
            if header:
                tables.append((header, rows))
                header, rows = None, []
            continue
        cells = [_clean(c) for c in s.strip("|").split("|")]
        nxt = lines[i + 1].strip() if i + 1 < len(lines) else ""
        is_rule = set(nxt.replace("|", "").replace(" ", "")) <= {"-", ":"} and "-" in nxt
        if set("".join(cells).replace(" ", "")) <= {"-", ":"} and cells:
            continue                      # the |---|---| rule itself
        if is_rule:
            if header:
                tables.append((header, rows))
            header, rows = cells, []
        elif header:
            rows.append(cells)
    if header:
        tables.append((header, rows))
    return tables


def ledger_cell(n, row_prefix, col_name):
    """One cell of one ledger table, by row label prefix and column name."""
    for header, rows in ledger_tables(n):
        if col_name not in header:
            continue
        c = header.index(col_name)
        for row in rows:
            if row and row[0].lower().startswith(row_prefix.lower()) and c < len(row):
                return row[c]
    warn(f"ledger section {n}: no cell for row '{row_prefix}' column '{col_name}'")
    return None


def cite(value, section, nd=3, what=""):
    """Check a value we are about to plot appears in the ledger section the paper
    cites it from. Formatting matches make_ledger.py's, so a mismatch means the
    figure and the ledger have genuinely diverged -- regenerate the ledger."""
    if value is None:
        return None
    s = f"{value:.{nd}f}"
    if s not in ledger_section(section):
        warn(f"{what or 'value'} {s} is not in ledger section {section} "
             f"-- figure and ledger disagree; regenerate the ledger")
    return value


def cite_int(value, section, what=""):
    if value is None:
        return None
    if str(int(value)) not in ledger_section(section):
        warn(f"{what or 'value'} {int(value)} is not in ledger section {section}")
    return value


# ------------------------------------------------------------------ plotting
# Okabe-Ito. Role -> (facecolour, hatch, marker, linestyle).
BLUE, VERM, GREEN, ORANGE, SKY = "#0072B2", "#D55E00", "#009E73", "#E69F00", "#56B4E9"
GREY, DARK, FAINT = "#9E9E9E", "#333333", "#EDEDED"

RC = {
    "pdf.fonttype": 42,          # TrueType; Type 3 (the default) is rejected by
    "ps.fonttype": 42,           # several venues' PDF checks
    "font.family": "serif",
    "font.serif": ["STIXGeneral", "DejaVu Serif"],
    "mathtext.fontset": "stix",
    "font.size": 8,
    "axes.titlesize": 8,
    "axes.labelsize": 8,
    "xtick.labelsize": 7.5,
    "ytick.labelsize": 7.5,
    "legend.fontsize": 6.8,
    "axes.linewidth": 0.6,
    "xtick.major.width": 0.6,
    "ytick.major.width": 0.6,
    "xtick.major.size": 2.5,
    "ytick.major.size": 2.5,
    "lines.linewidth": 1.2,
    "patch.linewidth": 0.6,
    "hatch.linewidth": 0.5,
    "legend.frameon": False,
    "legend.handlelength": 1.6,
    "legend.handletextpad": 0.5,
    "legend.columnspacing": 1.2,
    "legend.borderaxespad": 0.2,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "savefig.transparent": False,
    "figure.facecolor": "white",
}
plt.rcParams.update(RC)

CHANCE = 1.0 / 6.0  # six-way steering chance, ledger preamble


def chance_line(ax, x0=None, x1=None, label=True, y=CHANCE):
    ax.axhline(y, color="black", lw=0.8, ls=(0, (1, 1.6)), zorder=1)
    if label:
        ax.text(ax.get_xlim()[1], y, " chance .167", va="center", ha="left",
                fontsize=6.2, color="black")


def bar_labels(ax, xs, ys, nd=3, fs=5.8, dy=0.012, color=DARK):
    for x, y in zip(xs, ys):
        if y is None:
            continue
        ax.text(x, y + dy, f"{y:.{nd}f}".lstrip("0"), ha="center", va="bottom",
                fontsize=fs, color=color)


def save(fig, name, png_dir=None):
    os.makedirs(OUTDIR, exist_ok=True)
    path = os.path.join(OUTDIR, name + ".pdf")
    # metadata: no CreationDate -> byte-identical output across runs
    fig.savefig(path, format="pdf", metadata={
        "CreationDate": None, "Creator": "paper/figures/make_figures.py",
        "Producer": "matplotlib", "Title": name})
    if png_dir:
        os.makedirs(png_dir, exist_ok=True)
        fig.savefig(os.path.join(png_dir, name + ".png"), dpi=220)
    plt.close(fig)
    print(f"wrote {path}")
    return path


# =============================================================== F1: schematic
def fig1(png_dir=None):
    """F1 -- the task and the intervention. FULL WIDTH (figure*, 6.3in).

    A diagram, not a result: the only numbers on it are structural (six
    segments, K, +/-B) and the token geometry of the vision arm (24 interleaved
    runs of 7 at stride 42), which is ledger section 6 and main.tex Method.
    """
    fig, axes = plt.subplots(1, 2, figsize=(WIDTH_FULL, 2.45))
    for ax, mode in zip(axes, ("audio", "vision")):
        _f1_panel(ax, mode)
    fig.subplots_adjust(left=0.005, right=0.995, top=0.99, bottom=0.01, wspace=0.06)
    return save(fig, "fig1_task_and_intervention", png_dir)


def _f1_panel(ax, mode):
    audio = mode == "audio"
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")
    target = 3  # zero-based: the fourth segment, the running example in the paper

    ax.text(0.0, 0.995, ("(a) audio: six speakers in turn" if audio
                         else "(b) vision: a six-panel comic strip"),
            fontsize=8, fontweight="bold", va="top", ha="left")

    # --- row 1: the stimulus, six segments -------------------------------
    x0, w, gap = 0.02, 0.145, 0.018
    for k in range(6):
        x = x0 + k * (w + gap)
        sel = k == target
        ax.add_patch(FancyBboxPatch(
            (x, 0.80), w, 0.12, boxstyle="round,pad=0,rounding_size=0.012",
            facecolor=(SKY if sel else "white"), alpha=(0.45 if sel else 1.0),
            edgecolor=(BLUE if sel else DARK), lw=(1.0 if sel else 0.6)))
        ax.text(x + w / 2, 0.86, f"{'S' if audio else 'P'}{k + 1}",
                ha="center", va="center", fontsize=7)
    ax.text(0.02, 0.775, ("six speech clips, distinct speaker and topic"
                          if audio else "six panels, tiled side by side"),
            fontsize=6.4, color=DARK, va="top")

    # --- the query -------------------------------------------------------
    qx = x0 + target * (w + gap) + w / 2
    ax.annotate("", xy=(qx, 0.80), xytext=(qx, 0.70),
                arrowprops=dict(arrowstyle="-|>", lw=0.8, color=BLUE,
                                shrinkA=0, shrinkB=0, mutation_scale=6))
    ax.text(qx + 0.015, 0.70, "query names segment $k$", fontsize=6.6,
            color=BLUE, va="bottom", ha="left")

    # --- row 2: the token sequence ---------------------------------------
    ty, th = 0.50, 0.075
    tx0, tw = 0.02, 0.72
    ax.add_patch(Rectangle((tx0, ty), tw, th, facecolor="white",
                           edgecolor=DARK, lw=0.6))
    if audio:                     # segment k is one contiguous span
        sx = tx0 + tw * target / 6.0
        ax.add_patch(Rectangle((sx, ty), tw / 6.0, th, facecolor=SKY,
                               alpha=0.75, edgecolor=BLUE, lw=0.8))
        for k in range(1, 6):     # segment boundaries, exact by construction
            ax.plot([tx0 + tw * k / 6.0] * 2, [ty, ty + th], color=DARK,
                    lw=0.4, ls=":")
        span_note = "segment $k$ = one contiguous token run"
    else:                         # panel k is 24 interleaved runs of 7, stride 42
        nrun, stride = 24, 42
        rw = tw / (nrun * stride) * 7
        for r in range(nrun):
            sx = tx0 + tw * (r * stride + target * 7) / (nrun * stride)
            ax.add_patch(Rectangle((sx, ty), rw, th, facecolor=SKY,
                                   alpha=0.9, edgecolor="none"))
        span_note = f"panel $k$ = {nrun} interleaved runs of 7 (stride {stride})"
    ax.add_patch(Rectangle((tx0 + tw, ty), 0.24, th, facecolor="white",
                           edgecolor=DARK, lw=0.6, hatch="////"))
    ax.text(tx0 + tw + 0.12, ty + th / 2, "prompt", ha="center", va="center",
            fontsize=6.4, bbox=dict(facecolor="white", edgecolor="none", pad=0.8))
    ax.text(tx0, ty + th + 0.018,
            ("audio tokens" if audio else "image tokens"), fontsize=6.6, va="bottom")
    ax.text(0.98, ty + th + 0.018, "text tokens: untouched", fontsize=6.4,
            va="bottom", ha="right", color=DARK)
    ax.text(tx0, ty - 0.025, span_note, fontsize=6.4, va="top", color=BLUE)

    # --- row 3: the intervention -----------------------------------------
    by, bh = 0.26, 0.075
    ax.plot([tx0, 0.98], [by] * 2, color=DARK, lw=0.6)
    for k in range(6):
        lo = tx0 + tw * k / 6.0
        hi = tx0 + tw * (k + 1) / 6.0
        y = by + bh if k == target else by - bh
        c = BLUE if k == target else VERM
        ax.plot([lo, lo, hi, hi], [by, y, y, by], color=c, lw=1.1,
                ls=("-" if k == target else (0, (3, 1.4))))
    ax.plot([tx0 + tw, 0.98], [by, by], color=DARK, lw=1.1)
    tgt_mid = tx0 + tw * (target + 0.5) / 6.0
    ax.text(tgt_mid, by + bh + 0.015, "$+B$ on the target span", fontsize=6.6,
            color=BLUE, ha="center", va="bottom")
    ax.text(tx0 + tw / 12.0, by - bh - 0.015, "$-B$ on every other segment",
            fontsize=6.6, color=VERM, ha="left", va="top")
    ax.text(tx0 + tw + 0.12, by + 0.012, "0", fontsize=6.4, color=DARK,
            ha="center", va="bottom")
    ax.text(tx0, 0.055,
            "added to the pre-softmax attention logits of the $K$ selected heads",
            fontsize=6.4, va="bottom", color=DARK)


# =============================================================== F2: inversion
ARMS = [
    dict(key="qwen2_audio", label="Qwen2-Audio", block="audio",
         run="confirm/qwen2_audio_{c}", ledger_row="qwen2_audio"),
    dict(key="salmonn", label="SALMONN", block="audio",
         run="confirm/salmonn_{c}", ledger_row="salmonn"),
    dict(key="ultravox", label="Ultravox", block="audio",
         run="confirm/ultravox_{c}", ledger_row="ultravox"),
    dict(key="qwen2_vl", label="Qwen2-VL", block="vision",
         run="vlm/e2/qwen2_vl_{c}", ledger_row="Qwen2-VL"),
    dict(key="bunny_grid", label="Bunny-grid", block="vision",
         run="vlm/e2/bunny_grid_{c}", ledger_row="Bunny-grid"),
    dict(key="bunny_strip", label="Bunny-strip", block="vision",
         run="vlm/e2/bunny_strip_{c}", ledger_row="Bunny-strip"),
]


def fig2(png_dir=None):
    """F2 -- what separates the two head sets. FULL WIDTH (figure*). Ledger 3, 11.

    REBUILT 2026-08-19. The previous version drew the inversion: the raw score
    against five layer-histogram-matched draws it sat below. Ten draws now span
    .007-.663 and two land at or below the raw score, so that claim is withdrawn
    (ledger section 11) and a figure asserting it would be false.

    What replaced it is a two-panel contrast, because the finding IS the
    contrast: on accuracy (left) the raw score sits inside the control
    distribution, and on unattributability (right) it sits above every draw.
    Same runs, same heads, two readouts. Panel (a) is deliberately the one that
    does NOT separate them -- showing the negative first is the honest order and
    it is what makes (b) mean anything.

    RELABELLED 2026-09-12 (ledger 26). The ten draws copy the SELECTIVITY
    hundred's per-layer counts (mean layer 22.8); the raw-score hundred sits at
    mean layer 6.2. They are not a control for the raw score, so the figure no
    longer calls them layer-matched, no longer counts draws "at or below theirs",
    and no longer marks "highest of any draw".
    """
    ours, theirs, rband = [], [], []
    for arm in ARMS:
        ours.append(cite(steer_acc(arm["run"].format(c="top100")), 3,
                         what=f"{arm['label']} ours"))
        theirs.append(cite(steer_acc(arm["run"].format(c="rawtop100")), 3,
                           what=f"{arm['label']} theirs"))
        rband.append(cite(steer_acc(arm["run"].format(c="random100")), 3,
                          what=f"{arm['label']} random(band)"))

    # REBUILT AGAIN 2026-09-12 after simulated reads: even relabelled, a strip of
    # draws at the selectivity hundred's depth plotted beside the mass bar invites
    # the unmatched comparison. The draws are in Appendix A. Panel (b) now sets
    # the mass ranking against NO intervention (ledger 25), which is matched on
    # items and needs no depth argument.
    u_ours = cite(unattr("confirm/ultravox_top100"), 3, what="unattr ours")
    u_theirs = cite(unattr("confirm/ultravox_rawtop100"), 3, what="unattr theirs")
    u_none = cite(unattr("confirm/ultravox_none0"), 25, what="unattr none")

    fig, (axa, axb) = plt.subplots(
        1, 2, figsize=(WIDTH_FULL, 2.95), gridspec_kw=dict(width_ratios=[2.35, 1],
                                                           wspace=0.26))

    # ---- (a) steering accuracy, all six arms -------------------------------
    x = np.arange(len(ARMS), dtype=float)
    w = 0.26
    offs = (-w, 0.0, w)
    ui = [a["key"] for a in ARMS].index("ultravox")
    axa.axvspan(x[ui] - 0.50, x[ui] + 0.50, color=FAINT, zorder=0)
    axa.bar(x + offs[0], ours, w, label="selectivity ranking",
            color=BLUE, edgecolor=DARK, zorder=3)
    axa.bar(x + offs[1], theirs, w, label="mass ranking",
            color=VERM, edgecolor=DARK, hatch="///", zorder=3)
    axa.bar(x + offs[2], rband, w, label="random, band-matched (one draw)",
            color=GREY, edgecolor=DARK, hatch="...", zorder=3)
    bar_labels(axa, x + offs[0], ours)
    bar_labels(axa, x + offs[1], theirs)
    axa.set_xticks(x)
    axa.set_xticklabels([a["label"] for a in ARMS])
    for tick, arm in zip(axa.get_xticklabels(), ARMS):
        if arm["key"] == "ultravox":
            tick.set_fontweight("bold")
    axa.set_ylabel("steering accuracy (held-out, $K$=100)")
    axa.set_ylim(0, 1.13)
    axa.set_yticks([0, 0.167, 0.5, 1.0])
    axa.set_yticklabels(["0", ".167", ".50", "1.0"])
    axa.set_xlim(-0.55, len(ARMS) - 0.45)
    chance_line(axa, label=False)
    nb = sum(1 for a in ARMS if a["block"] == "audio")
    axa.axvline(nb - 0.5, color=DARK, lw=0.5, ls=(0, (2, 2)))
    axa.set_title("(a)  steering accuracy", fontsize=7.4, loc="left", pad=3)
    axa.legend(loc="lower left", bbox_to_anchor=(-0.005, 1.055), ncol=3)

    # ---- (b) unattributability, Ultravox ------------------------------------
    if None not in (u_ours, u_theirs, u_none):
        vals = [u_ours, u_none, u_theirs]
        axb.bar([0], [u_ours], 0.55, color=BLUE, edgecolor=DARK, zorder=3)
        axb.bar([1], [u_none], 0.55, color="white", edgecolor=DARK, zorder=3)
        axb.bar([2], [u_theirs], 0.55, color=VERM, edgecolor=DARK, hatch="///",
                zorder=3)
        for xx, v in enumerate(vals):
            axb.text(xx, v + 0.018, f"{v:.3f}".lstrip("0"), ha="center",
                     va="bottom", fontsize=6.4, color=DARK)
        axb.set_xticks([0, 1, 2])
        axb.set_xticklabels(["selectivity", "no\nintervention", "mass"])
        axb.set_xlim(-0.6, 2.6)
        axb.set_ylim(0, 0.82)
        axb.set_ylabel("output the judge cannot place\non any segment")
        axb.set_title("(b)  Ultravox: unplaceable output", fontsize=7.4, loc="left",
                      pad=3)
    else:
        warn("F2b: an unattributed rate is missing -- panel (b) omitted")

    fig.subplots_adjust(left=0.085, right=0.985, top=0.800, bottom=0.135)
    return save(fig, "fig2_failure_mode", png_dir)


# ============================================================= F3: inheritance
PAIRS = [
    dict(label="Ultravox\nvs Llama-3.1-8B", a="ultravox/scores.npz",
         b="text/llama31_8b/scores.npz", rel="reliability/ultravox_audio",
         row="Ultravox audio"),
    dict(label="SALMONN\nvs Vicuna-13B", a="salmonn/scores.npz",
         b="text/vicuna13b/scores.npz", rel=None, row="SALMONN audio"),
    dict(label="Qwen2-VL\nvs Qwen2-7B", a="vlm/qwen2_vl/scores.npz",
         b="text/qwen2_7b_captions/scores.npz", rel="reliability/qwen2vl_image",
         row="Qwen2-VL image"),
    dict(label="Bunny-grid\nvs Bunny-LM", a="vlm/bunny_grid/scores.npz",
         b="text/bunny_phi2/scores.npz", rel=None, row="Bunny-grid image"),
    dict(label="Bunny-strip\nvs Bunny-LM", a="vlm/bunny_strip/scores.npz",
         b="text/bunny_phi2/scores.npz", rel=None, row="Bunny-strip image"),
    dict(label="Qwen2-Audio\nvs Qwen1.5-7B", a="qwen2_audio_v2/scores.npz",
         b="text/qwen15_7b_chat/scores.npz", rel=None, row="Qwen2-Audio audio"),
]


def fig3(png_dir=None):
    """F3 -- inheritance. FULL WIDTH (figure*, 6.3in). Ledger sections 1 and 2.

    Observed top-100 overlap against the layer-matched null (mean, whisker to
    the 95th percentile), with the split-half ceiling drawn as a BOUND -- the
    region above it is shaded as unreachable, not marked as a target. The
    ceiling exists for two of the six pairs (ledger section 2 has three
    rankings); the other four get no ceiling rather than a borrowed one.
    """
    obs, nmean, n95, ceil = [], [], [], []
    for p in PAIRS:
        o = overlap(p["a"], p["b"])
        if o is None:
            warn(f"F3: scores missing for {p['label']} -- pair omitted")
        obs.append(o)
        m, q = layer_matched_null(p["a"], p["b"])
        nmean.append(m)
        n95.append(q)
        if o is not None and f"{o}/100" not in ledger_section(1):
            warn(f"F3: overlap {o}/100 for {p['label']} is not in ledger section 1")
        if m is not None and f"{m:.1f}" not in ledger_section(1):
            warn(f"F3: null mean {m:.1f} for {p['label']} is not in ledger section 1")
        c = overlap(f"{p['rel']}_A/scores.npz", f"{p['rel']}_B/scores.npz") if p["rel"] else None
        if c is None:
            warn(f"F3: no split-half ceiling for {p['label']} "
                 f"(no reliability run; ledger section 1 says n/a) -- bound not drawn")
        else:
            cite_int(c, 2, what=f"{p['label']} ceiling")
        ceil.append(c)

    fig, ax = plt.subplots(figsize=(WIDTH_FULL, 2.65))
    x = np.arange(len(PAIRS), dtype=float)
    w = 0.3

    for i, c in enumerate(ceil):          # the bound, drawn first and behind
        if c is None:
            continue
        ax.add_patch(Rectangle((x[i] - 0.46, c), 0.92, 100 - c, facecolor="none",
                               edgecolor=GREY, hatch="\\\\\\", lw=0.0, alpha=0.55,
                               zorder=0))
        ax.plot([x[i] - 0.46, x[i] + 0.46], [c, c], color=DARK, lw=1.0,
                ls=(0, (4, 1.5)), zorder=4)
        ax.text(x[i] + 0.46, c + 1.5, f"ceiling {c}", fontsize=6.0, ha="right",
                va="bottom", color=DARK)

    ax.bar(x - w / 2, [o if o is not None else 0 for o in obs], w,
           color=BLUE, edgecolor=DARK, zorder=3, label="observed overlap")
    ax.bar(x + w / 2, [m if m is not None else 0 for m in nmean], w,
           color=GREY, edgecolor=DARK, hatch="...", zorder=3,
           label="layer-matched null (mean; whisker to 95th pct)")
    for i, (m, q) in enumerate(zip(nmean, n95)):
        if m is None:
            continue
        ax.plot([x[i] + w / 2] * 2, [m, q], color=DARK, lw=0.9, zorder=4)
        ax.plot([x[i] + w / 2 - w / 4, x[i] + w / 2 + w / 4], [q, q], color=DARK,
                lw=0.9, zorder=4)
        ax.text(x[i] + w / 2, q + 1.2, f"{m:.1f}", ha="center", va="bottom",
                fontsize=5.8, color=DARK)
    for i, o in enumerate(obs):
        if o is not None:
            ax.text(x[i] - w / 2, o + 1.2, str(o), ha="center", va="bottom",
                    fontsize=6.2, color=DARK, fontweight="bold")

    ax.set_xticks(x)
    ax.set_xticklabels([p["label"] for p in PAIRS])
    ax.set_ylabel("top-100 head overlap (of 100)")
    ax.set_ylim(0, 104)
    ax.set_yticks([0, 25, 50, 75, 100])
    ax.set_xlim(-0.55, len(PAIRS) - 0.45)
    handles, labels = ax.get_legend_handles_labels()
    handles.append(Line2D([], [], color=DARK, lw=1.0, ls=(0, (4, 1.5))))
    labels.append("split-half ceiling: an upper BOUND on any overlap here\n"
                  "(hatched region above it is unreachable, not a target)")
    ax.legend(handles, labels, loc="upper right", ncol=1)
    fig.tight_layout(pad=0.4)
    return save(fig, "fig3_inheritance", png_dir)


# ==================================================================== F4: AC1
SPLITS = [
    dict(label="Qwen2-Audio", key="qwen2_audio", sh=66, ao=34,
         full="confirm/qwen2_audio_top100",
         shared="crosstask/qwen2_audio_shared66",
         only="crosstask/qwen2_audio_only34",
         shctl="crosstask/qwen2_audio_randmatched66",
         shctl_seeds="mseeds/qwen2_audio_randmatched66_s{s}",
         t100="t100/qwen2_audio_t100rand66_s{s}"),
    dict(label="Ultravox", key="ultravox", sh=71, ao=29,
         full="confirm/ultravox_top100",
         shared="crosstask/ultravox_shared71",
         only="crosstask/ultravox_audioonly29",
         shctl="crosstask/ultravox_randmatched71",
         shctl_seeds="mseeds/ultravox_randmatched71_s{s}",
         t100="t100/ultravox_t100rand71_s{s}"),
    dict(label="SALMONN", key="salmonn", sh=74, ao=26,
         full="confirm/salmonn_top100",
         shared="crosstask/salmonn_shared74",
         only="crosstask/salmonn_only26",
         shctl="crosstask/salmonn_randmatched74",
         shctl_seeds="mseeds/salmonn_randmatched74_s{s}",
         t100="t100/salmonn_t100rand74_s{s}"),
]


def fig4(png_dir=None):
    """F4 -- the sufficiency result, three arms. FULL WIDTH. Ledger 19 and 21.

    REBUILT 2026-08-19. The previous version was Ultravox only and annotated the
    audio-only subset as falling "below its own matched control". That claim is
    WITHDRAWN: five draws of that control span .008-.650 and the subset sits
    inside them (ledger section 19). A figure carrying the annotation would
    assert something the ledger now denies.

    What it draws instead is the comparison section 4 actually leads with, on
    all three arms, in the order the argument runs:
      full 100 -> shared subset -> a random draw of the SAME SIZE from the same
      100 -> the audio-only remainder -> the matched-random control as a strip
      over all five of its draws.
    The third bar is the one that constrains the reading, and putting it beside
    the second is the whole point: any two-thirds of this head set carries most
    of the behaviour.
    """
    fig, ax = plt.subplots(figsize=(WIDTH_FULL, 2.75))
    x = np.arange(len(SPLITS), dtype=float)
    w = 0.165
    offs = (-2 * w, -w, 0.0, w, 2 * w)

    full, sh, t100, ao = [], [], [], []
    for spec in SPLITS:
        full.append(cite(steer_acc(spec["full"]), 19, what=f"{spec['label']} full"))
        sh.append(cite(steer_acc(spec["shared"]), 19, what=f"{spec['label']} shared"))
        ao.append(cite(steer_acc(spec["only"]), 19, what=f"{spec['label']} audio-only"))
        draws = [steer_acc(spec["t100"].format(s=i)) for i in range(3)]
        draws = [d for d in draws if d is not None]
        if not draws:
            warn(f"F4: no within-top-100 draws for {spec['label']} -- the bar "
                 "that constrains the reading is missing")
            t100.append(None)
        else:
            t100.append(float(np.mean(draws)))

    ax.bar(x + offs[0], full, w, label="full discovered 100",
           color=SKY, edgecolor=DARK, zorder=3)
    ax.bar(x + offs[1], sh, w, label="shared with the text backbone",
           color=BLUE, edgecolor=DARK, zorder=3)
    ax.bar(x + offs[2], t100, w,
           label="random draw, same size, same 100",
           color="white", edgecolor=DARK, hatch="xxx", zorder=3)
    ax.bar(x + offs[3], ao, w, label="audio-only remainder",
           color=ORANGE, edgecolor=DARK, hatch="///", zorder=3)
    for o, vals in zip(offs, (full, sh, t100, ao)):
        bar_labels(ax, x + o, vals)

    # the matched-random control, as all five of its draws
    for i, spec in enumerate(SPLITS):
        d = [steer_acc(spec["shctl"])] + [
            steer_acc(spec["shctl_seeds"].format(s=k)) for k in (1, 2, 3, 4)]
        d = [v for v in d if v is not None]
        if len(d) < 5:
            warn(f"F4: {spec['label']} matched control has {len(d)} draws, not 5 "
                 "-- the strip understates its spread")
        if not d:
            continue
        xs = x[i] + offs[4]
        ax.plot([xs] * len(d), d, ls="none", marker="D", ms=2.6, mfc="white",
                mec=DARK, mew=0.7, zorder=6,
                label="matched random, all draws" if i == 0 else None)
        ax.plot([xs, xs], [min(d), max(d)], color=DARK, lw=0.9, zorder=5)

    ax.set_xticks(x)
    ax.set_xticklabels([f"{sp['label']}\n{sp['sh']} shared / {sp['ao']} audio-only"
                        for sp in SPLITS])
    ax.set_ylabel("steering accuracy (held-out)")
    ax.set_ylim(0, 1.13)
    ax.set_yticks([0, 0.167, 0.5, 1.0])
    ax.set_yticklabels(["0", ".167", ".50", "1.0"])
    ax.set_xlim(-0.62, len(SPLITS) - 0.38)
    chance_line(ax, label=False)
    ax.text(len(SPLITS) - 0.42, CHANCE + 0.02, "six-way chance .167", fontsize=6.2,
            va="bottom", ha="right")
    ax.legend(loc="lower left", bbox_to_anchor=(-0.005, 1.005), ncol=3)
    fig.subplots_adjust(left=0.075, right=0.995, top=0.775, bottom=0.175)
    return save(fig, "fig4_sufficiency", png_dir)


# =========================================================== F5: dose-response
BS = [1, 2, 5, 10, 100, 10000]


def fig5(png_dir=None):
    """F5 -- bias dose-response. SINGLE COLUMN (3.15in). Ledger section 11.

    Panel (a) is the accuracy sweep (ledger section 11). Panel (b) is the
    attention-rerouting cost over the same bias axis (ledger section 18,
    discovery/measure_rerouting.py). Panel (b) draws itself only if BOTH the run
    file and a ledger section reporting it exist; otherwise the vertical space
    stays reserved and empty, and a warning goes to stderr. The canvas height is
    the same either way, so the LaTeX layout does not move.
    """
    series = []
    for label, tag, colour, marker, ls in (
            ("Ultravox", "ultravox", BLUE, "o", "-"),
            ("Qwen2-Audio", "qwen2_audio", VERM, "s", (0, (4, 1.5)))):
        ys = []
        for B in BS:
            v = steer_acc(f"dose/{tag}_top100_B{B}")
            if v is None:
                warn(f"F5: {label} B={B} was not run -- the curve is BROKEN across "
                     f"it, not interpolated")
            else:
                cite(v, 11, what=f"{label} B={B}")
            ys.append(v)
        series.append((label, ys, colour, marker, ls))

    fig = plt.figure(figsize=(WIDTH_COL, 3.75))
    gs = fig.add_gridspec(2, 1, height_ratios=[1.0, 0.95], hspace=0.36,
                          left=0.185, right=0.965, top=0.945, bottom=0.10)
    ax = fig.add_subplot(gs[0, 0])

    for label, ys, colour, marker, ls in series:
        # draw contiguous runs only: a missing cell breaks the line
        run_x, run_y = [], []
        for B, y in list(zip(BS, ys)) + [(None, None)]:
            if y is None:
                if len(run_x) > 1:
                    ax.plot(run_x, run_y, color=colour, ls=ls, marker=marker,
                            ms=3.4, mfc=colour, mec=colour, label=None, zorder=3)
                elif len(run_x) == 1:
                    ax.plot(run_x, run_y, color=colour, ls="none", marker=marker,
                            ms=3.4, mfc="white", mec=colour, zorder=3)
                run_x, run_y = [], []
                continue
            run_x.append(B)
            run_y.append(y)
        if len(run_x) > 1:
            ax.plot(run_x, run_y, color=colour, ls=ls, marker=marker, ms=3.4,
                    mfc=colour, mec=colour, zorder=3)
        elif len(run_x) == 1:
            ax.plot(run_x, run_y, color=colour, ls="none", marker=marker, ms=3.4,
                    mfc="white", mec=colour, zorder=3)
        ax.plot([], [], color=colour, ls=ls, marker=marker, ms=3.4, label=label)

    missing = [(lab, B) for lab, ys, *_ in series
               for B, y in zip(BS, ys) if y is None]
    for lab, B in missing:
        ax.axvline(B, color=DARK, lw=0.6, ls=(0, (1, 2)), zorder=1)
        ax.text(B * 0.86, 0.52, f"{lab} $B$={B:g} not run", fontsize=5.8,
                color=DARK, ha="center", va="center", rotation=90)

    ax.axvline(5, color=GREEN, lw=0.7, ls=(0, (4, 1.5)), zorder=1)
    ax.text(5.8, 1.17, "saturates at $B$=5", fontsize=6.2, color=GREEN,
            ha="left", va="top")
    _log_B_axis(ax)
    ax.set_ylim(0, 1.20)
    ax.set_yticks([0, 0.167, 0.5, 1.0])
    ax.set_yticklabels(["0", ".167", ".50", "1.0"])
    ax.set_ylabel("steering accuracy")
    ax.axhline(CHANCE, color="black", lw=0.8, ls=(0, (1, 1.6)), zorder=1)
    ax.text(0.85, CHANCE + 0.02, "chance .167", fontsize=6.0, va="bottom")
    ax.legend(loc="lower right", ncol=1)
    ax.set_title("(a) steering accuracy vs bias magnitude", fontsize=7.5, loc="left",
                 pad=3)

    _fig5_panel_b(fig.add_subplot(gs[1, 0]))
    return save(fig, "fig5_dose_response", png_dir)


def _log_B_axis(ax):
    ax.set_xscale("log")
    ax.set_xticks(BS)
    ax.set_xticklabels(["1", "2", "5", "10", "100", "$10^4$"])
    ax.minorticks_off()
    ax.set_xlim(0.8, 16000)


# panel (b): the cost of the same bias, on the same B axis. Ledger section 18.
REROUTE = os.path.join("rerouting", "ultravox", "rerouting_report.json")
REROUTE_SETS = [("top", "discovered top-100", BLUE, "o", "-"),
                ("matched_random", "layer-histogram-matched random", GREY, "D",
                 (0, (4, 1.5)))]


def _fig5_panel_b(ax):
    """Attention-rerouting cost vs B, or reserved empty space if it is not yet
    both measured AND in the ledger. Never drawn from a file the ledger does not
    report -- standing rule 1 applies to figures too."""
    rep = j(REROUTE) or {}
    sec18 = ledger_section(18)
    if not rep or "summary" not in rep or "DM" not in sec18:
        if not rep:
            warn("F5: panel (b) (attention-rerouting cost vs B) has no result file "
                 "yet -- vertical space is reserved and left empty")
        else:
            warn(f"F5: {REROUTE} exists but no ledger section reports it -- "
                 f"regenerate RESULTS_LEDGER.md; panel (b) left reserved")
        ax.set_axis_off()
        ax.add_patch(Rectangle((0.0, 0.0), 1.0, 1.0, transform=ax.transAxes,
                               facecolor="none", edgecolor=GREY, lw=0.6,
                               ls=(0, (3, 3))))
        ax.text(0.5, 0.5, "(b) reserved: attention-rerouting cost vs $B$\n"
                          "(discovery/measure_rerouting.py -- measurement pending)",
                transform=ax.transAxes, ha="center", va="center", fontsize=6.4,
                color=GREY)
        return

    summ = rep["summary"]
    grid = [float(b) for b in rep.get("B_sweep", [])]
    for key, label, colour, marker, ls in REROUTE_SETS:
        xs, ys = [], []
        for B in grid:
            cell = summ.get(f"{key}|B={B:g}")
            if cell is None:
                warn(f"F5b: no rerouting cell for {key} at B={B:g} -- point omitted")
                continue
            xs.append(B)
            # Pr(DM <= -0.10): SKOP's own threshold, the statistic main.tex leads
            # with, and on the same 0-1 scale as panel (a) so the two read together
            ys.append(cite(cell["pr_dm_le_-0.10"], 18,
                           what=f"{key} Pr(DM<=-.10) at B={B:g}"))
        ax.plot(xs, ys, color=colour, ls=ls, marker=marker, ms=3.4, mfc=colour,
                mec=colour, label=label, zorder=3)
    for B in BS:                      # the cost sweep's grid is not the accuracy
        if B not in grid:             # sweep's grid; say so rather than imply it
            warn(f"F5b: B={B:g} is not on the rerouting sweep's grid "
                 f"({', '.join(f'{g:g}' for g in grid)}) -- no point drawn there")
    # focus-set sizes (9.5 vs 5.2 keys) are in ledger section 18 and in the
    # caption of main.tex's Table "reroute"; not repeated on the panel.

    _log_B_axis(ax)
    ax.axvline(5, color=GREEN, lw=0.7, ls=(0, (4, 1.5)), zorder=1)
    ax.set_ylim(0, 1.08)
    ax.set_yticks([0, 0.25, 0.5, 0.75, 1.0])
    ax.set_yticklabels(["0", ".25", ".50", ".75", "1.0"])
    ax.set_xlabel("attention-logit bias $B$ (log scale)")
    ax.set_ylabel("Pr($\\Delta M \\leq -.10$)")
    ax.legend(loc="lower right", ncol=1)
    ax.set_title("(b) cost: triples losing $\\geq$.10 of their focus-set mass",
                 fontsize=7.5, loc="left", pad=3)


# ========================================================= F6: layout transfer
LAYOUT = [
    dict(long="cross-layout heads\n(horizontal $\\rightarrow$ vertical)",
         short="cross-layout", run="vlm/e2/qwen2_vl_vertical_crosslayout100",
         colour=BLUE, hatch="", marker="o", ls="-"),
    dict(long="vertical-native heads\n(vertical $\\rightarrow$ vertical)",
         short="vertical-native", run="vlm/e2/qwen2_vl_vertical_vtop100",
         colour=GREEN, hatch="///", marker="s", ls=(0, (4, 1.5))),
    dict(long="random heads\n($\\rightarrow$ vertical)",
         short="random control", run="vlm/e2/qwen2_vl_vertical_vrandom100",
         colour=GREY, hatch="...", marker="^", ls=(0, (1, 1.6))),
]


def fig6(png_dir=None):
    """F6 -- layout transfer. FULL WIDTH (figure*, 6.3in). Ledger sections 6 and 13.

    Top: the three steering cells. Bottom: the per-target accuracy profile of
    the same three cells, which is where the positional confound is answered --
    the random control dips in the middle and recovers, ours do not.
    """
    accs, profiles = [], []
    for cell in LAYOUT:
        run = cell["run"]
        a = cite(steer_acc(run), 6, what=cell["short"])
        if a is None:
            warn(f"F6: {run} missing -- bar omitted")
        accs.append(a)
        p = per_target(run)
        if p is None:
            warn(f"F6: no per-target accuracy for {run} -- profile omitted")
        else:
            for v in p:
                cite(v, 13, nd=2, what=f"{cell['short']} per-target")
        profiles.append(p)

    ov = overlap("vlm/qwen2_vl/scores.npz", "vlm/qwen2_vl_vertical/scores.npz")
    if ov is None or f"{ov}/100" not in ledger_section(6):
        warn("F6: vertical-vs-horizontal head-set overlap unavailable or not in "
             "ledger section 6 -- annotation omitted")
        ov = None

    fig = plt.figure(figsize=(WIDTH_FULL, 3.5))
    gs = fig.add_gridspec(2, 1, height_ratios=[0.78, 1.0], hspace=0.46,
                          left=0.215, right=0.985, top=0.94, bottom=0.11)

    ax = fig.add_subplot(gs[0, 0])
    y = np.arange(len(LAYOUT))[::-1].astype(float)
    for yi, cell, a in zip(y, LAYOUT, accs):
        ax.barh([yi], [a or 0], 0.55, color=cell["colour"], edgecolor=DARK,
                hatch=cell["hatch"], zorder=3)
        if a is not None:
            ax.text(a + 0.008, yi, f"{a:.3f}".lstrip("0"), va="center", ha="left",
                    fontsize=6.6, color=DARK)
    ax.set_yticks(y)
    ax.set_yticklabels([c["long"] for c in LAYOUT], fontsize=7)
    ax.set_xlim(0, 0.70)
    ax.set_xticks([0, 0.2, 0.4, 0.6])
    ax.set_xticklabels(["0", ".20", ".40", ".60"])
    ax.set_ylim(-0.75, len(LAYOUT) - 0.25)
    ax.set_xlabel("steering accuracy (Qwen2-VL, vertical strips, $K$=100)", labelpad=1)
    ax.axvline(CHANCE, color="black", lw=0.8, ls=(0, (1, 1.6)), zorder=1)
    ax.text(CHANCE + 0.006, len(LAYOUT) - 0.30, "chance .167", fontsize=6.2,
            va="top")
    ax.spines["left"].set_visible(False)
    ax.tick_params(axis="y", length=0)
    if ov is not None:
        ax.text(0.695, -0.70, f"head-set overlap, horizontal vs vertical: {ov}/100",
                fontsize=6.4, ha="right", va="bottom", color=DARK)
    ax.set_title("(a) the mechanism transfers across token geometry", fontsize=7.5,
                 loc="left", pad=3)

    ax2 = fig.add_subplot(gs[1, 0])
    xs = np.arange(1, 7)
    for cell, p in zip(LAYOUT, profiles):
        if p is None:
            continue
        ax2.plot(xs, p, color=cell["colour"], ls=cell["ls"], marker=cell["marker"],
                 ms=3.6, mfc=cell["colour"], mec=cell["colour"], zorder=3,
                 label=f"{cell['short']} (spread {max(p) - min(p):.2f})".replace("0.", "."))
    ax2.set_xticks(xs)
    ax2.set_xlabel("target panel index (vertical strip, top to bottom)", labelpad=1)
    ax2.set_ylabel("per-target accuracy")
    ax2.set_ylim(0, 0.82)
    ax2.set_yticks([0, 0.167, 0.4, 0.6, 0.8])
    ax2.set_yticklabels(["0", ".167", ".40", ".60", ".80"])
    ax2.set_xlim(0.7, 6.3)
    ax2.axhline(CHANCE, color="black", lw=0.8, ls=(0, (1, 1.6)), zorder=1)
    ax2.text(6.28, CHANCE + 0.012, "chance .167", fontsize=6.0, ha="right", va="bottom")
    ax2.legend(loc="lower left", ncol=3, bbox_to_anchor=(-0.005, -0.03),
               fontsize=6.4)
    ax2.set_title("(b) the positional check: flat for the discovered heads, "
                  "a mid-strip dip for the random control", fontsize=7.5, loc="left",
                  pad=3)
    return save(fig, "fig6_layout_transfer", png_dir)


# ===================================================================== driver
FIGS = {"f1": fig1, "f2": fig2, "f3": fig3, "f4": fig4, "f5": fig5, "f6": fig6}


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("which", nargs="*", default=[], help="f1 ... f6 (default: all)")
    ap.add_argument("--png-dir", default=None,
                    help="also write PNG copies here, for on-screen QA only")
    a = ap.parse_args()
    which = [w.lower() for w in a.which] or list(FIGS)
    unknown = [w for w in which if w not in FIGS]
    if unknown:
        ap.error(f"unknown figure(s): {', '.join(unknown)}")
    if not os.path.exists(LEDGER):
        sys.exit(f"missing ledger: {LEDGER}")
    for w in which:
        FIGS[w](a.png_dir)
    if WARNINGS:
        print(f"\n{len(WARNINGS)} omission(s)/warning(s) above -- each is an element "
              f"left out for lack of a citable number, not a defect to paper over.",
              file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(main())
