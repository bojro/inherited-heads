#!/usr/bin/env python3
"""Check that every number in the paper traces to the results ledger.

Every figure in the paper must appear in experiments/RESULTS_LEDGER.md, which is
generated from the run outputs by experiments/discovery/make_ledger.py. This script
flags any numeric token in paper/tex/main.tex that does not appear in the ledger
(or, for protocol constants such as item ranges, in the pre-registration). Not
every flag is a defect: derived quantities, structural numbers such as K=100 and
years, and cited results from other papers are legitimate, and each is listed in
ALLOW below with its reason.

  python paper/check_numbers.py            # report
  python paper/check_numbers.py --strict   # exit 1 if any unexplained number appears
"""
import os, re, sys

HERE = os.path.dirname(os.path.abspath(__file__))
TEXES = [os.path.join(HERE, "tex", "main.tex")]
LEDGER = os.path.join(HERE, "..", "experiments", "RESULTS_LEDGER.md")
# Secondary source, for protocol constants only (item ranges, corpus sizes, split
# points, bootstrap resamples). Results may never come from here.
PREREG = os.path.join(HERE, "..", "experiments", "PREREGISTRATION.md")

# number -> why it is legitimately absent from the ledger
ALLOW = {
    "0.012": "derived: 0.662 (their published) - 0.650 (ours), stated as a gap",
    "4.0": "derived: 0.990 - 0.950, the Ultravox shortfall stated in accuracy POINTS (ledger 19)",
    "13.2": "derived: 0.907 - 0.775, the SALMONN shortfall stated in accuracy POINTS (ledger 19)",
    "1.3": "rounded from 1.31x, ledger 20 'ours vs all' column, Qwen2-Audio",
    "3.2": "rounded from 3.21x, ledger 20 'ours vs all' column, SALMONN",
    "192": "external: the slice size in Bektursun arXiv:2605.00333, cited not measured",
    "4.8": "derived: 0.828 collateral / 0.173 intended at B=5, ledger 18, stated as a ratio",
    "33.5": "strip duration as stated in the v1 paper text; the experiments used 29.5 s strips (README, Errata)",
    "448": "corpus constant: comic panel size in px, testbed/build_comic_strips.py --size default",
    "150": "corpus constant: comic items built, testbed/build_comic_strips.py --n-items default",
    "0.020": "derived: 0.990 - 0.970, stated as a bound on the norm-weighting shift",
    "1":  "structural", "2": "structural", "3": "structural", "4": "structural",
    "5":  "structural", "6": "structural", "7": "structural", "8": "structural",
    "9":  "structural", "10": "structural", "12": "structural",
    "100": "K, the head-set size", "50": "item count", "20": "item count",
    "24": "layout: interleaved runs per panel", "42": "layout: token stride",
    "1.0": "the no-signal reference value for a real-vs-shuffled ratio",
    "2026": "year", "2025": "year", "2024": "year", "2020": "year",
    "0.95": "confidence level", "0.05": "significance level",
    # literature values used to calibrate our overlap range -- other people's
    # results, so they belong in citations, not in our ledger
    "51": "Li et al., backbone-to-VLM retrieval-head overlap (cited)",
    "78": "Merullo et al. 2310.08744, cross-task overlap for one circuit (cited)",
    # PAHQ's published measurement of 4-bit vs 16-bit circuit-discovery quality,
    # quoted in Limitations. Another paper's result, so it is not in our ledger.
    "0.61": "PAHQ 2510.23264 Table 5, 4-bit AUC-ROC (cited)",
    # prose renders these ledger §18 decimals as percentages for readability
    "76": "ledger §18: Pr(DM<=-.10)=0.758 for our heads at B=5, as a percentage",
    "37": "ledger §18: gain-source share from prompt text at B=1e4, 0.368",
    "56": "ledger §18: gain-source share from the sink at B=1e4, 0.563",
    "82": "ledger §18: gain-source share from other segments at B=2, 0.821",
    "31": "SKOP's own Pr(DM<=-.10)=31% at their strongest setting (cited)",
    "0.96": "PAHQ 2510.23264 Table 5, 16-bit AUC-ROC (cited)",
    "120": "trial count: 20 strips x 6 targets",
    "300": "trial count: 50 strips x 6 targets",
    "40": "architecture: SALMONN's 40 layers x 40 heads",
    "83.1": "the seed paper's published steering accuracy (Gandikota & Bau 2026, cited)",
}

def norm(x):
    x = x.replace(",", "")
    return (x.lstrip("0") or "0") if "." in x else x

def main():
    led = open(LEDGER, encoding="utf-8").read()
    pre = open(PREREG, encoding="utf-8").read()
    ledger_nums = {norm(m) for m in re.findall(r"\d[\d,]*\.?\d*", led)}
    ledger_nums |= {norm(m) for m in re.findall(r"\d[\d,]*\.?\d*", pre)}
    allow = {norm(k) for k in ALLOW}

    # A percentage is the same number as its ledger decimal: the paper writes
    # "85.0%" where the ledger says 0.850, and forcing decimals into prose reads
    # badly. But this only applies when the token is ACTUALLY WRITTEN as a
    # percentage -- an earlier version accepted any token whose value/100 rounded
    # to something in the ledger, which let 0.267 match .00 and silently turned
    # the guard green on four withdrawn figures. Require a following % sign, and
    # require an exact 3- or 4-decimal match, never a 2-decimal rounding.
    def pct_ok(tok, is_pct):
        if not is_pct:
            return False
        try:
            v = float(tok.replace(",", ""))
        except ValueError:
            return False
        if not (0 < v <= 100):
            return False
        return bool({norm(f"{v / 100:.{d}f}") for d in (3, 4)} & ledger_nums)

    def ok(tok, is_pct):
        return (norm(tok) in ledger_nums or norm(tok) in allow
                or pct_ok(tok, is_pct))

    bad = {}
    for path in TEXES:
        tex = open(path, encoding="utf-8").read()
        # strip comments; then undo LaTeX digit grouping so 10{,}000 reads as one
        # number rather than as "10" followed by "000"
        tex = re.sub(r"(?<!\\)%.*", "", tex)
        tex = tex.replace("{,}", ",")
        rel = os.path.relpath(path, HERE)
        for m in re.finditer(r"(?<![\w.\\])\d[\d,]*(?:\.\d+)?(?![\w])", tex):
            tok = m.group()
            after = tex[m.end():m.end() + 2]
            is_pct = after.startswith("%") or after.startswith("\\%")
            if ok(tok, is_pct):
                continue
            line = tex[:m.start()].count("\n") + 1
            bad.setdefault(tok, []).append(f"{rel}:{line}")

    if not bad:
        print(f"OK: every number in {len(TEXES)} source file(s) traces to the ledger "
              "(results) or the preregistration (protocol constants), or is an "
              "allowed exception.")
        return 0
    print("Numbers not found in RESULTS_LEDGER.md and not in the allow-list:\n")
    for tok, lines in sorted(bad.items()):
        print(f"  {tok:>12}   {', '.join(lines)}")
    print("\nEither correct them against the ledger, regenerate the ledger, or "
          "add them to ALLOW in this script with a reason.")
    return 1 if "--strict" in sys.argv else 0

if __name__ == "__main__":
    sys.exit(main())
