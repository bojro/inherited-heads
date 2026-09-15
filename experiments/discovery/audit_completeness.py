"""Did every expected run actually produce a result?

The failure mode this exists for: a step dies, its chain echoes DONE anyway, and
the missing cell is invisible. That happened to H11 on 2026-08-17 (CUDA "device
not ready"; the inheritance chain printed INHERIT_DONE and released the
confirmatory suite while H11 had produced nothing). A marker is not evidence.

Distinguishes four states, because "hasn't run yet" and "failed" are not the
same thing and only one of them needs action:
  OK          report file present
  RUNNING     no report, but answers.jsonl was written in the last 20 min
  PENDING     no report, and the chain that owns it has not finished yet
  MISSING     no report, and its chain HAS reported DONE -> a real gap

    python audit_completeness.py
"""
import json, time
from pathlib import Path

ARMS = ["qwen2_audio", "salmonn", "ultravox"]
# the frozen list from PREREGISTRATION.md plus the two disclosed additions
CONFIRM_TAGS = (["none0"] + [f"top{k}" for k in (25, 50, 100, 250)]
                + [f"random{k}" for k in (50, 100, 250)]
                + [f"random{k}_s1" for k in (50, 100, 250)]
                + [f"masstop{k}" for k in (50, 100)] + ["all0"]
                + ["gaze_lowmass", "gaze_highmass", "mass_bandmatched"]
                + ["top100_p1", "top100_p2"]
                + ["gaze_lowmass_dm", "gaze_highmass_dm"])
FRESH = 20 * 60

# which chain owns each result group, and the marker that means it finished.
# A group whose chain has not printed its marker cannot have "missing" cells yet.
OWNER = {
    "confirm":  ("confirm.log", "CONFIRM_DONE"),
    "nonmono":  ("riskfirst.log", "RISK_DONE"),
    "e3":       ("riskfirst.log", "RISK_DONE"),
    "dual":     ("e1_raw.log", "E1_RAW_DONE"),
    "e3b":      ("followups.log", "FOLLOWUPS_DONE"),
    "gencond":  ("overnight2.log", "PHASE2_DONE"),
    "text/llama31_8b":     ("text.log", "TEXT_DONE"),
    "text/vicuna13b":      ("followups.log", "FOLLOWUPS_DONE"),
    "text/qwen15_7b_chat": ("followups.log", "FOLLOWUPS_DONE"),
    "vlm/qwen2_vl":        ("v2.log", "V2_DONE"),
}


def chain_finished(key):
    """True if the chain owning `key` has printed its done marker."""
    import os
    ent = OWNER.get(key)
    if ent is None:
        return True
    log, marker = ent
    path = os.path.expanduser("~/" + log)
    try:
        with open(path) as f:
            return marker in f.read()
    except OSError:
        return False


def state(d: Path, report_name: str, owner_key: str):
    if (d / report_name).exists():
        return "OK"
    for probe in ("answers.jsonl", "scores.npz"):
        f = d / probe
        if f.exists() and time.time() - f.stat().st_mtime < FRESH:
            return "RUNNING"
    return "MISSING" if chain_finished(owner_key) else "PENDING"


def main():
    root = Path("discovery/results")
    rows, missing = [], []

    for arm in ARMS:
        for tag in CONFIRM_TAGS:
            d = root / "confirm" / f"{arm}_{tag}"
            s = state(d, "steer_report.json", "confirm")
            rows.append(("confirm", f"{arm}_{tag}", s))
            if s == "MISSING":
                missing.append(f"confirm/{arm}_{tag}")

    for arm in ARMS:
        for sub, rep in (("nonmono", "report.json"), ("e3", "report.json"),
                         ("e3b", "report.json"), ("gencond", "report.json"),
                         ("dual", "report.json")):
            d = root / sub / arm
            s = state(d, rep, sub)
            rows.append((sub, arm, s))
            if s == "MISSING":
                missing.append(f"{sub}/{arm}")

    for name, d, rep in (("text/llama31_8b", root / "text/llama31_8b", "report.json"),
                         ("text/vicuna13b", root / "text/vicuna13b", "report.json"),
                         ("text/qwen15_7b_chat", root / "text/qwen15_7b_chat", "report.json"),
                         ("vlm/qwen2_vl", root / "vlm/qwen2_vl", "report.json")):
        s = state(d, rep, name)
        rows.append(("single", name, s))
        if s == "MISSING":
            missing.append(name)

    by = {}
    for group, name, s in rows:
        by.setdefault(group, []).append((name, s))
    for group in by:
        c = {k: sum(1 for _, s in by[group] if s == k)
             for k in ("OK", "RUNNING", "PENDING", "MISSING")}
        extra = "  ".join(f"{v} {k.lower()}" for k, v in c.items()
                          if v and k != "OK")
        print(f"{group:10} {c['OK']:3}/{len(by[group])} OK   {extra}")
    print()
    if missing:
        print(f"REAL GAPS ({len(missing)}) — their chain finished but they produced nothing:")
        for m in missing:
            print("   ", m)
        print()
        print("Re-run before interpreting anything: a marker in a chain log is not")
        print("evidence that its steps succeeded.")
    else:
        print("no gaps: every expected run is complete, in flight, or still queued")


if __name__ == "__main__":
    main()
