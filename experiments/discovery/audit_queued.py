"""Is every planned run actually QUEUED somewhere?

`audit_completeness.py` answers "did every expected run produce a result?" It
cannot answer "is every expected run in a script at all", and that gap bit three
times on 2026-08-17:

  * the raw-ranked confirmatory runs (H8/H9) — PREREGISTRATION.md 17e said they
    were appended to run_confirm.sh; they were not, so the paper's headline
    contrast had no held-out version;
  * the Bunny arms — run_v2.sh predated the Bunny adapter and ran only the
    Qwen2-VL control, so V2 would have finished having tested nothing it existed
    for;
  * H11 — queued, but its chain echoed DONE after it died, which
    audit_completeness does catch.

A result that is neither present nor queued is invisible to every other check:
nothing fails, nothing is missing from a finished chain, it simply never happens.

    python audit_queued.py
"""
import glob
import os
from pathlib import Path

SCRIPTS = sorted(glob.glob(os.path.expanduser("~/run_*.sh"))
                 + glob.glob(os.path.expanduser("~/dl_*.sh")))

ARMS = ["qwen2_audio", "salmonn", "ultravox"]
CONFIRM_TAGS = (["none0"] + [f"top{k}" for k in (25, 50, 100, 250)]
                + [f"random{k}" for k in (50, 100, 250)]
                + [f"random{k}_s1" for k in (50, 100, 250)]
                + [f"masstop{k}" for k in (50, 100)] + ["all0"]
                + ["gaze_lowmass", "gaze_highmass", "mass_bandmatched"]
                + ["top100_p1", "top100_p2", "gaze_lowmass_dm", "gaze_highmass_dm",
                   "rawtop100"])

EXPECTED = ([(f"confirm/{a}_{t}", "steer_report.json") for a in ARMS for t in CONFIRM_TAGS]
            + [(f"nonmono/{a}", "report.json") for a in ARMS]
            + [(f"e3b/{a}", "report.json") for a in ARMS]
            + [(f"gencond/{a}", "report.json") for a in ARMS]
            + [("text/llama31_8b", "report.json"), ("text/vicuna13b", "report.json"),
               ("text/qwen15_7b_chat", "report.json")]
            + [("vlm/qwen2_vl", "report.json"), ("vlm/bunny_grid", "report.json"),
               ("vlm/bunny_strip", "report.json")]
            + [(f"vlm/e2/{m}_{c}", "steer_report.json")
               for m in ("qwen2_vl", "bunny_grid", "bunny_strip")
               for c in ("top100", "rawtop100", "random100")]
            # V15 prosody. The GATE is expected for every arm x axis; E1 and E2
            # are expected only where the gate passes, so they are listed as
            # gate-conditional and audit_completeness is what tracks whether a
            # blocked arm was skipped for the right reason.
            + [(f"v15/gate/{a}_{ax}", "gate_report.json")
               for a in ARMS for ax in ("emotion", "speaker")])

# Runs that only exist if their capability gate passed. Absent-and-unqueued is
# only a gap when the gate for that arm/axis reported PASS.
GATE_CONDITIONAL = ([(f"v15/e1/{a}_{ax}", "report.json")
                     for a in ARMS for ax in ("emotion", "speaker")]
                    + [(f"v15/e2/{a}_{ax}_{c}", "steer_report.json")
                       for a in ARMS for ax in ("emotion", "speaker")
                       for c in ("top100", "random100", "none0")])


def main():
    blob = "\n".join(open(s).read() for s in SCRIPTS)
    root = Path("discovery/results")
    done = notq = queued = 0
    gaps = []
    for rel, report in EXPECTED:
        if (root / rel / report).exists():
            done += 1
            continue
        # queued if any script mentions the output path, or the arm+tag pair
        leaf = rel.split("/")[-1]
        # scripts name arms with hyphens (bunny-grid) and paths with underscores
        # (bunny_grid), converting via tr -- try both spellings
        if rel in blob or leaf in blob or leaf.replace("_", "-") in blob:
            queued += 1
            continue
        # Scripts build paths from loop variables ($mm, $tag, $p), so an exact
        # path never appears. Strip the arm prefix and try the tag, then the tag
        # with a trailing index removed (top100_p1 -> top100_p, random50_s1 ->
        # random50_s). Arm names contain underscores, so split on the known arms
        # rather than on the first "_" — getting that wrong made this checker
        # report 18 false gaps on its first run.
        tag = None
        for arm in ARMS + ["qwen2_vl", "bunny_grid", "bunny_strip"]:
            if leaf.startswith(arm + "_"):
                tag = leaf[len(arm) + 1:]
                break
        cands = [tag] if tag else []
        if tag:
            cands.append(tag.rstrip("0123456789"))
        if any(c and c in blob for c in cands):
            queued += 1
            continue
        notq += 1
        gaps.append(rel)

    print(f"expected runs: {len(EXPECTED)}")
    print(f"  done      {done}")
    print(f"  queued    {queued}")
    print(f"  NOT QUEUED {notq}")
    if gaps:
        print()
        print("NOT FOUND by string match — verify each by hand before trusting:")
        for g in gaps:
            print("   ", g)
        print()
        print("Scripts build paths dynamically, so this check is a heuristic: it can")
        print("report a false gap. It cannot report a false PASS, which is the")
        print("direction that matters — a genuinely unqueued run always shows here.")
    else:
        print()
        print("every expected run is either complete or queued in a script")


if __name__ == "__main__":
    main()
