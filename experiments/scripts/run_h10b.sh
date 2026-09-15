#!/bin/bash
# H10b -- extend the text-to-audio transfer cell to the other two arms.
#
# The strongest claim in section 4 is that heads chosen WITHOUT EVER LOOKING AT
# AUDIO steer audio. It was measured on Ultravox alone because run_inherit.sh
# only ran that arm, while the text discovery for both other backbones has been
# on disk since. A simulated adversarial read named this as the section's worst
# gap: n=1 where n=3 was free.
#
# Identical settings to the Ultravox cell so the three are comparable: K=100,
# B default (10^4), 20 items, NO item-offset (items 0-19), same strips.
cd "$(dirname "$0")/.."
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
echo "=== H10b starting $(date +%H:%M:%S) ==="

run () {  # run <model> <text-scores-dir> <out-tag>
  local o=discovery/results/e2/$3_texttop100
  if [ -f $o/steer_report.json ]; then echo "  skip $3 (done)"; return; fi
  echo "=== $3 steered by TEXT-derived heads, start $(date +%H:%M:%S) ==="
  .venv/bin/python discovery/run_steer.py --model $1 --strips testbed/strips \
    --scores discovery/results/text/$2/scores.npz --condition top --k 100 \
    --n-items 20 --out $o 2>&1 \
    | grep -E 'hook check|items:|"accuracy"|"unattributed"|Error|Traceback'
  echo "=== $3 end $(date +%H:%M:%S) ==="
}

run qwen2-audio qwen15_7b_chat qwen2_audio
run salmonn     vicuna13b      salmonn

echo "=== H10b summary $(date +%H:%M:%S) ==="
.venv/bin/python - <<'PYEOF'
import json, os
R = "discovery/results"
def a(p):
    f = os.path.join(R, p, "steer_report.json")
    return json.load(open(f))["accuracy"] if os.path.exists(f) else None
print(f"  {'arm':14s} {'text-derived':>13s} {'ours (audio)':>13s} {'random':>8s} {'theirs':>8s}")
for arm in ("qwen2_audio", "salmonn", "ultravox"):
    row = [a(f"e2/{arm}_texttop100"), a(f"e2/{arm}_top100"),
           a(f"e2/{arm}_random100"), a(f"e2/{arm}_rawtop100")]
    print(f"  {arm:14s} " + " ".join("     --   " if v is None else f"{v:13.3f}" for v in row[:1])
          + " ".join("     --  " if v is None else f"{v:13.3f}" for v in row[1:2])
          + " ".join("    --  " if v is None else f"{v:8.3f}" for v in row[2:]))
print("\n  Heads chosen with no sight of audio, used on audio. Compare each arm to")
print("  its OWN random control, not across arms.")
PYEOF
echo H10B_DONE
