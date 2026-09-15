#!/bin/bash
# R2 — SPLIT-HALF RELIABILITY of the head ranking. Closes two audit gaps at once:
#
#  (a) THE CEILING. The inheritance claim rests on 71-74/100 overlap between a
#      model's modality gaze heads and its backbone's TEXT gaze heads. That
#      number is meaningless without knowing how much a ranking agrees with
#      ITSELF. If the same task on disjoint items only reproduces 60/100, then
#      cross-modal 71/100 is at or above the reliability ceiling and something is
#      wrong; if split-half is ~90, then 71 is a strong signal well inside it.
#
#  (b) RANKING STABILITY. PREREGISTRATION 17aa recorded an 81/100 top-100 shift
#      from a numerically EQUIVALENT change in how the forward pass was chunked,
#      on a single item. We criticise an unstable ranking; we have never
#      measured our own. This measures it on the axis that matters -- items.
#
# Same protocol, same prompt, same model; only the items differ (100-124 vs
# 125-149, both held out). Overlap is computed at several K because the
# inheritance claim is stated at K=100.
cd "$(dirname "$0")/.."
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
echo "=== R2 starting $(date +%H:%M) ==="

run_half () {   # $1=tag  $2=runner  $3..=args
  local tag=$1 runner=$2; shift 2
  for half in A B; do
    local off=100; [ $half = B ] && off=125
    local d=discovery/results/reliability/${tag}_$half
    [ -f $d/report.json ] || { echo "=== R2 $tag half$half (items $off-$((off+24))) $(date +%H:%M) ==="
      .venv/bin/python discovery/$runner "$@" --n-items 25 --item-offset $off --out $d 2>&1 \
        | grep -E 'items$|separation_ratio|real_vs_shuffled|Error|Traceback'; }
  done
}

run_half ultravox_audio  run_discovery.py      --model ultravox --strips testbed/strips
run_half llama31_text    run_text_discovery.py --model meta-llama/Llama-3.1-8B-Instruct --text testbed/text_strips
run_half qwen2vl_image   run_vlm_discovery.py  --model qwen2-vl --comics testbed/comics

echo "=== R2 overlap analysis $(date +%H:%M) ==="
.venv/bin/python - <<'PYEOF'
import numpy as np, os
from math import comb
def top(x, k): return set(np.argsort(x.ravel())[::-1][:k].tolist())
print(f"{'ranking':22s} {'K=25':>8s} {'K=50':>8s} {'K=100':>8s} {'K=200':>8s}   (self-agreement across disjoint items)")
for tag in ("ultravox_audio", "llama31_text", "qwen2vl_image"):
    pa = f"discovery/results/reliability/{tag}_A/scores.npz"
    pb = f"discovery/results/reliability/{tag}_B/scores.npz"
    if not (os.path.exists(pa) and os.path.exists(pb)):
        print(f"{tag:22s} MISSING"); continue
    a, b = np.load(pa)["scores"], np.load(pb)["scores"]
    row = [len(top(a, k) & top(b, k)) for k in (25, 50, 100, 200)]
    print(f"{tag:22s} " + " ".join(f"{v:8d}" for v in row))
print("\nCompare with the cross-modal inheritance overlaps at K=100:")
print("  Ultravox audio vs Llama-3.1 text 71 | SALMONN 74 | Qwen2-VL vs Qwen2-7B 73")
print("A cross-modal overlap ABOVE the within-task split-half number would be")
print("incoherent and would mean the inheritance figure is measuring something else.")
PYEOF
echo R2_DONE
