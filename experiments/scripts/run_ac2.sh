#!/bin/bash
# AC2 (retry) - does MEASUREMENT POINT explain the seed paper's Bunny null?
# Their score reads the FINAL PROMPT TOKEN. We already show that point
# understates: Qwen2-VL separates 1.240 at prefill vs 2.631 during generation.
# If Bunny's structure likewise only appears during generation, head selection
# at prefill picks the wrong heads and their 8.3% is a measurement artefact
# rather than a fact about Bunny.
# First attempt died: BunnyAdapter had no stream_generate. Added (two-phase
# prefill; a full [H,n,n] tensor does not fit alongside the fp16 model).
# Waits for V21 to release the GPU.
cd "$(dirname "$0")/.."
while pgrep -f "bash \./run_v21.sh" > /dev/null; do sleep 60; done
echo "=== V21 released the GPU; AC2 starting $(date +%H:%M) ==="

d=discovery/results/vlm/gencond/bunny_grid
[ -f $d/report.json ] || .venv-bunny/bin/python discovery/run_vlm_gencond.py \
  --model bunny-grid --comics testbed/comics --n-items 50 --item-offset 100 \
  --out $d --full-precision 2>&1 \
  | grep -E 'items,|separation_ratio|real_vs_shuffled|mean_panels|Error|Traceback'

if [ -f $d/report.json ]; then
  echo "=== AC2: does the ranking change between prefill and generation? ==="
  .venv/bin/python - <<'PYEOF'
import json, numpy as np
pre = json.load(open('discovery/results/vlm/bunny_grid/report.json'))
gen = json.load(open('discovery/results/vlm/gencond/bunny_grid/report.json'))
print(f"  separation       prefill {pre['separation_ratio']:.3f} -> generation {gen['separation_ratio']:.3f}")
print(f"  real vs shuffled prefill {pre['real_vs_shuffled_top100']:.3f} -> generation {gen['real_vs_shuffled_top100']:.3f}")
a = np.load('discovery/results/vlm/bunny_grid/scores.npz')['scores'].ravel()
b = np.load('discovery/results/vlm/gencond/bunny_grid/scores.npz')['scores'].ravel()
top = lambda x, k=100: set(np.argsort(x)[::-1][:k].tolist())
print(f"  top-100 overlap between the two measurement points: {len(top(a)&top(b))}/100")
print("  LOW overlap => prefill selection picks different heads than generation")
print("  selection, so measurement point is a live explanation for their 8.3%.")
print("  HIGH overlap => that explanation is ruled out.")
PYEOF
fi
echo AC2_DONE
