#!/bin/bash
cd "$(dirname "$0")/.."
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
until grep -q E2_SETS_DONE ~/e2_sets.log; do sleep 60; done
echo "=== sets chain done; re-running E1 with BOTH score definitions $(date +%H:%M) ==="
for m in qwen2-audio ultravox salmonn; do
  mm=$(echo $m | tr '-' '_')
  echo "=== $m E1 dual-score $(date +%H:%M) ==="
  .venv/bin/python discovery/run_discovery.py --model $m --n-items 50 --prompt 0 --strips testbed/strips --out discovery/results/dual/$mm 2>&1 | grep -v "Warning\|warn\|Loading weights\|Fetching\|deprecated\|WeightNorm\|meta device" | grep -E '"separation_ratio"|"raw_separation_ratio"|"real_vs_shuffled_top100"|"raw_real_vs_shuffled|Error|Traceback'
done
echo E1_RAW_DONE
