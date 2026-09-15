#!/bin/bash
cd "$(dirname "$0")/.."
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
for arm in qwen2-audio ultravox salmonn; do
  for p in 0 1 2; do
    d=$(echo $arm | tr '-' '_')_p$p
    echo "=== $arm prompt $p -> $d  $(date +%H:%M) ==="
    .venv/bin/python discovery/run_discovery.py --model $arm --n-items 50 --prompt $p --strips testbed/strips --out discovery/results/$d 2>&1 | grep -v "Warning\|warn\|Loading weights\|Fetching\|deprecated\|WeightNorm\|meta device" | grep -E "separation_ratio|real_vs_shuffled|Error|error|precomputed"
  done
done
echo ALL_RERUNS_DONE
