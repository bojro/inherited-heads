#!/bin/bash
cd "$(dirname "$0")/.."
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
run() { arm=$1; p=$2; d=$(echo $arm | tr '-' '_')_p$p
  echo "=== $arm prompt $p -> $d  $(date +%H:%M) ==="
  .venv/bin/python discovery/run_discovery.py --model $arm --n-items 50 --prompt $p --strips testbed/strips --out discovery/results/$d 2>&1 | grep -v "Warning\|warn\|Loading weights\|Fetching\|deprecated\|WeightNorm\|meta device" | grep -E '"separation_ratio"|"real_vs_shuffled_top100"|Error|error'; }
run salmonn 0; run salmonn 1; run salmonn 2; run qwen2-audio 1; run ultravox 1
echo ALL_RERUNS_DONE
