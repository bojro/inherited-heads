#!/bin/bash
cd "$(dirname "$0")/.."
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
S=discovery/results/qwen2_audio_v2/scores.npz
run() { cond=$1; k=$2; seed=${3:-0}; d=discovery/results/e2/qwen2_audio_${cond}${k}$( [ "$seed" != "0" ] && echo _s$seed )
  echo "=== $cond K=$k seed=$seed $(date +%H:%M) ==="
  .venv/bin/python discovery/run_steer.py --model qwen2-audio --scores $S --condition $cond --k $k --seed $seed --n-items 20 --strips testbed/strips --out $d 2>&1 | grep -v "Warning\|warn\|Loading weights\|Fetching\|deprecated\|generation flags\|Setting" | grep -E '"accuracy"|"unattributed_rate"|Error|Traceback'; }
run none 0
run random 100 0
run random 100 1
run random 100 2
run all 0
for k in 5 10 25 50 250 500; do run top $k; done
echo E2_QWEN_DONE
