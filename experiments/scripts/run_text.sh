#!/bin/bash
cd "$(dirname "$0")/.."
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
until grep -q E1_RAW_DONE ~/e1_raw.log 2>/dev/null; do sleep 60; done
echo "=== dual-score done; TEXT inheritance test $(date +%H:%M) ==="
.venv/bin/python discovery/run_text_discovery.py --model meta-llama/Llama-3.1-8B-Instruct --text testbed/text_strips --n-items 50 --item-offset 100 --out discovery/results/text/llama31_8b 2>&1 | grep -v "Warning\|warn\|Loading weights\|Fetching\|deprecated\|meta device" | grep -E 'items$|separation_ratio|real_vs_shuffled|n_heads_gt|Error|Traceback'
echo TEXT_DONE
