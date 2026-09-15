#!/bin/bash
# H11 final attempt. Vicuna-13B text discovery failed THREE times (run_inherit.sh
# 03:14, run_followups.sh 17:20 and 17:21) with "CUDA driver error: device not
# ready". That was diagnosed 2026-08-17 17:35 as NOT a race: run_text_discovery.py
# loaded every model with plain NF4 and fp16 embed+lm_head resident, which for a
# 13B is ~7.79GB on an 8188MiB card. bitsandbytes surfaces that allocation
# failure as a driver error, not OutOfMemoryError. The script now sizes the load
# from the config and offloads embed/lm_head for anything over the budget
# (Vicuna 7.79 -> 6.55GB resident; Llama-3.1-8B 6.03 and Qwen1.5-7B 6.13 stay on
# the old path, so no completed run changes). See PREREGISTRATION.md 17z.
# Still runs last, still settles, and this time the FULL traceback is kept.
cd "$(dirname "$0")/.."
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
until grep -q H14_DONE ~/h14.log 2>/dev/null; do sleep 300; done
d=discovery/results/text/vicuna13b
[ -f $d/report.json ] && { echo "H11 already done"; echo H11_DONE; exit 0; }
for attempt in 1 2; do
  echo "=== H11 Vicuna-13B text discovery, final attempt $attempt $(date +%H:%M) ==="
  sleep 120
  echo -n "  free VRAM before load: "; nvidia-smi --query-gpu=memory.free --format=csv,noheader
  .venv/bin/python discovery/run_text_discovery.py \
    --model ~/models/vicuna-13b-v1.1-safetensors \
    --text testbed/text_strips --n-items 50 --item-offset 100 --out $d \
    > /tmp/h11_attempt$attempt.log 2>&1
  grep -E 'large model|items$|separation_ratio|real_vs_shuffled|n_heads_gt' /tmp/h11_attempt$attempt.log
  [ -f $d/report.json ] && break
  echo "  --- attempt $attempt failed, last 25 lines of /tmp/h11_attempt$attempt.log ---"
  tail -25 /tmp/h11_attempt$attempt.log
done
[ -f $d/report.json ] || echo "H11 FAILED after the memory fix too — report as a gap, do not infer"
echo H11_DONE
