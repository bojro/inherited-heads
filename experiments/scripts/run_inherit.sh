#!/bin/bash
# H10/H11 — is the text->audio inheritance CAUSAL, and does it hold for the
# other frozen backbone? Gated on RISK_DONE so it never shares the GPU.
cd "$(dirname "$0")/.."
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
until grep -q RISK_DONE ~/riskfirst.log 2>/dev/null; do sleep 60; done
echo "=== inheritance block starting $(date +%H:%M) ==="

# H10: steer Ultravox on AUDIO with the head set found purely from TEXT.
# items 0-19, matching the .983 / .017 already measured there.
d=discovery/results/e2/ultravox_texttop100
[ -f $d/steer_report.json ] || { echo "=== H10 ultravox steered by TEXT-derived heads $(date +%H:%M) ==="
  .venv/bin/python discovery/run_steer.py --model ultravox \
    --scores discovery/results/text/llama31_8b/scores.npz --condition top --k 100 \
    --n-items 20 --strips testbed/strips --out $d 2>&1 \
    | grep -E 'hook check|"accuracy"|"unattributed_rate"|Error|Traceback'; }

# H11: SALMONN's backbone is Vicuna-13B, already local. Text discovery on it,
# then overlap against SALMONN's audio gaze ranking.
d=discovery/results/text/vicuna13b
[ -f $d/report.json ] || { echo "=== H11 Vicuna-13B text discovery $(date +%H:%M) ==="
  .venv/bin/python discovery/run_text_discovery.py \
    --model ~/models/vicuna-13b-v1.1-safetensors \
    --text testbed/text_strips --n-items 50 --item-offset 100 --out $d 2>&1 \
    | grep -E 'items$|separation_ratio|real_vs_shuffled|n_heads_gt|Error|Traceback'; }
echo INHERIT_DONE
