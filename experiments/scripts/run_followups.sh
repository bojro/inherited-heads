#!/bin/bash
# Runs AFTER everything else (gated on PHASE2_DONE) so it can never contend for
# the GPU with the confirmatory suite.
#   H12  - Qwen2-Audio's backbone was UPDATED during audio training. If text
#          inheritance is a property of FROZEN backbones, its text/audio head
#          overlap should be well below Ultravox's 71/100. Predicted <40.
#   R1'  - retargeting redone with sustained output. The first attempt returned
#          follow_rate 0.00 for a trivial reason: the one-sentence prompt ended
#          generation before the switch at token 25, leaving an empty second
#          half. Results in results/e3/ are INVALID for that reason.
cd "$(dirname "$0")/.."
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
until grep -q PHASE2_DONE ~/overnight2.log 2>/dev/null; do sleep 300; done
echo "=== follow-ups starting $(date +%H:%M) ==="

# ---- H8/H9 CONFIRMATORY (promised in PREREGISTRATION.md 17e, never queued) ----
# The headline contrast -- their raw-score ranking vs ours -- exists only on
# exploratory items 0-19 (.733/.875/.017 vs .958/.892/.983). 17e said a
# confirmatory version was appended to run_confirm.sh; it was not. Held-out
# items 100-149, same head sets (raw_ranked.npz built during the D3 follow-up,
# frozen from exploratory data like every other ranking).
for m in ultravox qwen2-audio salmonn; do
  mm=$(echo $m | tr '-' '_')
  R=discovery/results/sets/$mm/raw_ranked.npz
  [ -f "$R" ] || { echo "H8/H9 confirm: MISSING $R — skipping $m"; continue; }
  d=discovery/results/confirm/${mm}_rawtop100
  [ -f $d/steer_report.json ] || { echo "=== H8/H9 confirm $m RAW-ranked top-100 $(date +%H:%M) ==="
    .venv/bin/python discovery/run_steer.py --model $m --scores $R --condition top --k 100 \
      --n-items 50 --item-offset 100 --strips testbed/strips --out $d 2>&1 \
      | grep -E 'hook check|"accuracy"|"unattributed_rate"|Error|Traceback'; }
done
echo "=== H8/H9 confirmatory done $(date +%H:%M) ==="

# H12 - needs the Qwen1.5-7B-Chat download to have finished
if grep -q DL_QWEN15_DONE ~/dl_qwen15.log 2>/dev/null; then
  d=discovery/results/text/qwen15_7b_chat
  [ -f $d/report.json ] || { echo "=== H12 Qwen1.5-7B-Chat text discovery $(date +%H:%M) ==="
    .venv/bin/python discovery/run_text_discovery.py --model Qwen/Qwen1.5-7B-Chat \
      --text testbed/text_strips --n-items 50 --item-offset 100 --out $d 2>&1 \
      | grep -E 'items$|separation_ratio|real_vs_shuffled|n_heads_gt|Error|Traceback'; }
else
  echo "H12 SKIPPED: Qwen1.5-7B-Chat not downloaded yet"
fi

# H11 RETRY. Its first attempt (run_inherit.sh, 03:14) died with
# "CUDA driver error: device not ready" -- almost certainly Vicuna-13B starting
# its load before the previous model's VRAM was released. The chain echoed
# INHERIT_DONE regardless, so H11 silently dropped. Settle first, then retry once.
d=discovery/results/text/vicuna13b
[ -f $d/report.json ] || { for attempt in 1 2; do
    echo "=== H11 Vicuna-13B text discovery, attempt $attempt $(date +%H:%M) ==="
    sleep 30                       # let VRAM from the previous step be reclaimed
    .venv/bin/python discovery/run_text_discovery.py \
      --model ~/models/vicuna-13b-v1.1-safetensors \
      --text testbed/text_strips --n-items 50 --item-offset 100 --out $d 2>&1 \
      | grep -E 'items$|separation_ratio|real_vs_shuffled|n_heads_gt|Error|Traceback|RuntimeError'
    [ -f $d/report.json ] && break
  done; }

# R1' - corrected retargeting: narration prompt, 120 tokens, switch at 60
for m in qwen2-audio salmonn ultravox; do
  mm=$(echo $m | tr '-' '_')
  G=discovery/results/$( [ $mm = qwen2_audio ] && echo qwen2_audio_v2 || echo $mm )/scores.npz
  d=discovery/results/e3b/$mm
  [ -f $d/report.json ] || { echo "=== R1' $m retargeting, narrate/120tok $(date +%H:%M) ==="
    .venv/bin/python discovery/run_retarget.py --model $m --scores $G --k 100 \
      --prompt-style narrate --switch-at 60 --max-new 120 \
      --n-items 20 --item-offset 100 --strips testbed/strips --out $d 2>&1 \
      | grep -E 'hook check|items:|stay|follow|latency|Error|Traceback'; }
done
echo FOLLOWUPS_DONE
