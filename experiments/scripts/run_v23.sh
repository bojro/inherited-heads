#!/bin/bash
# THE FULL STACK, in value order. Every item was named by a sweep or by a
# pre-committed reading; none is exploratory.
#
# V23  centre-crop -- the ONLY surviving explanation for the seed paper's Bunny
#      8.3% now that AC2 ruled out measurement point (72/100 overlap, 17az).
#      Geometry checked before running: centre-cropping the 6:1 strip keeps ONLY
#      panels 2 and 3, each ~49% visible; four of six panels leave the image.
#      Prediction fixed here: steering collapses toward their 8.3%. If it does
#      NOT collapse, the centre-crop hypothesis dies and their null is unexplained.
# V24  bias dose-response on a SECOND arm -- does saturation at B=5 generalise
#      beyond Ultravox, or is it an Ultravox quirk? (17az)
# V25  three more seeds on the layer-histogram-matched control -- two seeds gave
#      .393/.283 and that spread is too wide to quote (17az).
# V26  headline cells re-run at B=5, the saturating on-distribution setting.
cd "$(dirname "$0")/.."
echo "=== STACK starting $(date +%H:%M) ==="

# ---- V23: Bunny under centre-crop ------------------------------------------
d=discovery/results/vlm/bunny_strip_crop
[ -f $d/report.json ] || { echo "=== V23 discovery, centre-cropped $(date +%H:%M) ==="
  .venv-bunny/bin/python discovery/run_vlm_discovery.py --model bunny-strip-crop \
    --comics testbed/comics --n-items 50 --item-offset 0 --out $d --full-precision 2>&1 \
    | grep -E 'items|separation_ratio|real_vs_shuffled|panel_token|Error|Traceback'; }
R=discovery/results/vlm/bunny_strip_raw_ranked.npz
for spec in "top $d/scores.npz top100" "top $R rawtop100" "random $d/scores.npz random100"; do
  set -- $spec
  o=discovery/results/vlm/e2/bunny_strip_crop_$3
  [ -f $o/steer_report.json ] || { echo "--- V23 steer $3 ---"
    .venv-bunny/bin/python discovery/run_vlm_steer.py --model bunny-strip-crop \
      --comics testbed/comics --scores $2 --condition $1 --k 100 \
      --n-items 20 --item-offset 100 --out $o --full-precision 2>&1 \
      | grep -E '"accuracy"|"unattributed"|Error|Traceback'; }
done

# ---- V24: does B=5 saturation generalise? ----------------------------------
G2=discovery/results/qwen2_audio_v2/scores.npz
for B in 1 2 5 10 10000; do
  o=discovery/results/dose/qwen2_audio_top100_B$B
  [ -f $o/steer_report.json ] || { echo "--- V24 qwen2-audio B=$B ---"
    .venv/bin/python discovery/run_steer.py --model qwen2-audio --strips testbed/strips \
      --scores $G2 --condition top --k 100 --B $B --n-items 20 --item-offset 100 \
      --out $o 2>&1 | grep -E '"accuracy"|Error|Traceback'; }
done

# ---- V25: more seeds for the histogram-matched control ---------------------
G=discovery/results/ultravox/scores.npz
for seed in 2 3 4; do
  n=ultravox_randmatched100_s$seed
  [ -f discovery/results/sets/$n.npz ] || .venv/bin/python discovery/make_layer_matched.py \
      --scores $G --k 100 --seed $seed --out discovery/results/sets/$n.npz
  o=discovery/results/confirm/$n
  [ -f $o/steer_report.json ] || { echo "--- V25 histogram-matched seed $seed ---"
    .venv/bin/python discovery/run_steer.py --model ultravox --strips testbed/strips \
      --scores discovery/results/sets/$n.npz --condition top --k 100 \
      --n-items 50 --item-offset 100 --out $o 2>&1 | grep -E '"accuracy"|Error'; }
done

# ---- V26: headline cells at the saturating B=5 -----------------------------
for spec in "ultravox $G top100" "ultravox discovery/results/sets/ultravox/raw_ranked.npz rawtop100" \
            "qwen2-audio $G2 top100" "qwen2-audio discovery/results/sets/qwen2_audio/raw_ranked.npz rawtop100" \
            "salmonn discovery/results/salmonn/scores.npz top100" \
            "salmonn discovery/results/sets/salmonn/raw_ranked.npz rawtop100"; do
  set -- $spec
  o=discovery/results/confirm_b5/${1}_$3
  [ -f $o/steer_report.json ] || { echo "--- V26 $1 $3 at B=5 ---"
    .venv/bin/python discovery/run_steer.py --model $1 --strips testbed/strips \
      --scores $2 --condition top --k 100 --B 5 --n-items 50 --item-offset 100 \
      --out $o 2>&1 | grep -E '"accuracy"|Error|Traceback'; }
done

echo "=== STACK SUMMARY $(date +%H:%M) ==="
.venv/bin/python - <<'PYEOF'
import json, os, glob
a = lambda p: f"{json.load(open(p))['accuracy']:.3f}" if os.path.exists(p) else "MISSING"
print("V23 Bunny-strip CENTRE-CROPPED, 20 items (padded reference: ours .367 theirs .383 random .042; THEY PUBLISH .083)")
for c in ("top100", "rawtop100", "random100"):
    print(f"    {c:12s} {a(f'discovery/results/vlm/e2/bunny_strip_crop_{c}/steer_report.json')}")
print("\nV24 Qwen2-Audio bias dose-response, 20 items (Ultravox saturated at B=5)")
for B in (1, 2, 5, 10, 10000):
    print(f"    B={B:<7} {a(f'discovery/results/dose/qwen2_audio_top100_B{B}/steer_report.json')}")
print("\nV25 Ultravox layer-histogram-matched random-100, 50 items, 5 seeds")
for s in range(5):
    print(f"    seed {s}      {a(f'discovery/results/confirm/ultravox_randmatched100_s{s}/steer_report.json')}")
print("\nV26 headline cells at B=5, 50 items (B=1e4 references: ultravox .990/.017, salmonn .907/.807, qwen2_audio .983/.647)")
for m in ("ultravox", "qwen2-audio", "salmonn"):
    o, t = a(f'discovery/results/confirm_b5/{m}_top100/steer_report.json'), a(f'discovery/results/confirm_b5/{m}_rawtop100/steer_report.json')
    print(f"    {m:12s} ours {o}   theirs {t}")
PYEOF
echo STACK_DONE
