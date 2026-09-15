#!/bin/bash
# V22 - two fixes named by a literature review of causal methods as the main open objections.
#
# A) LAYER-HISTOGRAM-MATCHED RANDOM AT K=100. Our published "random" steering
#    control samples UNIFORMLY from the layer BAND the gaze set spans
#    (steer.random_k_heads). The stronger control matches the gaze set's per-
#    layer COUNTS. On Ultravox the two disagree sharply -- band-matched K=100
#    steers .453, histogram-matched K=71 steers .150 -- so the inversion claim
#    ("their raw score steers below a layer-matched random draw") is currently
#    resting on the weaker of the two controls. Two seeds.
#
# B) BIAS-MAGNITUDE DOSE-RESPONSE. The seed paper's +10,000 pre-softmax bias
#    drives attention to ~one-hot. PASTA (ICLR 2024) runs the same experiment
#    gently -- post-softmax, multiplicative, alpha=0.01 -- and warns that
#    alpha->0 causes "performance degeneration" by removing all other context.
#    +10,000 is numerically that case. If the effect appears at B=2, the
#    intervention is on-distribution and both the OOD objection and the
#    dormant-pathway objection weaken. If it ONLY appears at B=10,000, that is
#    a real limitation and gets reported. 20 items per cell (shape, not a
#    headline estimate) -- the K=100 reference at 50 items is .990.
#    NOTE: this is BIAS-magnitude dose-response. It is unrelated to the
#    MODALITY-MASS dose-response claim withdrawn in 17al.
cd "$(dirname "$0")/.."
while pgrep -f "bash \./run_ac2.sh" > /dev/null; do sleep 60; done
echo "=== V22 starting $(date +%H:%M) ==="
G=discovery/results/ultravox/scores.npz
S=discovery/results/sets

for seed in 0 1; do
  n=ultravox_randmatched100_s$seed
  [ -f $S/$n.npz ] || .venv/bin/python discovery/make_layer_matched.py \
      --scores $G --k 100 --seed $seed --out $S/$n.npz
  o=discovery/results/confirm/$n
  [ -f $o/steer_report.json ] || { echo "--- histogram-matched random 100, seed $seed ---"
    .venv/bin/python discovery/run_steer.py --model ultravox --strips testbed/strips \
      --scores $S/$n.npz --condition top --k 100 --n-items 50 --item-offset 100 \
      --out $o 2>&1 | grep -E '"accuracy"|"unattributed"|Error|Traceback'; }
done

echo "=== V22b bias dose-response $(date +%H:%M) ==="
for B in 1 2 5 10 100; do
  o=discovery/results/dose/ultravox_top100_B$B
  [ -f $o/steer_report.json ] || { echo "--- B=$B ---"
    .venv/bin/python discovery/run_steer.py --model ultravox --strips testbed/strips \
      --scores $G --condition top --k 100 --B $B --n-items 20 --item-offset 100 \
      --out $o 2>&1 | grep -E '"accuracy"|"unattributed"|Error|Traceback'; }
done
o=discovery/results/dose/ultravox_top100_B10000
[ -f $o/steer_report.json ] || { echo "--- B=10000 (the published setting, 20 items for comparability) ---"
  .venv/bin/python discovery/run_steer.py --model ultravox --strips testbed/strips \
    --scores $G --condition top --k 100 --B 1e4 --n-items 20 --item-offset 100 \
    --out $o 2>&1 | grep -E '"accuracy"|"unattributed"|Error|Traceback'; }

echo "=== V22 SUMMARY $(date +%H:%M) ==="
.venv/bin/python - <<'PYEOF'
import json, os
def acc(p):
    return f"{json.load(open(p))['accuracy']:.3f}" if os.path.exists(p) else "MISSING"
print("  A) random-100 controls, 50 items, held-out")
print(f"    band-matched (published)      {acc('discovery/results/confirm/ultravox_random100/steer_report.json')}")
for s in (0, 1):
    print(f"    layer-histogram-matched s{s}   {acc(f'discovery/results/confirm/ultravox_randmatched100_s{s}/steer_report.json')}")
print("    their raw top-100             0.017   ours 0.990   chance 0.167")
print("\n  B) bias dose-response, top-100, 20 items")
for B in (1, 2, 5, 10, 100, 10000):
    print(f"    B={B:<7} {acc(f'discovery/results/dose/ultravox_top100_B{B}/steer_report.json')}")
PYEOF
echo V22_DONE
