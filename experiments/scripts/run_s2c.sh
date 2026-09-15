#!/bin/bash
# S2c -- finish the job S2b started: give EVERY random-head control five draws.
#
# S2b showed the Ultravox K=29 control spans .008-.650 across five identically
# built draws, which flipped the sign of a result the paper reported with an
# interval excluding zero. Two families of control were still single-draw:
#
#  1. THE PRIMARY SUFFICIENCY CONTROLS (K=71/66/74). These carry +0.900/+0.800/
#     +0.583 -- the paper's headline. The differences are large enough that the
#     SIGN is not in doubt, but the magnitudes are quoted off one draw each and
#     the paper should report a distribution.
#
#  2. THE K=100 LAYER-MATCHED CONTROL ON ULTRAVOX. Five draws exist (.147-.663)
#     and contribution 1 rests on the raw score's .017 sitting below ALL of them.
#     S2b then produced a .008 draw at K=29 -- proof that the lower tail of a
#     layer-matched control can reach below .017. Five more K=100 draws test
#     whether the inversion survives a properly sampled control. If any draw
#     lands below .017, the inversion as worded is gone and we need to know that
#     before it is written, not after review.
#
# Gated on S2B_DONE so it never shares the GPU.
cd "$(dirname "$0")/.."
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
until grep -q S2B_DONE ~/s2b.log 2>/dev/null; do sleep 60; done
echo "=== S2c starting $(date +%H:%M:%S) ==="

# ---- 1. sufficiency controls, four more draws each, matching the originals ----
G=discovery/results/sets/mseeds
run () {  # run <model> <set-name> <k>
  local o=discovery/results/mseeds/$2
  if [ -f $o/steer_report.json ]; then echo "  skip $2 (done)"; return; fi
  echo "=== $2 (K=$3) start $(date +%H:%M:%S) ==="
  .venv/bin/python discovery/run_steer.py --model $1 --strips testbed/strips \
    --scores $G/$2.npz --condition top --k $3 --n-items 20 --item-offset 100 \
    --out $o 2>&1 | grep -E 'hook check|items:|"accuracy"|"unattributed"|Error|Traceback'
  echo "=== $2 end $(date +%H:%M:%S) ==="
}
for sd in 1 2 3 4; do run ultravox    ultravox_randmatched71_s$sd 71; done
for sd in 1 2 3 4; do run qwen2-audio qwen2_audio_randmatched66_s$sd 66; done
for sd in 1 2 3 4; do run salmonn     salmonn_randmatched74_s$sd 74; done

# ---- 2. five more K=100 layer-matched draws, the inversion's control ----------
# 50 items and item-offset 100, matching seeds 0-4 exactly, or the draws are not
# comparable to the band the paper already quotes.
E=discovery/results/ultravox/scores.npz
S=discovery/results/sets
for seed in 5 6 7 8 9; do
  n=ultravox_randmatched100_s$seed
  [ -f $S/$n.npz ] || .venv/bin/python discovery/make_layer_matched.py \
      --scores $E --k 100 --seed $seed --out $S/$n.npz
  o=discovery/results/confirm/$n
  [ -f $o/steer_report.json ] || { echo "=== K=100 layer-matched seed $seed start $(date +%H:%M:%S) ==="
    .venv/bin/python discovery/run_steer.py --model ultravox --strips testbed/strips \
      --scores $S/$n.npz --condition top --k 100 --n-items 50 --item-offset 100 \
      --out $o 2>&1 | grep -E '"accuracy"|"unattributed"|Error|Traceback'
    echo "=== K=100 layer-matched seed $seed end $(date +%H:%M:%S) ==="; }
done

echo "=== S2c summary $(date +%H:%M:%S) ==="
.venv/bin/python - <<'PYEOF'
import json, os
import numpy as np
R = "discovery/results"
def acc(p):
    f = os.path.join(R, p, "steer_report.json")
    return json.load(open(f))["accuracy"] if os.path.exists(f) else None
print("\n-- primary sufficiency controls, all draws --")
for tag, k, orig, subj in (("ultravox", 71, "crosstask/ultravox_randmatched71", "crosstask/ultravox_shared71"),
                           ("qwen2_audio", 66, "crosstask/qwen2_audio_randmatched66", "crosstask/qwen2_audio_shared66"),
                           ("salmonn", 74, "crosstask/salmonn_randmatched74", "crosstask/salmonn_shared74")):
    d = [acc(orig)] + [acc(f"mseeds/{tag}_randmatched{k}_s{s}") for s in (1, 2, 3, 4)]
    d = [x for x in d if x is not None]
    s_ = acc(subj)
    if d and s_ is not None:
        a = np.array(d)
        print(f"  {tag:12s} shared {s_:.3f}   control {a.mean():.3f} "
              f"[{a.min():.3f}, {a.max():.3f}] over {len(d)}   diff vs mean {s_-a.mean():+.3f}")
print("\n-- K=100 layer-matched control, Ultravox: does any draw fall below the raw score's .017? --")
raw = acc("confirm/ultravox_rawtop100")
d = [(s, acc(f"confirm/ultravox_randmatched100_s{s}")) for s in range(10)]
d = [(s, v) for s, v in d if v is not None]
for s, v in d:
    print(f"  seed {s}  {v:.3f}" + ("   <-- BELOW the raw score" if raw is not None and v < raw else ""))
if d and raw is not None:
    a = np.array([v for _, v in d])
    below = int((a < raw).sum())
    print(f"\n  raw score {raw:.3f}   control mean {a.mean():.3f} [{a.min():.3f}, {a.max():.3f}] over {len(a)} draws")
    print(f"  draws below the raw score: {below}/{len(a)}")
    print("  ZERO below  -> the inversion survives a properly sampled control." if below == 0
          else "  AT LEAST ONE below -> the inversion as worded is NOT supported. Rewrite it.")
PYEOF
echo S2C_DONE
