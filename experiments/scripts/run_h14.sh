#!/bin/bash
# H14 — are the seed paper's VLM gaze heads also text heads? Needs V2's image
# scores, so gated on V2_DONE. Text side is Qwen2-7B (Qwen2-VL-7B's backbone).
cd "$(dirname "$0")/.."
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
until grep -q V2_DONE ~/v2.log 2>/dev/null; do sleep 300; done
echo "=== H14 starting $(date +%H:%M) ==="
d=discovery/results/text/qwen2_7b_captions
[ -f $d/report.json ] || { echo "=== H14 Qwen2-7B caption-text discovery $(date +%H:%M) ==="
  .venv/bin/python discovery/run_text_discovery.py --model Qwen/Qwen2-7B-Instruct \
    --text testbed/caption_strips --n-items 50 --item-offset 100 --out $d 2>&1 \
    | grep -E 'items$|separation_ratio|real_vs_shuffled|n_heads_gt|Error|Traceback'; }
[ -f $d/report.json ] || { echo "H14 ABORT: caption-text discovery produced no report"; echo H14_DONE; exit 0; }
[ -f discovery/results/vlm/qwen2_vl/scores.npz ] || { echo "H14 ABORT: V2 produced no image scores"; echo H14_DONE; exit 0; }
echo "=== H14 overlap analysis $(date +%H:%M) ==="
.venv/bin/python - <<'PYEOF'
import numpy as np
from math import comb
img = np.load('discovery/results/vlm/qwen2_vl/scores.npz')['scores'].ravel()
txt = np.load('discovery/results/text/qwen2_7b_captions/scores.npz')['scores'].ravel()
assert img.shape == txt.shape, (img.shape, txt.shape)
n = img.size
top = lambda x, k=100: set(np.argsort(x)[::-1][:k].tolist())
ov = len(top(img) & top(txt)); ch = 100*100/n
p = sum(comb(100,i)*comb(n-100,100-i) for i in range(ov,101))/comb(n,100)
print('H14: Qwen2-VL image gaze vs Qwen2-7B caption-text gaze')
print('  overlap %d/100 (chance %.1f, ratio %.1fx, hypergeometric p=%.3g)' % (ov, ch, ov/ch, p))
verdict = 'INHERITED (>=40)' if ov>=40 else ('MODALITY-SPECIFIC (<25)' if ov<25 else 'INTERMEDIATE (25-40)')
print('  pre-registered verdict:', verdict)
for k in (25,50,100,200):
    o=len(top(img,k)&top(txt,k)); print('    K=%-4d overlap %3d  chance %5.1f  ratio %.1fx' % (k,o,k*k/n,o/(k*k/n)))
PYEOF
echo H14_DONE
