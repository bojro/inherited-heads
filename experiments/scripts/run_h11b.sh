#!/bin/bash
# H11 attempt 4. The memory fix (17z) worked — it loaded — and then died on
# something else entirely: Vicuna-1.1 ships no tokenizer.chat_template, and
# run_text_discovery.py called apply_chat_template unconditionally. It now falls
# back to the FastChat "v1" format Vicuna was tuned on. See PREREGISTRATION 17ah.
cd "$(dirname "$0")/.."
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
# wait for the orphaned V15 gate (and anything else) to release the card
while pgrep -f 'discovery/run_[a-z_]*\.py' >/dev/null; do sleep 60; done
sleep 60
d=discovery/results/text/vicuna13b
[ -f $d/report.json ] && { echo "H11 already done"; echo H11B_DONE; exit 0; }
echo "=== H11 Vicuna-13B text discovery, attempt 4 $(date +%H:%M) ==="
nvidia-smi --query-gpu=memory.free --format=csv,noheader
.venv/bin/python discovery/run_text_discovery.py \
  --model ~/models/vicuna-13b-v1.1-safetensors \
  --text testbed/text_strips --n-items 50 --item-offset 100 --out $d \
  > /tmp/h11b.log 2>&1
grep -E 'large model|items$|separation_ratio|real_vs_shuffled|n_heads_gt' /tmp/h11b.log
[ -f $d/report.json ] || { echo "H11 FAILED again — last 25 lines:"; tail -25 /tmp/h11b.log; }
if [ -f $d/report.json ]; then
  echo "=== H11 overlap: Ultravox audio gaze vs Vicuna-13B TEXT gaze $(date +%H:%M) ==="
  .venv/bin/python - <<'PYEOF'
import numpy as np
from math import comb
# SALMONN's backbone is Vicuna-13B, so the inheritance test for SALMONN is this one.
a = np.load('discovery/results/salmonn/scores.npz')['scores'].ravel()
t = np.load('discovery/results/text/vicuna13b/scores.npz')['scores'].ravel()
assert a.shape == t.shape, (a.shape, t.shape)
n = a.size
top = lambda x, k=100: set(np.argsort(x)[::-1][:k].tolist())
ov = len(top(a) & top(t)); ch = 100*100/n
p = sum(comb(100,i)*comb(n-100,100-i) for i in range(ov,101))/comb(n,100)
print('H11: SALMONN audio gaze vs Vicuna-13B caption-free TEXT gaze')
print('  overlap %d/100 (chance %.1f, ratio %.1fx, hypergeometric p=%.3g)' % (ov, ch, ov/ch, p))
for k in (25,50,100,200):
    o=len(top(a,k)&top(t,k)); print('    K=%-4d overlap %3d  chance %5.1f  ratio %.1fx' % (k,o,k*k/n,o/(k*k/n)))
PYEOF
fi
echo H11B_DONE
