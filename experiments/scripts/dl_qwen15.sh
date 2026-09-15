#!/bin/bash
# H12's CORRECT text counterpart. Qwen2-Audio-7B's LLM is 32L/32H/4096hidden/
# 11008inter -- that is Qwen1.5-7B, NOT Qwen2-7B (28L/28H/3584). The Qwen2-7B
# download earlier tonight was a mistake. Gated behind the VLM downloads because
# V2 is the higher-priority item and they share one ~3.5 MB/s pipe.
until grep -q DL_VLM_DONE ~/dl_vlm.log 2>/dev/null; do sleep 120; done
cd "$(dirname "$0")/.."
echo "=== Qwen1.5-7B-Chat start $(date +%H:%M) ==="
.venv/bin/python - <<'PYEOF'
from huggingface_hub import snapshot_download
p = snapshot_download("Qwen/Qwen1.5-7B-Chat",
                      allow_patterns=["*.safetensors","*.json","*.txt","*.model"])
print("QWEN15_AT", p)
PYEOF
echo DL_QWEN15_DONE
