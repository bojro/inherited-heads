#!/bin/bash
# Qwen2-7B-Instruct — the text-only counterpart of Qwen2-Audio's backbone, for
# H12. Bandwidth only, no GPU: safe to run alongside the experiment queue.
cd "$(dirname "$0")/.."
.venv/bin/python - <<'PYEOF'
from huggingface_hub import snapshot_download
p = snapshot_download("Qwen/Qwen2-7B-Instruct",
                      allow_patterns=["*.safetensors", "*.json", "*.txt", "*.model"])
print("QWEN2_7B_AT", p)
PYEOF
echo DL_QWEN_DONE
