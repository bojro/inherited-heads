#!/bin/bash
# Bunny's vision tower is a SEPARATE repo loaded at runtime (config.json:
# mm_vision_tower = google/siglip-so400m-patch14-384). It was missing from
# dl_vlm.sh, so Bunny would have failed to load despite its own 6GB being present.
# Chained after DL_QWEN15_DONE so downloads stay serial on one ~3.5 MB/s pipe.
until grep -q DL_QWEN15_DONE ~/dl_qwen15.log 2>/dev/null; do sleep 120; done
cd "$(dirname "$0")/.."
echo "=== siglip-so400m-patch14-384 (Bunny vision tower) start $(date +%H:%M) ==="
.venv/bin/python - <<'PYEOF'
from huggingface_hub import snapshot_download
p = snapshot_download("google/siglip-so400m-patch14-384",
                      allow_patterns=["*.safetensors", "*.json", "*.txt", "*.model"])
print("SIGLIP_AT", p)
PYEOF
echo DL_SIGLIP_DONE
