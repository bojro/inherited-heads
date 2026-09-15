#!/bin/bash
# V2 — the seed paper's OWN modality. Downloads only: bandwidth and
# disk, no GPU, so this runs safely alongside the experiment queue.
# Sequential on purpose: parallel downloads split one ~3.5 MB/s pipe.
cd "$(dirname "$0")/.."
get() {  # repo, kind, note
  echo "=== $1 ($3) start $(date +%H:%M) ==="
  .venv/bin/python - <<PYEOF
from huggingface_hub import snapshot_download
try:
    p = snapshot_download("$1", repo_type="$2",
                          allow_patterns=["*.safetensors","*.json","*.txt","*.model",
                                          "*.py","*.parquet","*.bin"] if "$2"=="model"
                                     else None)
    print("OK $1 ->", p)
except Exception as e:
    print("FAILED $1:", type(e).__name__, e)
PYEOF
  echo "=== $1 done $(date +%H:%M) ==="
}

# 1. their data first — small, and nothing can be done without it
get baulab/openai-comic-strips dataset "500 six-panel comic strips, MIT"
# 2. the NULL model: 8.3% steering in their Table 2, i.e. at chance.
#    This is the target — does our normalised score find heads here?
get BAAI/Bunny-v1_0-3B model "their strongest null"
# 3. POSITIVE CONTROL: 66.2% in their Table 2. Without this, a null result on
#    Bunny is uninterpretable — it could just mean our VLM adapter is broken.
get Qwen/Qwen2-VL-7B-Instruct model "positive control"
# 4. second null, well supported in transformers (their table lists the 13B at 39.0%)
get llava-hf/llava-1.5-7b-hf model "second null"
echo DL_VLM_DONE
