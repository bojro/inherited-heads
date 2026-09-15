"""Re-shard a pytorch_model-*.bin checkpoint into ~1GB safetensors shards.

Why: transformers >= 5.x refuses `torch.load` on torch < 2.6 (CVE-2025-32434),
and lmsys/vicuna-13b-v1.1 (SALMONN's backbone) ships only .bin shards. This
box runs torch 2.5.1+cu121 (the newest cu121 wheel that matches the driver
stack we validated), so convert once instead of upgrading torch under a
working bitsandbytes/torchaudio install.

Low-RAM by construction (WSL2 here has 7.7GB): each .bin is opened with
mmap=True and tensors are cloned into <=1GB groups before each save_file.

    python discovery/tools/reshard_to_safetensors.py lmsys/vicuna-13b-v1.1 \
        ~/models/vicuna-13b-v1.1-safetensors

Then point SalmonnAdapter at it via SALMONN_VICUNA_PATH=<out dir>.
"""
import gc, glob, json, os, shutil, sys

import torch
from huggingface_hub import snapshot_download
from safetensors.torch import save_file

repo, out = sys.argv[1], os.path.expanduser(sys.argv[2])
snap = snapshot_download(repo, allow_patterns=["*.json", "*.model", "*.bin"])
os.makedirs(out, exist_ok=True)
for f in ["config.json", "generation_config.json", "tokenizer.model",
          "tokenizer_config.json", "special_tokens_map.json"]:
    if os.path.exists(os.path.join(snap, f)):
        shutil.copy(os.path.join(snap, f), out)

LIMIT = 1_000_000_000
weight_map, shard_i, total = {}, 0, 0
buf, size = {}, 0


def flush():
    global shard_i, buf, size
    if not buf:
        return
    name = f"model-{shard_i:05d}.safetensors"
    save_file(buf, os.path.join(out, name), metadata={"format": "pt"})
    for k in buf:
        weight_map[k] = name
    shard_i += 1
    buf, size = {}, 0
    gc.collect()


for bin_path in sorted(glob.glob(os.path.join(snap, "pytorch_model-*.bin"))):
    sd = torch.load(bin_path, map_location="cpu", weights_only=True, mmap=True)
    for k, v in sd.items():
        t = v.clone().contiguous()
        buf[k] = t
        size += t.numel() * t.element_size()
        total += t.numel()
        if size >= LIMIT:
            flush()
    flush()
    del sd
    gc.collect()
    print("done", os.path.basename(bin_path), "shards so far", shard_i, flush=True)

json.dump({"metadata": {"total_size": total * 2}, "weight_map": weight_map},
          open(os.path.join(out, "model.safetensors.index.json"), "w"), indent=1)
print("tensors", len(weight_map), "params %.2fB" % (total / 1e9), "shards", shard_i)
