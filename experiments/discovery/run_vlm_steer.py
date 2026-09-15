"""E2 steering on the seed paper's own modality — experiment V2 (see PREREGISTRATION.md).

Mirrors run_steer.py: bias the chosen head set toward one panel (+B on its image
tokens, -B on the other five), ask a prompt that names no panel, and judge which
panel the answer describes. Chance = 1/6.

This is a SEPARATE file rather than a flag on run_steer.py because run_steer.py
was executing the 12-hour confirmatory suite when this was written and must not
be edited under a running job. `steer.py` and `judge.py` are imported and used
read-only. Once the suite is done the two runners should be unified — the only
real differences are the loader and that ground truth is `caption` rather than
`transcript`.

    python run_vlm_steer.py --model qwen2-vl --scores results/vlm/qwen2_vl/scores.npz \
        --condition top --k 100 --n-items 20 --item-offset 100 --out results/vlm/e2/qwen2_vl_top100
"""
import argparse, json
from pathlib import Path

import numpy as np

import steer
import steer_prehook
from judge import attribute
from run_vlm_discovery import load_comics, make_adapter
from vlm_regions import set_bias_regions


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="qwen2-vl")
    ap.add_argument("--comics", type=Path, default=Path("../testbed/comics"))
    ap.add_argument("--scores", type=Path, required=True)
    ap.add_argument("--condition", choices=("top", "random", "all", "none"), default="top")
    ap.add_argument("--k", type=int, default=100)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--B", type=float, default=1e4)
    ap.add_argument("--n-items", type=int, default=20)
    ap.add_argument("--item-offset", type=int, default=0)
    ap.add_argument("--max-new", type=int, default=50)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--full-precision", action="store_true")
    a = ap.parse_args()

    import pandas as pd
    meta = pd.read_parquet(a.comics / "metadata.parquet")
    scores = np.load(a.scores)["scores"]
    adapter = make_adapter(a.model, load_4bit=not a.full_precision)
    strips = list(load_comics(a.comics, a.n_items, a.item_offset))

    # hook-equivalence check, same discipline as run_steer.py. With the bias off
    # the patched attention must reproduce stock eager; this assertion is what
    # caught both critical E2 bugs in the audio arms.
    imgs0, _, panels0 = strips[0]
    inp0 = adapter.prepare(imgs0, None, adapter.NEUTRAL)
    ref = adapter.forward_last_token_attn(inp0)
    mech = getattr(adapter, "steer_mechanism", "interface")
    if mech == "prehook":
        # Bunny: vendored attention bypasses ALL_ATTENTION_FUNCTIONS
        layers = steer_prehook.find_layers_generic(adapter.model)
        n_layers, n_heads = steer_prehook.install(layers, scores.shape[1])
        steer_prehook.clear()
        set_bias = steer_prehook.set_bias_regions
    else:
        n_layers, n_heads = steer.install(adapter.model)
        steer.clear()
        set_bias = set_bias_regions
    dmax = float(np.abs(ref - adapter.forward_last_token_attn(inp0)).max())
    print(f"[hook check:{mech}] max |attn_ref - attn_hooked| = {dmax:.2e}", flush=True)
    assert dmax < 1e-2, "the bias hook with bias OFF must equal unhooked attention"
    assert (n_layers, n_heads) == scores.shape, (
        f"located {(n_layers, n_heads)} but the score matrix is {scores.shape} — "
        "find_layers_generic may have grabbed the wrong ModuleList")

    if a.condition == "top":
        heads = steer.top_k_heads(scores, a.k)
    elif a.condition == "random":
        heads = steer.random_k_heads(scores, a.k, steer.top_k_heads(scores, a.k), a.seed)
    elif a.condition == "all":
        heads = steer.all_heads(scores)
    else:
        heads = []

    a.out.mkdir(parents=True, exist_ok=True)
    conf = np.zeros((6, 7), dtype=int)
    with open(a.out / "answers.jsonl", "w") as f:
        for i, (imgs, _, panels) in enumerate(strips):
            item = i + a.item_offset
            caps = meta[meta.item_id == item].sort_values("panel_idx").caption.tolist()
            assert len(caps) == 6, (item, len(caps))
            targets = [0] if a.condition == "none" else range(6)
            for t in targets:
                inputs = adapter.prepare(imgs, None, adapter.NEUTRAL)
                _, _, positions = adapter.panel_regions(inputs)
                if heads:
                    # region-indexed because a tiled strip's panels are column
                    # bands, not contiguous spans; writes steer._State directly so
                    # the same hook applies it (see vlm_regions.set_bias_regions)
                    set_bias(heads, positions, t,
                             int(inputs["inputs_embeds"].shape[1])
                             if "inputs_embeds" in inputs
                             else int(inputs["input_ids"].shape[1]), n_heads, a.B)
                ans = adapter.generate(inputs, max_new=a.max_new)
                j, sc = attribute(ans, caps)
                conf[t, j if j >= 0 else 6] += 1
                f.write(json.dumps(dict(item=item, target=t, judged=j, answer=ans,
                                        scores=[round(x, 2) for x in sc])) + "\n")
            (steer_prehook if mech == "prehook" else steer).clear()
            if (i + 1) % 5 == 0:
                d = np.trace(conf[:, :6]) / max(conf.sum(), 1)
                print(f"  {i+1} items: acc {d:.3f}", flush=True)

    rep = dict(model=a.model, modality="image", condition=a.condition, k=a.k,
               B=a.B, n_items=len(strips), item_offset=a.item_offset,
               prompt=adapter.NEUTRAL, chance=1 / 6,
               accuracy=float(np.trace(conf[:, :6]) / conf.sum()),
               unattributed_rate=float(conf[:, 6].sum() / conf.sum()),
               per_target_accuracy=[float(conf[t, t] / max(conf[t].sum(), 1)) for t in range(6)],
               confusion_target_x_judged=conf.tolist(),
               heads=[[int(l), int(h)] for l, h in heads][:500])
    (a.out / "steer_report.json").write_text(json.dumps(rep, indent=2))
    print(json.dumps({k: v for k, v in rep.items() if k != "heads"}, indent=2))


if __name__ == "__main__":
    main()
