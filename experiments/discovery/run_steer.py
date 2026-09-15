"""E2 runner — steer generation to a chosen segment by biasing head attention.

    python run_steer.py --model qwen2-audio --scores results/qwen2_audio_v2/scores.npz \
        --condition top --k 100 --n-items 20 --out results/e2/qwen2_audio_top100

Per strip and per target segment t in 0..5: install the bias (+B on t's audio
tokens, -B on the other segments' audio tokens, on the chosen heads), ask a
NEUTRAL prompt that names no segment, decode greedily, attribute the answer
with judge.attribute. Steering accuracy = P(judged == t); chance 1/6.
Conditions: top (top-K gaze heads), random (K heads from the same layer band,
excluding the gaze set), all (every head), none (unsteered baseline).
"""
import argparse, json
from pathlib import Path

import numpy as np

import steer
from judge import attribute
from judge_prosody import attribute_emotion, attribute_gender
from run_discovery import load_strips, make_adapter

NEUTRAL = ("The audio contains six segments spoken by different people, "
           "separated by silences. Describe what one of the speakers talks "
           "about, in one sentence.")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--strips", type=Path, default=Path("../testbed/strips"))
    ap.add_argument("--model", default="qwen2-audio")
    ap.add_argument("--scores", type=Path, required=True, help="scores.npz of the E1 run (head ranking)")
    ap.add_argument("--condition", choices=["top", "random", "all", "none"], default="top")
    ap.add_argument("--k", type=int, default=100)
    ap.add_argument("--B", type=float, default=1e4)
    ap.add_argument("--n-items", type=int, default=20)
    ap.add_argument("--item-offset", type=int, default=0, help=">=50 selects held-out data")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--max-new", type=int, default=50)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--full-precision", action="store_true")
    # V15. Defaults leave the validated path bit-identical.
    ap.add_argument("--judge", choices=["lexical", "emotion", "gender"], default="lexical",
                    help="lexical = transcript overlap (the audio strips). On the "
                         "prosody corpora every segment says the SAME sentence, so "
                         "lexical overlap is uniform and carries no information: use "
                         "emotion (chance 1/6) or gender (chance 1/2).")
    ap.add_argument("--prompt-text", default=None,
                    help="override the NEUTRAL prompt (the prosody corpora need a "
                         "'how does it sound' question, not 'what is it about')")
    a = ap.parse_args()
    import pandas as pd
    meta = pd.read_parquet(a.strips / "metadata.parquet")
    neutral = a.prompt_text or NEUTRAL
    chance = 0.5 if a.judge == "gender" else 1 / 6
    scores = np.load(a.scores)["scores"]
    adapter = make_adapter(a.model, load_4bit=not a.full_precision)
    strips = list(load_strips(a.strips, a.n_items, a.item_offset))
    if hasattr(adapter, "precompute"):
        adapter.precompute(strips)
    # make sure the LLM exists (SALMONN loads lazily), then install the hook
    if hasattr(adapter, "_ensure_llm"):
        adapter._ensure_llm()
    backbone = getattr(adapter, "llm", None) or adapter.model
    # equivalence check: with the bias inactive, the hooked model must reproduce
    # the unhooked model's last-token attention on the first strip (this caught
    # a missing causal mask once — never trust the hook without it)
    wav0, sr0, b0 = strips[0]
    inp0 = adapter.prepare(wav0, sr0, neutral)
    ref = adapter.forward_last_token_attn(inp0)
    n_layers, n_heads = steer.install(backbone)
    steer.clear()
    hooked = adapter.forward_last_token_attn(inp0)
    dmax = float(np.abs(ref - hooked).max())
    print(f"[hook check] max |attn_ref - attn_hooked| = {dmax:.2e}", flush=True)
    assert dmax < 1e-2, "eager_bias with bias off must equal eager"
    assert (n_layers, n_heads) == scores.shape, ((n_layers, n_heads), scores.shape)
    top = steer.top_k_heads(scores, a.k)
    if a.condition == "top":
        heads = top
    elif a.condition == "random":
        heads = steer.random_k_heads(scores, a.k, top, a.seed)
    elif a.condition == "all":
        heads = steer.all_heads(scores)
    else:
        heads = []
    print(f"[{a.model}] condition={a.condition} K={len(heads)} B={a.B} items={len(strips)}", flush=True)

    a.out.mkdir(parents=True, exist_ok=True)
    conf = np.zeros((6, 7), dtype=int)   # target x judged (col 6 = unattributed)
    # The gender endpoint does not identify a SEGMENT — three segments share the
    # target's gender by construction — so it gets its own counter rather than
    # being forced into the confusion matrix, and its chance line is 1/2.
    gender_counts = {"correct": 0, "wrong": 0, "unattributed": 0}
    with open(a.out / "answers.jsonl", "w") as f:
        for item, (wav, sr, bounds) in enumerate(strips):
            g = meta[meta.item_id == item + a.item_offset].sort_values("seg_idx")
            tr = g.transcript.tolist()
            seg_labels = g.seg_label.tolist() if "seg_label" in g.columns else None
            genders = g.gender.tolist() if "gender" in g.columns else None
            inputs = adapter.prepare(wav, sr, neutral)
            spans = adapter.segment_token_spans(inputs, bounds)
            P = int(inputs["input_ids"].shape[1])
            for t in range(6):
                if heads:
                    steer.set_bias(heads, spans, t, P, n_heads, a.B, device="cuda")
                else:
                    steer.clear()
                ans = adapter.generate(inputs, max_new=a.max_new)
                if a.judge == "emotion":
                    j, sc = attribute_emotion(ans, seg_labels)
                elif a.judge == "gender":
                    ok, pred = attribute_gender(ans, genders[t])
                    gender_counts["unattributed" if ok is None else
                                  ("correct" if ok else "wrong")] += 1
                    j, sc = -1, []
                else:
                    j, sc = attribute(ans, tr)
                if a.judge != "gender":
                    conf[t, j if j >= 0 else 6] += 1
                f.write(json.dumps(dict(item=item + a.item_offset, target=t, judged=j, answer=ans,
                                        scores=[round(x, 2) for x in sc])) + "\n")
                if a.condition == "none":
                    break   # unsteered: one generation per strip is enough
            steer.clear()
            if item % 5 == 4:
                if a.judge == "gender":
                    gn = sum(gender_counts.values())
                    print(f"  {item+1} items: acc {gender_counts['correct']/gn:.3f} "
                          f"unattributed {gender_counts['unattributed']/gn:.3f} (chance 0.5)", flush=True)
                else:
                    print(f"  {item+1} items: acc {np.trace(conf[:, :6]) / conf.sum():.3f} unattributed {conf[:,6].sum()/conf.sum():.3f}", flush=True)
    if a.judge == "gender":
        n = sum(gender_counts.values())
        rep = dict(model=a.model, condition=a.condition, k=len(heads), B=a.B,
                   n_items=len(strips), prompt=neutral, judge=a.judge, chance=chance,
                   heads=heads if len(heads) <= 500 else "all",
                   accuracy=float(gender_counts["correct"] / n),
                   unattributed_rate=float(gender_counts["unattributed"] / n),
                   gender_counts=gender_counts)
        (a.out / "steer_report.json").write_text(json.dumps(rep, indent=2))
        print(json.dumps({k: v for k, v in rep.items() if k != "heads"}, indent=2))
        return
    n = conf.sum()
    rep = dict(model=a.model, condition=a.condition, k=len(heads), B=a.B, n_items=len(strips),
               prompt=neutral, judge=a.judge, chance=chance,
               heads=heads if len(heads) <= 500 else "all",
               accuracy=float(np.trace(conf[:, :6]) / n),
               unattributed_rate=float(conf[:, 6].sum() / n),
               per_target_accuracy=[float(conf[t, t] / conf[t].sum()) if conf[t].sum() else None for t in range(6)],
               confusion_target_x_judged=conf.tolist(),
               judged_marginal=conf.sum(0).tolist())
    (a.out / "steer_report.json").write_text(json.dumps(rep, indent=2))
    print(json.dumps({k: v for k, v in rep.items() if k != "heads"}, indent=2))


if __name__ == "__main__":
    main()
