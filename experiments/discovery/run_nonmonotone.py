"""Non-monotone steering evaluation — the §2.2 "monotonicity trap".

If every steering demo is transcription, the result reads as a
rediscovery of ASR alignment heads (Whisper cross-attention + DTW), which is
already production technology. The claim only survives if steering also works
on tasks where the model must JUMP AROUND the audio rather than read it
left-to-right. Three such prompts here, none answerable by transcribing:

  compare   cross-segment contrast — must hold two distant places at once
  summarise select-and-abstract, not transcribe
  identify  search: "one speaker mentions {cue}" where the cue comes from the
            TARGET segment (steering and the prompt agree)
  override  the strong test: the cue comes from a DIFFERENT segment than the
            one we steer to, so the semantic pointer and the attention bias
            conflict. If the answer follows the bias, steering overrides
            content-based search — which no alignment-head account predicts.

Steering to segment t should move the answer to segment t under all three,
not just under a describe-this prompt. Scored by the same transcript-overlap
judge; chance 1/6.

    python run_nonmonotone.py --model qwen2-audio --scores <gaze.npz> --k 100 \
        --n-items 20 --item-offset 100 --out results/nonmono/qwen2_audio
"""
import argparse, json
from pathlib import Path

import numpy as np

import steer
from judge import attribute, _tokens
from run_discovery import load_strips, make_adapter

PROMPTS = {
    "compare": ("The audio contains six segments spoken by different people. "
                "The speakers talk about different things. Describe what ONE "
                "of them talks about, and say how it differs from the others."),
    "summarise": ("Six different people speak in this audio. In one sentence, "
                  "summarise what a single one of them discusses."),
    "identify": ("Six different people speak in this audio. One of them "
                 "mentions '{cue}'. What is that speaker talking about?"),
}


def pick_cue(transcript):
    """A distinctive content word from the target segment, for the identify task."""
    t = [w for w in _tokens(transcript) if len(w) > 5]
    return t[0] if t else (_tokens(transcript) or ["it"])[0]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--strips", type=Path, default=Path("../testbed/strips"))
    ap.add_argument("--model", default="qwen2-audio")
    ap.add_argument("--scores", type=Path, required=True)
    ap.add_argument("--k", type=int, default=100)
    ap.add_argument("--B", type=float, default=1e4)
    ap.add_argument("--n-items", type=int, default=20)
    ap.add_argument("--item-offset", type=int, default=0)
    ap.add_argument("--max-new", type=int, default=50)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--full-precision", action="store_true")
    a = ap.parse_args()
    import pandas as pd
    meta = pd.read_parquet(a.strips / "metadata.parquet")
    scores = np.load(a.scores)["scores"]
    adapter = make_adapter(a.model, load_4bit=not a.full_precision)
    strips = list(load_strips(a.strips, a.n_items, a.item_offset))
    if hasattr(adapter, "precompute"):
        adapter.precompute(strips)
    if hasattr(adapter, "_ensure_llm"):
        adapter._ensure_llm()
    backbone = getattr(adapter, "llm", None) or adapter.model
    # hook-equivalence check, ported from run_steer.py: with the bias OFF the
    # patched attention must reproduce stock eager. This assertion is what
    # caught both critical E2 bugs (missing causal mask; re-routing Qwen's
    # Whisper tower). A runner that installs the hook without checking it can
    # emit a complete set of silently wrong numbers.
    wav0, sr0, _ = strips[0]
    inp0 = adapter.prepare(wav0, sr0, PROMPTS["compare"])
    ref = adapter.forward_last_token_attn(inp0)
    n_layers, n_heads = steer.install(backbone)
    steer.clear()
    dmax = float(np.abs(ref - adapter.forward_last_token_attn(inp0)).max())
    print(f"[hook check] max |attn_ref - attn_hooked| = {dmax:.2e}", flush=True)
    assert dmax < 1e-2, "eager_bias with bias off must equal eager"
    assert (n_layers, n_heads) == scores.shape, ((n_layers, n_heads), scores.shape)
    heads = steer.top_k_heads(scores, a.k)
    a.out.mkdir(parents=True, exist_ok=True)
    def build_prompt(task, t, tr):
        """-> (prompt, cue_segment). identify points the prompt AT the target;
        override points it at a different segment so prompt and bias conflict."""
        if task == "identify":
            return PROMPTS["identify"].format(cue=pick_cue(tr[t])), t
        if task == "override":
            cs = (t + 3) % 6              # a fixed derangement, never the target
            return PROMPTS["identify"].format(cue=pick_cue(tr[cs])), cs
        return PROMPTS[task], None

    res = {}
    tasks = list(PROMPTS) + ["override"]
    with open(a.out / "answers.jsonl", "w") as f:
        for task in tasks:
            conf = np.zeros((6, 7), dtype=int)          # steered
            ctl = np.zeros((6, 7), dtype=int)           # unsteered control
            cue_wins = ctl_cue_wins = 0
            for item, (wav, sr, bounds) in enumerate(strips):
                tr = meta[meta.item_id == item + a.item_offset].sort_values("seg_idx").transcript.tolist()

                # UNSTEERED CONTROL, one target per item (item % 6 cycles all six
                # over 20 items). Two jobs, both needed to read this experiment:
                #   * compare/summarise: the judge's FLOOR. Their prompts invite
                #     mentioning other segments, which inflates rival scores and
                #     so raises ties -> attribute() returns -1. Without this
                #     control a judge artefact is indistinguishable from
                #     "steering fails on non-monotone tasks", and that is the
                #     direction that would wrongly kill the framing.
                #   * identify/override: does the cue work as a semantic pointer
                #     AT ALL? If the model cannot follow the cue unsteered, then
                #     "bias beats cue" is vacuous.
                t_ctl = item % 6
                pr, cue_seg = build_prompt(task, t_ctl, tr)
                inputs = adapter.prepare(wav, sr, pr)
                steer.clear()
                ans = adapter.generate(inputs, max_new=a.max_new)
                j, sc = attribute(ans, tr)
                ctl[t_ctl, j if j >= 0 else 6] += 1
                if cue_seg is not None and j == cue_seg:
                    ctl_cue_wins += 1
                f.write(json.dumps(dict(task=task, steered=False, item=item + a.item_offset,
                                        target=t_ctl, cue_segment=cue_seg, judged=j, answer=ans,
                                        scores=[round(x, 2) for x in sc])) + "\n")

                for t in range(6):
                    pr, cue_seg = build_prompt(task, t, tr)
                    inputs = adapter.prepare(wav, sr, pr)
                    spans = adapter.segment_token_spans(inputs, bounds)
                    steer.set_bias(heads, spans, t, int(inputs["input_ids"].shape[1]), n_heads, a.B)
                    ans = adapter.generate(inputs, max_new=a.max_new)
                    j, sc = attribute(ans, tr)
                    conf[t, j if j >= 0 else 6] += 1
                    if task == "override" and j == cue_seg:
                        cue_wins += 1
                    # per-segment judge scores are recorded so a tie can be told
                    # apart from a wrong answer without re-running the GPU work
                    f.write(json.dumps(dict(task=task, steered=True, item=item + a.item_offset,
                                            target=t, cue_segment=cue_seg, judged=j, answer=ans,
                                            scores=[round(x, 2) for x in sc])) + "\n")
                steer.clear()
            res[task] = dict(accuracy=float(np.trace(conf[:, :6]) / conf.sum()),
                             unattributed=float(conf[:, 6].sum() / conf.sum()),
                             confusion=conf.tolist(),
                             unsteered_accuracy=float(np.trace(ctl[:, :6]) / ctl.sum()),
                             unsteered_unattributed=float(ctl[:, 6].sum() / ctl.sum()),
                             unsteered_confusion=ctl.tolist())
            if task == "override":
                # bias_wins = answer follows the attention bias; cue_wins = it
                # follows the semantic pointer in the prompt instead
                res[task]["cue_wins"] = float(cue_wins / conf.sum())
                res[task]["bias_wins"] = res[task]["accuracy"]
                res[task]["unsteered_cue_wins"] = float(ctl_cue_wins / ctl.sum())
            if task == "identify":
                res[task]["unsteered_cue_wins"] = float(ctl_cue_wins / ctl.sum())
            print(f"[{a.model}] {task}: acc {res[task]['accuracy']:.3f} "
                  f"unattr {res[task]['unattributed']:.3f} | unsteered acc "
                  f"{res[task]['unsteered_accuracy']:.3f} unattr "
                  f"{res[task]['unsteered_unattributed']:.3f}", flush=True)
    (a.out / "report.json").write_text(json.dumps(
        dict(model=a.model, k=a.k, n_items=len(strips), item_offset=a.item_offset, tasks=res), indent=2))


if __name__ == "__main__":
    main()
