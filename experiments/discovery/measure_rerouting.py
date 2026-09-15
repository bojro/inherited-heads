"""Attention-rerouting cost of the steering bias — the measurement SKOP asks for.

WHY THIS EXISTS. Our intervention is an additive pre-softmax logit bias. SKOP
(OpenReview BYTQ4nQRha) shows that steering of exactly this form reroutes
attention away from the small "focus set" each head relies on, with severity
rising monotonically in steering strength, and that rerouting is NOT separable
from efficacy. So the objection to us is not that the gain is an artefact --- for
a logit-space edit rerouting IS the mechanism --- it is that we report no
measurement of what the bias COSTS the tokens the heads were reading. This
computes that cost.

WHAT IS MEASURED, and the two deviations from SKOP, both deliberate:

  focus set H(l,h): the minimal set of key positions covering tau=0.8 of that
  head's attention mass ON THE CLEAN RUN. SKOP's tau_high=0.8 is reused so our
  numbers sit beside theirs.

  DM = (steered mass on H) - (clean mass on H), per (head, item, target).
  Reported as Pr(DM <= -x) for x in {0.10, 0.15, 0.25} -- SKOP's own thresholds,
  so our row is comparable to their 31 / 22 / 10 %.

  DEVIATION 1: SKOP aggregates over (head, DECODING STEP). We measure at the
  final prompt token during prefill -- the same measurement point the gaze score
  itself uses. Aggregation is therefore over (head, item, target).
  DEVIATION 2: SKOP builds H from a held-out utility corpus. We build it from
  the clean run of the same item. Both are disclosed in the paper.

Also computed, because a bare DM cannot separate intended redistribution from
collateral damage: a three-way decomposition of where the target segment's
GAINED mass came from -- the other five segments (intended), the prompt/
instruction text (collateral), and position 0, the attention sink (free).

The control matters as much as the number: the same measurement runs on a
layer-histogram-matched random head set, so we can say whether the cost is
specific to the discovered heads or is what any K heads would incur.

  python discovery/measure_rerouting.py --model ultravox \
      --scores results/ultravox/scores.npz --k 100 \
      --B 1 2 5 10 10000 --n-items 20 --item-offset 100 \
      --out results/rerouting/ultravox
"""
import argparse, json
from pathlib import Path

import numpy as np

import steer
from run_discovery import load_strips, make_adapter

NEUTRAL = ("The audio contains six segments spoken by different people, "
           "separated by silences. Describe what one of the speakers talks "
           "about, in one sentence.")


def focus_set(row, tau=0.8):
    """Minimal key-index set covering tau of this head's attention mass."""
    order = np.argsort(row)[::-1]
    csum = np.cumsum(row[order])
    n = int(np.searchsorted(csum, tau * row.sum()) + 1)
    return order[:n]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--strips", type=Path, default=Path("../testbed/strips"))
    ap.add_argument("--model", default="ultravox")
    ap.add_argument("--scores", type=Path, required=True)
    ap.add_argument("--k", type=int, default=100)
    ap.add_argument("--B", type=float, nargs="+", default=[1, 2, 5, 10, 1e4])
    ap.add_argument("--n-items", type=int, default=20)
    ap.add_argument("--item-offset", type=int, default=100, help=">=100 is held-out")
    ap.add_argument("--tau", type=float, default=0.8, help="SKOP's tau_high")
    ap.add_argument("--control-heads", type=Path, default=None,
                    help="steer_report.json whose 'heads' field is the "
                         "layer-histogram-matched random set to reuse verbatim")
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--full-precision", action="store_true")
    a = ap.parse_args()

    scores = np.load(a.scores)["scores"]
    adapter = make_adapter(a.model, load_4bit=not a.full_precision)
    strips = list(load_strips(a.strips, a.n_items, a.item_offset))
    if hasattr(adapter, "precompute"):
        adapter.precompute(strips)
    if hasattr(adapter, "_ensure_llm"):
        adapter._ensure_llm()
    backbone = getattr(adapter, "llm", None) or adapter.model
    n_layers, n_heads = steer.install(backbone)
    steer.clear()
    assert (n_layers, n_heads) == scores.shape, ((n_layers, n_heads), scores.shape)

    top = steer.top_k_heads(scores, a.k)
    sets = {"top": top}
    if a.control_heads:
        ctrl = [tuple(h) for h in json.loads(a.control_heads.read_text())["heads"]]
        assert len(ctrl) == len(top), (len(ctrl), len(top))
        sets["matched_random"] = ctrl
    print(f"[{a.model}] head sets: { {k: len(v) for k, v in sets.items()} } "
          f"B={a.B} items={len(strips)} tau={a.tau}", flush=True)

    a.out.mkdir(parents=True, exist_ok=True)
    rows = []
    for item, (wav, sr, bounds) in enumerate(strips):
        inputs = adapter.prepare(wav, sr, NEUTRAL)
        spans = adapter.segment_token_spans(inputs, bounds)
        P = int(inputs["input_ids"].shape[1])
        steer.clear()
        A_clean = adapter.forward_last_token_attn(inputs)        # [L, H, P]
        seg_idx = [np.arange(s, e) for (s, e) in spans]
        in_any = np.zeros(A_clean.shape[-1], dtype=bool)
        for ix in seg_idx:
            in_any[ix] = True
        text_idx = np.where(~in_any)[0]
        text_idx = text_idx[text_idx != 0]                       # exclude the sink

        # focus sets come from the CLEAN run, once per item, per head
        F = {name: [focus_set(A_clean[l, h], a.tau) for (l, h) in hs]
             for name, hs in sets.items()}

        for name, hs in sets.items():
            for B in a.B:
                for t in range(6):
                    steer.set_bias(hs, spans, t, P, n_heads, B, device="cuda")
                    A_st = adapter.forward_last_token_attn(inputs)
                    for i, (l, h) in enumerate(hs):
                        Hs = F[name][i]
                        dm = float(A_st[l, h][Hs].sum() - A_clean[l, h][Hs].sum())
                        gain = float(A_st[l, h][seg_idx[t]].sum() - A_clean[l, h][seg_idx[t]].sum())
                        others = [j for j in range(6) if j != t]
                        from_seg = float(sum(A_clean[l, h][seg_idx[j]].sum()
                                             - A_st[l, h][seg_idx[j]].sum() for j in others))
                        from_txt = float(A_clean[l, h][text_idx].sum() - A_st[l, h][text_idx].sum())
                        from_sink = float(A_clean[l, h][0] - A_st[l, h][0])
                        rows.append(dict(item=item + a.item_offset, set=name, B=B, target=t,
                                         layer=int(l), head=int(h), n_focus=int(len(Hs)),
                                         dm=dm, target_gain=gain, from_segments=from_seg,
                                         from_text=from_txt, from_sink=from_sink))
                    steer.clear()
        if item % 2 == 1:
            print(f"  {item+1}/{len(strips)} items", flush=True)

    with open(a.out / "rerouting.jsonl", "w") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")

    # summary at SKOP's thresholds
    summ = {}
    for name in sets:
        for B in a.B:
            sel = [r for r in rows if r["set"] == name and r["B"] == B]
            dm = np.array([r["dm"] for r in sel])
            gain = np.array([r["target_gain"] for r in sel])
            src = np.array([[r["from_segments"], r["from_text"], r["from_sink"]] for r in sel])
            tot = src.sum(0)
            summ[f"{name}|B={B:g}"] = {
                "n_pairs": len(sel),
                "mean_focus_set_size": float(np.mean([r["n_focus"] for r in sel])),
                # SKOP's own thresholds, so our row sits beside their 31 / 22 / 10 %
                "pr_dm_le_-0.10": float((dm <= -0.10).mean()),
                "pr_dm_le_-0.15": float((dm <= -0.15).mean()),
                "pr_dm_le_-0.25": float((dm <= -0.25).mean()),
                "mean_dm": float(dm.mean()),
                "median_dm": float(np.median(dm)),
                "mean_target_gain": float(gain.mean()),
                "gain_source_share": {
                    "segments": float(tot[0] / tot.sum()) if tot.sum() else None,
                    "text": float(tot[1] / tot.sum()) if tot.sum() else None,
                    "sink": float(tot[2] / tot.sum()) if tot.sum() else None},
            }
    rep = dict(model=a.model, k=a.k, tau=a.tau, n_items=len(strips),
               item_offset=a.item_offset, B_sweep=a.B,
               measurement_point="final prompt token (prefill)",
               deviations_from_skop=["aggregated over (head,item,target) not (head,decoding step)",
                                     "focus set from the clean run of the same item, not a held-out corpus"],
               summary=summ)
    (a.out / "rerouting_report.json").write_text(json.dumps(rep, indent=2))
    print(json.dumps(summ, indent=2))


if __name__ == "__main__":
    main()
