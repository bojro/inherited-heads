"""Do steer.py and steer_prehook.py apply the SAME intervention?

`steer.py` registers `eager_bias_forward` into transformers' attention interface.
`steer_prehook.py` adds the bias to the `attention_mask` kwarg instead, because
Bunny's vendored PhiAttention never reaches that interface. Both are meant to be
"+B on the target region's key positions, -B on the others, pre-softmax, per
head" — but that was an argument, not a measurement, and every Bunny steering
number depends on it.

This runs both on the SAME small model where both mechanisms work, on CPU, so it
needs no GPU and cannot disturb a running suite. Any Qwen2-architecture model
qualifies: it routes through ALL_ATTENTION_FUNCTIONS and exposes
num_key_value_groups.

    python verify_steer_equivalence.py                     # Qwen/Qwen2.5-0.5B, cpu
    python verify_steer_equivalence.py --model <id> --device cuda
"""
import argparse

import numpy as np
import torch

import steer
import steer_prehook


def last_token_attn(model, ids, device):
    with torch.no_grad():
        out = model(input_ids=ids, attention_mask=torch.ones_like(ids),
                    output_attentions=True, use_cache=False)
    return torch.stack([a[0, :, -1, :] for a in out.attentions]).float().cpu().numpy()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Qwen/Qwen2.5-0.5B")
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--n-tokens", type=int, default=48)
    ap.add_argument("--tol", type=float, default=1e-4)
    a = ap.parse_args()

    from transformers import AutoModelForCausalLM, AutoTokenizer
    tok = AutoTokenizer.from_pretrained(a.model)
    model = AutoModelForCausalLM.from_pretrained(
        a.model, attn_implementation="eager", dtype=torch.float32).eval().to(a.device)

    text = ("Alpha beta gamma delta epsilon zeta eta theta iota kappa lambda mu "
            "nu xi omicron pi rho sigma tau upsilon phi chi psi omega. Summarise.")
    ids = tok(text, return_tensors="pt").input_ids[:, :a.n_tokens].to(a.device)
    L = ids.shape[1]

    # six contiguous "regions" over the middle of the sequence, mimicking the
    # six segments/panels of the real task
    r0, per = 4, (L - 8) // 6
    spans = [(r0 + k * per, r0 + (k + 1) * per) for k in range(6)]
    positions = [list(range(s, e)) for s, e in spans]
    target, B = 2, 1e4

    base = last_token_attn(model, ids, a.device)

    # ---- mechanism B first: pre-hook on a stock eager model ----
    layers = steer_prehook.find_layers_generic(model)
    n_layers, n_heads = steer_prehook.install(layers, model.config.num_attention_heads)
    heads = [(0, 0), (1, 3), (n_layers - 1, n_heads - 1), (n_layers // 2, 1)]
    steer_prehook.set_bias_regions(heads, positions, target, L, n_heads, B, device=a.device)
    attn_prehook = last_token_attn(model, ids, a.device)
    steer_prehook.clear(); steer_prehook.remove()

    # ---- mechanism A: the attention-interface hook ----
    n_layers_a, n_heads_a = steer.install(model)
    assert (n_layers_a, n_heads_a) == (n_layers, n_heads), (
        (n_layers_a, n_heads_a), (n_layers, n_heads))
    steer.set_bias(heads, spans, target, L, n_heads, B, device=a.device)
    attn_interface = last_token_attn(model, ids, a.device)
    steer.clear()
    layers[0].self_attn.config._attn_implementation = "eager"     # undo install

    d_ab = float(np.abs(attn_interface - attn_prehook).max())
    moved_a = float(np.abs(attn_interface - base).max())
    moved_b = float(np.abs(attn_prehook - base).max())

    print(f"model {a.model} on {a.device}: {n_layers} layers x {n_heads} heads, seq {L}")
    print(f"biased heads {heads}, target region {target}, B={B:g}")
    print()
    print(f"  interface hook vs baseline : max |diff| = {moved_a:.4f}   (must be >0)")
    print(f"  pre-hook       vs baseline : max |diff| = {moved_b:.4f}   (must be >0)")
    print(f"  interface hook vs PRE-HOOK : max |diff| = {d_ab:.3e}   (must be ~0)")
    print()
    ok = moved_a > 1e-3 and moved_b > 1e-3 and d_ab < a.tol
    if not ok:
        if moved_b <= 1e-3:
            print("PRE-HOOK DID NOTHING — attention_mask was probably None at the "
                  "hook, so the bias had nothing to add onto. steer_prehook needs a "
                  "fallback that CREATES a mask.")
        elif d_ab >= a.tol:
            print("MECHANISMS DISAGREE — they are not the same intervention. No Bunny "
                  "steering number may be compared against the audio arms.")
    print("EQUIVALENCE:", "PASS" if ok else "FAIL")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
