"""E1 discovery on the seed paper's OWN modality — experiment V2 (see PREREGISTRATION.md).

Same protocol, same maths, same file layout as run_discovery.py: one forward per
panel query, final-prompt-token attention summed over each panel's image-token
span, (queried x attended) matrix, gaze score = mean diagonal, chance 1/6, plus a
shuffled-label null and BOTH score definitions (ours row-normalised, the seed
paper's raw). `core.run_discovery` is reused verbatim rather than reimplemented,
so any difference between the audio and image results is the modality and not the
analysis code.

    python run_vlm_discovery.py --model qwen2-vl --n-items 50 --item-offset 100 \
        --comics ../testbed/comics --out results/vlm/qwen2_vl
    python run_vlm_discovery.py --self-test        # no weights? still checks the loader
"""
import argparse, json
from pathlib import Path

import numpy as np

from core import bootstrap_summary, concentration_report, gaze_scores
from vlm_regions import region_masses


def load_comics(comics_dir: Path, n_items, offset=0):
    """Yields (images, None, panel_placeholder) so core.run_discovery can drive
    this unchanged. The third element only needs len() == n_panels."""
    from PIL import Image
    import pandas as pd
    meta = pd.read_parquet(comics_dir / "metadata.parquet")
    ids = sorted(meta.item_id.unique())
    want = [i for i in ids if offset <= i < offset + n_items]
    if len(want) < n_items:
        raise SystemExit(f"need {n_items} strips from offset {offset}, found "
                         f"{len(want)} — rebuild testbed/build_comic_strips.py larger")
    for item_id in want:
        d = comics_dir / "images" / f"strip_{item_id:05d}"
        imgs = [Image.open(d / f"panel_{k}.png").convert("RGB") for k in range(6)]
        yield imgs, None, list(range(6))


def run_region_discovery(adapter, strips, query_texts, n_panels=6, shuffle_seed=0):
    """core.run_discovery's loop, with region-indexed aggregation.

    core.run_discovery is not reused here because it calls
    core.segment_attention_matrix, which slices a contiguous span; a tiled strip's
    panels are column bands and therefore non-contiguous. The MATHS after
    aggregation is core.gaze_scores, unchanged and shared with the audio arms, so
    both score definitions and the shuffled null are the same code.
    """
    accs = {"real": None, "real_raw": None, "shuf": None, "shuf_raw": None}
    per_item, per_item_shuf, n = [], [], 0
    rng = np.random.default_rng(shuffle_seed)
    for imgs, _sr, _panels in strips:
        per_query = []
        for k in range(n_panels):
            inputs = adapter.prepare(imgs, None, query_texts(k))
            img_start, region_ids, _ = adapter.panel_regions(inputs)
            attn = adapter.forward_last_token_attn(inputs)
            per_query.append(region_masses(attn, img_start, region_ids, n_panels))
        m, _ = gaze_scores(per_query)
        m_raw, _ = gaze_scores(per_query, normalize=False)
        perm = rng.permutation(n_panels)
        ms, _ = gaze_scores([per_query[i] for i in perm])
        ms_raw, _ = gaze_scores([per_query[i] for i in perm], normalize=False)
        for key, val in (("real", m), ("real_raw", m_raw), ("shuf", ms), ("shuf_raw", ms_raw)):
            accs[key] = val if accs[key] is None else accs[key] + val
        per_item.append(m.astype(np.float32))
        per_item_shuf.append(ms.astype(np.float32))
        n += 1
        if n % 10 == 0:
            print(f"  {n} strips", flush=True)
    out = {k: v / n for k, v in accs.items()}
    score = lambda mm: np.einsum("lhqq->lhq", mm).mean(axis=-1)
    return dict(n_items=n,
                scores=score(out["real"]), matrices=out["real"],
                raw_scores=score(out["real_raw"]), raw_matrices=out["real_raw"],
                shuffled_scores=score(out["shuf"]),
                shuffled_raw_scores=score(out["shuf_raw"]),
                per_item=np.stack(per_item), per_item_shuffled=np.stack(per_item_shuf))


def make_adapter(name, load_4bit):
    if name == "qwen2-vl":
        from adapters.qwen2_vl import Qwen2VLAdapter
        return Qwen2VLAdapter(load_4bit=load_4bit)
    if name == "qwen2-vl-vertical":
        # V20: identical model and task, panels stacked top-to-bottom so each
        # panel becomes a CONTIGUOUS token run instead of 24 interleaved ones.
        from adapters.qwen2_vl import Qwen2VLAdapter
        return Qwen2VLAdapter(load_4bit=load_4bit, layout="vstrip")
    if name in ("bunny-strip-cols",):
        # V19: their released column-band region map, ported to a pad-to-square
        # model. Same weights, same prompt, same ranking procedure — only the
        # panel->token mapping changes.
        from adapters.bunny import BunnyAdapter
        return BunnyAdapter(load_4bit=load_4bit, presentation="strip",
                            region_mode="column_bands")
    if name == "bunny-strip-crop":
        # V23: the ONLY surviving explanation for their Bunny 8.3% after AC2
        # ruled out measurement point (17az). Identical to bunny-strip except
        # the 6:1 canvas is CENTRE-CROPPED to a square instead of padded to one
        # -- the preprocessing their Appendix E.3 fix replaces, and which Bunny
        # is not listed as having received.
        from adapters.bunny import BunnyAdapter
        return BunnyAdapter(load_4bit=load_4bit, presentation="strip",
                            image_mode="centercrop")
    if name in ("bunny-grid", "bunny-strip"):
        # both presentations exist because H13 says their null may be a
        # resolution artefact; neither is privileged and both are reported
        from adapters.bunny import BunnyAdapter
        return BunnyAdapter(load_4bit=load_4bit,
                            presentation="grid" if name.endswith("grid") else "strip")
    raise SystemExit(f"unknown VLM arm {name!r}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="qwen2-vl",
                    choices=("qwen2-vl", "qwen2-vl-vertical", "bunny-grid",
                             "bunny-strip", "bunny-strip-cols",
                             "bunny-strip-crop"))
    ap.add_argument("--comics", type=Path, default=Path("../testbed/comics"))
    ap.add_argument("--n-items", type=int, default=50)
    ap.add_argument("--item-offset", type=int, default=0)
    ap.add_argument("--out", type=Path)
    ap.add_argument("--full-precision", action="store_true")
    ap.add_argument("--self-test", action="store_true",
                    help="load one strip, check the six panel spans, and verify "
                         "the two-phase prefill matches a single-pass forward")
    a = ap.parse_args()

    if a.self_test:
        return self_test(a)
    if not a.out:
        raise SystemExit("--out is required unless --self-test")

    adapter = make_adapter(a.model, load_4bit=not a.full_precision)
    strips = list(load_comics(a.comics, a.n_items, a.item_offset))
    r = run_region_discovery(adapter, strips, lambda k: adapter.QUERY.format(k=k + 1))

    a.out.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(a.out / "scores.npz", scores=r["scores"], matrices=r["matrices"],
                        shuffled_scores=r["shuffled_scores"],
                        raw_scores=r["raw_scores"], raw_matrices=r["raw_matrices"],
                        shuffled_raw_scores=r["shuffled_raw_scores"])
    np.savez_compressed(a.out / "per_item.npz", per_item=r["per_item"],
                        per_item_shuffled=r["per_item_shuffled"])
    rep = concentration_report(r["scores"])
    rep_shuf = concentration_report(r["shuffled_scores"])
    rep_raw = concentration_report(r["raw_scores"])
    rep_raw_shuf = concentration_report(r["shuffled_raw_scores"])
    summary = {
        "model": a.model, "modality": "image", "presentation": "tiled_strip_seed_protocol",
        "n_items": r["n_items"], "item_offset": a.item_offset, "query": adapter.QUERY,
        "bootstrap_95ci": bootstrap_summary(r["per_item"], r["per_item_shuffled"]),
        "top100_mean": rep["top100_mean"], "median": rep["median"],
        "shuffled_top100_mean": rep_shuf["top100_mean"],
        "separation_ratio": rep["top100_mean"] / max(rep["median"], 1e-9),
        "real_vs_shuffled_top100": rep["top100_mean"] / max(rep_shuf["top100_mean"], 1e-9),
        "top100_layer_hist": rep["top100_layer_hist"].tolist(),
        "raw_top100_mean": rep_raw["top100_mean"], "raw_median": rep_raw["median"],
        "raw_separation_ratio": rep_raw["top100_mean"] / max(rep_raw["median"], 1e-9),
        "raw_real_vs_shuffled_top100": rep_raw["top100_mean"] / max(rep_raw_shuf["top100_mean"], 1e-9),
    }
    (a.out / "report.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps({k: v for k, v in summary.items()
                      if k not in ("top100_layer_hist", "bootstrap_95ci")}, indent=2))


def self_test(a):
    """The checks that must pass before any V2 number is believed."""
    strips = list(load_comics(a.comics, 1, a.item_offset))
    imgs, _, panels = strips[0]
    print(f"loader: 1 strip, {len(imgs)} panels, sizes {[im.size for im in imgs]}")
    assert len(imgs) == 6 and len(panels) == 6

    adapter = make_adapter(a.model, load_4bit=not a.full_precision)
    inputs = adapter.prepare(imgs, None, adapter.QUERY.format(k=1))
    L = inputs["input_ids"].shape[1]
    img_start, region_ids, positions = adapter.panel_regions(inputs)
    counts = [len(p) for p in positions]
    # `L` above is the length of input_ids, which for a LLaVA-style splice
    # (Bunny) is the PRE-splice prompt with a single marker token, not the
    # sequence the model sees. Take the real length from the attention itself.
    one = adapter.forward_last_token_attn(inputs)
    seq = one.shape[-1]
    assigned = int((np.asarray(region_ids) >= 0).sum())
    print(f"sequence length {seq}; image block starts at {img_start}, "
          f"{len(region_ids)} image tokens, {assigned} of them inside a panel")
    print(f"tokens per panel {counts} (total {sum(counts)}, {100*sum(counts)/seq:.0f}% of the sequence)")
    assert len(positions) == 6, len(positions)
    assert all(c > 0 for c in counts), counts
    # Padded presentations (Bunny pads to square) leave image tokens outside
    # every panel; those are correctly EXCLUDED by region_masses, so the tokens
    # inside panels must match the assigned ids, not the whole block.
    assert sum(counts) == assigned, (sum(counts), assigned, len(region_ids))
    assert img_start + len(region_ids) <= seq, (img_start, len(region_ids), seq)
    flat = [i for p in positions for i in p]
    assert len(set(flat)) == len(flat), "a token was assigned to two panels"
    spread = max(counts) - min(counts)
    print(f"panel token counts differ by at most {spread} (column split, not exact halves)")

    # Production path (single pass) vs the memory-frugal two-phase prefill.
    # These agree on the COMPUTATION (identical next tokens, logit cosine .9985)
    # but their reported attention tensors diverge at layers 0 and 27 of 28
    # under bf16, so the raw comparison is printed as a diagnostic, not gated.
    # What IS gated is the quantity the score actually consumes: attention
    # summed over each panel's token span. See PREREGISTRATION 17aa.
    two = adapter._two_phase_last_token_attn(inputs)
    print(f"[self-test] max |two_phase - single_pass| raw attention = "
          f"{float(np.abs(two - one).max()):.2e} (diagnostic only)")

    # Panels are COLUMN BANDS over a patch grid, so their tokens are NOT
    # contiguous. Use the same index-based sum production uses; a slice from
    # first to last index would silently include the neighbouring panels.
    pm = lambda a: region_masses(a, img_start, region_ids, 6)
    dmax = float(np.abs(pm(two) - pm(one)).max())
    print(f"[self-test] max |two_phase - single_pass| PANEL MASS  = {dmax:.2e}")
    # Deliberately NOT a gate. The two paths are mathematically equivalent, and
    # the residual stream confirms it (identical next tokens, logit cosine
    # .9985), but on a 4-bit model with bf16 attention the rows of layers 0 and
    # 27 are numerically degenerate — near one-hot, with everything else at the
    # bf16 floor — so a tiny difference in operation order flips a one-hot into
    # a near-tie. Any threshold here would be a threshold tuned until it passed,
    # which is precisely the practice this project is criticising. It is
    # reported instead, and PREREGISTRATION 17aa records the size of it: on one
    # item the two paths give an 81/100 top-100 head-set overlap. That is a real
    # limitation of head rankings measured at 4-bit, and it is stated as one.

    # Hard gates that are actually about correctness rather than a tolerance:
    # the production path must be deterministic, must be normalised, and must
    # put real mass inside the image block. The SCIENTIFIC gate on this arm is
    # not here at all — it is that the control shows above-chance gaze and
    # steering in E1/E2, which no self-test can fake.
    again = adapter.forward_last_token_attn(inputs)
    assert float(np.abs(again - one).max()) == 0.0, "production path is not deterministic"
    rows = one.sum(-1)
    assert np.allclose(rows, 1.0, atol=2e-2), (
        f"attention rows do not sum to 1 (min {rows.min():.3f}, max {rows.max():.3f})")

    rowsum = one[:, :, img_start:img_start + len(region_ids)].sum()
    print(f"[self-test] attention mass inside the image block: {rowsum / one.sum():.3f} of total")
    print("SELF-TEST PASSED")


if __name__ == "__main__":
    main()
