"""E1 runner — gaze-score discovery over the testbed.

    python run_discovery.py --strips ../testbed/strips --model qwen2-audio \
        --n-items 50 --out results/qwen2_audio
    python run_discovery.py --mock          # CPU-only harness validation
    python run_discovery.py --self-test --model qwen2-audio
        # loads the real model, runs ONE strip, checks audio-token count vs
        # duration * token rate — run this before burning GPU-hours.

Decision rule (fixed in advance in the project plan): a concentrated head set
exists iff the top heads separate cleanly from the bulk (heavy right tail)
AND real scores beat the shuffled-label baseline. Outputs: scores.npz,
report.json, curve.png.
"""

import argparse
import json
from pathlib import Path

import numpy as np

from core import run_discovery, concentration_report, bootstrap_summary

QUERY = ("The audio contains six segments spoken by different people, "
         "separated by silences. What does the speaker of segment {k} "
         "(counting from 1) talk about?")
# paraphrases for a prompt-robustness check (--prompt 1 / 2); 0 = the original
_ORD = {1: "first", 2: "second", 3: "third", 4: "fourth", 5: "fifth", 6: "sixth"}


class _OrdinalQuery(str):
    """QUERIES[1] needs an ordinal word; .format(k=3) -> 'third'."""
    def format(self, k):
        return str.format(self, k=_ORD[k])


QUERIES = [
    QUERY,
    _OrdinalQuery("This recording has six parts, each read by a different speaker, with "
                  "short pauses between them. Summarize what the {k} speaker says."),
    ("Six different people speak one after another in this audio. Tell me "
     "the topic of speaker number {k}."),
    # 3, 4: V15 prosody corpora. Every segment says the SAME sentence, so a
    # "what do they talk about" query is degenerate there — the question has to
    # be about the voice. Appended, so indices 0-2 and every run that used them
    # are untouched.
    ("The audio contains six segments, separated by silences. Each is a "
     "different speaker reading the same sentence. Describe the voice of the "
     "speaker in segment {k} (counting from 1): is it a man or a woman, and "
     "what does the voice sound like?"),
    ("The audio contains six segments, separated by silences. Each is the same "
     "sentence spoken with a different emotion. What emotion is the speaker "
     "expressing in segment {k} (counting from 1)?"),
]


def load_strips(strips_dir: Path, n_items, offset=0):
    """Items [offset, offset+n_items). offset exists so the confirmatory phase
    can run on strips the exploratory phase never touched: all work up to
    2026-08-17 used items 0-49, so items >= 50 are held out (see
    experiments/PREREGISTRATION.md)."""
    import pandas as pd
    import soundfile as sf
    meta = pd.read_parquet(strips_dir / "metadata.parquet")
    for item_id, g in meta.groupby("item_id"):
        if item_id < offset:
            continue
        if item_id >= offset + n_items:
            break
        wav, sr = sf.read(strips_dir / "audio" / f"strip_{item_id:05d}.wav")
        bounds = list(zip(g.sort_values("seg_idx").start_s, g.sort_values("seg_idx").end_s))
        yield wav, sr, bounds


def make_adapter(name, load_4bit):
    if name == "qwen2-audio":
        from adapters.qwen2_audio import Qwen2AudioAdapter
        return Qwen2AudioAdapter(load_4bit=load_4bit)
    if name == "salmonn":
        from adapters.salmonn import SalmonnAdapter
        return SalmonnAdapter(load_4bit=load_4bit)
    if name == "ultravox":
        from adapters.ultravox import UltravoxAdapter
        return UltravoxAdapter(load_4bit=load_4bit)
    raise SystemExit(f"no adapter for {name} yet (see adapters/)")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--strips", type=Path, default=Path("../testbed/strips"))
    ap.add_argument("--model", default="qwen2-audio")
    ap.add_argument("--n-items", type=int, default=50)
    ap.add_argument("--item-offset", type=int, default=0,
                    help="first strip index; >=50 selects held-out data")
    ap.add_argument("--out", type=Path, default=Path("results/run"))
    ap.add_argument("--full-precision", action="store_true")
    ap.add_argument("--mock", action="store_true")
    ap.add_argument("--self-test", action="store_true")
    ap.add_argument("--prompt", type=int, default=0, help="index into QUERIES (0 = original)")
    a = ap.parse_args()
    global QUERY
    QUERY = QUERIES[a.prompt]

    if a.mock:
        from test_mock import run as mock_run
        return mock_run()

    adapter = make_adapter(a.model, load_4bit=not a.full_precision)
    strips = list(load_strips(a.strips, 1 if a.self_test else a.n_items, a.item_offset))

    if a.self_test:
        # Lesson from the v1 run: a loose count tolerance passed spans that were
        # wrong (33.5s strip, 30s encoder window -> 750 tokens, 10% off, still
        # "PASS"). Check three things instead: (1) the strip fits the encoder
        # window, (2) the implied token rate on a 10s excerpt matches the
        # adapter's stated rate within 2%, (3) full-strip count matches too.
        wav, sr, bounds = strips[0]
        dur = bounds[-1][1]
        window = getattr(adapter, "encoder_window_s", None)
        checks = {}
        if window is not None:
            checks[f"strip {dur:.1f}s fits encoder window {window:.0f}s"] = dur < window
        excerpt = wav[: int(10.0 * sr)]
        inp10 = adapter.prepare(excerpt, sr, QUERY.format(k=1))
        span10 = adapter.segment_token_spans(inp10, [(0.0, 10.0)])
        n10 = span10[0][1] - span10[0][0]
        rate = n10 / 10.0
        checks[f"implied rate {rate:.2f} tok/s vs stated {adapter.tokens_per_second}"] = \
            abs(rate - adapter.tokens_per_second) / adapter.tokens_per_second < 0.02
        inputs = adapter.prepare(wav, sr, QUERY.format(k=1))
        spans = adapter.segment_token_spans(inputs, bounds)
        n_full = spans[-1][1] - spans[0][0]
        exp_full = dur * adapter.tokens_per_second
        checks[f"full-strip tokens {n_full} vs expected {exp_full:.0f}"] = \
            abs(n_full - exp_full) / exp_full < 0.02
        for name, ok in checks.items():
            print(("PASS" if ok else "FAIL"), "-", name)
        print("SELF-TEST", "PASS" if all(checks.values())
              else "FAIL — do not run discovery; fix testbed length or adapter span logic")
        return

    if hasattr(adapter, "precompute"):
        # SALMONN: encoders and the 13B LLM cannot share an 8GB card — encode
        # every strip first, free the encoders, then let the LLM load lazily.
        adapter.precompute(strips)
    real, shuf = run_discovery(
        adapter, strips, lambda k: QUERY.format(k=k + 1), shuffle_baseline_seed=0
    )
    a.out.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(a.out / "scores.npz", scores=real.scores, matrices=real.matrices,
                        shuffled_scores=shuf.scores, shuffled_matrices=shuf.matrices,
                        raw_scores=real.raw_scores, raw_matrices=real.raw_matrices,
                        shuffled_raw_scores=shuf.raw_scores)
    # per-item matrices are large (7-20MB) and reproducible in minutes: kept out of git
    np.savez_compressed(a.out / "per_item.npz", per_item=real.per_item, per_item_shuffled=shuf.per_item)
    rep = concentration_report(real.scores)
    rep_shuf = concentration_report(shuf.scores)
    rep_raw = concentration_report(real.raw_scores)
    rep_raw_shuf = concentration_report(shuf.raw_scores)
    boot = bootstrap_summary(real.per_item, shuf.per_item)
    summary = {
        "model": a.model, "n_items": real.n_items, "item_offset": a.item_offset,
        "prompt": a.prompt, "query": QUERY,
        "bootstrap_95ci": boot,
        "top100_mean": rep["top100_mean"], "median": rep["median"],
        "shuffled_top100_mean": rep_shuf["top100_mean"],
        "separation_ratio": rep["top100_mean"] / max(rep["median"], 1e-9),
        "real_vs_shuffled_top100": rep["top100_mean"] / max(rep_shuf["top100_mean"], 1e-9),
        "top100_layer_hist": rep["top100_layer_hist"].tolist(),
        # seed-paper definition (un-normalized): the criterion whose flatness is
        # used to declare a head set absent, reported for like-for-like comparison
        "raw_top100_mean": rep_raw["top100_mean"], "raw_median": rep_raw["median"],
        "raw_separation_ratio": rep_raw["top100_mean"] / max(rep_raw["median"], 1e-9),
        "raw_real_vs_shuffled_top100": rep_raw["top100_mean"] / max(rep_raw_shuf["top100_mean"], 1e-9),
    }
    (a.out / "report.json").write_text(json.dumps(summary, indent=2))
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots(figsize=(6, 4))
        ax.plot(rep["curve"], label="real")
        ax.plot(rep_shuf["curve"], label="shuffled-label baseline")
        ax.set(xlabel="head rank", ylabel="gaze score", title=f"{a.model}: concentration curve")
        ax.legend(); fig.tight_layout(); fig.savefig(a.out / "curve.png", dpi=150)
    except ImportError:
        pass
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
