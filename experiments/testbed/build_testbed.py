"""E0 — build the 'audio comic strips' testbed.

500 items, each 6 concatenated ~5s LibriSpeech utterances from 6 distinct
speakers, separated by silence. Segment boundaries are exact by construction
and recorded in the metadata, so no forced alignment is needed for E1.

Usage (needs ~2GB disk for LibriSpeech test-clean + dev-clean and ~500MB output):
    python build_testbed.py --out ./strips --n-items 500 --seed 0

Output:
    strips/audio/strip_00000.wav ... (16 kHz mono PCM16)
    strips/metadata.parquet  — one row per segment:
        item_id, seg_idx, speaker_id, ls_utterance_id, transcript,
        start_s, end_s, start_sample, end_sample
    strips/datasheet.md
"""

import argparse
import io
import random
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
import soundfile as sf
from datasets import load_dataset, concatenate_datasets

SR = 16_000
# Whisper-family encoders (Qwen2-Audio, SALMONN, Ultravox) see a 30s window
# and silently truncate beyond it, so a strip must total < 30s:
# 6 x 4.5s + 5 x 0.5s = 29.5s.
SEG_SECONDS = 4.5          # target segment length (crop; skip shorter utterances)
SILENCE_SECONDS = 0.5      # separator between segments
N_SEGMENTS = 6
MAX_STRIP_SECONDS = 30.0


def _split(split):
    # Hit the per-split parquet directly: the "clean" config builder otherwise
    # materializes train.100 + train.360 (~29GB) before returning ~2GB.
    return load_dataset(
        "parquet",
        data_files={split: f"hf://datasets/openslr/librispeech_asr@refs%2Fconvert%2Fparquet/clean/{split}/*.parquet"},
        split=split,
    )


def load_pool():
    """LibriSpeech test-clean + dev-clean: ~5.4h+5.4h, 40+40 speakers."""
    parts = [_split("test"), _split("validation")]
    ds = concatenate_datasets(parts)
    by_speaker = defaultdict(list)
    for i, ex in enumerate(ds):
        if len(ex["audio"]["array"]) >= int(SEG_SECONDS * SR):
            by_speaker[ex["speaker_id"]].append(i)
    return ds, {s: idxs for s, idxs in by_speaker.items() if len(idxs) >= 3}


def crop_segment(audio: np.ndarray, rng: random.Random) -> np.ndarray:
    """Random 5s crop, snapped to start to keep speech onset natural."""
    need = int(SEG_SECONDS * SR)
    start = rng.randint(0, max(0, len(audio) - need))
    seg = audio[start : start + need].astype(np.float32)
    # short fades to avoid clicks at concatenation joints
    fade = int(0.01 * SR)
    ramp = np.linspace(0.0, 1.0, fade, dtype=np.float32)
    seg[:fade] *= ramp
    seg[-fade:] *= ramp[::-1]
    return seg


def build(out: Path, n_items: int, seed: int):
    total = N_SEGMENTS * SEG_SECONDS + (N_SEGMENTS - 1) * SILENCE_SECONDS
    if total >= MAX_STRIP_SECONDS:
        raise SystemExit(f"strip would be {total}s; must be < {MAX_STRIP_SECONDS}s encoder window")
    rng = random.Random(seed)
    ds, by_speaker = load_pool()
    speakers = sorted(by_speaker)
    if len(speakers) < N_SEGMENTS:
        raise SystemExit(f"only {len(speakers)} usable speakers")

    (out / "audio").mkdir(parents=True, exist_ok=True)
    silence = np.zeros(int(SILENCE_SECONDS * SR), dtype=np.float32)
    rows = []

    for item in range(n_items):
        chosen = rng.sample(speakers, N_SEGMENTS)
        pieces, cursor = [], 0
        for seg_idx, spk in enumerate(chosen):
            ex = ds[rng.choice(by_speaker[spk])]
            seg = crop_segment(np.asarray(ex["audio"]["array"]), rng)
            # peak-normalize per segment so no speaker dominates by level
            peak = np.abs(seg).max()
            if peak > 0:
                seg = seg * (0.7 / peak)
            start_sample = cursor
            pieces.append(seg)
            cursor += len(seg)
            rows.append(dict(
                item_id=item, seg_idx=seg_idx, speaker_id=spk,
                ls_utterance_id=ex["id"], transcript=ex["text"],
                start_s=start_sample / SR, end_s=cursor / SR,
                start_sample=start_sample, end_sample=cursor,
            ))
            if seg_idx < N_SEGMENTS - 1:
                pieces.append(silence)
                cursor += len(silence)
        wav = np.concatenate(pieces)
        sf.write(out / "audio" / f"strip_{item:05d}.wav", wav, SR, subtype="PCM_16")

    pd.DataFrame(rows).to_parquet(out / "metadata.parquet", index=False)
    (out / "datasheet.md").write_text(DATASHEET.format(
        n_items=n_items, seed=seed,
        total_min=n_items * (N_SEGMENTS * SEG_SECONDS + (N_SEGMENTS - 1) * SILENCE_SECONDS) / 60,
    ))
    print(f"wrote {n_items} strips to {out}")


DATASHEET = """# Audio comic strips — datasheet

- {n_items} items x 6 segments; segment = 4.5s crop of a LibriSpeech
  (test-clean + dev-clean) utterance; 0.5s silence separators; 16kHz mono;
  29.5s per strip (< the 30s Whisper-family encoder window — v1 strips were
  33.5s and lost most of segment 6 to truncation).
- 6 distinct speakers per item, sampled without replacement per item; peak
  normalized to 0.7; 10ms edge fades. Seed: {seed}. Total audio: ~{total_min:.0f} min.
- Segment boundaries are exact by construction (see metadata.parquet) — the
  index->time map needs no forced alignment.
- Mirrors the seed paper's testbed (500 six-panel comic strips, MIT) in the
  audio domain. LibriSpeech is CC-BY-4.0.
"""

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, default=Path("strips"))
    ap.add_argument("--n-items", type=int, default=500)
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    build(a.out, a.n_items, a.seed)
