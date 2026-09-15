"""V15 — content-matched strips where segment identity is PURELY acoustic.

Every existing testbed (LibriSpeech strips, text strips, comic strips) lets a
head disambiguate segments by CONTENT: segment 3 is "the one about the canyon".
That makes it impossible to say whether a gaze head routes acoustic information
or is doing text-style topic tracking over a transcript the encoder handed it.

RAVDESS removes content from the equation. Every clip is one of two fixed
sentences, read by 24 actors in 8 emotions at 2 intensities, twice. So a strip
can be built in which all six segments say THE SAME WORDS and differ only in:

  speaker axis  (testbed/prosody_speaker) — six different actors, one fixed
                sentence/emotion/intensity. Segment identity = who is talking.
  emotion axis  (testbed/prosody_emotion) — one actor, one fixed sentence, six
                different emotions. Segment identity = how it is said.

Transcript-overlap attribution is meaningless here by construction; that is the
point, and it is why V16's judge scores speaker/emotion attributes instead.

Deliberately conservative choices, both recorded in the datasheet:
  * every segment is RMS-normalised to a common level, so loudness — the
    cheapest possible cue, and a real confound on the emotion axis where angry
    is louder than sad — cannot carry segment identity. Anything that works
    must use pitch, timbre, or rate.
  * leading/trailing silence is trimmed before concatenation, so the recorded
    boundaries bracket speech rather than room tone.

Usage:
    python build_prosody_strips.py --out ./prosody_speaker --axis speaker --n-items 150
    python build_prosody_strips.py --out ./prosody_emotion --axis emotion --n-items 150

Output mirrors build_testbed.py so the discovery/steering runners take it with
--strips and no code change:
    <out>/audio/strip_00000.wav       16 kHz mono PCM16
    <out>/metadata.parquet            one row per segment
    <out>/datasheet.md
"""

import argparse
import glob
import io
import os
import random
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.parquet as pq
import soundfile as sf

SR = 16_000
SILENCE_SECONDS = 0.5
N_SEGMENTS = 6
MAX_STRIP_SECONDS = 30.0          # Whisper-family encoders truncate past 30s
TARGET_RMS = 0.05                 # common level for every segment
TRIM_DB = 35.0                    # silence floor below the clip peak

EMOTIONS = {"01": "neutral", "02": "calm", "03": "happy", "04": "sad",
            "05": "angry", "06": "fearful", "07": "disgust", "08": "surprised"}
STATEMENTS = {"01": "KIDS ARE TALKING BY THE DOOR",
              "02": "DOGS ARE SITTING BY THE DOOR"}
# RAVDESS actor parity: odd = male, even = female (datasheet of the original set)
GENDER = lambda actor: "male" if int(actor) % 2 == 1 else "female"


def load_pool():
    """path -> waveform, keyed by the RAVDESS filename fields. The HF export
    ships the same 1440 clips twice (two parquet shards, one per fold), so
    dedupe on the filename."""
    snaps = glob.glob(os.path.expanduser(
        "~/.cache/huggingface/hub/datasets--quinnlue--ravdess_emotional_speech_audio/snapshots/*"))
    if not snaps:
        raise SystemExit("RAVDESS not downloaded — download quinnlue/ravdess_emotional_speech_audio from the Hugging Face Hub first")
    rows = {}
    for f in sorted(glob.glob(snaps[0] + "/data/*.parquet")):
        for w in pq.read_table(f, columns=["wf"]).to_pydict()["wf"]:
            rows.setdefault(w["path"], w["bytes"])
    pool = {}
    for path, blob in rows.items():
        mod, voc, emo, inten, stmt, rep, actor = os.path.splitext(path)[0].split("-")
        pool[(actor, emo, inten, stmt, rep)] = (path, blob)
    return pool


def decode(blob):
    a, sr = sf.read(io.BytesIO(blob), dtype="float32")
    if a.ndim > 1:
        a = a.mean(axis=1)
    assert sr == SR, sr
    return a


def trim_and_level(a):
    """Strip room tone from both ends, then set a common RMS. Returns None if
    the clip has no usable speech."""
    if a.size == 0:
        return None
    peak = np.abs(a).max()
    if peak <= 0:
        return None
    thresh = peak * (10.0 ** (-TRIM_DB / 20.0))
    loud = np.flatnonzero(np.abs(a) > thresh)
    if loud.size == 0:
        return None
    pad = int(0.05 * SR)
    a = a[max(0, loud[0] - pad): min(a.size, loud[-1] + pad)]
    rms = float(np.sqrt(np.mean(a.astype(np.float64) ** 2)))
    if rms <= 0:
        return None
    a = a * (TARGET_RMS / rms)
    m = np.abs(a).max()
    if m > 0.99:                    # never clip; scale the whole segment down
        a = a * (0.99 / m)
    return a.astype(np.float32)


def speaker_items(pool, n_items, rng):
    """Six actors, one fixed (statement, emotion, intensity, repetition)."""
    actors = sorted({k[0] for k in pool})
    combos = sorted({(k[1], k[2], k[3], k[4]) for k in pool})
    out = []
    while len(out) < n_items:
        emo, inten, stmt, rep = combos[rng.randrange(len(combos))]
        avail = [a for a in actors if (a, emo, inten, stmt, rep) in pool]
        if len(avail) < N_SEGMENTS:
            continue
        # balance gender within a strip where the pool allows it, so "the male
        # one" is never a unique identifier
        males = [a for a in avail if GENDER(a) == "male"]
        females = [a for a in avail if GENDER(a) == "female"]
        if len(males) >= 3 and len(females) >= 3:
            pick = rng.sample(males, 3) + rng.sample(females, 3)
        else:
            pick = rng.sample(avail, N_SEGMENTS)
        rng.shuffle(pick)
        out.append([dict(key=(a, emo, inten, stmt, rep), label=f"actor {int(a)}",
                         attr=GENDER(a)) for a in pick])
    return out


def emotion_items(pool, n_items, rng):
    """Six emotions, one fixed (actor, statement, repetition). Intensity is
    pinned to 01 because 'neutral' exists only at normal intensity, and mixing
    intensities would reintroduce a level cue the RMS match is meant to kill."""
    actors = sorted({k[0] for k in pool})
    emos = sorted(EMOTIONS)
    out = []
    while len(out) < n_items:
        actor = actors[rng.randrange(len(actors))]
        stmt = rng.choice(sorted(STATEMENTS))
        rep = rng.choice(["01", "02"])
        avail = [e for e in emos if (actor, e, "01", stmt, rep) in pool]
        if len(avail) < N_SEGMENTS:
            continue
        pick = rng.sample(avail, N_SEGMENTS)
        rng.shuffle(pick)
        out.append([dict(key=(actor, e, "01", stmt, rep), label=EMOTIONS[e],
                         attr=EMOTIONS[e]) for e in pick])
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--axis", choices=["speaker", "emotion"], required=True)
    ap.add_argument("--n-items", type=int, default=150)
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()

    pool = load_pool()
    print(f"pool: {len(pool)} clips")
    rng = random.Random(a.seed)
    items = (speaker_items if a.axis == "speaker" else emotion_items)(pool, a.n_items, rng)

    (a.out / "audio").mkdir(parents=True, exist_ok=True)
    gap = np.zeros(int(SILENCE_SECONDS * SR), dtype=np.float32)
    rows, dropped, durs = [], 0, []
    item_id = 0
    for segs in items:
        clips = []
        for s in segs:
            path, blob = pool[s["key"]]
            w = trim_and_level(decode(blob))
            if w is None:
                break
            clips.append((s, path, w))
        if len(clips) < N_SEGMENTS:
            dropped += 1
            continue
        total = sum(len(w) for _, _, w in clips) + len(gap) * (N_SEGMENTS - 1)
        if total > MAX_STRIP_SECONDS * SR:
            dropped += 1
            continue

        buf, cur = [], 0
        for k, (s, path, w) in enumerate(clips):
            start = cur
            buf.append(w)
            cur += len(w)
            if k < N_SEGMENTS - 1:
                buf.append(gap)
                cur += len(gap)
            actor, emo, inten, stmt, rep = s["key"]
            rows.append(dict(
                item_id=item_id, seg_idx=k, speaker_id=int(actor),
                transcript=STATEMENTS[stmt],
                start_s=start / SR, end_s=(start + len(w)) / SR,
                start_sample=start, end_sample=start + len(w),
                axis=a.axis, seg_label=s["label"], seg_attr=s["attr"],
                emotion=EMOTIONS[emo], intensity=int(inten),
                statement=int(stmt), repetition=int(rep),
                gender=GENDER(actor), ravdess_file=path))
        strip = np.concatenate(buf)
        durs.append(len(strip) / SR)
        sf.write(a.out / "audio" / f"strip_{item_id:05d}.wav", strip, SR, subtype="PCM_16")
        item_id += 1

    meta = pd.DataFrame(rows)
    meta.to_parquet(a.out / "metadata.parquet", index=False)
    d = np.array(durs)
    uniq = meta.groupby("item_id").seg_label.nunique()
    print(f"wrote {item_id} items ({dropped} dropped), {len(meta)} segment rows")
    print(f"strip seconds: min {d.min():.1f} med {np.median(d):.1f} max {d.max():.1f}")
    print(f"distinct labels per item: min {uniq.min()} (must be 6)")
    assert uniq.min() == N_SEGMENTS, "a strip repeated a label — identity would be ambiguous"
    assert meta.groupby("item_id").transcript.nunique().max() == 1, \
        "a strip mixed statements — content would disambiguate segments"

    (a.out / "datasheet.md").write_text(f"""# V15 prosody strips — {a.axis} axis

Source: RAVDESS (quinnlue/ravdess_emotional_speech_audio), 24 actors x 8
emotions x 2 statements x 2 repetitions, 16 kHz. {item_id} items, {N_SEGMENTS}
segments each, built with seed {a.seed}.

**All six segments of a strip speak the identical sentence.** Segment identity is
carried only by {'speaker' if a.axis == 'speaker' else 'emotional prosody'}, so
transcript-overlap attribution is meaningless by construction — score with a
{'speaker-attribute' if a.axis == 'speaker' else 'emotion'} judge instead.

Conservative preprocessing, both of which make the test HARDER:
- every segment RMS-normalised to {TARGET_RMS} (peak-guarded), so loudness cannot
  carry identity — decisive on the emotion axis, where angry is naturally louder
  than sad;
- silence trimmed at {TRIM_DB} dB below clip peak with a 50 ms margin, so the
  recorded boundaries bracket speech, not room tone.
{'- gender balanced 3/3 within a strip wherever the pool allows, so "the male one" is never a unique identifier.' if a.axis == 'speaker' else '- intensity pinned to normal (01), because neutral exists only at that intensity and mixing intensities would reintroduce the level cue.'}

Strip duration {d.min():.1f}-{d.max():.1f} s (median {np.median(d):.1f}), under
the {MAX_STRIP_SECONDS:.0f} s Whisper-family encoder window.

Columns: item_id, seg_idx, speaker_id, transcript, start_s, end_s, start_sample,
end_sample, axis, seg_label, seg_attr, emotion, intensity, statement,
repetition, gender, ravdess_file.
""")
    print(f"datasheet -> {a.out / 'datasheet.md'}")


if __name__ == "__main__":
    main()
