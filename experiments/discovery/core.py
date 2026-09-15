"""E1 core — gaze-score discovery, model-agnostic.

Protocol (pinned from the seed paper, Gandikota & Bau 2026, arXiv:2606.14703,
and its released code):
  for each strip, for each segment query k: one forward pass; take the FINAL
  prompt token's post-softmax attention; sum it over each segment's audio-token
  span -> per-head matrix M[query k, attended j]; gaze score per head = mean
  diagonal of M (row-normalized over segments).

Memory strategy for 8GB cards: never materialize full LxL attention.
Two-phase prefill — tokens [0..N-1] with use_cache (no attention output),
then the final prompt token alone with output_attentions=True, giving
per-layer (heads x 1 x L). Identical math, megabytes of attention.

An adapter (see adapters/) supplies model specifics:
  prepare(strip_wav, sr, query_text) -> ModelInputs
  segment_token_spans(inputs, boundaries_s) -> list of (start_idx, end_idx)
      token-index span of each audio segment in the *decoder input sequence*
  forward_last_token_attn(inputs) -> np.ndarray [layers, heads, seq_len]
"""

from dataclasses import dataclass

import numpy as np


@dataclass
class HeadScores:
    """Per-head gaze scores and the raw query x segment attention tensors."""
    scores: np.ndarray            # [layers, heads]
    matrices: np.ndarray          # [layers, heads, n_seg (query), n_seg (attended)]
    n_items: int
    per_item: np.ndarray = None   # [n_items, layers, heads, n_seg, n_seg] row-normalised, if kept


def segment_attention_matrix(last_tok_attn, spans):
    """[layers, heads, L] + segment spans -> [layers, heads, n_seg] mass per segment."""
    out = np.stack(
        [last_tok_attn[:, :, s:e].sum(axis=-1) for (s, e) in spans], axis=-1
    )
    return out  # [layers, heads, n_seg]


def gaze_scores(per_query_masses, normalize=True):
    """per_query_masses: [n_seg(query), layers, heads, n_seg(attended)]
    -> matrix and mean-diagonal score per head.

    normalize=True (ours): divide each row by its sum, so the score is "of the
    attention this head spends on audio, what share lands on the queried
    segment". Chance = 1/n_seg, and the score is insensitive to how much the
    head attends to audio overall.
    normalize=False: the seed paper's definition ("no normalization beyond
    softmax") — raw attention mass on the queried segment, which is
    therefore partly a measure of total audio attention.
    Both are reported so the concentration curve can be compared like-for-like.
    """
    m = np.transpose(np.asarray(per_query_masses), (1, 2, 0, 3))  # [L,H,q,a]
    if normalize:
        denom = m.sum(axis=-1, keepdims=True)
        m = np.divide(m, denom, out=np.zeros_like(m), where=denom > 0)
    diag = np.einsum("lhqq->lhq", m)
    return m, diag.mean(axis=-1)  # matrices [L,H,q,a], scores [L,H]


def run_discovery(adapter, strips, query_texts, shuffle_baseline_seed=None, keep_per_item=True):
    """strips: iterable of (wav, sr, boundaries_s: list[(start_s, end_s)]).
    query_texts: fn(seg_idx) -> prompt asking about segment seg_idx.
    Returns (HeadScores, HeadScores|None) — real and shuffled-label baseline.
    """
    acc, acc_shuf, n = None, None, 0
    acc_raw, acc_shuf_raw = None, None
    items, items_shuf = [], []
    rng = np.random.default_rng(shuffle_baseline_seed)
    for wav, sr, boundaries in strips:
        per_query = []
        for k in range(len(boundaries)):
            inputs = adapter.prepare(wav, sr, query_texts(k))
            spans = adapter.segment_token_spans(inputs, boundaries)
            attn = adapter.forward_last_token_attn(inputs)  # [L, H, seq]
            per_query.append(segment_attention_matrix(attn, spans))
        matrices, scores = gaze_scores(per_query)
        raw_m, _ = gaze_scores(per_query, normalize=False)
        acc = matrices if acc is None else acc + matrices
        acc_raw = raw_m if acc_raw is None else acc_raw + raw_m
        if keep_per_item:
            items.append(matrices.astype(np.float32))
        if shuffle_baseline_seed is not None:
            # shuffled-label baseline: permute which query row is "correct"
            perm = rng.permutation(len(boundaries))
            m_s, _ = gaze_scores([per_query[i] for i in perm])
            acc_shuf = m_s if acc_shuf is None else acc_shuf + m_s
            m_s_raw, _ = gaze_scores([per_query[i] for i in perm], normalize=False)
            acc_shuf_raw = m_s_raw if acc_shuf_raw is None else acc_shuf_raw + m_s_raw
            if keep_per_item:
                items_shuf.append(m_s.astype(np.float32))
        n += 1
    matrices = acc / n
    scores = np.einsum("lhqq->lhq", matrices).mean(axis=-1)
    real = HeadScores(scores=scores, matrices=matrices, n_items=n,
                      per_item=np.stack(items) if items else None)
    real.raw_matrices = acc_raw / n
    real.raw_scores = np.einsum("lhqq->lhq", real.raw_matrices).mean(axis=-1)
    shuf = None
    if acc_shuf is not None:
        ms = acc_shuf / n
        shuf = HeadScores(np.einsum("lhqq->lhq", ms).mean(axis=-1), ms, n,
                          per_item=np.stack(items_shuf) if items_shuf else None)
        ms_raw = acc_shuf_raw / n
        shuf.raw_matrices = ms_raw
        shuf.raw_scores = np.einsum("lhqq->lhq", ms_raw).mean(axis=-1)
    return real, shuf


def bootstrap_summary(per_item, per_item_shuf, n_boot=2000, seed=0, top_k=100):
    """per_item: [n, L, H, q, a] row-normalised matrices. Resample items with
    replacement; recompute the headline statistics each time. Returns dict of
    (point, lo95, hi95) for separation ratio, real/shuffled top-k, top-k mean,
    n heads > 0.25, plus a per-head permutation p-value against the shuffled
    distribution."""
    rng = np.random.default_rng(seed)
    n = per_item.shape[0]
    diag = np.einsum("nlhqq->nlhq", per_item).mean(-1)          # [n, L, H] per-item scores
    diag_s = np.einsum("nlhqq->nlhq", per_item_shuf).mean(-1)

    def stats(idx):
        sc = diag[idx].mean(0); ss = diag_s[idx].mean(0)
        flat = np.sort(sc.ravel())[::-1]; fs = np.sort(ss.ravel())[::-1]
        return dict(top_k_mean=flat[:top_k].mean(),
                    separation=flat[:top_k].mean() / np.median(flat),
                    real_vs_shuffled=flat[:top_k].mean() / fs[:top_k].mean(),
                    n_gt_025=float((sc > 0.25).sum()), top1=flat[0])
    point = stats(np.arange(n))
    boots = [stats(rng.integers(0, n, n)) for _ in range(n_boot)]
    out = {}
    for k in point:
        v = np.array([b[k] for b in boots])
        out[k] = (float(point[k]), float(np.quantile(v, .025)), float(np.quantile(v, .975)))
    # per-head: fraction of shuffled per-item scores' means (over items) that
    # exceed the real mean is not a valid null for a *specific* head; instead
    # use the pooled shuffled distribution of head means as the null
    null = diag_s.mean(0).ravel()
    real = diag.mean(0).ravel()
    p = np.array([(null >= r).mean() for r in real])
    out["n_heads_p_lt_0.001"] = int((p < 1e-3).sum())
    out["n_heads_p_eq_0"] = int((p == 0).sum())      # above every shuffled head
    return out


def concentration_report(scores):
    """The decision-rule inputs: sorted score curve + top-k separation stats."""
    flat = np.sort(scores.ravel())[::-1]
    layer_of_top100 = np.argsort(scores.ravel())[::-1][:100] // scores.shape[1]
    return {
        "n_heads": flat.size,
        "top10_mean": float(flat[:10].mean()),
        "top100_mean": float(flat[:100].mean()),
        "median": float(np.median(flat)),
        "p95_over_median": float(flat[int(0.05 * flat.size)] / max(np.median(flat), 1e-9)),
        "curve": flat,
        "top100_layer_hist": np.bincount(layer_of_top100, minlength=scores.shape[0]),
    }
