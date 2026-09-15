"""Tiled-strip presentation and non-contiguous region handling for V2 / V2a.

Why this module exists. The seed paper presents a strip as ONE horizontally
tiled image and assigns each image token to a panel by its column in the merged
patch grid (`gaze_heads/regions.py::assign_panels_to_tokens`). Panel tokens are
therefore NOT contiguous in row-major order — a panel is a column band spanning
every row. V2 exists to test their method on their data, so the presentation has
to match theirs; feeding six separate images (the first V2 draft) is a different
experiment.

`core.segment_attention_matrix` and `steer.set_bias` both assume a segment is a
contiguous slice. Rather than edit either — the confirmatory suite is running on
them — this module provides region-indexed equivalents:

  * `region_masses`      the aggregation, by bincount over per-token region ids
  * `set_bias_regions`   writes `steer._State.bias` directly by index, so the
                         SAME hook (`steer.eager_bias_forward`) applies it

`core.gaze_scores` and `core.concentration_report` are reused unchanged: they
take per-query mass arrays and never touch token layout.

The column-split logic is ported from theirs deliberately, including its quirk of
splitting the full merged width in proportion to PANEL widths while ignoring the
inter-panel gaps. `tests_match_seed_regions()` checks ours against theirs
numerically when their repo is available.
"""
from __future__ import annotations

import numpy as np
import torch

import steer


# ---- presentation -------------------------------------------------------
def tile_strip(panels, target_height: int = 256, gap: int = 6):
    """Horizontally tile panels at a common height. Returns (PIL.Image, widths).

    Defaults are the seed paper's DEFAULT_TARGET_HEIGHT / DEFAULT_GAP.
    """
    from PIL import Image
    resized, widths = [], []
    for im in panels:
        w = max(1, int(round(im.width * target_height / im.height)))
        resized.append(im.resize((w, target_height), Image.LANCZOS))
        widths.append(w)
    total_w = sum(widths) + gap * (len(resized) - 1)
    strip = Image.new("RGB", (total_w, target_height), (255, 255, 255))
    x = 0
    for im in resized:
        strip.paste(im, (x, 0))
        x += im.width + gap
    return strip, widths


# ---- token -> panel -----------------------------------------------------
def _monotonic_boundaries(total_columns: int, widths) -> list[int]:
    """Ported from gaze_heads/regions.py so V2 matches their assignment exactly."""
    n = len(widths)
    if n == 0:
        return [0, total_columns]
    total_width = max(1, int(sum(widths)))
    raw = [0.0]
    running = 0
    for w in widths:
        running += int(w)
        raw.append(total_columns * running / total_width)
    bounds = [0]
    for i in range(1, n):
        proposed = int(round(raw[i]))
        remaining = n - i
        lo = bounds[-1] + (1 if total_columns >= n else 0)
        hi = total_columns - remaining if total_columns >= n else total_columns
        bounds.append(min(max(proposed, lo), hi))
    bounds.append(total_columns)
    return bounds


def assign_panels_to_tokens(image_grid_thw, panel_widths, spatial_merge: int):
    """-> (region_ids per image token, (t, mh, mw), column ranges)."""
    t, h, w = [int(x) for x in np.asarray(image_grid_thw)[0].tolist()]
    merge = max(1, int(spatial_merge))
    mh, mw = h // merge, w // merge
    bounds = _monotonic_boundaries(mw, panel_widths)
    ranges = [(bounds[i], bounds[i + 1]) for i in range(len(panel_widths))]
    col_to_panel = np.empty((mw,), dtype=int)
    for i, (c0, c1) in enumerate(ranges):
        col_to_panel[c0:c1] = i
    return np.tile(col_to_panel, t * mh), (t, mh, mw), ranges



def tile_column(panels, target_width: int = 256, gap: int = 6):
    """VERTICALLY tile panels at a common width. Returns (PIL.Image, heights).

    The vertical counterpart of tile_strip, for V20. Under row-major flattening
    of the patch grid, a vertical layout makes panel k a CONTIGUOUS run of
    tokens, where the horizontal layout makes it 24 interleaved runs. That is
    the whole point: it varies token contiguity while holding the task, the
    model and the panels fixed.
    """
    from PIL import Image
    resized, heights = [], []
    for im in panels:
        h = max(1, int(round(im.height * target_width / im.width)))
        resized.append(im.resize((target_width, h), Image.LANCZOS))
        heights.append(h)
    total_h = sum(heights) + gap * (len(resized) - 1)
    canvas = Image.new("RGB", (target_width, total_h), (255, 255, 255))
    y = 0
    for im, h in zip(resized, heights):
        canvas.paste(im, (0, y))
        y += h + gap
    return canvas, heights


def assign_panels_to_rows(image_grid_thw, panel_heights, spatial_merge: int):
    """Row-band counterpart of assign_panels_to_tokens.

    Same monotonic proportional split, applied to ROWS, then repeated across
    every column — `np.repeat` where the column version uses `np.tile`. Because
    the grid is flattened row-major, each panel is one contiguous token run.
    """
    t, h, w = [int(x) for x in np.asarray(image_grid_thw)[0].tolist()]
    merge = max(1, int(spatial_merge))
    mh, mw = h // merge, w // merge
    bounds = _monotonic_boundaries(mh, panel_heights)
    ranges = [(bounds[i], bounds[i + 1]) for i in range(len(panel_heights))]
    row_to_panel = np.empty((mh,), dtype=int)
    for i, (r0, r1) in enumerate(ranges):
        row_to_panel[r0:r1] = i
    return np.tile(np.repeat(row_to_panel, mw), t), (t, mh, mw), ranges

def region_positions(img_start: int, region_ids: np.ndarray, n_regions: int):
    """Absolute sequence positions per region."""
    return [(np.where(region_ids == r)[0] + img_start).tolist() for r in range(n_regions)]


# ---- aggregation (replaces core.segment_attention_matrix for regions) ----
def region_masses(attn: np.ndarray, img_start: int, region_ids: np.ndarray,
                  n_regions: int) -> np.ndarray:
    """attn [L, H, seq] -> [L, H, n_regions] summed raw attention per region.

    Equivalent to the seed paper's einsum over a region one-hot; bincount is used
    so the memory footprint stays linear in tokens.
    """
    usable = min(attn.shape[-1] - img_start, region_ids.shape[0])
    block = attn[:, :, img_start:img_start + usable].astype(np.float64)
    ids = region_ids[:usable]
    L, H, _ = block.shape
    out = np.zeros((L, H, n_regions), dtype=np.float64)
    for r in range(n_regions):
        out[:, :, r] = block[:, :, ids == r].sum(axis=-1)
    return out


# ---- intervention (replaces steer.set_bias for regions) -----------------
def set_bias_regions(heads, positions, target: int, prompt_len: int, n_heads: int,
                     B: float = 1e4, device: str = "cuda"):
    """+B on the target region's tokens, -B on every other region's, per head.

    Writes `steer._State` directly so `steer.eager_bias_forward` — the hook whose
    equivalence assertion has caught two real bugs — applies it unchanged. This is
    the seed paper's `boost_suppress` mode; text tokens are untouched.
    """
    steer._State.bias = {}
    idx = [torch.as_tensor(p, dtype=torch.long, device=device) for p in positions]
    for (l, h) in heads:
        if l not in steer._State.bias:
            steer._State.bias[l] = torch.zeros(n_heads, prompt_len, dtype=torch.float32,
                                               device=device)
        v = steer._State.bias[l][h]
        for j, ii in enumerate(idx):
            v[ii] = B if j == target else -B
    steer._State.active = True


# ---- 2-D grid presentation, for fixed-square-input models ---------------
def tile_grid(panels, cols: int = 3, rows: int = 2, cell: int = 256, gap: int = 6):
    """Tile panels into a cols x rows grid. Returns (PIL.Image, cell_boxes).

    Needed because a 6:1 horizontal strip is the wrong shape for models with a
    FIXED SQUARE vision input. Bunny-v1_0-3B uses siglip-so400m-patch14-384 with
    `image_aspect_ratio: pad`, so a 1566x256 strip is padded to 1566x1566 before
    the 384x384 resize — the content then occupies ~16% of the frame, leaving
    roughly 20 usable tokens per panel. A 3x2 grid fills the frame instead.

    This DEVIATES from the seed paper's horizontal `build_strip`. Their released
    code only supports Qwen3-VL (dynamic resolution), so how they presented strips
    to their fixed-input null models (LLaVA, Bunny) is not public; a choice has to
    be made and disclosed. Robustness across presentations should be reported.
    """
    from PIL import Image
    assert cols * rows == len(panels), (cols, rows, len(panels))
    sq = [im.resize((cell, cell), Image.LANCZOS) for im in panels]
    W = cols * cell + gap * (cols - 1)
    H = rows * cell + gap * (rows - 1)
    canvas = Image.new("RGB", (W, H), (255, 255, 255))
    boxes = []
    for i, im in enumerate(sq):
        r, c = divmod(i, cols)
        x, y = c * (cell + gap), r * (cell + gap)
        canvas.paste(im, (x, y))
        boxes.append((x, y, x + cell, y + cell))
    return canvas, boxes


def assign_panels_to_grid_blocks(merged_h: int, merged_w: int, boxes, image_size,
                                 pad_to_square: bool = False):
    """Assign each image token to a panel by which box its grid-cell CENTRE falls in.

    Mirrors the seed paper's `bbox_to_token_positions` rule (a token belongs to a
    region when its cell centre is inside), generalised from one box to a
    partition. Tokens in no box (gaps, or padding when pad_to_square) get -1 and
    are excluded from every region, exactly as gap columns are in the 1-D case.

    Returns region_ids of length merged_h*merged_w, row-major.
    """
    W, H = image_size
    if pad_to_square:
        side = max(W, H)
        off_x, off_y = (side - W) // 2, (side - H) // 2
        W = H = side
    else:
        off_x = off_y = 0
    col_c = (np.arange(merged_w) + 0.5) * W / merged_w
    row_c = (np.arange(merged_h) + 0.5) * H / merged_h
    cx, cy = np.meshgrid(col_c, row_c)
    ids = np.full((merged_h, merged_w), -1, dtype=int)
    for k, (x0, y0, x1, y1) in enumerate(boxes):
        x0, x1 = x0 + off_x, x1 + off_x
        y0, y1 = y0 + off_y, y1 + off_y
        ids[(cx >= x0) & (cx <= x1) & (cy >= y0) & (cy <= y1)] = k
    return ids.reshape(-1)
