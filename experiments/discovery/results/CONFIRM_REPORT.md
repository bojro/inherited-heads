# Confirmatory report — pre-registered criteria applied mechanically

Held-out items 100-149 (confirmatory). Primary endpoint: steering accuracy, chance = 0.167.
Item bootstrap, 10000 resamples, 95% percentile CIs; differences PAIRED.
Criteria are quoted from PREREGISTRATION.md and applied without adjustment.
PENDING = that run has not landed yet; it is never treated as a pass.

## H1 (replication) — top-100 CI lower bound > 1/6 in all three arms

| arm | unsteered | gaze top-100 | lower > 1/6 |
|---|---|---|---|
| Qwen2-Audio-7B | 0.060 [0.000, 0.140] | 0.983 [0.970, 0.997] | yes |
| SALMONN-13B | 0.160 [0.060, 0.260] | 0.907 [0.877, 0.937] | yes |
| Ultravox-8B | 0.080 [0.020, 0.160] | 0.990 [0.977, 1.000] | yes |

**H1: PASS**

## H2 (specificity) — gaze top-K > random-K, >= 8 of 9 cells with CI excluding 0

| arm | K | gaze top-K | random-K (2 seeds) | paired difference | excl. 0 |
|---|---|---|---|---|---|
| Qwen2-Audio-7B | 50 | 0.707 [0.657, 0.757] | 0.320 [0.285, 0.355] | +0.387 [+0.332, +0.442] p=0.0001 | yes |
| Qwen2-Audio-7B | 100 | 0.983 [0.970, 0.997] | 0.107 [0.080, 0.135] | +0.877 [+0.847, +0.905] p=0.0001 | yes |
| Qwen2-Audio-7B | 250 | 0.987 [0.973, 0.997] | 0.225 [0.195, 0.255] | +0.762 [+0.727, +0.795] p=0.0001 | yes |
| SALMONN-13B | 50 | 0.793 [0.753, 0.830] | 0.275 [0.248, 0.302] | +0.518 [+0.480, +0.555] p=0.0001 | yes |
| SALMONN-13B | 100 | 0.907 [0.877, 0.937] | 0.178 [0.150, 0.207] | +0.728 [+0.690, +0.765] p=0.0001 | yes |
| SALMONN-13B | 250 | 0.923 [0.893, 0.950] | 0.015 [0.007, 0.023] | +0.908 [+0.880, +0.935] p=0.0001 | yes |
| Ultravox-8B | 50 | 0.643 [0.587, 0.697] | 0.610 [0.575, 0.648] | +0.033 [-0.030, +0.097] p=0.3214 | no |
| Ultravox-8B | 100 | 0.990 [0.977, 1.000] | 0.490 [0.450, 0.530] | +0.500 [+0.460, +0.540] p=0.0001 | yes |
| Ultravox-8B | 250 | 0.900 [0.867, 0.933] | 0.507 [0.467, 0.545] | +0.393 [+0.343, +0.440] p=0.0001 | yes |

Cells with CI excluding 0 in the predicted direction: **8 of 9** (criterion: >= 8).
Holm-Bonferroni across the 9 cells at alpha 0.05: **8 of 9** survive.
  - not surviving Holm: Ultravox-8B K=50 (p=0.3214)

**H2: PASS**

## H3 (gaze is not audio mass) — gaze top-100 > mass top-100 in Qwen and Ultravox, NOT in SALMONN

| arm | gaze top-100 | mass top-100 | paired difference | predicted | met |
|---|---|---|---|---|---|
| Qwen2-Audio-7B | 0.983 [0.970, 0.997] | 0.790 [0.743, 0.837] | +0.193 [+0.143, +0.243] p=0.0001 | excludes 0, positive | yes |
| SALMONN-13B | 0.907 [0.877, 0.937] | 0.887 [0.853, 0.917] | +0.020 [-0.017, +0.057] p=0.3342 | includes 0 | yes |
| Ultravox-8B | 0.990 [0.977, 1.000] | 0.027 [0.010, 0.043] | +0.963 [+0.943, +0.980] p=0.0001 | excludes 0, positive | yes |

**H3: PASS**

## H4 (mass and depth controlled) — Qwen2-Audio

Clause 1: gaze_highmass > mass_bandmatched (CI excludes 0).
Clause 2 was REPLACED after being contradicted on exploratory data
(Deviations 2026-08-17): re-test of gaze_lowmass > gaze_highmass.
Depth-matched pair is a disclosed addition (Deviations 2026-08-17b): layer
histogram identical by construction, gaze matched, mass differs ~2x.

| arm | comparison | paired difference | criterion | met |
|---|---|---|---|---|
| Qwen2-Audio-7B | gaze_highmass - mass_bandmatched | +0.413 [+0.343, +0.483] p=0.0001 | excludes 0, positive | yes (primary) |
| Qwen2-Audio-7B | gaze_lowmass - gaze_highmass | +0.147 [+0.083, +0.210] p=0.0001 | excludes 0, positive (re-test) | yes (primary) |
| Qwen2-Audio-7B | gaze_lowmass_dm - gaze_highmass_dm | +0.323 [+0.253, +0.390] p=0.0001 | depth-matched, disclosed addition | yes |
| SALMONN-13B | gaze_highmass - mass_bandmatched | +0.067 [+0.003, +0.127] p=0.0406 | excludes 0, positive | yes |
| SALMONN-13B | gaze_lowmass - gaze_highmass | -0.057 [-0.130, +0.020] p=0.1634 | excludes 0, positive (re-test) | NO |
| SALMONN-13B | gaze_lowmass_dm - gaze_highmass_dm | +0.103 [+0.043, +0.160] p=0.0001 | depth-matched, disclosed addition | yes |
| Ultravox-8B | gaze_highmass - mass_bandmatched | +0.630 [+0.570, +0.687] p=0.0001 | excludes 0, positive | yes |
| Ultravox-8B | gaze_lowmass - gaze_highmass | +0.203 [+0.130, +0.277] p=0.0001 | excludes 0, positive (re-test) | yes |
| Ultravox-8B | gaze_lowmass_dm - gaze_highmass_dm | +0.010 [-0.057, +0.073] p=0.8060 | depth-matched, disclosed addition | NO |

**H4 (Qwen2-Audio clauses only): PASS**

## H5 (concentration does not predict steerability) — Ultravox not lowest at K=100

| arm | E1 separation ratio | gaze top-100 accuracy |
|---|---|---|
| Qwen2-Audio-7B | see results/<arm>/report.json | 0.983 [0.970, 0.997] |
| SALMONN-13B | see results/<arm>/report.json | 0.907 [0.877, 0.937] |
| Ultravox-8B | see results/<arm>/report.json | 0.990 [0.977, 1.000] |

**H5: PASS** (Ultravox is not lowest)

## H6 (censoring signature) — random-K rises with K in Ultravox, falls in Qwen and SALMONN

| arm | random-50 | random-100 | random-250 | 250 - 50 | predicted | met |
|---|---|---|---|---|---|---|
| Qwen2-Audio-7B | 0.320 [0.285, 0.355] | 0.107 [0.080, 0.135] | 0.225 [0.195, 0.255] | -0.095 [-0.138, -0.050] p=0.0001 | decrease | yes |
| SALMONN-13B | 0.275 [0.248, 0.302] | 0.178 [0.150, 0.207] | 0.015 [0.007, 0.023] | -0.260 [-0.287, -0.235] p=0.0001 | decrease | yes |
| Ultravox-8B | 0.610 [0.575, 0.648] | 0.490 [0.450, 0.530] | 0.507 [0.467, 0.545] | -0.103 [-0.150, -0.057] p=0.0002 | increase | NO |

**H6: **FAIL****

## H7 (inheritance) — the one genuine prediction

Ultravox audio gaze top-100 vs bare Llama-3.1-8B-Instruct TEXT gaze top-100.
Audio ranking is frozen from exploratory items 0-49 by design; the text run
uses held-out passages 100-149, so a shared-passage confound cannot inflate
the overlap. Pre-committed threshold: >= 25 of 100 (chance = 100^2/1024 = 9.77).

Overlap: **71 of 100** heads (chance 9.77, hypergeometric one-sided p = 1.19e-61).


**H7: PASS**

## What the failures mean (from PREREGISTRATION.md, not written after the fact)

- H3 or H4 fails -> withdraw "the gaze score is not a proxy for audio
  attentiveness"; the result reduces to a causal replication of the seed
  paper in a new modality.
- H5 fails -> withdraw the methodological claim that concentration criteria
  have false negatives.
- H7 fails -> the frozen backbone's tracking is not inherited text machinery
  and needs another explanation.
- None of these is to be reframed as a success.
