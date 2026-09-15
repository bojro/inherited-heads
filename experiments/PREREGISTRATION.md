# Pre-registration — confirmatory phase

> Written on 2026-08-17, before any held-out data were touched, and committed to
> the author's working repository before the confirmatory runs began. This public
> repository was created afterwards, so its own history does not timestamp the
> document. The plan is reproduced as written, apart from this note and two
> references reworded to point at files in this repository. The outcome
> of every hypothesis, and every deviation from the plan, is recorded in
> [`OUTCOMES.md`](OUTCOMES.md). Code comments refer to entries of the dated
> research log kept alongside this plan during the work (codes such as `17al`);
> that log is not published, and what it recorded is summarised in `OUTCOMES.md`
> and `RESULTS_LEDGER.md`.

## Why this document exists

Everything run up to this commit is **exploratory**. E1 and E2's core protocol
and controls were specified in advance (in the project plan), but everything after the
first E2 result was designed in response to what the previous run showed:
the audio-mass control, the disjoint head sets, the layer-matched rebuild, the
fluency metric, the dual score definition, the text-inheritance test. That is
legitimate hypothesis generation, but it means the choice of condition, K,
arm, metric, and stopping point was made with the data visible. Effective
multiple comparisons are therefore uncounted and the reported effects may be
partly fitted to noise (Gelman & Loken's garden of forking paths; HARKing).

Two kinds of reactive work happened and only one carries bias:
- **Validity checks** (causal-mask bug, fluency artefact, normalisation
  discrepancy, query-blind adapters). These correct the instrument. Keep.
- **Hypothesis generation** (mass control, disjoint sets, inheritance test).
  These need confirmation on data that did not generate them.

## Data split

The testbed has 500 strips. Every run to date used items **0–49** (E1: 50
items; E2 and all controls: 20 items). Items **100–399** are untouched and are
reserved here as the confirmation set; 50–99 stay as a buffer for any further
exploration. `--item-offset` was added to every runner for this purpose.

Confirmatory runs use `--item-offset 100`. No confirmatory result may be
reported from items < 100. If a confirmatory run is repeated after any change
to code or analysis, that is disclosed as a deviation below.

## Hypotheses (frozen; each stated so it can fail)

Primary endpoint for H1–H6: **steering accuracy** (transcript-overlap judge,
chance = 1/6), 95% CI by bootstrap resampling of ITEMS (steer_ci.py).
Secondary, reported always, never substituted for the primary: **distinct-2**
fluency and the unattributable rate.

- **H1 (replication).** Biasing the top-100 gaze-ranked heads steers above
  chance in all three arms. Confirmed if the CI lower bound > 1/6 in each.
- **H2 (specificity).** gaze-top-K > random-K at matched K for K in
  {50, 100, 250}, in all three arms. Confirmed if the paired difference CI
  excludes 0 in >= 8 of 9 cells.
- **H3 (gaze is not audio-mass).** gaze-top-100 > mass-top-100 in Qwen2-Audio
  and Ultravox; NOT in SALMONN. Confirmed if the first two difference CIs
  exclude 0 and SALMONN's includes 0. Exploratory values: +.158, +.967, +.017.
- **H4 (mass and depth controlled).** With audio mass and mean layer matched,
  gaze still wins: gaze_highmass > mass_bandmatched, and
  gaze_lowmass ~ gaze_highmass. Confirmed if the first difference CI excludes
  0 and the second includes 0, in Qwen2-Audio.
- **H5 (concentration does not predict steerability).** Across arms, E1
  separation ratio (1.90 / 1.45 / 1.10) does not rank-order steering accuracy
  at K=100. Confirmed if Ultravox — lowest concentration — is not lowest in
  steering accuracy.
- **H6 (censoring signature).** random-K accuracy INCREASES with K in
  Ultravox and DECREASES in Qwen2-Audio and SALMONN, over K in {50,100,250}.
- **H7 (inheritance, no exploratory value yet).** Ultravox's audio gaze
  top-100 overlaps the pure-text gaze top-100 of the same bare
  Llama-3.1-8B backbone above chance (chance = 100^2/1024 ~ 9.8 heads).
  Confirmed if overlap >= 25/100. This is the one hypothesis with no prior
  look at the answer; it is a genuine prediction.

## Conditions to run (fixed list; no additions without disclosure)

Per arm (qwen2-audio, salmonn, ultravox), items 100-149 (n=50), prompt 0:
  unsteered; gaze top-K for K in {25,50,100,250}; random-K for K in
  {50,100,250} x 2 seeds; mass-top-K for K in {50,100}; all-heads;
  gaze_lowmass / gaze_highmass / mass_bandmatched (K=50).
Robustness: repeat gaze top-100 and random-100 under prompt 1 and prompt 2.
Text arm: pure-text discovery on Llama-3.1-8B, items 100-149, for H7.

Head RANKINGS are taken from the exploratory phase (items 0-49) and NOT
recomputed on held-out data — the ranking is the hypothesis being tested, so
recomputing it would leak the confirmation set into the thing under test.

## Analysis, fixed in advance

- One primary metric (accuracy); fluency reported but never swapped in.
- Item-level bootstrap, 10,000 resamples, 95% percentile CIs.
- H2 spans 9 cells: Holm-Bonferroni across those 9, alpha 0.05.
- No optional stopping: all listed conditions run to completion before any
  are interpreted.
- Every deviation from this document is recorded, with the reason and the date
  (now in `OUTCOMES.md`).

## What a failure means

If H3 or H4 fails, the claim "the gaze score is not a proxy for audio
attentiveness" is withdrawn and the result reduces to a causal replication of
the seed paper in a new modality. If H5 fails, the methodological claim about
concentration having false negatives is withdrawn. If H7 fails, the frozen
backbone's tracking is not inherited text machinery and needs another
explanation. None of these outcomes is to be reframed as a success.
