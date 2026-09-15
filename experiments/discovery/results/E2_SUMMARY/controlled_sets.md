# Controlled head-set comparison (S4) — exploratory, items 0-19

Sets are 50 heads each, built by make_disjoint_sets.py from the gaze top-100.

| arm | set | gaze | audio mass | mean layer | steering | 95% CI |
|---|---|---|---|---|---|---|
| qwen2_audio | gaze_lowmass | 0.322 | 0.376 | 21.5 | 0.900 | [0.850, 0.950] |
| qwen2_audio | gaze_highmass | 0.317 | 0.785 | 23.5 | 0.642 | [0.567, 0.717] |
| qwen2_audio | mass_bandmatched | 0.217 | 0.730 | 23.4 | 0.308 | [0.242, 0.375] |
| salmonn | gaze_lowmass | 0.251 | 0.154 | 16.7 | 0.550 | [0.467, 0.633] |
| salmonn | gaze_highmass | 0.244 | 0.545 | 19.8 | 0.733 | [0.658, 0.808] |
