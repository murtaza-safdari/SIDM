# The 2Mu2E-displaced cross-cleaning difference between the two Run 3 signal productions

The first comparison of the two productions (2026-11-12; 20,000 events per production in 2022 and 2023) found the
pass fraction of the DSA-PF segment-match veto lower in the second production (v2, EXONanoAOD, rebuilt segment
count) than in the first (v1, LLPNanoAOD) for the 2Mu2E displaced set: -0.92 +- 0.31 points in 2022 and 2023
combined, while seven other sets agreed. This study repeats the comparison with five times the events, two more
eras and all column samples, and applies a decision rule fixed before the run.

## Verdicts

Rule, fixed before the run, for the combined 2Mu2E-displaced difference d +- s of the eight-set comparison
and, separately, of the pooled column samples: |d|/s < 2 "not confirmed"; |d|/s > 3 with the same sign in at
least three of the four eras "confirmed"; anything else "inconclusive".

- **Eight sets on all events, 2Mu2E displaced (`2Mu2E_200GeV_1p2GeV_4p8mm`, 100,000 events per era and
  production, four eras): not confirmed by the rule, 0.04 below its threshold.** d = -0.194 +- 0.099 points,
  d/s = -1.96; 4 of 4 eras negative. With the variance clustered by event, s = 0.111 and d/s = -1.74.
  - The two eras of the first comparison alone, on all their events (2022 + 2023): -0.31 +- 0.14 (d/s = -2.21).
    The veto "all + charge" over four eras: -0.22 +- 0.10 (-2.20). The rule was not written for these two
    numbers; both are beyond 2.
  - Without the files of the first comparison (j040-j059; 100 of 120 files and 80,000 of 100,000 events per era
    remain): -0.16 +- 0.11 (-1.43), +- 0.12 with the variance clustered by event.
  - Reading: the 0.9-point difference is not reproduced. A difference of about -0.2 points in this one sample is
    not excluded.
- **All column samples, 2Mu2E, 18 samples pooled, four eras: not confirmed.** d = +0.007 +- 0.024 points,
  d/s = +0.30; 2 of 4 eras negative. The 72 per-sample differences: 34 below / 38 above zero, chi2 98.7 / 72
  (p = 0.020) with binomial s; the event-clustered s is about 1.12 times larger for 2Mu2E, which scales chi2 to
  about 78 / 72 (p about 0.3; scaling estimated from the pooled s_evt / s, not recomputed per sample).
- The 4Mu control: 18 samples pooled, +0.00 +- 0.02 points (d/s +0.27); per-sample chi2 85.8 / 72 (p = 0.13).
  The 4Mu displaced set on all events: -0.05 +- 0.07.
- The 2Mu2E prompt-like set, which the rule does not cover: +0.36 +- 0.19 points over four eras (d/s = +1.95),
  above zero in 4 of 4 eras. Its samples have 4 files of 500 events per point, all of them used; in 2022 and
  2023 these are the events of the first comparison.
- With both verdicts "not confirmed", the search for where the productions first differ was not run.

The first 20,000 events moved both productions: in 2022 v1 gives 0.3245 on the events of the first comparison
against 0.3198 on all events, v2 with the rebuilt count 0.3149 against 0.3182.

## Not tested

- 2024: the first production has no 2024 samples, so neither this comparison nor check (a) below exists there.
- The effect of a difference of this size on the lepton-jet efficiency or on a signal yield.
- The rebuilt count on background simulation or on data.
- The per-sample chi2 with event-clustered uncertainties (scaled from the pooled samples, not recomputed).

## Method

- Code: this directory (the README has
  the command lines). The veto is evaluated with the `"all"` and `"all + charge"` lambdas of `sidm/definitions/cuts.py`
  on a `good_matched_muons` collection built as `sidm_processor.py` builds it; v1 uses the stored `nSegments`, v2
  `nDTSegments + ceil(trkNumCSCHits / 6)`.
- Population: "pre" DSA muons (pT > 10 GeV, abs(eta) < 2.4, displaced ID). Files from the location YAMLs
  (`signal_{4mu,2mu2e}_run3.yaml` for v1, `signal_run3_v2_<era>.yaml` for v2); 34,944 files read, 0 read failures
  (per-file logs of `xclean_rows.py`); every column sample has 120 files and 100,000
  events per era in both productions (event counts in the tables of the eight sets).
- Uncertainties: binomial per muon, productions independent; `s_evt` clusters the variance by
  event. Combinations are inverse-variance weighted means; chi2 is sum (d / s)^2 against zero.
- Displaced / prompt-like split for the column samples: all 36 column samples have ctau scaled with the boost, median gen Lxy of the
  gen-matched "pre" DSA muons 17.4 to 22.8 cm (table), so all fall on the displaced side of the first comparison's
  definition. Its prompt-like points are grid-only samples (4 files of 500 events); the eight-set comparison uses all their
  events, which are the events of the first comparison.
- The sections from "The eight sets on all events" to "Verdicts of the decision rule" are the output of
  `xclean_stats.py`, unedited; d / s there uses the binomial s.

## Reproduction of the first comparison's table (same files and events; `xclean_first_table.py`)

All 40 values of the first comparison's table reproduce to four digits (columns "all" v1, v2, v2 fixed, "all + charge" v1, v2;
the extra column is "all + charge" with the rebuilt count). The "pre" counts of the 2Mu2E displaced sets (22,135
and 22,183 in v1) equal those of the first comparison. Combination of the two 2Mu2E displaced eras: -0.92 +- 0.31
points (-2.9), and -0.92 +- 0.35 (-2.6) with the event-clustered variance.

| sample | events v1 / v2 | pre v1 / v2 | all v1 | all v2 | all v2 fixed | all+q v1 | all+q v2 | all+q v2 fixed |
|---|---|---|---|---|---|---|---|---|
| 2022 4Mu displaced | 20,000 / 20,000 | 44,283 / 44,469 | 0.3200 | 0.3321 | 0.3193 | 0.3446 | 0.3482 | 0.3444 |
| 2023 4Mu displaced | 20,000 / 20,000 | 44,308 / 44,464 | 0.3248 | 0.3367 | 0.3243 | 0.3480 | 0.3515 | 0.3484 |
| 2022 4Mu prompt-like | 12,000 / 12,000 | 25,652 / 25,560 | 0.1009 | 0.1028 | 0.0969 | 0.1262 | 0.1238 | 0.1233 |
| 2023 4Mu prompt-like | 12,000 / 12,000 | 25,620 / 25,578 | 0.1023 | 0.1074 | 0.1008 | 0.1306 | 0.1296 | 0.1296 |
| 2022 2Mu2E displaced | 20,000 / 20,000 | 22,135 / 22,293 | 0.3245 | 0.3275 | 0.3149 | 0.3494 | 0.3433 | 0.3398 |
| 2023 2Mu2E displaced | 20,000 / 20,000 | 22,183 / 22,156 | 0.3266 | 0.3306 | 0.3179 | 0.3495 | 0.3459 | 0.3426 |
| 2022 2Mu2E prompt-like | 12,000 / 12,000 | 12,907 / 12,825 | 0.0972 | 0.1076 | 0.1016 | 0.1223 | 0.1288 | 0.1288 |
| 2023 2Mu2E prompt-like | 12,000 / 12,000 | 12,892 / 12,819 | 0.0967 | 0.1048 | 0.0978 | 0.1243 | 0.1250 | 0.1242 |

## The eight sets on all events, four eras

Displaced = `<chan>_200GeV_1p2GeV_4p8mm`, all 120 files (100,000 events); prompt-like = the six points `<chan>_<M>GeV_1p2GeV_<ctau>mm` of the first comparison, all 4 files each (12,000 events: these are grid-only samples, so the first comparison already used all their events). Veto "all".

### 2Mu2E displaced

| era (events v1 / v2) | pre v1 | pre v2 | v1 | v2 rebuilt | v2 - v1 (points) | s (points) | pull | s_evt (points) |
|---|---|---|---|---|---|---|---|---|
| 2022 (100,000 / 100,000; files 120 / 120) | 111,101 | 110,778 | 0.3198 | 0.3182 | -0.16 | 0.20 | -0.8 | 0.22 |
| 2022EE (100,000 / 100,000; files 120 / 120) | 111,263 | 111,606 | 0.3194 | 0.3184 | -0.10 | 0.20 | -0.5 | 0.22 |
| 2023 (100,000 / 100,000; files 120 / 120) | 110,883 | 111,028 | 0.3243 | 0.3198 | -0.45 | 0.20 | -2.3 | 0.22 |
| 2023BPix (100,000 / 100,000; files 120 / 120) | 111,230 | 110,846 | 0.3231 | 0.3225 | -0.06 | 0.20 | -0.3 | 0.22 |

| combination | d +- s (points) | d / s | s_evt (points) | eras or samples below / above zero | chi2 / ndof | p(chi2) |
|---|---|---|---|---|---|---|
| four eras | -0.19 +- 0.10 | -1.96 | 0.11 | 4 / 0 | 6.3 / 4 | 0.180 |
| 2022 + 2023 | -0.31 +- 0.14 | -2.21 | 0.16 | 2 / 0 | 5.9 / 2 | 0.052 |
| 2022EE + 2023BPix | -0.08 +- 0.14 | -0.56 | 0.16 | 2 / 0 | 0.3 / 2 | 0.846 |
| four eras, without the files of the first comparison (j040-j059) | -0.16 +- 0.11 | -1.43 | 0.12 | 3 / 1 | 3.6 / 4 | 0.467 |
| "all + charge", four eras | -0.22 +- 0.10 | -2.20 | 0.11 | 4 / 0 | 7.1 / 4 | 0.128 |

### 2Mu2E prompt-like

| era (events v1 / v2) | pre v1 | pre v2 | v1 | v2 rebuilt | v2 - v1 (points) | s (points) | pull | s_evt (points) |
|---|---|---|---|---|---|---|---|---|
| 2022 (12,000 / 12,000; files 24 / 24) | 12,907 | 12,825 | 0.0972 | 0.1016 | +0.44 | 0.37 | +1.2 | 0.40 |
| 2022EE (12,000 / 12,000; files 24 / 24) | 12,849 | 13,050 | 0.0920 | 0.0982 | +0.62 | 0.36 | +1.7 | 0.39 |
| 2023 (12,000 / 12,000; files 24 / 24) | 12,892 | 12,819 | 0.0967 | 0.0978 | +0.11 | 0.37 | +0.3 | 0.39 |
| 2023BPix (12,000 / 12,000; files 24 / 24) | 12,795 | 12,827 | 0.1009 | 0.1035 | +0.26 | 0.38 | +0.7 | 0.40 |

| combination | d +- s (points) | d / s | s_evt (points) | eras or samples below / above zero | chi2 / ndof | p(chi2) |
|---|---|---|---|---|---|---|
| four eras | +0.36 +- 0.19 | +1.95 | 0.20 | 0 / 4 | 4.9 / 4 | 0.298 |
| 2022 + 2023 | +0.28 +- 0.26 | +1.05 | 0.28 | 0 / 2 | 1.5 / 2 | 0.471 |
| 2022EE + 2023BPix | +0.45 +- 0.26 | +1.70 | 0.28 | 0 / 2 | 3.4 / 2 | 0.183 |
| "all + charge", four eras | +0.25 +- 0.21 | +1.23 | 0.22 | 1 / 3 | 2.9 / 4 | 0.568 |

### 4Mu displaced

| era (events v1 / v2) | pre v1 | pre v2 | v1 | v2 rebuilt | v2 - v1 (points) | s (points) | pull | s_evt (points) |
|---|---|---|---|---|---|---|---|---|
| 2022 (100,000 / 100,000; files 120 / 120) | 221,632 | 221,642 | 0.3190 | 0.3176 | -0.15 | 0.14 | -1.0 | 0.16 |
| 2022EE (100,000 / 100,000; files 120 / 120) | 221,157 | 221,532 | 0.3184 | 0.3183 | -0.01 | 0.14 | -0.1 | 0.16 |
| 2023 (100,000 / 100,000; files 120 / 120) | 221,470 | 221,033 | 0.3231 | 0.3224 | -0.07 | 0.14 | -0.5 | 0.16 |
| 2023BPix (100,000 / 100,000; files 120 / 120) | 221,018 | 221,394 | 0.3232 | 0.3236 | +0.04 | 0.14 | +0.3 | 0.16 |

| combination | d +- s (points) | d / s | s_evt (points) | eras or samples below / above zero | chi2 / ndof | p(chi2) |
|---|---|---|---|---|---|---|
| four eras | -0.05 +- 0.07 | -0.65 | 0.08 | 3 / 1 | 1.5 / 4 | 0.835 |
| 2022 + 2023 | -0.11 +- 0.10 | -1.10 | 0.11 | 2 / 0 | 1.4 / 2 | 0.509 |
| 2022EE + 2023BPix | +0.02 +- 0.10 | +0.19 | 0.11 | 1 / 1 | 0.1 / 2 | 0.950 |
| four eras, without the files of the first comparison (j040-j059) | -0.06 +- 0.08 | -0.81 | 0.09 | 3 / 1 | 1.5 / 4 | 0.833 |
| "all + charge", four eras | -0.04 +- 0.07 | -0.55 | 0.08 | 3 / 1 | 2.0 / 4 | 0.732 |

### 4Mu prompt-like

| era (events v1 / v2) | pre v1 | pre v2 | v1 | v2 rebuilt | v2 - v1 (points) | s (points) | pull | s_evt (points) |
|---|---|---|---|---|---|---|---|---|
| 2022 (12,000 / 12,000; files 24 / 24) | 25,652 | 25,560 | 0.1009 | 0.0969 | -0.40 | 0.26 | -1.5 | 0.28 |
| 2022EE (12,000 / 12,000; files 24 / 24) | 25,608 | 25,710 | 0.0974 | 0.0988 | +0.14 | 0.26 | +0.5 | 0.28 |
| 2023 (12,000 / 12,000; files 24 / 24) | 25,620 | 25,578 | 0.1023 | 0.1008 | -0.15 | 0.27 | -0.6 | 0.28 |
| 2023BPix (12,000 / 12,000; files 24 / 24) | 25,228 | 25,438 | 0.1029 | 0.1037 | +0.08 | 0.27 | +0.3 | 0.29 |

| combination | d +- s (points) | d / s | s_evt (points) | eras or samples below / above zero | chi2 / ndof | p(chi2) |
|---|---|---|---|---|---|---|
| four eras | -0.08 +- 0.13 | -0.63 | 0.14 | 2 / 2 | 3.0 / 4 | 0.557 |
| 2022 + 2023 | -0.28 +- 0.19 | -1.48 | 0.20 | 2 / 0 | 2.6 / 2 | 0.267 |
| 2022EE + 2023BPix | +0.11 +- 0.19 | +0.59 | 0.20 | 0 / 2 | 0.4 / 2 | 0.833 |
| "all + charge", four eras | -0.04 +- 0.15 | -0.25 | 0.16 | 3 / 1 | 1.9 / 4 | 0.760 |

## All column samples (18 per channel), four eras

Every column sample has 120 files (100,000 events) per era and production. All 36 have ctau scaled with the boost (lab decay length of order 40 cm), so all are "displaced" in the sense of the first comparison; its prompt-like points are grid-only samples (previous section). Median generated Lxy of the gen-matched (dR < 0.3) "pre" DSA muons is listed per sample (v1, 2022).

### 2Mu2E: per sample and era

| sample | median gen Lxy (cm) | 2022: v2 - v1 (points), pull | 2022EE: v2 - v1 (points), pull | 2023: v2 - v1 (points), pull | 2023BPix: v2 - v1 (points), pull | chi2 / 4 |
|---|---|---|---|---|---|---|
| 2Mu2E_1000GeV_0p25GeV_0p2mm | 17.5 | +0.27 +- 0.25, +1.1 | -0.45 +- 0.24, -1.8 | +0.03 +- 0.25, +0.1 | +0.17 +- 0.24, +0.7 | 5.1 |
| 2Mu2E_1000GeV_1p2GeV_0p96mm | 19.3 | +0.11 +- 0.22, +0.5 | +0.35 +- 0.22, +1.6 | -0.14 +- 0.22, -0.6 | +0.00 +- 0.22, +0.0 | 3.1 |
| 2Mu2E_1000GeV_5p0GeV_4p0mm | 20.2 | +0.03 +- 0.21, +0.1 | +0.17 +- 0.21, +0.8 | +0.41 +- 0.21, +1.9 | -0.29 +- 0.21, -1.3 | 6.0 |
| 2Mu2E_100GeV_0p25GeV_2p0mm | 22.8 | +0.10 +- 0.20, +0.5 | -0.02 +- 0.20, -0.1 | +0.19 +- 0.20, +0.9 | +0.24 +- 0.20, +1.2 | 2.5 |
| 2Mu2E_100GeV_1p2GeV_9p6mm | 22.8 | -0.11 +- 0.21, -0.5 | +0.26 +- 0.21, +1.2 | -0.52 +- 0.22, -2.4 | +0.02 +- 0.22, +0.1 | 7.6 |
| 2Mu2E_100GeV_5p0GeV_40p0mm | 22.7 | +0.19 +- 0.22, +0.9 | +0.40 +- 0.22, +1.9 | -0.02 +- 0.22, -0.1 | -0.08 +- 0.22, -0.3 | 4.4 |
| 2Mu2E_150GeV_0p25GeV_1p3mm | 21.2 | -0.07 +- 0.19, -0.4 | +0.23 +- 0.19, +1.2 | +0.18 +- 0.19, +0.9 | +0.40 +- 0.19, +2.1 | 6.9 |
| 2Mu2E_150GeV_1p2GeV_6p4mm | 22.2 | -0.39 +- 0.20, -1.9 | -0.15 +- 0.20, -0.7 | +0.02 +- 0.20, +0.1 | -0.13 +- 0.20, -0.6 | 4.7 |
| 2Mu2E_150GeV_5p0GeV_27p0mm | 22.4 | -0.02 +- 0.20, -0.1 | +0.06 +- 0.21, +0.3 | +0.33 +- 0.21, +1.6 | -0.10 +- 0.21, -0.5 | 2.9 |
| 2Mu2E_200GeV_0p25GeV_1p0mm | 21.2 | +0.06 +- 0.19, +0.3 | -0.04 +- 0.19, -0.2 | +0.06 +- 0.19, +0.3 | -0.12 +- 0.19, -0.6 | 0.6 |
| 2Mu2E_200GeV_1p2GeV_4p8mm | 22.0 | -0.16 +- 0.20, -0.8 | -0.10 +- 0.20, -0.5 | -0.45 +- 0.20, -2.3 | -0.06 +- 0.20, -0.3 | 6.3 |
| 2Mu2E_200GeV_5p0GeV_20p0mm | 21.8 | +0.09 +- 0.20, +0.5 | +0.35 +- 0.20, +1.8 | +0.38 +- 0.20, +1.9 | -0.02 +- 0.20, -0.1 | 7.0 |
| 2Mu2E_500GeV_0p25GeV_0p4mm | 19.3 | +0.15 +- 0.20, +0.7 | -0.15 +- 0.20, -0.8 | +0.04 +- 0.20, +0.2 | -0.68 +- 0.20, -3.4 | 12.5 |
| 2Mu2E_500GeV_1p2GeV_1p9mm | 20.5 | -0.10 +- 0.20, -0.5 | +0.02 +- 0.20, +0.1 | -0.24 +- 0.20, -1.2 | +0.25 +- 0.20, +1.3 | 3.3 |
| 2Mu2E_500GeV_5p0GeV_8p0mm | 20.9 | +0.30 +- 0.20, +1.5 | -0.24 +- 0.20, -1.2 | +0.36 +- 0.20, +1.8 | +0.06 +- 0.20, +0.3 | 7.1 |
| 2Mu2E_800GeV_0p25GeV_0p25mm | 18.0 | +0.08 +- 0.23, +0.4 | -0.40 +- 0.23, -1.8 | -0.15 +- 0.23, -0.7 | -0.01 +- 0.23, -0.0 | 3.6 |
| 2Mu2E_800GeV_1p2GeV_1p2mm | 19.7 | -0.15 +- 0.21, -0.7 | -0.21 +- 0.21, -1.0 | -0.38 +- 0.21, -1.8 | -0.26 +- 0.21, -1.2 | 6.3 |
| 2Mu2E_800GeV_5p0GeV_5p0mm | 20.4 | +0.06 +- 0.21, +0.3 | +0.47 +- 0.21, +2.3 | -0.22 +- 0.21, -1.0 | +0.32 +- 0.21, +1.5 | 8.7 |

### 2Mu2E: pooled over the 18 samples

| era | pre v1 | pre v2 | v1 | v2 rebuilt | v2 - v1 (points) | s (points) | pull | s_evt (points) |
|---|---|---|---|---|---|---|---|---|
| 2022 | 1,892,449 | 1,892,531 | 0.3428 | 0.3429 | +0.02 | 0.05 | +0.4 | 0.05 |
| 2022EE | 1,891,777 | 1,893,016 | 0.3424 | 0.3427 | +0.03 | 0.05 | +0.6 | 0.05 |
| 2023 | 1,888,855 | 1,889,430 | 0.3461 | 0.3461 | -0.00 | 0.05 | -0.1 | 0.05 |
| 2023BPix | 1,888,040 | 1,887,065 | 0.3460 | 0.3459 | -0.01 | 0.05 | -0.3 | 0.05 |

| combination | d +- s (points) | d / s | s_evt (points) | eras or samples below / above zero | chi2 / ndof | p(chi2) |
|---|---|---|---|---|---|---|
| pooled samples, four eras | +0.01 +- 0.02 | +0.30 | 0.03 | 2 / 2 | 0.6 / 4 | 0.963 |
| per-sample differences (72 = 18 samples x 4 eras) | +0.01 +- 0.02 | +0.43 | 0.03 | 34 / 38 | 98.7 / 72 | 0.020 |

### 4Mu: per sample and era

| sample | median gen Lxy (cm) | 2022: v2 - v1 (points), pull | 2022EE: v2 - v1 (points), pull | 2023: v2 - v1 (points), pull | 2023BPix: v2 - v1 (points), pull | chi2 / 4 |
|---|---|---|---|---|---|---|
| 4Mu_1000GeV_0p25GeV_0p2mm | 17.4 | +0.05 +- 0.17, +0.3 | -0.08 +- 0.17, -0.4 | -0.31 +- 0.17, -1.8 | -0.07 +- 0.17, -0.4 | 3.6 |
| 4Mu_1000GeV_1p2GeV_0p96mm | 19.4 | -0.00 +- 0.16, -0.0 | -0.21 +- 0.16, -1.3 | +0.27 +- 0.16, +1.7 | +0.23 +- 0.16, +1.5 | 6.9 |
| 4Mu_1000GeV_5p0GeV_4p0mm | 20.2 | +0.14 +- 0.15, +0.9 | +0.07 +- 0.15, +0.4 | +0.14 +- 0.15, +0.9 | +0.25 +- 0.15, +1.7 | 4.5 |
| 4Mu_100GeV_0p25GeV_2p0mm | 22.7 | -0.12 +- 0.14, -0.9 | -0.13 +- 0.14, -0.9 | -0.22 +- 0.14, -1.5 | +0.01 +- 0.14, +0.1 | 3.9 |
| 4Mu_100GeV_1p2GeV_9p6mm | 22.7 | +0.08 +- 0.15, +0.5 | +0.09 +- 0.15, +0.6 | -0.41 +- 0.15, -2.7 | -0.03 +- 0.15, -0.2 | 7.9 |
| 4Mu_100GeV_5p0GeV_40p0mm | 22.7 | +0.18 +- 0.15, +1.2 | +0.18 +- 0.15, +1.2 | -0.25 +- 0.16, -1.6 | +0.15 +- 0.16, +0.9 | 6.4 |
| 4Mu_150GeV_0p25GeV_1p3mm | 21.1 | +0.04 +- 0.13, +0.3 | +0.09 +- 0.13, +0.7 | -0.21 +- 0.13, -1.6 | -0.08 +- 0.13, -0.6 | 3.3 |
| 4Mu_150GeV_1p2GeV_6p4mm | 22.2 | +0.15 +- 0.14, +1.1 | +0.06 +- 0.14, +0.4 | -0.10 +- 0.14, -0.7 | -0.01 +- 0.14, -0.1 | 1.8 |
| 4Mu_150GeV_5p0GeV_27p0mm | 22.5 | -0.02 +- 0.15, -0.1 | +0.09 +- 0.15, +0.6 | -0.18 +- 0.15, -1.3 | +0.06 +- 0.15, +0.4 | 2.2 |
| 4Mu_200GeV_0p25GeV_1p0mm | 21.2 | +0.20 +- 0.13, +1.5 | +0.19 +- 0.13, +1.4 | +0.10 +- 0.13, +0.7 | -0.13 +- 0.13, -0.9 | 5.6 |
| 4Mu_200GeV_1p2GeV_4p8mm | 22.0 | -0.15 +- 0.14, -1.0 | -0.01 +- 0.14, -0.1 | -0.07 +- 0.14, -0.5 | +0.04 +- 0.14, +0.3 | 1.5 |
| 4Mu_200GeV_5p0GeV_20p0mm | 21.6 | -0.14 +- 0.14, -1.0 | -0.11 +- 0.14, -0.8 | -0.02 +- 0.14, -0.1 | -0.08 +- 0.14, -0.5 | 1.9 |
| 4Mu_500GeV_0p25GeV_0p4mm | 19.6 | +0.05 +- 0.14, +0.3 | -0.06 +- 0.14, -0.4 | -0.23 +- 0.14, -1.6 | +0.18 +- 0.14, +1.3 | 4.5 |
| 4Mu_500GeV_1p2GeV_1p9mm | 20.5 | +0.02 +- 0.14, +0.1 | +0.25 +- 0.14, +1.8 | +0.13 +- 0.14, +0.9 | -0.24 +- 0.14, -1.7 | 7.1 |
| 4Mu_500GeV_5p0GeV_8p0mm | 20.8 | -0.13 +- 0.14, -0.9 | -0.08 +- 0.14, -0.5 | +0.10 +- 0.14, +0.7 | -0.28 +- 0.14, -2.0 | 5.5 |
| 4Mu_800GeV_0p25GeV_0p25mm | 17.9 | +0.06 +- 0.16, +0.4 | -0.30 +- 0.16, -1.8 | +0.32 +- 0.16, +2.0 | +0.11 +- 0.16, +0.7 | 8.0 |
| 4Mu_800GeV_1p2GeV_1p2mm | 19.9 | +0.07 +- 0.15, +0.4 | +0.05 +- 0.15, +0.3 | +0.18 +- 0.15, +1.2 | -0.25 +- 0.15, -1.6 | 4.4 |
| 4Mu_800GeV_5p0GeV_5p0mm | 20.5 | +0.28 +- 0.15, +1.9 | +0.23 +- 0.15, +1.6 | +0.07 +- 0.15, +0.5 | +0.10 +- 0.15, +0.7 | 6.9 |

### 4Mu: pooled over the 18 samples

| era | pre v1 | pre v2 | v1 | v2 rebuilt | v2 - v1 (points) | s (points) | pull | s_evt (points) |
|---|---|---|---|---|---|---|---|---|
| 2022 | 3,758,289 | 3,759,656 | 0.3446 | 0.3450 | +0.04 | 0.03 | +1.1 | 0.04 |
| 2022EE | 3,761,434 | 3,763,989 | 0.3444 | 0.3446 | +0.02 | 0.03 | +0.7 | 0.04 |
| 2023 | 3,750,580 | 3,752,542 | 0.3490 | 0.3486 | -0.04 | 0.03 | -1.1 | 0.04 |
| 2023BPix | 3,751,821 | 3,752,170 | 0.3488 | 0.3488 | -0.00 | 0.03 | -0.1 | 0.04 |

| combination | d +- s (points) | d / s | s_evt (points) | eras or samples below / above zero | chi2 / ndof | p(chi2) |
|---|---|---|---|---|---|---|
| pooled samples, four eras | +0.00 +- 0.02 | +0.27 | 0.02 | 2 / 2 | 2.8 / 4 | 0.593 |
| per-sample differences (72 = 18 samples x 4 eras) | +0.00 +- 0.02 | +0.26 | 0.02 | 33 / 39 | 85.8 / 72 | 0.127 |

## Verdicts of the decision rule

Rule: |d|/s < 2 not confirmed; |d|/s > 3 with the same sign in at least three of the four eras confirmed; otherwise inconclusive.

- **eight sets on all events, 2Mu2E displaced: not confirmed.** d = -0.194 +- 0.099 points, d/s = -1.96 (event-clustered s: 0.111 points, d/s_evt = -1.74), 4 of 4 eras with the sign of d.
- **2Mu2E pooled column samples: not confirmed.** d = +0.007 +- 0.024 points, d/s = +0.30 (event-clustered s: 0.027 points, d/s_evt = +0.28), 2 of 4 eras with the sign of d.

## Checks on the rows (`xclean_checks.py`)

(a) the rebuilt count on the first production, where the stored count is correct; (b) the pooled column-sample
difference with the variance clustered by file inside each sample; (c) the file-level design effect
sum_f (k_f - p n_f)^2 / (N p (1 - p)) per production, era and sample. A bootstrap of files over the pooled samples
is not used: files hold unequal numbers of events and resampling them changes the sample mix.

```
(a) first production, rebuilt count in place of the stored one, four eras, all samples
    pre muons 22,736,796; decisions changed 199 (pass to fail 199, fail to pass 0); rebuilt = stored: all 0.9982, with CSC segments 0.9957 (of 9,641,411); rebuilt > stored: 0
(b) pooled v2 - v1, 18 column samples per channel, four eras (points)
    2Mu2E (samples per era [18]): d = +0.007 +- 0.024 (binomial), +- 0.026 (file-clustered); d/s = +0.30, +0.28
    4Mu (samples per era [18]): d = +0.005 +- 0.017 (binomial), +- 0.019 (file-clustered); d/s = +0.27, +0.20
(c) file-level design effect per (production, era, sample)
    cells 288: mean 1.18, median 1.18, min 0.72, max 1.79; per-sample means (36 samples, 8 cells each) 0.95 to 1.31
    2Mu2E v1: 72 cells, mean 1.16
    2Mu2E v2: 72 cells, mean 1.19
    4Mu v1: 72 cells, mean 1.18
    4Mu v2: 72 cells, mean 1.19
```
