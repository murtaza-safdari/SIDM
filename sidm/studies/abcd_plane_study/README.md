# ABCD plane-choice study (2018 MC)

Which ABCD plane, working points, and event-cut menu should the SIDM background
estimate use — demonstrated, not assumed. Everything here is MC-only (2018); data
control regions are deliberately deferred until the region definitions are settled and
signal contamination is shown to be small.

## What is being decided

The background estimate predicts the signal-region yield from sidebands via
`A_pred = B·C/D` on a 2D plane. Candidate planes (per channel):

| channel | candidates |
|---|---|
| 2mu2e | iso(mu-LJ) × iso(egm-LJ) *(incumbent)*, iso × \|Δφ\|, iso × mJJ, iso × displacement, displacement × \|Δφ\|, \|Δφ\| × mJJ |
| 4mu | iso(LJ₀) × iso(LJ₁) *(incumbent)*, iso × \|Δφ\|, iso × displacement, disp × disp, \|Δφ\| × mJJ |

Whatever is not a plane axis becomes an event-level cut (SR candidates: |Δφ| ≥ 2.0,
mJJ ≥ 150 GeV, standard displacement, iso ≤ WP).

## Pre-registered decision rule

Declared before any campaign output was inspected (see git history of this file).
Lexicographic gates, evaluated per plane and channel, on **even-numbered events only**:

1. **Independence** — factorization-fit goodness-of-fit p > 0.05 for each background
   process (QCD, DY, TTJets, diboson) AND the total, under all three isolation-quirk
   prescriptions (below). The fit runs on an adaptively rebinned grid with
   n_eff ≥ 10 per super-bin; χ² validated with toys (`test_abcd_tools.py`).
2. **Statistical health** — n_eff = (Σw)²/Σw² > 10 in B, C, and D at the working point.
3. **Closure trend** — fitted R(cut-tightness), extrapolated to the SR point, consistent
   with 1 within max(2σ_fit, 0.25). Scan-point correlations from a per-bin bootstrap.

Survivors are ranked by Asimov Z (with the non-closure systematic folded into the
background uncertainty); tiebreak = largest min n_eff(B, C, D); the chosen working
point must sit on a plateau (verdict unchanged under ±1-bin boundary shifts).
Displacement planes are screened-only (quasi-boolean axes admit no closure trend).
The single chosen configuration is then confirmed once on **odd-numbered events**
(notebook 04) — those numbers are not inspected before the choice is frozen.

If every plane fails gate 3 while its rebinned factorization fit gives χ²/ndf < 1.5,
the multi-bin factorization prediction (extended ABCD) is promoted to baseline.

## The isolation quirk

LJ isolation is `(E_jet/E_LJ)·(1 − lepton fraction)` of the nearest AK4 jet within
ΔR < 0.4; when **no jet matches, the processor silently records isolation = 0**
("perfectly isolated"). In the scan histograms these LJs sit in a dedicated sentinel
bin ([−0.02, 0)) instead. Three prescriptions are compared everywhere: (i) sentinel
merged into the first real bin (reproduces current behavior), (ii) sentinel events
dropped, (iii) sentinel excluded from the plane (gap at the axis). Measured rates:
~3% of signal mu-LJs, ~0% of egm-LJs (AK4 jets do cluster muons, so mu-LJ isolation
is usually a real value). If failed-match events exceed ~20% of region A for a plane,
prescription (i) is disqualified there — no sideband constrains that population.

## How the histograms work

One processing pass fills N-dimensional histograms (channels `4mu_abcd_scan` /
`2mu2e_abcd_scan`, hist collection `abcd_scan`) whose axes carry every scan variable:
iso (0.025-wide bins, sentinel + catch-all), |Δφ| (edges include 2.0), mJJ (edges
include 150), displacement categories, and an event-parity axis (the even/odd split).
Every promised cut value is an exact bin edge, so all ABCD-boundary scans AND
event-cut scans happen offline, from the same files, with per-bin sumw². Displacement
axes use PF-muon pixel hits with two special categories: −1 = no pixel info (DSA-only
LJ, or a trackless PF muon — both auto-pass every displacement convention in use) and,
on the egm side, lostHits = 999 = photon-only LJ (~63% of signal 2mu2e events — the
collimated ee pair often reconstructs photon-like; these auto-pass the missing-hits
requirement).

## Selection baked into the channels

Trigger (4-path DoubleL2Mu OR) + PV filter; LJ pT > 30 GeV, |η| < 2.4; ≥2 LJs;
channel classifier on the leading two LJs; and the team-final DSA-muon cross-cleaning
`"all + segment fraction < 1"` (identical definition adopted on Dongyub's and Maria's
branches, 2026-07-09). No cosmic veto (MC-only round; cosmics are a data-side issue).

## Normalization (read before quoting any yield)

- Campaign hists are per-chunk scaled and merged with
  `sidm/scripts/merge_coffea_chunks_eos.py` (the chunk-unwrap + renormalization path
  is regression-tested in `test_merge_unwrap.py`).
- **Final per-sample normalization happens offline** (`abcd_tools.offline_norm_factor`):
  hists × `scaled_sum_weights`(merged metadata) / Σw_pre, where Σw_pre is the
  pre-skim / pre-filter `genEventSumw` from the file census. This matters:
  - the 2018_v2 background skims carry **no Runs tree** (Σw_pre is not recoverable
    from the skims; the unskimmed census provides it), and
  - the v10 signal ntuples are **gen-filtered** (only ~10% of generated events reach
    the ntuple, point-dependent) — normalizing by processed Σw would overstate signal
    ~10× and distort point-to-point comparisons.
- `processed_fraction.json` records the fraction of each sample's post-skim events
  actually staged (census-vetoed files are all empty ⇒ f_w = 1 unless noted).
- **Campaign cutflows are relative-only** (the condor path hardcodes skim_factor = 1);
  never quote absolute yields from them.
- TTJets is produced at the repo's 471.7 pb; the NNLO+NNLL 831.76 pb move is a
  pending, explicit sign-off (`abcd_tools.ttjets_xsec_rescale`), not a silent default.
- Signal cross section is the 1 fb reference convention.

## Reproducing

```bash
# processing (condor, from the branch worktree)
python condor/make_job_args.py ...            # see condor/job_args_abcd_scan.txt provenance
condor_submit condor/submit_abcd_scan.sub     # channels + abcd_scan + weighted mode
python sidm/scripts/merge_coffea_chunks_eos.py --filelists-dir condor/filelists_abcd \
    --selections 4mu_abcd_scan,2mu2e_abcd_scan --hist-collections abcd_scan \
    --output-eos-dir /store/group/lpcmetx/SIDM/coffea_outputs/murtazas/abcd_plane_study
# statistics self-tests
python -m pytest sidm/studies/abcd_plane_study/
```

## Notebooks

| notebook | contents |
|---|---|
| 01_plane_atlas | normalization + xsec table, quirk rates, plane distributions per process, correlations, displacement-convention check, mJJ-sculpting inputs |
| 02_closure_scans | factorization-fit gates, boundary + event-cut scans with bootstrap covariance, guard bands, κ study, non-closure systematic, round-2 validation region |
| 03_signal_sensitivity | per-point efficiency, sideband leakage, prediction bias vs signal strength, Asimov ranking |
| 04_selection_verdict | decision matrix (even events), chosen SR + WPs, odd-event confirmation |

Merged inputs + `.meta.yaml` sidecars:
`/store/group/lpcmetx/SIDM/coffea_outputs/murtazas/abcd_plane_study/`.

## Verdict of the 2018 MC round (summary — details in notebook 04)

- Weighted MC cannot validate tight-WP closure for any plane (single large-weight
  QCD events give effective counts ≈ 1 in every tight region); the decisive closure
  test moves to data sidebands in round 2. The gates quantify this honestly instead
  of quoting a fragile closure number.
- The incumbent iso×iso plane fails total-background factorization at preselection
  (p = 0.006, DY + process mixture) with presel closure R = 0.45 ± 0.20 — do not use
  as-is. **muiso × mJJ** factorizes for every process and the total (p ≈ 0.9), closes
  at presel, and provides the low/high-mass two-region split natively — the leading
  candidate. muiso × |Δφ| shows real ~2× presel non-closure — disfavored.
- Failed-jet-match (isolation = 0) events are 95–100% of the naive tight-SR
  background and unconstrained by any isolation sideband; a jet-matched SR removes
  them for 0–4% signal cost at short/mid cτ (30–50% at the longest lifetimes) —
  recommended baseline, with the no-jet population as an explicit separate category.
- 4mu is effectively background-free at the working points → counting treatment.
- The extended-ABCD (per-LJ fake-factor) estimator is unbiased with smaller variance
  than B·C/D on toys and reproduces it at preselection; its current implementation is
  unstable in ultra-sparse tight MC regions, so it is a data-round upgrade candidate
  (sparsity disappears there), with plain ABCD as the cross-check.

## Known deltas vs the study design doc

- The legacy `abcd_base` lj_lj 2D histograms ride along but use unwrapped |φ₁−φ₀|
  (range 0–2π) — do not mix them with the scan hists' wrapped |Δφ| without folding.
- Dongyub's 2D iso histograms are not duplicated here; they are projections of the
  scan hists.
- The `"mJJ >= 150 GeV"` event cut in `cuts.py` is staged for the eventual SR menu;
  nothing in this study's channels consumes it (scans use the mJJ axis instead).
