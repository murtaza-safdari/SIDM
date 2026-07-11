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
| 2mu2e | iso(mu-LJ) × iso(egm-LJ) *(incumbent)*, muiso × \|Δφ\|, egmiso × \|Δφ\|, muiso × mJJ, egmiso × mJJ, iso × displacement, displacement × \|Δφ\|, \|Δφ\| × mJJ |
| 4mu | iso(LJ₀) × iso(LJ₁) *(incumbent)*, iso × \|Δφ\|, iso × mJJ, iso × displacement, disp × disp, \|Δφ\| × mJJ |

Whatever is not a plane axis becomes an event-level cut (SR candidates: |Δφ| ≥ 2.0,
mJJ ≥ 150 GeV, standard displacement, iso ≤ WP). Derived/discrete planes (N_tight,
N_jet-matched, photon-only category) are evaluated offline from the same histograms.

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
Each frozen configuration is then read once on **odd-numbered events** at its
ladder anchor stage (notebook 04, which keeps a look log of every odd-half and
full-statistics inspection relative to the primary designation). Seven declared
amendments (MC-statistics rescopes of gates 1–3, prescription matching, the DY
exclusion, the primary-designation override, and the sentinel-disqualifier
rescope) are recorded in notebook 04 next to the gate evaluation.

If every plane fails gate 3 while its rebinned factorization fit gives χ²/ndf < 1.5,
the multi-bin factorization prediction (extended ABCD) is promoted to baseline.

## The isolation quirk

LJ isolation is `(E_jet/E_LJ)·(1 − lepton fraction)` of the nearest AK4 jet within
ΔR < 0.4; when **no jet matches, the processor silently records isolation = 0**
("perfectly isolated"). In the scan histograms these LJs sit in a dedicated sentinel
bin ([−0.02, 0)) instead. Three prescriptions are compared everywhere: (i) sentinel
merged into the first real bin (reproduces current behavior), (ii) sentinel events
dropped, (iii) sentinel excluded from the plane (gap at the axis). Measured rates:
0.01–36% of signal mu-LJs depending on ctau (~0% at the shortest lifetimes, rising
steeply for displaced LJs), ~0% of egm-LJs. If failed-match events exceed ~20% of region A for a plane,
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
on the egm side, lostHits = 999 = photon-only LJ (strongly lifetime-dependent:
1.6–89% of signal 2mu2e events across the grid — the collimated ee pair often
reconstructs photon-like; these auto-pass the missing-hits requirement).

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

All four notebooks carry the full figure program (mplhep CMS style, from the shared
`plot_helpers.py`): stacked scan-variable distributions, plane views with the
A/B/C/D regions drawn, shape-invariance panels, factorization pull maps and
p-value matrices, closure ladders with the acceptance band, leakage bar charts,
and the decision matrix.

| notebook | contents |
|---|---|
| 01_plane_atlas | normalization + xsec table, quirk rates (bkg + signal vs cτ), scan-variable stacks, plane distributions per process, correlations, shape-invariance panels, displacement-convention check, mJJ spectra + the σ(m_B) relevance curve |
| 02_closure_scans | factorization-fit gates with p-value matrices and pull maps, staged loose→tight ladders, guard bands, sentinel-share + jet-match signal cost, low/high-mass regions, inverted-\|Δφ\| validation region |
| 03_signal_sensitivity | which masses carry the search, per-point efficiency, sideband-leakage bars, prediction bias vs signal strength, Asimov ranking |
| 04_selection_verdict | decision matrix (even events), declared primary + mechanical survivor, plateau maps, odd-event confirmations |

Merged inputs + `.meta.yaml` sidecars:
`/store/group/lpcmetx/SIDM/coffea_outputs/murtazas/abcd_plane_study/`.

## Verdict of the 2018 MC round (summary — details, figures and the look log in notebook 04)

- **Primary: muiso × mJJ under prescription (i), single high-mass SR
  (mJJ ≥ 150 GeV — a benchmark-scoped choice pending collaboration sign-off),
  low-mJJ side as validation region.** Several planes factorize with the
  failed-jet-match (sentinel) population included; P4 (p = 0.762/0.756 under both
  prescriptions) is the only non-screened one that combines that with a measurable
  closure ladder AND signal-pristine sidebands (mu-iso sideband ≤ 2.1% of SR
  signal for every m_B ≥ 500 point; low-mJJ sideband leakage < 10⁻³). No jet-match
  requirement is needed, so none of its lifetime-dependent signal cost is paid
  (≤ 5% short cτ, 12–32% long in 2mu2e; up to 48% in 4mu) and DSA-only lepton-jets
  stay in the search. In the orthogonal inverted-|Δφ| VR it keeps factorizing
  (p = 0.18) where the incumbent-as-operated fails (p ≈ 0.000); its closure there
  is inconclusive at MC statistics and is repeated in data. Its ladder anchor
  closes (R = 0.77 ± 0.26, carried systematic ~49%; one-look odd-half consistency
  R = 1.08 ± 0.40). The promotion over the mechanical gate survivor is amendment 6;
  the rescope of the registered sentinel disqualifier that would otherwise reject
  this primary is amendment 7 — both recorded openly in notebook 04.
- **The load-bearing sentinel hypothesis is stated as such**: sentinel events (95%
  of the naive tight-SR background) can only populate regions A and C here, and
  predicting them through the mJJ direction is untested at weighted-MC statistics
  (the factorization gate has no power on it; notebook 02 shows the preselection
  shape comparison). The decisive test is data sidebands in round 2.
- **mJJ regions**: at the dijet-constrained m_A = 750 GeV benchmark,
  σ(m_B ≤ 200) < 0.05 fb (< 3 events produced in 2018), so the m_B = 100 samples
  are illustrations. The single high-mass SR forecloses m_B ≤ 200 limits under
  that scope — a collaboration-level decision; the two-region machinery is
  retained as the fallback.
- **Cross-check: jet-matched iso×iso (prescription iii)** — the mechanical
  survivor of the pre-registered gates (p = 0.48, anchor R = 0.60 ± 0.24) — kept
  with its lifetime-dependent signal cost quoted.
- **Excluded**: muiso × |Δφ| (p = 0.011 + ~2× non-closure), |Δφ| × mJJ
  (p = 0.000), the egm-iso axis in any pairing (its fail-sideband holds 5–86% of
  the SR signal for displaced points; egmiso × mJJ also fails factorization,
  TTJets p = 0.009), displacement planes (quasi-boolean, signal-defining).
- Weighted MC cannot validate tight-WP closure directly for any plane (effective
  counts ≈ 1 from single large-weight QCD events); the gates quantify this honestly
  and the decisive re-test moves to (unweighted) data sidebands in round 2, with
  the expected data counts per region quoted in notebook 03.
- 4mu: effectively background-free → counting-experiment SR model whose Poisson
  mean is constrained by observed sideband counts via likelihood-ABCD rateParams
  on the Q6 muiso × mJJ structure — never an assumed zero. Q6 fails the registered
  per-process gate for QCD (p = 0.037/0.044 at 1–2 ndf) while the total passes
  (0.51/0.57); it is adopted as a scaffold to be tested on data, stated plainly in
  notebook 04.
- The extended-ABCD (per-LJ fake-factor) estimator is a data-round upgrade candidate
  (unbiased + smaller variance on toys, matches at presel; unstable in ultra-sparse
  regions).
- DY: rogue generator weights found in both powheg samples (M10to50 repaired via a
  rogue-free denominator; M50 excluded and bounded — its skimmed events are
  contaminated). Superseded by the DYJetsToLL migration. Rogue-file list saved for
  the production team.

## Known deltas vs the study design doc

- The legacy `abcd_base` lj_lj 2D histograms ride along but use unwrapped |φ₁−φ₀|
  (range 0–2π) — do not mix them with the scan hists' wrapped |Δφ| without folding.
- Dongyub's 2D iso histograms are not duplicated here; they are projections of the
  scan hists.
- The `"mJJ >= 150 GeV"` event cut in `cuts.py` is staged for the eventual SR menu;
  nothing in this study's channels consumes it (scans use the mJJ axis instead).
