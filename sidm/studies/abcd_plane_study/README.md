# ABCD plane-choice study (2018 MC)

Which ABCD plane, working points, and event-cut menu should the SIDM background
estimate use — demonstrated, not assumed, on 2018 Monte Carlo. Everything here is MC-only:
signal and background simulation, no collision data. The measurements that
require real data -- the ABCD closures, the cosmic rejection, and the
windowed-fit backgrounds -- are named throughout as the continuation of this
study, not shown here.

## Start here: what was done, and what to continue

This branch is the **Monte-Carlo foundation** of the SIDM ABCD background estimate: it fixes the
estimation *method* on simulation and hands the measurements that need real data to the
continuation. Everything here is MC-only (notebooks 07–09 even carry a guard that refuses any
collision-data file); no real-data result is quoted.

**What the MC round established** (detail and numbers in the Verdict section and notebooks 07–09):

- **4mu has no second ABCD axis.** The isolated corner is an instrumental fake, and the three
  structural candidates for an independent second axis — dimuon mass equality, opposite-sign
  charge, vertex DCA — all sit in the muon-reconstruction family and correlate with isolation
  (nb 07). A muon-only final state has no independent third family.
- **4mu is therefore a windowed dimuon-resonance mass fit** (nb 07), not an ABCD: blind a mass
  window at each hypothesis, fit the smooth sidebands, integrate; the window widths follow the
  signal mJJ cores shown there.
- **The within-LJ dimuon vertex (normChi2 < 5) cleans the 4mu fake and spares signal** (58–93%
  efficient across the grid, gentle lifetime decline; nb 08).
- **The cosmic collinearity veto is signal-safe** (≤ 0.21% signal cost across the grid; nb 09),
  unlike a plain back-to-back cut, which self-vetoes 4mu signal.
- **2mu2e can do an ABCD** because it has an independent third family: egmiso × mudisp factorizes
  on MC (κ ≈ 0.95, unweighted; nb 07). Notebook 04's MC-round primary is muiso × mJJ;
  egmiso × mudisp is the plane adopted going forward.

**What the continuation must do on data** (none of it is on this branch):

- Measure the **physical ABCD closures** on data sidebands — the MC checks here are
  variable-independence only, not weighted closures.
- Measure the **4mu vertex fake-rejection** in a data sideband (the fake is MC-invisible).
- Measure the **cosmic rejection** in a cosmic-enriched data control region, and evaluate the
  vertex-consistency `coll` variant (`COSMIC_VERTEX_CONSISTENCY.md`).
- Build and validate the **windowed-fit backgrounds** on data.
- Evaluate the **same-sign tag as a 4mu background-cleaning veto** — its signal cost is
  characterized here (nb 07); its background rejection is a data question.

**How to read it:** this file, then the notebook contents table and Verdict below; the new
material is notebooks 07 → 08 → 09. All notebooks are committed with outputs (figures and printed
numbers), so results are visible without re-running. MC inputs are on EOS under
`/store/group/lpcmetx/SIDM/coffea_outputs/murtazas/{abcd_plane_study, abcd_plane_study_member,
abcd_plane_study_pairres}` (all MC-only).


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
`"all + segment fraction < 1"`. Cosmic rejection is a data-side measurement; the
signal-safe collinearity veto and its (near-zero) signal cost are characterized in
notebook 09.

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

Notebook 01 opens with a first-principles primer (lepton jets, PF vs DSA muons,
isolation and the sentinel, the ABCD method, closure R, n_eff, event-parity
blinding) — the notebooks are written to read clearly from senior-undergrad level
up, with jargon defined at first use in each notebook.

All notebooks carry the full figure program (mplhep CMS style, from the shared
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
| 05_member_isolation | member-lepton (PF) isolation as a jet-match-free alternative axis: distributions, sentinel rates at the SR working point, jet-vs-member migration, ROC, factorization of member-iso planes, egm parity. Conclusion: it relabels the sentinel (larger, 5.8% vs 0.1% at the SR WP), discriminates worse, and fails 2mu2e factorization — the jet-based plane is retained; its value is labeling the displaced DSA-only sentinel population for notebook 06 |
| 06_mothers_and_cosmics | gen-origin (genPartFlav) composition per ABCD region (dominant prompt/b transport consistently along mJJ; subdominant c and light/DIF are inconclusive on partial stats) and cosmic-veto input distributions/costs (dz spread; min cos α — note this fires on the 4mu di-LJ back-to-back topology, not only cosmics, so it needs signal-LJ exclusion before it is a clean veto; rejection to be measured in a data sideband) |
| 07_beyond_isolation_candidate_axes | the three structural candidate axes for a second 4mu ABCD axis (dimuon mass equality, opposite-sign charge, vertex DCA) on signal MC: their signal discrimination and its mass/lifetime pattern, the two-family argument for why none is independent of isolation, and the windowed dimuon-resonance mass fit that replaces the ABCD, with the signal mJJ line-shape cores that set the windows |
| 08_vertex_fake_killer | the within-LJ dimuon vertex (normChi2 < 5) that cleans the isolated-4mu instrumental fake, shown signal-preserving on MC (58–93% across the grid, gentle lifetime decline) and contrasted with a resolution-limited DCA cut; the fake rejection is a data measurement |
| 09_cosmic_collinearity_veto | the signal-safe cosmic veto — a third-muon collinearity tag that excludes the signal lepton-jets (plugging the cos α self-veto flagged in notebook 06) and its vertex-consistency variant — with its signal cost measured on MC; cosmic rejection deferred to a data control region |

Merged inputs + `.meta.yaml` sidecars:
`/store/group/lpcmetx/SIDM/coffea_outputs/murtazas/abcd_plane_study/` (notebooks 01–04) and `.../abcd_plane_study_member/` (notebooks 05–06). Notebooks 07–09 read the
pair-resonance / vertex / cosmic-tag signal-and-background MC from
`.../abcd_plane_study_pairres/` (MC-only; contains no collision data).

## Verdict: the 4mu and 2mu2e background methods

This branch is the Monte-Carlo foundation of the background estimate. The
isolation-plane survey (notebooks 01–06) is summarized here; the two follow-up
questions it left open — is there a second 4mu axis, and does the vertex/cosmic
machinery spare signal — are answered on MC in notebooks 07–09; and the
measurements that require real data are named as the continuation.

**2mu2e — muiso × mJJ in the MC round, egmiso × mudisp adopted going forward.**
Within the isolation-plane survey the best plane, and the one notebook 04
designates the MC-round primary, is muiso × mJJ (P4: factorizes at
p = 0.762/0.756, signal-pristine sidebands, ladder anchor R = 0.77 ± 0.26). The plane the analysis
adopts, however, pairs two variables from *independent families*: electron/
photon isolation (egm) and muon displacement (mudisp). The EGM leg is a third
observable family — one a muon-only final state does not have — so these axes
share no latent variable, exactly the condition an ABCD needs. On background MC
the two variables factorize (notebook 07: unweighted A·D/(B·C) ≈ 0.95 at
mJJ ≥ 150 — a variable-independence check, not a physically-weighted closure,
since weighted 2mu2e MC is n_eff-limited). The physical closure, measured with real sideband
statistics, is a data-sideband measurement and is the continuation of this study.

**4mu — a windowed mass fit, because no ABCD exists.** Every isolation ABCD
plane fails in 4mu: the isolated corner is saturated and dominated by an
instrumental DSA-fake (notebooks 02, 04 — n_eff(A) ≈ 1, empty sidebands).
Notebooks 07–08 close the question of whether *any* second axis independent of
isolation exists: the three structural candidates — dimuon mass equality,
opposite-sign charge, vertex DCA — all live in the muon-reconstruction family
and are correlated with isolation; there is no third family to draw an
independent axis from. The 4mu background is therefore estimated by a **windowed
dimuon-resonance mass fit** (notebook 07): blind a mass window at each hypothesis,
fit the smooth sidebands, integrate — a smoothness assumption in place of the
factorization one 4mu cannot satisfy. The within-LJ vertex fake-killer
(notebook 08, normChi2 < 5) defines the clean region the fit runs on and is
signal-preserving (58–93% across the grid); its rejection of the fake is a data
measurement.

**Cosmics (notebook 09).** MC has no cosmics, so rejection is a data-sideband
measurement; what MC establishes is that the signal-safe collinearity veto — a
third-muon back-to-back tag that excludes the signal lepton-jets, and its
vertex-consistency variant — costs essentially no signal, unlike a plain cos α
cut, which self-vetoes the back-to-back 4mu topology (notebook 06).

**Scope and honest limits.**
- The single high-mass SR (mJJ ≥ 150 GeV) is a benchmark-scoped choice pending
  collaboration sign-off; it forecloses m_B ≤ 200 limits (at m_A = 750 GeV,
  σ(m_B ≤ 200) < 0.05 fb), and the windowed search is what recovers low-mass
  points. The two-region machinery is retained as the fallback.
- Weighted MC cannot validate tight-working-point closure directly for any plane
  (effective counts ≈ 1 from single large-weight QCD events); the gates quantify
  this honestly, and every closure normalization is a data-sideband measurement.
- The sentinel (failed-jet-match) isolation population — 95% of the naive tight-SR
  background — and its prediction through the mJJ direction is untested at
  weighted-MC statistics; the decisive test is data.
- The jet-matched iso × iso plane (prescription iii, anchor R = 0.60 ± 0.24) is
  the mechanical survivor of the pre-registered gates, kept as a cross-check with
  its lifetime-dependent signal cost quoted. Excluded: muiso × |Δφ| (p = 0.011),
  |Δφ| × mJJ (p = 0.000), the egm-iso axis in any *isolation* pairing (fail-
  sideband holds 5–86% of displaced signal), pure displacement planes
  (quasi-boolean, signal-defining).
- DY rogue generator weights (both powheg samples) were repaired (M10to50) or
  excluded and bounded (M50); superseded by the DYJetsToLL migration. Rogue-file
  list saved for the production team.

## Known deltas vs the study design doc

- The legacy `abcd_base` lj_lj 2D histograms ride along but use unwrapped |φ₁−φ₀|
  (range 0–2π) — do not mix them with the scan hists' wrapped |Δφ| without folding.
- Dongyub's 2D iso histograms are not duplicated here; they are projections of the
  scan hists.
- The `"mJJ >= 150 GeV"` event cut in `cuts.py` is staged for the eventual SR menu;
  nothing in this study's channels consumes it (scans use the mJJ axis instead).
