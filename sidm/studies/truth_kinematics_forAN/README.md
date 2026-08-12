# Truth kinematics for the analysis note

Generator-level characterization of the 2018 v10 signal samples for the
`signal_kinematics` section of AN-23-107: proper-lifetime validation, lepton
impact parameter, boost, dark-photon polarization, reconstruction migration,
and the production gen-filter story. Three executed notebooks with their
figures in `figures/` (vector PDF, CMS Simulation style; the repo-wide
`*.pdf` ignore keeps them untracked — they regenerate on notebook execution,
render inline in the committed notebooks, and are copied into the analysis
note repository).

## The canonical output

All notebooks read one production:
`/store/group/lpcmetx/SIDM/coffea_outputs/murtazas/truth_kinematics_forAN/`
(180 per-sample `.coffea` files + metadata sidecars; full statistics, every
file of every sample; unweighted histograms). Load it with
`_lifetime_refit.load_truthkin()`, which caches locally.

Channels, defined in `sidm/configs/selections.yaml`:

- `genOnly` — status-1 generator leptons as the only object definitions;
  **no event cuts at all** (no trigger, no PV filter, no vertex requirements).
- `genOnly_born` — the status-23 variant, for rest-frame comparisons.
- `genFilterEmulation` / `genFilterEmulation_isHardProcess` — re-apply the
  central production gen-filter cut string (two statusFlags variants). Their
  cutflows are re-pass rates on already-filtered events (0.84-1.00 across the
  grid, median ~0.97) that validate the filter form — never a filter
  efficiency (see below).
- `baseNoLj_noTrigger` / `baseNoLjNoLjsource_noTrigger` — analysis LJ-source
  object definitions with/without quality cuts, PV filter, **no HLT**; used
  for the reconstruction-migration maps.

Hist collections: `genA_lifetime`, `gen_truth` (new: the pure-generator subset
of `genE_base`+`genMu_base`, plus `fromA` pT/dxy variants), 
`genBS_genA_kinematics`, `genA_base`, `lepton_genA_base`.

Produced with the standard condor pipeline: `condor/submit_truthkin.sub`
plus job arguments regenerated from the committed sample lists with
`condor/make_job_args.py` (`--files-per-job 5 --replace-xcache`, once per
channel YAML; 1833 jobs), merged with
`sidm/scripts/merge_coffea_chunks_eos.py`. The Dask scripts
`_truthkin_gen_run.py` / `_recomigration_run.py` document the same
configuration and serve for small reruns; a full-grid Dask driver does not
survive the login node.

## Notebooks

- `lifetime_forAN.ipynb` — proper-lifetime faithfulness and the
  acceptance-corrected recovery, at ~50x the statistics of the original
  `lifetime_study` pass. Faithful-regime closure: median measured/nominal
  = 0.9999 (N=130). Acceptance-corrected: median 1.002, 119/180 within 5%.
  Fitted lab cap R_max = 814 cm over the heavily-truncated samples; the
  truncation-onset test covers all 36 mass points.
- `truth_kinematics_forAN.ipynb` — gen lepton |d0| across the lifetime grid
  (the variable the NoVtx triggers and displaced reconstruction respond to),
  the boost map across the mass grid, and the gen-filter efficiency figure.
- `polarization_migration_forAN.ipynb` — the polarization-fit summary
  (transverse alpha ~ 1 in the analysis sweet spot; muon velocity suppression
  at M_Zd = 0.25 GeV) and the reconstruction-migration maps
  (electron-to-photon and PF-to-DSA handoff vs Lxy).

Builder scripts (`_build_*_notebook.py`) regenerate each notebook
deterministically; execute with
`jupyter nbconvert --to notebook --execute --inplace <name>.ipynb`.

## The production gen filter

The v10 samples were produced **with the central gen filter applied**: at
least 4 e/mu with pT > 5 GeV, |eta| < 2.4, and production vertex within
rho < 740 cm, |z| < 960 cm. Evidence: the samples' own `GenFilterInfo`
bookkeeping (e.g. tried 4500 / passed 2325, eff = 0.517, in the retained
`4Mu MBs-500 MDp-1.2 ctau-19mm` AODSIM), and the kinematic walls in the
stored events sitting exactly on the filter thresholds. Re-applying the cut
string at the ntuple level passes 0.84-1.00 of stored events depending on the
mass point (lowest at the lightest masses, where leptons sit near the pT
threshold) — a form validation on filtered events, not an efficiency. The `CutDecayFalse`
sample-name prefix refers to the MadGraph run-card flag (which removes
LHE-level cuts on the decay leptons and is indeed False); it does not mean
the samples are unfiltered. Note `edmProvDump` does not reveal this filter;
`GenFilterInfo` does.

Two consequences:

1. The lifetime truncation modeled in `lifetime_forAN.ipynb` (R_max ~ 8 m) is
   this filter's vertex cylinder, approximated by a single radius.
2. **Normalization**: the stored sum of generator weights is post-filter. Any
   absolute signal normalization must use sigma x eff_filter with the
   per-point efficiency — otherwise yields are overstated by 1.1-5.2x.

Per-point central-production efficiencies (with uncertainties) are committed
here as `central_genFilterEfficiencies.yml`
(source: `phylsix/Firefighter`, `ffConfig/python/production/Autumn18/sigmc/central/`).
Two points were re-measured from scratch with the exact production
configuration (gridpack from cvmfs, CMSSW_10_2_16_UL, GenFilterInfo counting)
and agree: 0.8607 +- 0.0035 vs 0.8593 recorded, and 0.6060 +- 0.0049 vs
0.5994, for the 2Mu2E mXX-1000 mA-0.25 prompt and lxy-300 points.

## Framework changes in this branch

- `sidm/definitions/cuts.py`: `_gen_filter_count` helper + the two
  gen-filter event cuts.
- `sidm/configs/selections.yaml`: the four new channels above.
- `sidm/configs/hist_collections.yaml`: the `gen_truth` collection.
- `sidm/definitions/hists.py`: fix — eight leading/subleading gen dxy
  histograms masked the leptons but not the PV reference, so their fills
  silently skipped whenever any event failed the event mask; also new
  `fromA` pT/dxy histogram definitions.
- `sidm/tools/selection.py`: fix — the cut-application excepts now re-raise
  I/O errors instead of swallowing them, so a transient xrootd failure during
  lazy-array materialization fails the chunk loudly rather than silently
  corrupting the selection state. Downstream handling is the pipeline's:
  with `skipbadfiles=True` coffea skips chunks whose error text matches its
  bad-file patterns and otherwise kills the job for manual resubmission —
  it does not retry. The event-cut warning now prints the exception.
- `sidm/definitions/hists.py`: fix — the four 2D `*_nearGenA_n_genA_lxy`
  histograms mixed a per-dark-photon axis with a per-event axis, silently
  dropping every processing chunk containing an event with other than one
  matching dark photon (all 4Mu content and ~10% of 2Mu2E events). Both axes
  now count per dark photon; the canonical production carries the fixed
  definitions.
