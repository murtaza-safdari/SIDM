# Background estimation for the SIDM search — documentation series

Self-contained notebooks documenting the data-driven background estimate for the
4mu and 2mu2e channels on the full 2018 dataset, written to be readable without
prior knowledge of the analysis: every sample, cut, and variable is defined
before it is used.

| Notebook | Content |
|---|---|
| `01_abcd_from_first_principles.ipynb` | The signal and the lepton-jet final states; the samples; the event selection; the ABCD method and its assumptions; the ABCD variables; the blinding scheme; the deployed 2mu2e estimate with its closure measurement and caveats |
| `02_closure_tests_and_the_4mu_verdict.ipynb` | The full closure-test compendium at 2018 statistics: why every isolation-based ABCD plane fails in 4mu (dilution effect, conditioned and cross-object planes); the instrumental artifact in the isolated 4mu corner; the dimuon-vertex discriminant and its validations; era stability and the alternative-plane cross-check in 2mu2e; the 4mu verdict |
| `03_method_tests_and_recommendation.ipynb` | The three non-ABCD methods (smooth-shape fits, transfer factors, MC-assisted) tested in both channels on a dedicated fine-binned mJJ campaign, every extrapolation validated in never-blinded regions; the failure of all three simple forms, the local-window construction that passes, and the unified windowed-resonance-search recommendation |
| `04_windowed_pass_first_results.ipynb` | The approved windowed blinding (pre-registered mass windows replacing the fixed box), its verification, and first results of the windowed data pass: the pass-region spectra, in-region local-fit validation with the measured 4mu bias, first per-window background estimates, and the deployed 2mu2e ABCD's first data confrontation |
| `05_frozen_prescription_expected_limits.ipynb` | The frozen background prescription (per-fitter bias corrections, bootstrap statistical errors, family systematics; the coarse wide-range fitter for the sparse 2mu2e windows with its own open-region validation) and the first median expected 95% CLs limits on σ×BR per signal point — including the generator-filter correction to the signal efficiencies (Runs genEventCount vs stored events, ~10%) that the σ×BR normalization requires |
| `06_pair_resonance_candidate_axes.ipynb` | Whether 4mu admits a second background axis independent of isolation: three structural candidates — lepton-jet mass equality, muon-pair charge, and dimuon-vertex DCA — measured against isolation on the open sidebands (the vertex-cleaned test is the honest one; the raw mixture fakes factorization by dilution) and across the signal grid. All three fail, with mass equality the closest miss, confirming the windowed fit as the 4mu method; the pass also surfaces a same-sign veto that would clean ~40% of the 4mu background, quantified with its lifetime-dependent signal cost and left as an unapplied recommendation |

The ABCD region counts, closure ratios, predictions, survival fractions, and
method-test results in the notebooks are recomputed at execution time from the
merged histograms of the blinded Condor campaigns (`AbcdVtx_Run2018_v1`,
`AbcdVtxCross_Run2018_v2` for Parts 1–2; `AbcdMjjFine_Run2018_v1` for Part 3;
`AbcdMjjWin_Run2018_v1` for Part 4);
a few quoted figures (the Z → μμ vertexing control, the cosmic-ray study) come
from standalone campaign scripts and are cited as such where used. Part 3's
weighted MC composition additionally reads the background file census
(per-sample generated-event counts of the unskimmed parents). Data never
fills signal-region bins (see the blinding section of notebook 01). The
histogram and cut definitions live in `sidm/definitions/hists.py` and
`sidm/definitions/cuts.py`; the campaign submission machinery is under
`condor/`.

The notebooks read local copies of the merged campaign outputs (the `CACHE`
variables at the top of each notebook). The shared masters live on EOS under
`/store/group/lpcmetx/SIDM/coffea_outputs/murtazas/abcd_vtx_run2018`,
`.../abcd_vtxcross_run2018_v2`, `.../abcd_mjjfine_run2018`, and
`.../abcd_mjjwin_run2018`; to re-execute
elsewhere, `xrdcp` those directories to local disk (reads work with a Kerberos
ticket) and point the `CACHE` paths at the copies. The merges themselves were
produced with `sidm/scripts/merge_coffea_chunks_eos.py` from the per-chunk
campaign outputs.
