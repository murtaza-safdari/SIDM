# Background estimation for the SIDM search — documentation series

Self-contained notebooks documenting the data-driven background estimate for the
4mu and 2mu2e channels on the full 2018 dataset, written to be readable without
prior knowledge of the analysis: every sample, cut, and variable is defined
before it is used.

| Notebook | Content |
|---|---|
| `01_abcd_from_first_principles.ipynb` | The signal and the lepton-jet final states; the samples; the event selection; the ABCD method and its assumptions; the ABCD variables; the blinding scheme; the deployed 2mu2e estimate with its closure measurement and caveats |
| `02_closure_tests_and_the_4mu_verdict.ipynb` | The full closure-test compendium at 2018 statistics: why every isolation-based ABCD plane fails in 4mu (dilution effect, conditioned and cross-object planes); the instrumental artifact in the isolated 4mu corner; the dimuon-vertex discriminant and its validations; era stability and the alternative-plane cross-check in 2mu2e; the 4mu verdict |
| `03_*` (forthcoming) | Non-ABCD methods tested on a dedicated fine-binned mJJ campaign, in both channels: bump-hunt, MC-assisted shapes, transfer factors — aiming at a single unified method |

The ABCD region counts, closure ratios, predictions, and survival fractions in
the notebooks are recomputed at execution time from the merged histograms of
the blinded Condor campaigns (`AbcdVtx_Run2018_v1`, `AbcdVtxCross_Run2018_v2`);
a few quoted figures (the Z → μμ vertexing control, the cosmic-ray study) come
from standalone campaign scripts and are cited as such where used. Data never
fills signal-region bins (see the blinding section of notebook 01). The
histogram and cut definitions live in `sidm/definitions/hists.py` and
`sidm/definitions/cuts.py`; the campaign submission machinery is under
`condor/`.

The notebooks read local copies of the merged campaign outputs. The shared
masters live on EOS under
`/store/group/lpcmetx/SIDM/coffea_outputs/murtazas/abcd_vtx_run2018` and
`.../abcd_vtxcross_run2018_v2`; to re-execute elsewhere, `xrdcp` those
directories to local disk (reads work with a Kerberos ticket) and point the
`CACHE*` paths at the copies. The merges themselves were produced with
`sidm/scripts/merge_coffea_chunks_eos.py` from the per-chunk campaign outputs.
