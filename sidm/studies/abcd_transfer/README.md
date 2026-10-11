# Prompt-to-displaced transfer of the ABCD factors

Can the factors of the ABCD background estimate (pass/fail ratios of one axis of the plane at a
fixed state of the other) be measured with prompt lepton jets and used with displaced ones?
`transfer_factors.ipynb` answers that in 2018 data for the analysis plane (|Δφ| between the
lepton jets against the leading muon LJ's isolation, the other LJ isolated at the event level)
and for the companion plane of the two isolations, by comparing each ratio and the shape behind
it between the planes that differ in one lepton jet's displacement, in the open regions of the
search plane, with buffered regions (isolation pass < 0.25, fail ≥ 0.5; |Δφ| pass ≥ 2.4, fail
< 2.0) and a NaN-blinded search region. A last section shows what the 150 GeV pair-mass
requirement does to the plane and whether per-mass-bin factors are measurable. Section 6 of
the notebook carries the verdict.

`plane_search.ipynb` takes the next step: which open plane is compatible with the search plane
and can validate the estimate. It compares the open ratios, fail-slice shapes and egamma-LJ
composition of every candidate plane (the four displacement planes and their splits by pixel
hits of the muon LJ and by electron/photon class of the egamma LJ) with the search plane, tests
the closure of the candidates against muon class, pair mass and boundary variants, tests the
prompt-to-displaced step of the muon LJ in the open populations, and assembles the estimate per
mass window and muon pixel-hit class. Section 7 carries the proposal.

`estimate.ipynb` assembles the estimate for both channels from a new set of histograms of the
same data and simulation (the `abcd8d` campaign, below): the six axes plus the leading muon LJ's
own mass and its DSA-muon count, which split the muon LJs into those with and without a
displaced-standalone muon. The hidden box is the search region itself (below), the ABCD
sidebands start at its edges (no buffer bands), and the estimate is made four ways — directly from
the search plane's own sidebands, and by borrowing A/B, A/C or A/D from the PD plane per DSA class
of the muon LJ — and compared. The independence assumption is tested in the open planes above
150 GeV (PP, PD, DP) and in the search plane itself below 150 GeV, against a 15 % target, and a
configuration scan (axes, working points, buffers) selected on PP and confirmed on PD and DP shows
that the earlier |Δφ| buffer was most of PP's non-closure while PD stays 11–27 % above one for
every unbuffered boundary. In 2μ2e PP closes (0.97 ± 0.10) and PD does not quite (1.20 ± 0.10),
both falling between the 150–300 GeV and the ≥ 300 GeV windows; the four routes read 46–72 events
and agree within their shared statistics, and the estimate is 61 ± 8 (stat.) ± 20 (syst.)
background events in the one-bin search region (41 ± 7 ± 14 at 150–300 GeV, 19 ± 4 ± 6 above
300 GeV), the 33 % systematic being the non-closure plus the precision of the two open tests of
the borrowed A/B (the PP → DP analog and the search plane below 150 GeV, 1.32 ± 0.27 and
1.00 ± 0.25). The 15 % target is not met. In
4μ the search plane's sidebands hold three events, no κ of it can be measured, and six
count-limited readings from the open neighbouring regions put the search region at about 1.7
events, uncertain by a factor two to three.
Section 7 carries the conclusions. The earlier version of the notebook (buffered sidebands, a
75 GeV cut on the muon LJ's mass, the proxy-muon validation region of Weinan Si's thesis with and
without a b-tagged jet) is in the history of this directory; the proxy-region histograms are still
produced by the campaign and merged under `$ABCD8D_LOCAL/proxy`, but not read.

## Inputs

Dongyub Lee's inclusive 6D histograms (campaign `ABCD_6D_inclusive_eHEM_skimv1`, merged
outputs on EOS under `/store/user/dlee3/sidm_condor/`): per event, the isolation and
displacement variable of both lepton jets, |Δφ| and the pair mass, with no isolation,
displacement, Δφ or mass requirement applied. `fetch_6d.sh` copies them read-only to
`$ABCD6D_LOCAL` (default `/uscms_data/d3/murtazas/abcd6d_local`; 68 data, 18 background and
180 signal files, about 1 MB each) and writes a manifest.

The `abcd8d` campaign re-ran the same skims (`ABCD_8D_inclusive_eHEM_skimv1_*` under
`/store/user/murtazas/sidm_condor/`) with Dongyub's framework copy plus the files in `campaign/`:
`abcd8d.py` adds the histograms `abcd8d_<channel>` (the 6D axes plus `lj0_mass` and `ndsa0`, in
4μ also `ndsa1`); `abcd8d_proxy.py` adds `abcd8d_proxy_<channel>` for the proxy-muon channels
(`abcd_proxy_2mu2e_eHEM`: one egamma LJ, no two-muon LJ, a lone muon; `abcd_proxy_4mu_eHEM`: one
muon LJ, no egamma LJ, a lone muon; axes as the 6D ones plus `ndsa0`/`ndsa1` and `nbjets`, the
number of tight DeepFlavB jets with ΔR > 0.4 from both objects), with the isolation clipped at zero
(the axis has no underflow bin and an LJ's isolation can be slightly negative); `patch_proxy.py`
adds the channels, the objects and the cuts to the framework copy; `patch_prefilter.py` adds a
raw-event pre-filter on the trigger and on two muons above 26 GeV, which every channel requires,
to halve the running time (designed to leave every histogram unchanged; section 1 of
`estimate.ipynb` finds the 8D histograms 0.1 % below the 6D ones — in the fail regions of 2μ2e and
outside the nominal regions in 4μ — which neither the pre-filter nor the one unmerged data chunk is shown to
account for). `merge_abcd8d.py` sums the per-chunk
outputs into one file per sample under `$ABCD8D_LOCAL` (default `/uscms_data/d3/murtazas/abcd8d_local`,
the search-channel cubes) and `$ABCD8D_LOCAL/proxy` (the proxy-region cubes), each with a manifest;
simulated chunks are scaled to luminosity × cross section with their own sum of generator weights, so
the merge weights each chunk by that sum (Σ H_j S_j / Σ S_j) instead of adding the chunks up.

Both campaigns apply the 2018 certified luminosity sections
(`Cert_314472-325175_13TeV_Legacy2018_Collisions18_JSON`, the same file as the repository's) to data,
and to simulation the generator weight and the HEM15/16 treatment of the `_eHEM` channels (a veto of
data events in runs ≥ 319077 with a jet or electron in the HEM15/16 region, a weight of 0.351 for such
simulated events). No pileup reweighting, L1-prefiring weight or lepton scale factor is applied; the
estimates are data-driven and simulation only enters the composition tables and the
signal-contamination checks.

## Files

| file | role |
|---|---|
| `abcd6d_io.py` | loads one file into a `Cube` (sumw, sumw2 with flow bins), the hidden box, projections, A/B/C/D regions, κ |
| `test_abcd6d_io.py` | projection, displacement-definition, sentinel, blinding (the box is exactly the search corner; the 4μ one-LJ-displaced corners) and closure tests on synthetic histograms (`pytest`) |
| `transfer_lib.py` | planes, buffered regions of both ABCD planes, the four pass/fail ratios with Clopper–Pearson intervals, pooled shape χ², scans, closures, search-region recipes, MC composition, signal contamination |
| `test_transfer_lib.py` | the data path is blinded and cached per hidden box, strip exclusion, interval and χ² values, table coverage |
| `transfer_plots.py` | the figures: isolation and \|Δφ\| shapes per plane, the ratios against the displacement variable, the ratios, κ and the region spectra against the pair mass |
| `_build_notebook.py` | writes `transfer_factors.ipynb` |
| `plane_search_lib.py` | split planes (`PD:mu3`, `DD:dsa1+`, `DD:g`, …), ABCD configurations with or without buffers and whether their sidebands stay outside the hidden box, regions, κ and pulls, the configuration scan, the estimate, simulation-composition and summary tables |
| `test_plane_search_lib.py` | region conventions and the hidden box on a synthetic cube (a configuration's sidebands are open exactly when `dd_open` says so), split planes refused outside their plane, exported 4D cubes against the full cube, table arithmetic |
| `plane_search_plots.py` | κ against the \|Δφ\| boundaries, open ratios per plane, factors against mass and against the muon pixel-hit class |
| `_build_plane_search.py` | writes `plane_search.ipynb` |
| `estimate_lib.py` | 6D/8D consistency, spectra per region, event-cut scans, class grids, the closure and final estimate tables, the four estimate routes (direct B C/D; B × A/B, C × A/C, D × A/D borrowed per DSA class and summed) with the per-class factors and the sideband test of the borrowing, reduced planes and the configuration closure scan with its summaries, signal yields per region, the count-limited 4μ tables (DSA-count grid, sideband and mass transfers, simulation per region, signal per region, counting-experiment limits) |
| `test_estimate_lib.py` | the tables split by b-tag and class, the pulls and the final table, flow folding of the spectra, the four routes against the observed A (the per-class sum differs from the inclusive form by construction), reduced cubes against the full one, the closure scan's flags, `parse_fmt`, the count-limited tables on a blinded 4μ cube, the Poisson upper limits |
| `estimate_plots.py` | spectra of a stored variable per region (data over the stacked groups), ABCD ratios against an object class; used by the earlier version of `estimate.ipynb`, not by the current one |
| `_build_estimate.py` | writes `estimate.ipynb` |
| `merge_abcd8d.py` | EOS per-chunk outputs → one file per sample and family, manifests, rescue lists; chunks with an exception or without the sample are left out and listed for rescue, a sample with such chunks is written only with `--allow-missing` |
| `test_merge_abcd8d.py` | weighted combination of chunks, mismatched histogram sets refused, the merge plan (listed chunks only, bad chunks to the rescue list) |
| `campaign/` | the histogram producers and patches the `abcd8d` campaign ran (reference copies, not imported here) |
| `fetch_6d.sh` | EOS → local copy |

## Running

On the LPC, in the notebook environment (`sidm_venv`), with a VOMS proxy for the fetch:

```
bash fetch_6d.sh                      # once; skips files already present
python _build_notebook.py
jupyter nbconvert --to notebook --execute --inplace transfer_factors.ipynb
python _build_plane_search.py
jupyter nbconvert --to notebook --execute --inplace plane_search.ipynb
python merge_abcd8d.py --kind data --workers 2   # and dy, signal, bkg_rest, dy_proxy, signal_proxy
python _build_estimate.py
jupyter nbconvert --to notebook --execute --inplace estimate.ipynb
```

The tests run anywhere with numpy, scipy, pandas and hist: `pytest -q`.

The notebooks cache the summed data cubes, blinded, under `$ABCD6D_LOCAL/summed/` and
`$ABCD8D_LOCAL/summed/`, keyed on the manifest and on the hidden box. Figures are saved as PDF under `figures/` (untracked); the
committed notebook carries them rendered.

## Blinding

The search region is the isolated (leading muon LJ isolation < 0.25, other LJ < 0.5),
back-to-back (|Δφ| ≥ 2.4) corner of the plane with both lepton jets displaced at pair mass
≥ 150 GeV. `transfer_lib.summed_cubes` is the only data path the notebooks use (`abcd6d_io.load_cubes`
reads any single file unblinded and is never called on data by a notebook), and it returns the cubes
with exactly that region set to NaN — in 4μ also the same corner of the two planes with one muon LJ
displaced, which hold a quarter to a half of the 4μ signal's search-region yield against tens of
background events. Any projection touching a hidden cell prints as "blind" or NaN, and no total
that includes a hidden cell is printed. Every other cell is open: the sidebands of the search plane
(iso0 ≥ 0.25 or |Δφ| < 2.4), everything below 150 GeV, and in 2μ2e the isolated corners of the
one-LJ-displaced planes; the signal in every region read from is tabulated in `estimate.ipynb`.

Until 2026-10-11 the box was larger — the search region plus a margin on both ABCD axes (leading
isolation < 0.5, |Δφ| ≥ 2) — to keep buffer bands between A and its sidebands hidden.
`transfer_factors.ipynb` and `plane_search.ipynb` were executed under that box and are not
re-executed; their committed outputs show it ("blind" in cells that are open now). The data cache
under `$ABCD6D_LOCAL/summed/` and `$ABCD8D_LOCAL/summed/` is keyed on the box and rebuilds itself.
The proxy-region cubes hide the same corner of the 4μ planes without a b-jet when they are loaded;
`estimate.ipynb` no longer loads them.
