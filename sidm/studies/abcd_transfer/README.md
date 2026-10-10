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

`estimate.ipynb` assembles the estimate for both channels from two new sets of histograms of
the same data and simulation (the `abcd8d` campaign, below): the six axes plus the leading muon
LJ's own mass and its DSA-muon count, which answer the Drell–Yan question of the validation
plane in the data and split the pixel-0 class; and the proxy-muon validation region of Weinan
Si's thesis (UC Riverside 2020, section 3.6.2), in which a lone muon stands in for one muon LJ,
so that every corner of its four displacement planes is open and the closure the search channel
cannot test on itself can be tested there, with and without the b-jet requirement of the thesis.
The target is closure within 10–15 % in every open validation region: the notebook scans the
family of ABCD configurations (axes, working points, buffers) for the ones that meet it, selecting
on one set of validation regions and confirming on a second set (not fully disjoint: the b-tagged
proxy DD plane is part of the proxy DD plane), and judges every region against the target before
assembling the estimates. Neither channel meets it with the 2018 data: in 2μ2e κ of the search
channel's PD plane is 1.27 ± 0.15 and, over every boundary choice that keeps the search region,
above one (1.19–1.90) when the muon LJ's isolation is an ABCD axis and below one (0.46–0.71) when
the egamma LJ's is; the proxy region populates the isolation axis differently from the search
channel, and the usable estimate is B × (A/B) of the PD plane per DSA class of the muon LJ, at
about 40 %. In 4μ the search plane holds one event outside A, so no κ of it can be measured; the
open neighbouring regions put the search region at about one event, uncertain by a factor two to
three, and the open low-mass corner and DSA-count split show no population the search region
could be hiding. Section 7 carries the conclusions.

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
estimates are data-driven and simulation only enters the Drell–Yan subtraction, the composition
tables and the signal-contamination checks.

## Files

| file | role |
|---|---|
| `abcd6d_io.py` | loads one file into a `Cube` (sumw, sumw2 with flow bins), the hidden box, projections, A/B/C/D regions, κ |
| `test_abcd6d_io.py` | projection, displacement-definition, sentinel, blinding and closure tests on synthetic histograms (`pytest`) |
| `transfer_lib.py` | planes, buffered regions of both ABCD planes, the four pass/fail ratios with Clopper–Pearson intervals, pooled shape χ², scans, closures, search-region recipes, MC composition, signal contamination |
| `test_transfer_lib.py` | the data path is blinded and cached per hidden box, strip exclusion, interval and χ² values, table coverage |
| `transfer_plots.py` | the figures: isolation and \|Δφ\| shapes per plane, the ratios against the displacement variable, the ratios, κ and the region spectra against the pair mass |
| `_build_notebook.py` | writes `transfer_factors.ipynb` |
| `plane_search_lib.py` | split planes (`PD:mu3`, `DD:g`, …), ABCD configurations with buffers, regions, κ and pulls, the configuration scan, the estimate and summary tables |
| `test_plane_search_lib.py` | region conventions and the hidden box on a synthetic cube, split planes refused outside their plane, exported 4D cubes against the full cube, table arithmetic |
| `plane_search_plots.py` | κ against the \|Δφ\| boundaries, open ratios per plane, factors against mass and against the muon pixel-hit class |
| `_build_plane_search.py` | writes `plane_search.ipynb` |
| `estimate_lib.py` | 6D/8D consistency, spectra per region, event-cut scans, proxy planes against the search plane, class grids, the closure and final estimate tables, the class-split recipe, reduced planes and the configuration closure scan with its summaries, signal yields per region, the count-limited 4μ tables (DSA-count grid, sideband and mass transfers, simulation per region, signal per region, counting-experiment limits) |
| `test_estimate_lib.py` | the proxy tables split by b-tag and class, the pulls and the final table, flow folding of the spectra, the class-split recipe against the observed A, reduced cubes against the full one, the closure scan's flags, `parse_fmt`, the count-limited tables on a blinded 4μ cube, the Poisson upper limits |
| `estimate_plots.py` | spectra of a stored variable per region (data over the stacked groups), ABCD ratios against an object class |
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

The notebook caches the summed data cube, blinded, under `$ABCD6D_LOCAL/summed/` keyed on the
manifest and on the hidden box. Figures are saved as PDF under `figures/` (untracked); the
committed notebook carries them rendered.

## Blinding

The search region is the isolated (leading muon LJ isolation < 0.25, other LJ < 0.5),
back-to-back (|Δφ| ≥ 2.4) corner of the plane with both lepton jets displaced at pair mass
≥ 150 GeV. `transfer_lib.summed_cubes` is the only data path the notebook uses (`abcd6d_io.load_cubes`
reads any single file unblinded and is never called on data by the notebook), and it returns the cubes
with that region plus a margin on both ABCD axes set to NaN (leading isolation < 0.5, |Δφ| ≥ 2,
other isolation < 0.5, mass ≥ 150 GeV, both displaced; in 4μ also the isolated corners of the two
one-LJ-displaced planes); any projection touching a hidden cell prints as "blind", and no total
over a blinded cube is printed. The fail slices of the search plane, its isolated corner at
|Δφ| < 2 and at mass < 150 GeV, and in 2μ2e the isolated corners of the one-LJ-displaced planes
are open and used; the signal in every open region is tabulated in the notebook.

The 8D cubes carry the same hidden box. The proxy-region cubes are open everywhere in 2μ2e
(the region holds under 1% of the 2μ2e signal); in 4μ the region without a b-jet holds about a
quarter of the 4μ signal, so the isolated corners of its DD, DP and PD planes above 150 GeV are
hidden, while the b-tagged part (Weinan's validation region) is open.
