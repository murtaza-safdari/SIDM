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

## Inputs

Dongyub Lee's inclusive 6D histograms (campaign `ABCD_6D_inclusive_eHEM_skimv1`, merged
outputs on EOS under `/store/user/dlee3/sidm_condor/`): per event, the isolation and
displacement variable of both lepton jets, |Δφ| and the pair mass, with no isolation,
displacement, Δφ or mass requirement applied. `fetch_6d.sh` copies them read-only to
`$ABCD6D_LOCAL` (default `/uscms_data/d3/murtazas/abcd6d_local`; 68 data, 18 background and
180 signal files, about 1 MB each) and writes a manifest.

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
| `fetch_6d.sh` | EOS → local copy |

## Running

On the LPC, in the notebook environment (`sidm_venv`), with a VOMS proxy for the fetch:

```
bash fetch_6d.sh                      # once; skips files already present
python _build_notebook.py
jupyter nbconvert --to notebook --execute --inplace transfer_factors.ipynb
python _build_plane_search.py
jupyter nbconvert --to notebook --execute --inplace plane_search.ipynb
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
