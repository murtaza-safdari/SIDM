# Prompt-to-displaced transfer of the ABCD factors

**Start here: `nominal.ipynb`** (2026-10-16). It presents the 2μ2e background estimate as the search
plane's own ABCD, A = B·C/D = 46 ± 15 (stat.) events with a systematic of 20% by its stated rule, or −20%/+31% as it recommends (the group's choice), validates it in every open region
of the data and in the seven validation regions of Dongyub Lee's programme recomputed from the same
histograms, and derives the 20% systematic from those regions by a rule stated in the notebook. It is
written for readers new to the analysis. The notebooks below are the longer record and are kept as
they were run; where their conclusions make the borrowed-factor estimate (61 ± 8) the headline,
`nominal.ipynb` supersedes them and makes it the cross-check.

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

`binning.ipynb` evaluates the same 2μ2e estimate per bin of the LJ-pair mass — B of the search
plane in the bin times A/B of PD, per DSA class, with the factor measured in the bin where PD
measures it to 20 % or better and otherwise taken from a fit of A/B against the mass
(exponential, fixed in advance; linear as the cross-check) with |measured − fitted| as a shape
systematic — for four binnings fixed in advance, and chooses among them by a decision rule written
down before any result (a toy asymptotic expected limit for the 500 and 800 GeV points, with a
count requirement per bin; nothing selected on κ or on closure; A of DD never read). A/B of PD
falls to 0.61 of its 150–200 GeV value by 400–600 GeV, in PP the same way (PP over PD flat at
0.54 ± 0.03), but neither form describes the seven bins (deviance 19/5 for the dsa0 class): above
600 GeV the measured A/B of PD and PP sit above the fall, and the fit there is the pooled average.
Summed over the bins the estimate reproduces the one-bin 61 ± 8 to 3 % with the measured factors,
so binning changes where the background sits, not how much there is. The rule chooses the
histograms' own bins (150–200, 200–300, 300–400, 400–600, ≥ 600 GeV; 25, 17, 10, 5 and 3 events):
a factor 6 in the expected limit of the 800 GeV points against one bin and 4.7 for the 500 GeV
point, of which a factor 2.6 comes from the split at 300 GeV alone. The gain beyond that rests on
a ≥ 600 GeV bin of four sideband events, which passes the rule's count requirement only with the
fitted factor (2.9 events against 4.7 measured, 3.05 with the linear form, where the rule leaves
the two windows) and whose A region in PD would hold 3–19 signal events at μ = 1 for the 800 GeV
points; whether to admit it is left to the group. A split at 400 GeV (150–400, ≥ 400 GeV), added
after the result and not a candidate of the rule, is evaluated beside the four: a factor four at
both decision masses from 13 sideband events and a factor PD measures in the bin to 16 %, with no
fitted dsa0 factor. Section 4.1 tests the route per bin where the data allow it (the search plane
below 150 GeV, where the closure splits into κ(DD)/κ(PD) times the C/D ratio and the two
compensate; its C/D against PD's per bin above 150 GeV; κ per bin in the open planes), on the
cube bins and on the split at 400 GeV: no bin fails, nothing tightens the 33 % per bin, which
stays assumed. The control region the factor is measured from, A of PD, is not signal-blind at
high mass (an 800 GeV point at μ = 1 puts up to 19 events among its 25 dsa0 events above
600 GeV); section 4.2 measures by signal injection what the estimate absorbs of a signal that is
in the data (up to 8 % of the signal in the search region in a ≥ 600 GeV bin with the measured
factor, 4 % in the split at 400 GeV, 3 % in one bin). Section 5.1 runs
the same counting experiments through combine (`binned_cards.py` writes one datacard per
binning, signal point and background model; the limits come from
`sidm/scripts/run_combine_limits.py`, blind asymptotic, every card with a fixed `--rmax 300`
because the runner's automatic range reads the first bin's signal only): the toy's relative
limits agree with combine's to 5 % with the same ordering, and a Gamma treatment of the
sideband counts (`gmN`) in place of the log-normal costs the binnings with a four-event bin
10–19 %, the split at 400 GeV 7 %. A third model (`cr`) puts the control region into the
likelihood, per bin and class a channel with A of PD observed, B of PD times the factor as its
background and the point's yield there as its signal, the factor a free parameter shared with
the search bin, so that a signal in the control region is part of the signal hypothesis instead
of a bias of the estimate. It costs the binnings with a ≥ 600 GeV bin 35 % of their expected limit
at 800 GeV: a factor 1.30 because the bin's factor is then what its own 25 and 16 control events
say (1.56 for dsa0) and not the fitted 0.90, and 1.04 (1.02 to 1.05 over the three 800 GeV
points) for the signal in the control region (measured by running the cr cards once more with
that signal set to zero, `control_signal_off.py`);
the split at 400 GeV pays 2 %, all of it the control-region signal. In that model the cube bins
and the split at 400 GeV tie at 0.28 of the one-bin limit for the 800 GeV points, while the rule's
own metric still ranks the cube bins first through the 500 GeV point. The cr model was built after
the binning result as a cross-check and departs from the rule's factor clause; with its factor the
≥ 600 GeV bin fails the rule's count requirement. Section 7 lays this out and leaves the choice
to the group. The absolute limits are on μ relative to the stored cross section,
without the dark-photon branching fractions. 4μ gains little: its three sideband events sit
below 600 GeV, where the heavy points have 5–17 % of their yield, and its estimate stays
count-limited. Section 7 carries the comparison with the one-bin estimate.

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
| `binning_lib.py` | the per-bin estimate: counts and A/B per mass bin, the fits of A/B against the mass, the B route per bin with the measured or the fitted factor, the signal yields per bin and per region read by the estimate, the per-bin validation (the open test below 150 GeV, the sideband test C/D against PD's, κ per bin), the signal-injection test, the toy asymptotic expected limit, the decision rule |
| `test_binning_lib.py` | the fit recovers a known slope, the binned sums equal the one-bin estimate for a flat factor and differ for a falling one, the rule takes the fitted factor where PD is imprecise, the hidden A stays hidden in every mass bin, the per-bin validation reads the open regions only, the injected signal leaks into the estimate by B × signal / B of the control plane, the toy limit against reference values |
| `binning_plots.py` | the four planes and the per-bin route as cartoons, signal and sideband spectra against the mass, A/B against the mass with its fits, background and signal per bin of each binning, the three per-bin validation tests against the mass, the toy limits per signal point |
| `nominal_lib.py` | the validation regions of `nominal.ipynb` (ours and Dongyub Lee's seven, with his working point), κ per region with the hidden-cell flag, the egamma-type and DSA-class shares per region, the κ drop between mass windows, the rule for the closure systematic with its alternatives, the four routes per window, the signal share per mass bin, the projected precision of an opened window |
| `nominal_plots.py` | the ABCD cartoon, the isolation × \|Δφ\| plane with the regions and the hidden box, κ per region (open planes, the seven validation regions at three settings, the displacement ladder, the mass windows, the systematic candidates with the band), the photon-only share per region, the signal share per mass bin |
| `_build_nominal.py` | writes `nominal.ipynb`; every number in its text is computed by the builder from the same functions the code cells call |
| `test_nominal_lib.py` | the seven region definitions and the hidden-cell flag on synthetic histograms, the systematic rule, the share tables, the signal shares (`pytest`) |
| `_build_binning.py` | writes `binning.ipynb` |
| `binned_cards.py` | the combine datacards of `binning.ipynb` section 5.1: one per binning, signal point and background model (`lnN`: the per-bin estimate with its asymmetric statistical and shape log-normals; `gmN`: the B count of each DSA class as a Gamma-distributed sideband with its factor as a log-normal; `cr`: the gmN model with a control channel per bin and class, A of PD observed, the factor a free `rateParam` shared with the search bin), the 33 % transfer systematic, the luminosity and the signal's statistical error, with their provenance sidecar; reads the runner's `limits.csv` back and compares the binnings relative to one bin against the toy |
| `test_binned_cards.py` | the lnN and gmN cards mirror the frames they are written from, the cr card puts the control region's counts and signal in their own channels with one shared factor, zero counts drop processes and bins, the card names parse in `run_combine_limits.py` with the right channel, normalisation and rates, the comparison is relative to the same model |
| `control_signal_off.py` | the cr cards of the decision points with the control channels' signal set to zero, the probe behind the two-part decomposition of section 5.1 |
| `binning_cards/` | the 105 datacards (five binnings, seven 2μ2e points, three models) and `datacards.meta.yaml` |
| `binning_limits/` | combine's expected limits on them, `limits.csv` with the sidecar `run_combine_limits.py` writes; `control_signal_off/` the same for the eight cards without the control-region signal |
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
python binned_cards.py --outdir binning_cards       # the datacards of binning.ipynb section 5.1
python ../../scripts/run_combine_limits.py --datacards binning_cards --pattern 'datacard_SR_2mu2e_*.txt' \
    --outdir binning_limits --rmax 300             # combine, in Sam's CMSSW area (see the runner)
python _build_binning.py
jupyter nbconvert --to notebook --execute --inplace binning.ipynb
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
