# Dimuon-vertex-fit cosmic veto, 2018C

Compares a per-lepton-jet dimuon-vertex-fit cosmic veto (`mu_ljs.vtx_chi2`) against the
`*Spread_*` veto used in the 1-LJ control region, on 2018C `DoubleMuon` data, the 2018
background stack and the v10 signal grid, and characterizes the >= 200 GeV DSA-muon tail
that survives either veto.

**Start here: [`vertexChi2CosmicVetoStudy.ipynb`](vertexChi2CosmicVetoStudy.ipynb).** It is the
write-up and reads only the `.coffea` outputs of the run notebooks below; it runs no
processor of its own.

## Two cosmic-enriched regions, and why they are never interchangeable

The study uses two different cosmic-enriched selections. They are named apart everywhere — in
the write-up prose, in the figure legends and here — because they are not the same region and
the difference decides what one of the results is conditional on.

| name used throughout | channel | mu-LJ cuts | uses `vtx_chi2`? |
|---|---|---|---|
| **spread-inverted cosmic region** | `cosmic_muons` | `lj_iso < 0.2`, `inverse_spread`, cosAlpha event cut inverted | no |
| **vertex-inverted cosmic region** | `cosmic_muons_vtxChi2` | `lj_iso < 0.2`, `no vtx or vtx_chi2 >= 5`, cosAlpha event cut inverted | yes |

The spread-inverted region is what makes the section-2b comparison a test of `vtx_chi2` rather
than a tautology: the variable is asked to flag objects selected with entirely different
quantities. The vertex-inverted region is where the azimuthal cosmic template of section 7 is
measured, so the 95% CL residual-cosmic bound is conditional on it — the write-up shows the
bound moves only from 6.2% to 7.8% even if the template's vertical fraction is degraded from
0.99 to 0.90.

## Notebooks

Every run notebook opens its own Dask cluster on LPC Condor, writes one `.coffea` per sample
to `<repo>/RunOutputFiles/<run_tag>/` together with a `.meta.yaml` provenance sidecar, and
copies both to `root://cmseos.fnal.gov//store/group/lpcmetx/SIDM/coffea_outputs/$USER/<run_tag>/`.
`sidm_path` is derived from the working directory, so run them from this directory.
Per-sample outputs already on disk are skipped, so a run can be resumed.

| notebook | what it runs | `run_tag` / output directory |
|---|---|---|
| `runDataBg2018.ipynb` | 2018C `DoubleMuon` plus the background stack (TTJets, DYJetsToLL M10to50 and M50, DYJetsToMuMu M10to50, eleven QCD `Pt` bins, WW/WZ/ZZ) through six channels: Before, the `vxySpread` and `dxySpread` variants of the Spread veto with cosAlpha, the vertex veto with and without cosAlpha, and the vertex-inverted cosmic region. Hist collection `cosmic_veto`. | `vtxChi2_vs_spread_2018C` (19 samples) |
| `runDataCosmic2018.ipynb` | 2018C `DoubleMuon` in the `cosmic_muons` channel only — the **spread-inverted** cosmic region, defined by inverted spread cuts and inverted cosAlpha, which uses no vertex information. Hist collection `cosmic_veto`. | `vtxCosmic_2018C` (1 sample) |
| `runDataTail2018.ipynb` | 2018C `DoubleMuon` through Before, Spread + cosAlpha, vertex + cosAlpha and `cosmic_muons_vtxChi2` (the **vertex-inverted** cosmic region), with hist collections `cosmic_veto` and `dsa_pf_response`. Adds the 2D DSA-muon pT versus nHits, \|eta\|, phi, \|dxy\| and sigma(pT)/pT histograms, and the matched DSA-to-PF response. | `vtxTail2_2018C` (1 sample) |
| `runMCTail2018.ipynb` | The MC counterpart of the tail check, same two hist collections, Before and Spread + cosAlpha only: the eight samples that carry the >= 200 GeV DSA tail (DYJetsToLL_M50 and QCD_Pt80To120 through QCD_Pt800To1000). | `vtxTailMC2_2018` (8 samples) |
| `runSignalVtx.ipynb` | The full v10 signal grid, 180 samples (90 4Mu, 90 2Mu2E), through `base_ljDisplacementIso`, `+ cosAlpha`, `+ Spread veto` and `+ vertex veto`. Hist collection `cosmic_veto` as it stood before the 2D nHits and \|eta\| histograms were added, so those are absent from these outputs. | `vtxChi2_signal_2018` (180 samples) |
| `runSignalVtxNhits.ipynb` | A 26-sample subset of the same grid (13 4Mu, 13 2Mu2E, mirrored mass and lifetime points), re-run after `mu_lj_dsaMuon_pt_nHits` and `mu_lj_dsaMuon_pt_abseta` joined the `cosmic_veto` collection. Used only for the signal cost of an nHits cut. | `vtxChi2_signal_2018_nhits` (26 samples) |
| `vertexChi2CosmicVetoStudy.ipynb` | The write-up. Loads all six output directories above and produces every number and figure of the study. | reads only; writes nothing |

## Provenance sidecars

Each `.coffea` is written with a `<sample>.meta.yaml` next to it via `sidm.tools.metadata`,
recording the full selection definitions, the hist-collection contents, the input ROOT file
list, the per-sample cross section, the SIDM git commit, the coffea version, the schema and
chunksize, and a UTC timestamp. Read one back with

```python
from sidm.tools.metadata import load_run_metadata
meta = load_run_metadata("<path to the .coffea>")   # it derives the .meta.yaml path itself
```

The outputs that already existed when the sidecar write was added carry **back-filled**
sidecars, written by `runlogs/backfill_meta.py` with no reprocessing. Those are labelled as
such: they contain a `backfill:` block naming where each field came from, and their
`sidm_commit` is `null`, because the commit that produced them was never recorded and is not
recoverable. Two of the eight output directories, `vtxTail_2018C` and `vtxTailMC_2018`, are
superseded first passes that no current run notebook produces; their sidecars say so and
derive the hist-collection list from the outputs themselves.

## Run order for a full reproduction from the samples

1. `runDataBg2018.ipynb` — the control-region yields and the `vtx_chi2` threshold scan.
2. `runDataCosmic2018.ipynb` — the spread-inverted cosmic region used as a `vtx_chi2` cross-check.
3. `runDataTail2018.ipynb` — the tail histograms and the DSA-PF response, data.
4. `runMCTail2018.ipynb` — the same histograms in MC. Must use the same channel list as 3.
5. `runSignalVtx.ipynb` — the 180-sample signal efficiency of both vetoes.
6. `runSignalVtxNhits.ipynb` — the 26-sample nHits subset. Needs the extended `cosmic_veto`
   collection, so it must be run from a checkout that has it.
7. `vertexChi2CosmicVetoStudy.ipynb` — the write-up, once 1 to 6 are on disk.

Steps 1 to 6 are independent of each other and can be run in any order or in parallel;
step 7 needs all of them. Signal samples read `../../configs/ntuples/signal_4mu_v10.yaml`
and `signal_2mu2e_v10.yaml`; `runSignalVtxNhits.ipynb` also reads a JSON veto list of
corrupt input files (16 files across 14 samples).

Steps 1, 3, 4 and 5 take hours each on a hundred Condor workers. All of them require a valid
VOMS proxy readable by the Condor schedd and `replace_xcache=True` in `make_fileset`, since
the location YAMLs carry `root://xcache//` URLs that only resolve on coffea-casa.

## A note on the two MC sets

Two different MC selections appear in the study, and the distinction matters when reading it,
because several headline numbers change completely between them:

- `vtxChi2_vs_spread_2018C` carries the full background stack and is what the control-region
  yields and the threshold scan use.
- `vtxTailMC2_2018` carries only the eight samples that populate the >= 200 GeV DSA tail, but
  is the only MC set with the 2D histograms.

Numbers that must always carry the name of the set they came from:

| quantity | full stack | eight-sample tail set |
|---|---|---|
| pre-veto MC tail (`pT >= 200` GeV) | 293.6 | 133.7 (46% of the stack's) |
| post-veto MC tail | 133.7 | 129.5 (97% of the stack's) |
| fraction of the MC tail the veto removes | **55%** | **3%** |
| pre-veto tail data/MC | 8.22 | 18.06 |
| post-veto tail data/MC | 2.31 | 2.385 +/- 0.426 |
| MC yield over all pT, Spread channel | 108137.7 | 57091.1 (52.8% of the stack's) |

The two sets agree after the veto because the tail set covers 97% of the MC tail there; they
disagree before it because the full stack's pre-veto tail contains one high-weight
`QCD_Pt30To50` event worth about 156 weighted entries, which the veto removes. (That the
eight-sample set's pre-veto tail, 133.7, equals the full stack's post-veto tail, also 133.7,
is a coincidence of two unrelated quantities.)

The tail set does **not** cover the bulk: over all pT it carries 52.8% of the full stack, the
missing half being the soft-QCD bins. Absolute data/MC in the bulk is therefore meaningless
with it, and so is anything built from its total yield — only shapes across bins are used
there. Section 8 of the write-up marks the one place where that restriction bites, and gives
the argument in a form (a ratio of rates) that does not depend on it.

## One more normalization caveat

The control region itself is not normalized, and no scale factor is applied anywhere in this
study. In the Before channel it holds 0.770 data mu-type lepton jets per MC one but only 0.556
data DSA muons per MC one; the ratio of those two is a DSA-muon multiplicity mismodelling, MC
carrying 0.351 DSA muons per mu-type lepton jet against 0.254 in data, 38% more. Veto
efficiencies and the azimuthal cosmic test are ratios within one sample and are untouched by
it; absolute tail data/MC ratios are not, which is why the write-up's load-bearing tail
statement is a ratio of rates rather than the raw 2.4 +/- 0.4. Section 3.1 of the write-up
sets this out in full.
