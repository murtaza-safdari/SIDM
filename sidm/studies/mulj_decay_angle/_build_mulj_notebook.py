"""Generate mulj_decay_angle.ipynb.

The notebook is a generated artefact: edit this script, re-run it, and re-execute the
notebook.  Editing the .ipynb by hand puts the two out of step.

    python _build_mulj_notebook.py                 # write the notebook next to this file
    jupyter nbconvert --to notebook --execute --inplace \
        --ExecutePreprocessor.kernel_name=sidm_venv mulj_decay_angle.ipynb

Every figure is produced by a function in `_mulj_lib`, so a single figure can be redrawn
by re-running its cell.
"""

import os

import nbformat
from nbformat.v4 import new_code_cell, new_markdown_cell, new_notebook

HERE = os.path.dirname(os.path.abspath(__file__))
OUTPUT = os.path.join(HERE, "mulj_decay_angle.ipynb")

SMOKE_INPUT = "/uscms_data/d3/murtazas/mulj_smoke"
CAMPAIGN_INPUT = "/uscms_data/d3/murtazas/mulj_merged"
PAIR_INPUT = "/uscms_data/d3/murtazas/mulj_pair_merged"
PLANES_INPUT = "/uscms_data/d3/murtazas/lj_angle_planes_merged"

def build():
    """Assemble the notebook cell by cell and write it out."""
    cells = []

    # ---------------------------------------------------------------- title and setup
    cells.append(new_markdown_cell("""# Decay angle of the leading muon pair in a muon lepton jet

A lepton jet is a group of collimated muons, electrons or photons clustered into one object, the
signature of a light, highly boosted dark photon (Zd) decaying to a lepton pair.  A muon lepton
jet is built from muons, an electromagnetic lepton jet from electrons and photons.  A fake lepton
jet is one not produced by a dark-photon decay, for example two unrelated muons that happen to be
clustered together.

This study takes every muon lepton jet with at least two constituent muons and measures the decay
angle theta* of its two leading muons: the angle of the leading muon in the rest frame of the
pair, relative to the pair's direction in the laboratory (section a).  It compares |cos(theta*)|
between simulated signal, simulated background and collision data in a set of control regions
(CR) and validation regions (VR), and asks whether the angle separates genuine from fake muon
lepton jets.  Signal samples are labelled by the mass m_XX of the heavy bound state that decays to
two dark photons, the dark-photon mass m_Zd and the dark photon's proper decay length c tau.

All loading, normalisation and drawing lives in `_mulj_lib.py` in this directory, and each figure
comes from one function there.  Only the configuration cell below needs to change when the inputs
change; the sample lists are read from whatever is in the input directory.

The histograms come from three production campaigns run on Condor, listed under "Selections used
in this study".  The first campaign fills one entry per muon lepton jet and feeds sections a to h.
The pair campaign records the two muon lepton jets of a four-muon event together and feeds
section i.  The decay-plane campaign feeds section j, which adds the electron pair of an
electromagnetic lepton jet and the angle between the two lepton jets' decay planes.  Sections i
and j were added after the first campaign, and each reads its own merged files, set in its own
configuration cell.

Each section states what its figures show and what follows from them.  Section g collects the
caveats that apply to every figure, and section h states the verdict."""))

    cells.append(new_code_cell("""import os
import sys

# locate the study directory whether the notebook was started in it or from the repository root
_candidates = []
_walk = os.getcwd()
while True:
    _candidates.append(_walk)
    _candidates.append(os.path.join(_walk, "sidm", "studies", "mulj_decay_angle"))
    _parent = os.path.dirname(_walk)
    if _parent == _walk:
        break
    _walk = _parent
_study_dir = next(d for d in _candidates if os.path.exists(os.path.join(d, "_mulj_lib.py")))
sys.path.insert(0, _study_dir)

import _mulj_lib as lib

%matplotlib inline"""))

    cells.append(new_markdown_cell("""## Configuration

`INPUT_DIR` points at a directory of merged, one-file-per-sample `.coffea` outputs.  It
accepts a local path or an EOS `root://` URL, in which case files are copied to a local
cache the first time they are read.

`LUMI_FB` is the integrated luminosity printed in the CMS label of figures that contain
data.  Leave it as `None` while running over a partial data set, so that no figure claims
a luminosity it does not have; set it to the processed luminosity once the full campaign
is in."""))

    cells.append(new_code_cell(f"""# ---------------------------------------------------------------------------------
# The only settings that change between a test run and the full campaign.
# ---------------------------------------------------------------------------------
# INPUT_DIR = "{SMOKE_INPUT}"          # reduced test inputs
INPUT_DIR = "{CAMPAIGN_INPUT}"     # merged campaign outputs
# INPUT_DIR = "root://cmseos.fnal.gov//store/group/lpcmetx/SIDM/coffea_outputs/murtazas/mulj_decay_angle"

LUMI_FB = 59.8                       # set to 59.8 for the full 2018 data set

M_ZD_VALUES = (0.25, 1.2, 5.0)       # dark-photon masses shown as separate columns
M_ZD_REFERENCE = 1.2                 # mass point used for the single-point figures
# ---------------------------------------------------------------------------------

lib.configure(input_dir=INPUT_DIR, lumi_fb=LUMI_FB)
lib.set_style(dpi=85)
print("input directory :", lib.INPUT_DIR)
print("figures written to:", lib.FIG_DIR)"""))

    cells.append(new_markdown_cell("""### What is in the input directory

Samples are classified by name: `4Mu_*` and `2Mu2E_*` are signal, `DoubleMuon_2018*` is
collision data, and anything else with a configured cross section is simulated background.
The table counts muon pairs, not events: one entry per muon lepton jet with at least two
muons."""))

    cells.append(new_code_cell("""print("signal    :", len(lib.signal_samples()), "samples")
print("data      :", len(lib.data_samples()), "samples")
print("background:", len(lib.background_samples()), "samples")
print()
_ = lib.inventory()"""))

    # ---------------------------------------------------------------- selections
    cells.append(new_markdown_cell(r"""## Selections used in this study

The histograms read here come from three Condor campaigns on 2018 data and simulation, run from
the branch `mulj-decay-angle-dev`:

| campaign | ran | merged outputs | read by |
| --- | --- | --- | --- |
| `MuljDecayAngle_Run2018_v1` | 2026-08-28 to 2026-08-31 | `mulj_merged` | sections a to h |
| `MuljPairAngle_Run2018_v1` | 2026-08-31 to 2026-09-01 | `mulj_pair_merged` | section i |
| `LjAnglePlanes_Run2018_v1` | 2026-09-02 to 2026-09-05 | `lj_angle_planes_merged` | section j |

The text calls them the first campaign, the pair campaign and the decay-plane campaign.  The
merged-outputs column names the directory that holds each campaign's merged files.

All three ran the channel definitions of `sidm/configs/selections.yaml` at commit `61224901` of
that branch.  Those definitions are kept in `sidm/configs/selections_mulj_decay_angle_as_run.yaml`,
with every YAML anchor resolved, every cut list in the order it is applied, and every channel under
the name stored in the histograms.  `00_produce_inputs.ipynb` and this study's Condor submit files
pass that file to the processor explicitly.

Three pieces of evidence tie the file to the inputs.  The jobs ran from a working copy with
uncommitted changes, and commit `61224901` was made on 2026-09-07, after all three campaigns had
run.  The metadata file written next to each merged output (152, 101 and 154 such files) records a
definition for every channel, and every recorded definition agrees with this file; but those
definitions, and the commit recorded with them, come from the checkout that did the merging, not
from the jobs.  For the third campaign the code shipped to the
jobs was kept and is identical, file by file, to commit `61224901`; the code shipped to the first
two campaigns was not kept.  The strongest evidence is a rerun: stored Condor outputs of all three
campaigns (signal chunks of each, data chunks of the first and third) rerun from this branch with
this file reproduce the stored histograms and cutflows bin for bin.

**How these differ from the analysis-wide definitions.**  `sidm/configs/selections.yaml` on main
(commit `5bb94cd`) changed the base cuts every channel inherits (PR #56) after these campaigns
ran, so a channel here that shares its name with a channel there selects different events:

* main applies an eta-phi veto to electrons, photons, muons and displaced-standalone muons; none
  of the channels here has it;
* main has `>=2 muons (pf or dsa) pT > 26 GeV` and `all cosAlpha(mu, mu) > -0.99` among its base
  event cuts.  Here the 26 GeV requirement is in `4mu`, `4mu_sr`, `2mu2e`, `2mu2e_sr` and the four
  spread-veto regions, and absent from the two `_noMuPtCut` variants, the uncut and cosmic-veto
  single-lepton-jet control regions and the electromagnetic-lepton-jet control region; the
  opening-angle veto is in the four spread-veto regions only;
* main cuts on `dzSpread` and `dxySpread` in the muon-lepton-jet cuts of its signal channels.
  None of the signal-like channels here has a spread cut; the four spread-veto regions cut on
  `dzSpread` and `vxySpread`, with main's thresholds (50 for all muons, 10 for particle-flow
  muons);
* main has since also required `DT + CSC hits > 3` (`trkNumDTHits + trkNumCSCHits > 3`) of every
  particle-flow muon at the lepton-jet source (the `muons_ljsource_cuts` anchor, commit
  `65b46a86`).  None of the channels here has it, and its effect on the shapes and counts of this
  study was not measured.

The four spread-veto regions, `data_control_region_1muLj_spread_cosAlpha_mu_veto`,
`cr_2lj_invDisplaced_spread_cosAlpha_mu_veto` and the two `test_VR_*` regions, are not defined on
main at all.

The two tables below are rendered from the files when the notebook runs.  The first lists, for each
channel, the cuts it adds to the baseline every channel shares.  The second compares each channel
with the channel of the same name in this checkout's `configs/selections.yaml`, as "as run /
there".  Generator-level particle selections are left out of both."""))
    cells.append(new_code_cell("""from IPython.display import Markdown, display

print("channel definitions:", os.path.relpath(lib.AS_RUN_SELECTIONS, _study_dir))
display(Markdown(lib.selections_table()))"""))
    cells.append(new_code_cell("""print("compared with:", os.path.relpath(lib.REFERENCE_SELECTIONS, _study_dir))
display(Markdown(lib.selections_vs_reference()))"""))

    # ---------------------------------------------------------------- (a) definitions
    cells.append(new_markdown_cell("""## a. Definitions and method

Every quantity below is computed per muon lepton jet, from the **two leading-transverse-momentum
constituent muons** of that lepton jet.  The constituent list mixes particle-flow muons, which
use the inner tracker, and displaced-standalone muons, reconstructed from the muon system alone so
that muons from decays far from the beam line are kept.  It is sorted by transverse momentum inside each lepton jet; only lepton
jets holding at least two muons contribute, which is imposed explicitly rather than inherited from a
channel's muon-multiplicity cut.

* Each of the two muons is given a four-vector with the muon mass, 0.105658 GeV.
* The **pair** is the sum of those two four-vectors; its invariant mass is the pair mass shown
  throughout.
* **The boost.** The leading muon is boosted into the rest frame of the pair, and the decay angle
  is the angle between that boosted momentum and the direction of the pair in the laboratory.
  The absolute value of its cosine is stored, since swapping the two muons flips the sign.
* The **momentum ratio** is the sub-leading over the leading muon transverse momentum in the
  laboratory, in the range 0 to 1.
* Companion quantities recorded alongside the angle for each pair: lepton-jet transverse momentum
  and isolation, pair mass, angular separation of the two muons, the smaller and larger of their
  transverse impact parameters, the number of displaced-standalone muons among the two (0, 1 or 2),
  each muon's transverse momentum, and the lepton-jet-level impact-parameter spreads used by the
  cleaning cuts.

All axes of a histogram are filled from one shared jagged array, so each pair contributes exactly
one entry per axis.  A single helper in `sidm/definitions/hists.py` computes the quantities, and the
per-axis functions only pick fields from it.

### The two-leading-muon convention

A muon lepton jet can hold more than two muons.  This study always takes the two hardest, and makes
no attempt to pick the pair that best reconstructs a dark photon.  The consequence is that in a
lepton jet with three or more muons the "pair" need not be the true decay pair, and the angle is then
a property of the reconstructed object rather than of a decay.  The number of displaced-standalone
muons in the pair is recorded so that this population can be separated out.

### Regions

Each region has one name in the text, given in the first column.  Figures and printed tables
abbreviate control region to CR, validation region to VR and muon lepton jet to mu-LJ, as in
"1 mu-LJ CR".  The validation regions and the two-lepton-jet control region are the signal-region
selection with the displacement requirement inverted, described below the table.

| region, as named in the text | channel | what it selects |
| --- | --- | --- |
| uncut single-lepton-jet control region | `data_control_region_1muLj` | events with exactly one lepton jet, a muon lepton jet, no cleaning cuts applied; the largest and least biased fake sample |
| cosmic-veto single-lepton-jet control region | `data_control_region_1muLj_cosmic_veto` | the same, with the cosmic-muon veto applied |
| cleaned single-lepton-jet control region | `data_control_region_1muLj_spread_cosAlpha_mu_veto` | the same, with the lepton-jet impact-parameter spread cuts and the muon-pair opening-angle veto applied |
| two-muon-two-electron validation region | `test_VR_2mu2e_invDisplaced_spread_cosAlpha_mu_veto` | the two-lepton-jet selection with one muon and one electromagnetic lepton jet, with the displacement requirement inverted on every lepton jet |
| four-muon validation region | `test_VR_4mu_invDisplaced_spread_cosAlpha_mu_veto` | the two-muon-lepton-jet selection with the displacement requirement inverted on every muon lepton jet |
| two-lepton-jet control region | `cr_2lj_invDisplaced_spread_cosAlpha_mu_veto` | at least two lepton jets of either type, the displacement requirement inverted on every one, with the spread cleaning and the opening-angle veto; no requirement on the pair mass or the azimuthal separation (section j) |
| electromagnetic-lepton-jet control region | `data_control_region_1egmLj_cosmic_veto` | events with exactly one lepton jet, an electromagnetic one, with the cosmic-muon veto (section j) |

The two validation regions are orthogonal to the signal region by construction: the signal region
requires every lepton jet to be displaced, and these require the exact negation (inverted
displacement).  A muon's pixel hits are the hits its track leaves in the silicon pixel detector,
the innermost part of the tracker; a muon from a decay beyond the pixel layers has few or none.
For a muon lepton jet the negation is that at least one of its muons has more than two pixel hits.
A displaced-standalone muon is given zero pixel hits, so that muon is always a particle-flow muon.
The requirement does not exclude displaced-standalone muons from the lepton jet, nor from the two
leading muons the angle is computed from.  The dependence of the angle on displacement is
measured in the uncut single-lepton-jet control region, which has no requirement on displacement.

The table below counts the data pairs of each control and validation region by the number of
displaced-standalone muons among the two, and gives for each class the fraction above
|cos(theta*)| = 0.4, the threshold quoted in sections f and h, with its binomial uncertainty.  It
includes the two-lepton-jet control region of section j, read from that campaign's files.
Classes with fewer than 20 pairs are marked."""))

    cells.append(new_code_cell("""_ = lib.dsa_composition_table(threshold=0.4)"""))
    cells.append(new_markdown_cell("""Mixed pairs, one particle-flow and one displaced-standalone
muon, are 29 percent of the data pairs in the four-muon validation region, 10 percent in the
two-muon-two-electron one and 41 percent in the two-lepton-jet control region; pairs of two
displaced-standalone muons are below one percent everywhere.  In the four-muon validation region
and the two-lepton-jet control region the two main classes have different shapes, 0.59 and 0.62
of the particle-flow pairs above 0.4 against 0.27 and 0.16 of the mixed ones.  The region
dependence found in sections e and h is not created by composition.  Within particle-flow pairs
alone the fraction above 0.4 runs from 0.12 in the cleaned single-lepton-jet control region to
0.59 and 0.62 in the four-muon validation region and the two-lepton-jet control region, while
mixed pairs stay between 0.13 and 0.27 in every region.  Any single number quoted for an
inverted-displacement region is therefore an average over two populations with unlike shapes."""))

    # ---------------------------------------------------------------- (b) reference
    cells.append(new_markdown_cell("""A muon lepton jet reconstructed from a dark-photon decay carries one internal degree of
freedom that the standard lepton-jet variables (pT, mass, dR, isolation, displacement) do
not use: where along the allowed kinematic band its two muons sit. In the rest frame of the
muon pair, a transversely polarized dark photon decaying to muons follows
dN/dcos(theta*) ~ 1 + alpha cos^2(theta*) with alpha = beta*^2 / (2 - beta*^2), where beta* is the
muon speed in the pair rest frame: alpha is 0.94 at m_Zd = 1.2 GeV, 0.996 at 5.0 GeV and 0.17 at
0.25 GeV, and exactly at threshold the muons are produced at rest in the pair frame and no decay
direction exists to measure. The dark photons of this signal are transverse because the mediator
couples to them through the field-strength operator (phi F F), which produces only transverse
states.

A fake muon lepton jet has no reason to follow this distribution.  In the laboratory the accessible observables are |cos(theta*)| and the momentum
ratio r = pT_sub / pT_lead; for a collimated pair r = (1 - beta* cos(theta*)) / (1 + beta* cos(theta*)),
so the narrow band in the (|cos(theta*)|, r) plane is a kinematic identity shared by any
collimated pair, and the discriminating information is the position along the band, not the
band itself.

This study measures that position for signal in its inclusive four-muon and
two-muon-two-electron channels, for simulated backgrounds, and for fake-enriched data control
and validation regions, and asks two questions:
does the shape separate genuine from fake muon lepton jets, and is it correlated with the
variables the background estimate already uses."""))

    cells.append(new_markdown_cell("""## b. Signal reference

Generator-level and reconstructed angle for one dark-photon mass, and the plane of angle against
momentum ratio with the two-body-decay band drawn on it.  All lifetimes and bound-state masses of
the dark-photon mass point are summed; the angular distribution of a two-body decay does not
depend on how far the parent travelled before decaying, so the sum is legitimate and enlarges the
sample.

The band is the ratio of laboratory momenta expected for a two-body decay of a fast parent,
(1 - beta* cos theta*) / (1 + beta* cos theta*), with beta* the muon speed in the pair rest frame
computed from the nominal dark-photon mass and the muon mass."""))

    cells.append(new_code_cell(
        """c_gen_reco = lib.fig_gen_vs_reco(m_zd_values=M_ZD_VALUES)"""))

    cells.append(new_code_cell("""c_ridge = lib.fig_ridge_map(m_zd=M_ZD_REFERENCE)"""))
    cells.append(new_markdown_cell("""How to read these figures: the left panels show the reconstructed pairs, the right panels the
generator-level truth, one row per mediator mass, and the dashed curve is the analytic band r =
(1 - beta* cos theta*)/(1 + beta* cos theta*) evaluated at the sample dimuon mass. The
reconstructed pairs track the analytic ridge bin for bin; the printed line under the figure
gives the agreement (median column deviation 0.6 percent, 23 of 25 columns within 4 percent,
the worst column at the sparse |cos(theta*)| = 0.98 edge).

At m_Zd = 1.2 and 5.0 GeV reconstruction changes the angle little: the hi/lo ratio (density
above 0.8 over density below 0.2) moves from 1.58 generated to 1.53 reconstructed at 1.2 GeV in
the 4mu channel, and by three to seven percent elsewhere. The generated curves at these two
masses are a sound reference: the precision check below reproduces the generated angle to about
one percent of alpha at every bound-state mass.

At m_Zd = 0.25 GeV the generated curve is not a sound reference, and the hi/lo it prints (1.83 in
4mu, 1.63 in 2mu2e) does not describe the decay. The generated histogram boosts each muon into
its dark photon's rest frame in single precision. A 0.25 GeV dark photon from a 500 or 1000 GeV
bound state has a Lorentz factor of 1,000 to 2,000, so 1 - beta^2 of the dark photon, 1e-6 to
3e-7, is only a few times the single-precision resolution of 6e-8, and the boost misplaces the
muon, more often towards |cos(theta*)| = 1. The expectation is alpha = 0.17 for beta* = 0.53, a
hi/lo of 1.13: the decay is nearly flat.

A reading of the angle that needs no boost,
|cos(theta*)| = |E1 - E2| / (beta* |p1 + p2|), confirms it in an offline check of one file
(check_gen_boost_precision.py in this directory), the 4mu sample with m_XX = 500 GeV,
m_Zd = 0.25 GeV and c tau = 0.004 mm (its file part-0, no selection, 16,674 dark photons with
two muons): alpha = 0.14 +- 0.03 with beta* taken from each dark photon's generated mass (0.19
expected for that mass distribution), while the processor's single-precision boost gives
0.51 +- 0.04 on the same muons and the same boost in double precision 0.15 +- 0.03. The
reconstructed hi/lo of 1.09 (4mu) and 0.90 (2mu2e) is close to the decay's 1.13, so the flat
reconstructed shape does not indicate a loss in reconstruction. It is the net of the decay, the
same single-precision boost applied to the stored muons, the detector resolution and the
acceptance, which this study does not separate; the table below gives what the arithmetic alone
does at truth level.

The overflow of the |cos(theta*)| axis holds the entries the arithmetic places outside it
(caveat 9): a few in a hundred thousand in data and at most 0.2 percent of the simulated entries
at m_Zd = 1.2 and 5.0 GeV, but 2 to 18 percent at m_Zd = 0.25 GeV with m_XX = 500 and
1000 GeV, as the table at the end of this section counts from the stored histograms.  Every
figure is built from the in-range bins."""))

    cells.append(new_markdown_cell("""### What the stored precision and the arithmetic can resolve

The generator record keeps GenPart pt, eta, phi and mass at 8 mantissa bits, a step of 0.4 to 0.8
percent of each value; for eta and phi of order one that is several times the 8e-4 rad median
opening angle between the two muons of a 0.25 GeV dark photon at m_XX = 500 GeV.  The generated
histograms do not use those branches: coffea builds the generator-level momenta from the px, py
and pz branches, which are stored in full single precision, and only the dark photon's mass is
read at 8 bits.  The reconstructed muons keep pt in full single precision and eta and phi at 12
mantissa bits, a step of up to 5e-4 rad for the values that occur.  Every boost is computed in
single precision.

The toy below generates XX -> Zd Zd -> 4 mu decays with the angular distributions of the model,
stores them at these precisions, and recomputes both angles with the processor's own functions.
The two angles are |cos(theta*)|, whose shape coefficient is alpha, and the azimuthal angle Phi
between the two dark photons' decay planes, whose coefficient A is defined in section j.  The
toy's docstring gives the kinematics.  There is no generator filter, no selection and no detector
resolution.  It reads no files and runs in about twenty seconds."""))

    cells.append(new_code_cell("""toy_rows = lib.storage_precision_toy()"""))

    cells.append(new_markdown_cell("""How to read this table: "exact" recomputes the angles in double
precision from the toy's four-vectors and recovers the input alpha and A at every point within two
standard deviations, which checks the toy and the fit.  "As the generated histograms" applies the
processor's functions to the generator record as coffea reads it.  At m_Zd = 1.2 and 5.0 GeV it
agrees with "exact" for both angles, within 0.011 in alpha (at m_XX = 1000 GeV, m_Zd = 1.2 GeV)
and exactly in A, so the generated references of sections b and j are sound there and the
reconstructed-over-generated ratios of section j need no correction.  At m_Zd = 0.25 GeV the
plane angle is unaffected, A_out = A_in, but |cos(theta*)| is not: alpha comes out 0.17 at
m_XX = 100 GeV (a Lorentz factor of about 200), 0.31 at 500 GeV and 1.7 at 1000 GeV against an
input of 0.17.  The boost-free "energy" reading stays at the input everywhere.

The toy does not reproduce the real file exactly.  In the offline check of one file
(check_gen_boost_precision.py in this directory) at m_XX = 500 GeV, m_Zd = 0.25 GeV the processor
gives alpha = 0.51 +- 0.04 and hi/lo 1.41 where the toy gives 0.31 and 1.23, and on the
corresponding m_XX = 1000 GeV file 2.2 against 1.7.  The toy uses a fixed m_Zd, while the
generated dark-photon mass has a broad lineshape (dimuon mass 0.226 to 0.366 GeV, 1 to 99
percent, on the checked file), and its spectrum of dark-photon energies is not the generator's
(median Lorentz factor 1,171 in the file against 1,066 in the toy at m_XX = 500 GeV); the
distortion grows steeply with the Lorentz factor.  The toy is not tuned to close the gap, so it
sizes the effect and does not correct the stored histograms, which cannot be refilled without
rerunning the campaign.  The stored generated histogram of the 4mu channel at m_XX = 500 GeV,
m_Zd = 0.25 GeV reads hi/lo 1.46 (table below), in line with the 1.41 of the unselected file.

"Reconstructed precision" is the reconstruction construction at truth level (each pair the sum
of its two stored muons, the angle that of the leading muon), with no detector resolution.  At
m_Zd = 0.25 GeV it is distorted as well (alpha 0.25 at m_XX = 500 GeV, 2.5 at 1000 GeV), so the
reconstructed angle at that mass carries the arithmetic on top of the detector resolution; at 1.2
and 5.0 GeV it equals the exact value, and the reconstructed plane angle keeps A to better than
one percent at those masses.  The last two columns of the Phi table give the correlation r of the
two signed helicity angles of section j, exact and as the generated histograms compute it.

The table below counts the overflow entries quoted above and in caveat 9 from the stored
histograms, flow bins included: in the data only in the uncut single-lepton-jet control region,
and in simulation in the inclusive channel of each mass point, all lifetimes summed."""))

    cells.append(new_code_cell("""overflow_rows = lib.overflow_table()"""))

    # ---------------------------------------------------------------- (c) acceptance
    cells.append(new_markdown_cell("""## c. Acceptance

The reconstructed angle in the two signal final states side by side, and where the sub-leading muon
of the pair sits in transverse momentum as a function of the angle.  The dashed lines on the maps
mark the two trigger-relevant thresholds; the top-right panel is the fraction of pairs below each of
them, drawn only where the angle bin holds at least 20 pairs.

The map is drawn on the bin index of the momentum axis, because that axis is variable-binned from
zero and a logarithmic axis would drop its first bin."""))

    cells.append(new_code_cell("""c_acceptance = lib.fig_acceptance(m_zd=M_ZD_REFERENCE)"""))
    cells.append(new_markdown_cell("""The reconstructed |cos(theta*)| shape is sculpted by the trigger and muon pT thresholds, and
the sculpting is channel dependent. Large |cos(theta*)| means a hard-soft pair: as
|cos(theta*)| -> 1 the sub-leading muon pT collapses (visible in the maps as the population
crossing below the 20-26 GeV lines). In the 2mu2e channel both level-2 trigger muons (23 or 25
GeV) must come from the single muon lepton jet, and the offline selection then requires two
muons above 26 GeV, so asymmetric pairs fail and the distribution is cut off well before
|cos(theta*)| = 1. In the 4mu channel the two required muons can come from different lepton
jets, and the asymmetric pairs survive even though the fraction of them whose sub-leading muon
falls below the 26 GeV requirement grows to about two thirds by |cos(theta*)| = 0.86. The same
mechanism acts in any selection requiring two muons above 26 GeV from one lepton jet, which is
why the signal shape in the cleaned single-lepton-jet control region below is depleted above
0.8. Consequence: none of the reconstructed shapes here is a polarization measurement; they are
shapes under a fixed selection, compared like for like."""))

    # ---------------------------------------------------------------- (d) signal vs fakes
    cells.append(new_markdown_cell("""## d. Signal, simulated background and data, region by region

Unit-area shapes so that only the shape is compared: rows are the five regions, columns the
dark-photon mass.  Signal curves sum the three bound-state masses with raw-count weights; the
per-mass table below shows the spread that sum hides.
Simulated backgrounds are combined with their cross-section weights before the shape is taken.  The
observed distribution carries asymmetric Poisson error bars.

Everything in a panel is drawn under the same selection as the data in that panel, which is the
comparison this study is built to make.  Panels with no entries are left empty and labelled.

Note on the two validation-region rows: their inverted-displacement requirement asks each muon lepton
jet for at least one muon with more than two pixel hits, which a displaced-standalone muon never has.
It does not exclude displaced-standalone muons from the pair: 29 percent of the four-muon and 10
percent of the two-muon-two-electron data pairs contain one (table in section a).  The row labels
state the requirement, and caveat 2 below spells it out."""))

    cells.append(new_code_cell(
        """c_shapes_cos = lib.fig_region_shapes("absCosTheta", m_zd_values=M_ZD_VALUES)"""))
    cells.append(new_code_cell(
        """c_shapes_ratio = lib.fig_region_shapes("ptRatio", m_zd_values=M_ZD_VALUES)"""))
    cells.append(new_markdown_cell("""How to read this figure: each row is one region, each column one dark-photon mass, every
curve is normalized to unit area, and the question is whether the fake-dominated data shape
differs from the signal shape under the same selection. The signal curves sum the three
bound-state masses m_XX = 100, 500 and 1000 GeV with raw-count weights; the printed table
under the figure gives every curve's hi/lo ratio and the per-mass split, and the spread
across m_XX is large (at m_Zd = 1.2 GeV in this control region the 4mu hi/lo runs from 0.07
at m_XX = 100 to 2.48 at m_XX = 1000, because heavier bound states make harder lepton jets
whose asymmetric pairs survive the thresholds). The data shape falls steeply with
|cos(theta*)| in the single-lepton-jet control regions and in the two-muon-two-electron
validation region (hi/lo 0.008 to 0.035) and only mildly in the four-muon validation region
(0.33). The signal curves as
drawn give hi/lo 1.24 and 1.16 in 4mu at m_Zd = 1.2 and 5.0 GeV; in 2mu2e they give 0.64
and 0.66, the tail above 0.8 removed by acceptance, with the separation from data living in
the middle of the axis, 0.4 < |cos(theta*)| < 0.8.

Three qualifications.  First, at
m_Zd = 0.25 GeV the signal column is nearly flat (hi/lo 0.73 and 0.72 as drawn), as the decay
itself is, 1 + 0.17 cos^2(theta*) (section b). Against the steeply falling single-lepton-jet
fakes a flat shape still separates: signal efficiency over the square root of the fake
efficiency for a cut at 0.4 is 1.6 to 1.7 against the cleaned single-lepton-jet control region
(section j), comparable to the 1.6 to 2.1 at the heavier masses. The reconstructed angle at this
mass carries the single-precision arithmetic of section b (4 to 12 percent of the four-muon
entries fall in the overflow at m_XX = 500 and 1000 GeV), so these numbers describe the variable
as computed, and it should be recomputed in double precision before it is used at this mass
point. Second, in regions that require two 26 GeV muons from the single
jet, the signal tail above 0.8 is removed by acceptance (previous section) and only the mid-axis
separation remains. Third, the weighted background simulation in the validation-region rows
rests on a few hundred effective entries dominated by a handful of high-weight QCD events;
it is shown for completeness, and the comparison is made with the data."""))

    # ---------------------------------------------------------------- (e) correlations
    cells.append(new_markdown_cell("""## e. What the angle correlates with

If the shape of the angle in a control region is to be carried into the signal region, it must not
depend on the quantities that separate the two.  Each panel below takes one companion quantity,
slices the data in it, and overlays the unit-area angular distribution of each slice; a change of
shape between the curves of one panel is a correlation.  The companions are split over two
figures each, four panels apiece, so that every panel is wide enough to read at the width the
figure is placed at.  The profile figures show the same
information as one number per companion bin, the mean angle, with the standard error of the mean and
only where the bin holds at least 20 pairs.

The measurement is made in the uncut single-lepton-jet control region, which is the only region that
contains pairs across the whole range of displacement, and the one with the most pairs containing
displaced-standalone muons in number.  The same set is repeated
for signal, where the angle is a property of a real decay."""))

    cells.append(new_code_cell("""c_corr_data_d = lib.fig_correlation_overlays(
    lib.data_samples(), lib.UNCUT_CR,
    "2018 data, uncut single-lepton-jet control region: isolation and impact parameters",
    "e1a_correlation_overlays_data_displacement", is_data=True,
    companions=lib.companion_group("displacement"))"""))
    cells.append(new_code_cell("""c_corr_data_k = lib.fig_correlation_overlays(
    lib.data_samples(), lib.UNCUT_CR,
    "2018 data, uncut single-lepton-jet control region: kinematics and muon type",
    "e1b_correlation_overlays_data_kinematics", is_data=True,
    companions=lib.companion_group("kinematics"))"""))
    cells.append(new_code_cell("""c_prof_data = lib.fig_correlation_profiles(
    lib.data_samples(), lib.UNCUT_CR,
    "2018 data, the three single-lepton-jet control regions",
    "e2_correlation_profiles_data", is_data=True,
    regions=lib.PROFILE_REGIONS)"""))
    cells.append(new_code_cell("""c_corr_sig_d = lib.fig_correlation_overlays(
    lib.signal_samples("4Mu", m_zd=M_ZD_REFERENCE), "4mu",
    f"signal $4\\\\mu$, $m_{{Z_d}}$ = {M_ZD_REFERENCE} GeV, four-muon channel: "
    "isolation and impact parameters",
    "e3a_correlation_overlays_signal_displacement",
    companions=lib.companion_group("displacement"))"""))
    cells.append(new_code_cell("""c_corr_sig_k = lib.fig_correlation_overlays(
    lib.signal_samples("4Mu", m_zd=M_ZD_REFERENCE), "4mu",
    f"signal $4\\\\mu$, $m_{{Z_d}}$ = {M_ZD_REFERENCE} GeV, four-muon channel: "
    "kinematics and muon type",
    "e3b_correlation_overlays_signal_kinematics",
    companions=lib.companion_group("kinematics"))"""))
    cells.append(new_code_cell("""c_prof_sig = lib.fig_correlation_profiles(
    lib.signal_samples("4Mu", m_zd=M_ZD_REFERENCE), "4mu",
    f"signal $4\\\\mu$, $m_{{Z_d}}$ = {M_ZD_REFERENCE} GeV, four-muon channel",
    "e4_correlation_profiles_signal")"""))
    cells.append(new_markdown_cell("""How to read these figures: each panel slices the control-region data in one companion variable;
identical normalized shapes across slices (overlays) and a flat profile of the mean (profiles)
would mean the decay angle carries information independent of that companion. The profile
figure overlays the three single-lepton-jet control regions: uncut, with the cosmic veto, and
cleaned by the spread cuts and the muon-pair opening-angle (cos alpha) veto.  Against lepton-jet
pT the mean rises in every region (Spearman rank correlation +0.98, +0.95, +0.31; from 0.31 at
40 GeV to 0.53 at 750 GeV in the uncut region): harder jets keep their asymmetric pairs, the
acceptance mechanism again. Against isolation the dependence is a moderate fall (mean 0.322 to
0.300 uncut; steeper, -0.93 in rank, after the spread cleaning). Against displacement the
answer depends on the cleaning, and that dependence is itself the finding. In the uncut region
the profile of the mean against the smaller |dxy| of the pair dips by about 13 percent below
one centimeter and then rises to 24 percent above its inner value in the outer bins; after the
cosmic veto the same profile falls by 20 percent (rank correlation -0.92, the visible
difference confined to the outermost bins), because the outer displacement bins of the uncut
region are dominated by the cosmic-ray population the veto removes, together with its |dxy| <=
40 cm requirement on the displaced-standalone muons; after the spread and opening-angle cleaning it
rises again (+63 percent, rank +0.68) on a much smaller residual sample. The size and even the
sign of the displacement dependence change with the cleaning applied, that is, with the
composition of the surviving fakes. The decay angle is therefore not measured to be independent
of displacement, and nothing in this study demonstrates that a fake shape measured in the
inverted-displacement regions transfers across the displacement axis. Any use of the variable in
a background estimate has to measure that dependence in the estimate's own regions."""))

    # ---------------------------------------------------------------- (f) cut scan
    cells.append(new_markdown_cell("""## f. What a straight cut on the angle would buy

Solid curves are the fraction of signal pairs a threshold keeps, measured in the signal channel of
each final state; dashed curves are the fraction of control-region pairs it rejects.  Bands are
Clopper-Pearson intervals.  Both directions of the cut are shown.

Signal efficiency and fake rejection are measured under different selections (signal in its own
channel, fakes in the data control region), so the two curves in a panel answer "what would this
threshold do here", not "what would it do to the same sample".

The signal curves sum the three bound-state masses with raw-count weights; the per-mass table below
shows the spread that sum hides.  The figure draws only the two single-lepton-jet control regions
as fakes, so the two validation regions are tabulated under it as well."""))

    cells.append(new_code_cell("""scan = lib.cut_scan(m_zd_values=M_ZD_VALUES)
c_scan = lib.fig_cut_scan(scan, m_zd_values=M_ZD_VALUES)"""))
    cells.append(new_code_cell("""_ = lib.efficiency_by_mediator_mass(m_zd_values=M_ZD_VALUES)
print()
_ = lib.validation_region_rejection()"""))
    cells.append(new_code_cell("""_ = lib.cut_scan_table(scan, direction="below")
print()
_ = lib.cut_scan_table(scan, direction="above")"""))
    cells.append(new_markdown_cell("""How to read this figure: solid curves are the fraction of signal pairs kept by a cut at the
threshold on the x axis, dashed curves the fraction of control-region data pairs rejected; both
directions of the cut are shown because the informative side differs by channel. The curves sum
the three bound-state masses; the printed tables under the figure give the per-mass
efficiencies and the validation-region rejections, and the mass dependence is decisive. A cut
keeping |cos(theta*)| > 0.4 rejects 79.6 percent of the data pairs in the two-muon-two-electron
validation region and 50.0 percent in the four-muon validation region (printed table; the figure
draws two of the single-lepton-jet control regions, where the same cut rejects 68 and 88
percent). The same cut
keeps, in the 2mu2e channel, 55 to 56 percent of signal pairs at m_XX = 500 GeV and 70 to 74
percent at 1000 GeV for the two heavier dark-photon masses, but only 5 percent at m_XX = 100
GeV (and 12 percent at m_Zd = 0.25 GeV there): the lightest bound state makes the softest
lepton jets, exactly the ones whose asymmetric pairs the trigger and the 26 GeV requirement
already remove. In the 4mu channel the efficiency is flatter, 57 to 71 percent across the grid.
These are per-lepton-jet numbers in fake-enriched displaced-vetoed regions, not per-event
signal-region predictions; in the 4mu channel a per-jet cut applies to both muon lepton jets of
an event, so the event-level signal cost is roughly the square of the per-jet efficiency. They
size the handle; they do not certify it."""))

    # ---------------------------------------------------------------- (g) caveats
    cells.append(new_markdown_cell("""## g. Caveats

These apply to every reading of the figures above.

1. **The reconstructed angle is shaped by the selection.** The high-to-low population ratio printed
   in the legends of the first figures is not the value an unsculpted 1 + cos^2 decay would give, and
   the two-muon-two-electron final state loses the region near an absolute cosine of one, because
   both muons that fire the dimuon trigger have to come from the one muon lepton jet and the
   sub-leading muon of a wide-angle pair falls below the trigger threshold.  The generator-level
   distribution in the same channel shows the same shape at m_Zd = 1.2 and 5.0 GeV, which is how
   the sculpting is identified as acceptance rather than physics; at m_Zd = 0.25 GeV the generated
   histogram is not a sound reference (caveat 9).  This study compares signal and fake shapes
   under an identical selection; it is not a measurement of the decay's angular distribution,
   which would require the efficiency to be corrected for.

2. **The inverted-displacement requirement does not exclude displaced-standalone muons.** In the
   two validation regions and the two-lepton-jet control region every muon lepton jet must contain
   at least one muon with more than two pixel hits.  The processor sets the pixel-hit count of
   a displaced-standalone track to zero, so that muon is a particle-flow muon, but the two leading
   muons the angle is computed from can include displaced-standalone ones.  In data the pairs split
   71, 29 and 0.2 percent into particle-flow, mixed and displaced-standalone pairs in the four-muon
   validation region, 90, 10 and 0.2 percent in the two-muon-two-electron one, 59, 41 and below 0.1
   percent in the two-lepton-jet control region, and 93, 6.5 and 0.8 percent in the uncut
   single-lepton-jet control region (table in section a).  Section e separates pairs by the
   number of displaced-standalone muons in the single-lepton-jet control regions; the table in
   section a gives the fraction above 0.4 for each class in every region.

3. **Impact parameters do not all share a reference.** The angle helper uses primary-vertex
   referenced transverse and longitudinal impact parameters for both muon types.  The lepton-jet
   spread quantities used by the cleaning cuts, which are kept exactly as the cuts define them, mix
   primary-vertex referenced particle-flow values with beamspot referenced displaced-standalone
   values; for a mixed pair the transverse spread is close to the displaced-standalone muon's own
   impact parameter rather than a separation between two vertices.  Do not describe it as a vertex
   separation, and slice by the number of displaced-standalone muons where the distinction matters.

4. **These are the as-run definitions, not main's.** Every channel here is defined by
   `selections_mulj_decay_angle_as_run.yaml`.  Main's channels of the same names add an eta-phi
   veto, the `DT + CSC hits > 3` requirement on the muons at the lepton-jet source, and the
   base-cut and spread-cut differences listed under "Selections used in this study",
   and under them the shapes would change as well as the yields: the cleaned single-lepton-jet
   control region, which differs from the uncut one by the spread and opening-angle vetoes alone,
   shows how strongly such cuts move the angle (data hi/lo 0.008 against 0.035).

5. **Histograms are raw counts; cutflows are not.** The campaign ran with unweighted histograms, so
   nothing in the histograms carries a cross-section or generator weight, while the cutflows in the
   same files are luminosity times cross-section scaled.  The two must never be combined.  A
   normalised simulation prediction is luminosity times cross section divided by the sum of
   generator weights over the events stored in the input files, the normalisation the processor
   itself applies and the one `lumi_xs_weight` and `planes_weight` return.  An earlier figure of
   about one event in ten kept by the signal skim, which would call for a generated count in the
   denominator instead, was an artefact: the per-file generated-event counter in the Runs tree is
   cumulative, so summing it over files overcounts the generated events by a factor of about ten.
   No generator filter is recorded in the files,
   and neither signal ntuple location file the campaigns read sets a `skim_factor`.  Shapes are
   unaffected in any case, because the factor cancels in a unit-area normalisation.

6. **Signal-region channels evaluate their event cuts on unfiltered lepton jets**, so an event whose
   lepton jets are all prompt can still pass a signal-region channel with no muon lepton jets left in
   it.  The orthogonality between the validation and signal regions holds at the level of histogram
   entries, not events.

7. **Two axes merge things that are not the same.** The lepton-jet count in the collection counts
   unfiltered lepton jets, that is, the event multiplicity.  Isolation fills a missing value with
   zero, so "no matched jet" and "isolation exactly zero" land in the same bin and in the first
   isolation slice.

8. **One warning to watch for in the campaign logs.** "Unable to apply event cuts to evt_weights.
   Skipping." means an event cut was not applied to the event weights.  Where it appears, the
   cutflows of that channel should not be trusted until it is understood.

9. **Numerical precision.** The generator record keeps GenPart pt, eta, phi and mass at 8
   mantissa bits, but the generated histograms build every momentum from the px, py and pz
   branches, which are full single precision, and read only the dark photon's mass at 8 bits.  The
   reconstructed muons keep pt in full single precision and eta and phi at 12 mantissa bits, a step
   of up to 5e-4 rad, against a median opening angle between the two muons of 8e-4 rad at
   m_XX = 500 GeV and 4e-4 rad at 1000 GeV for m_Zd = 0.25 GeV, and 6e-3 rad or more at the two
   heavier dark-photon masses.  All arithmetic is single precision.  Per mass point, from the
   table in section b: at m_Zd = 1.2 and 5.0 GeV the generated and reconstructed angles are computed
   to about one percent of alpha and the plane angle exactly, so every generated reference there
   can be used; at m_Zd = 0.25 GeV the plane angle and its generated coefficient are unaffected,
   but the generated |cos(theta*)| is pushed towards one by the single-precision boost at
   m_XX = 500 and 1000 GeV and is not a reference, and the reconstructed |cos(theta*)| is subject to
   the same arithmetic.

   The overflow bins come from the same arithmetic.  An absolute cosine cannot exceed one and the
   sub-leading momentum cannot exceed the leading one, yet the overflow bin of each histogram holds
   entries: 333 pairs of the 8.2 million in the uncut single-lepton-jet control region, four
   parts in a hundred thousand, and one pair for the momentum ratio (counted in the table at the
   end of section b).  In the data every one of these pairs has a computed dimuon mass either
   below twice the muon mass, which two muons cannot have and which only the finite precision of
   the computation produces, or above it, where the boosted muon is within single precision of
   parallel to the pair and the cosine comes out as exactly one, the upper edge of the axis,
   which the histogram counts as overflow.  272 of the 333 have a computed dimuon mass below
   0.2 GeV, the other 61 between 0.2 and 0.8 GeV, none above.  In simulation the
   |cos(theta*)| overflow depends on the mass point: at most 0.2
   percent of the reconstructed or generated entries at m_Zd = 1.2 and 5.0 GeV, but at
   m_Zd = 0.25 GeV 4 and 12 percent of the reconstructed four-muon entries at m_XX = 500 and
   1000 GeV (2 and 7 percent in two-muon-two-electron) and 4 and 18 percent of the generated ones
   (3 and 11 percent).  In an offline check of one unselected file of each of those two points
   (check_gen_boost_precision.py in this directory), the single-precision boost returns the
   cosine of 1,136 of 33,352 and 2,544 of 17,752 generated muons as not a number, which the
   histogram counts as overflow, and none at one or above; in double precision every one of
   those muons has a finite cosine.  Every figure and number here is built from the in-range
   bins alone.

10. **The data regions hold signal.**  Collision data is read in seven regions.  The three
    single-lepton-jet control regions (uncut, with the cosmic veto, and with the spread and
    opening-angle vetoes) ask for one muon lepton jet, and the electromagnetic-lepton-jet control
    region, read by the decay-plane campaign only, asks for one electromagnetic lepton jet.  The
    two validation regions and the two-lepton-jet control region are the signal-region selection
    with the displacement requirement inverted on every lepton jet and no isolation requirement.
    They form the prompt-prompt corner of the displacement plane, where every lepton jet fails the
    displacement requirement, with no isolation split and no ABCD cells (the four regions of the
    two-variable background estimate).  A signal with a short proper lifetime is prompt by that
    definition, so these three regions collect it: in the
    table below (column reg/ch, the region's entries over those of the sample's signal channel)
    0.851 of the four-muon channel's muon-lepton-jet entries of 4Mu_500GeV_5p0GeV_0p08mm fall in
    the four-muon validation region.

    The table counts each region as each campaign draws it, at a signal cross section of 1 fb:
    each raw entry of the campaign's own histogram is weighted by the luminosity times that cross
    section over the campaign's own sum of generator weights.  The first and the pair campaign
    count muon lepton jets, so their expected numbers are muon-lepton-jet entries, not events; the
    decay-plane campaign counts events.  For m_XX = 500 and 1000 GeV the largest shares of the
    data are 1.7289 percent in the four-muon validation region (4Mu_500GeV_5p0GeV_0p08mm; the
    pair and decay-plane campaigns give 1.7293 and 1.7305 percent for the same sample), 0.3452
    percent in the first campaign's two-muon-two-electron validation region, 0.5216 percent in
    the two-lepton-jet control region, at most 0.0005 percent in the single-lepton-jet control
    regions, and 10.6436 percent in the decay-plane campaign's two-muon-two-electron validation
    region (2Mu2E_500GeV_5p0GeV_0p08mm; 4 of 22 samples above 1 percent).  The decay-plane
    campaign's own single-lepton-jet control regions are not in the table: its muon one is the
    cleaned region above under the same definition, and its electromagnetic one is not counted.

    The expected count is proportional to the assumed cross section.  `cross_sections.yaml` lists
    1.264 fb at m_XX = 500 GeV and 0.342 fb at 1000 GeV, and the table prints each region's
    largest share at the listed value as well: 2.1854 percent in the four-muon validation region
    and 13.4536 percent in the decay-plane two-muon-two-electron one.

    The two-muon-two-electron validation region holds 7,403 muon-lepton-jet entries in the first
    campaign and 86 events in the decay-plane campaign.  The first campaign fills one entry per
    muon lepton jet with two or more muons, whatever the other lepton jet is.  The decay-plane
    histograms are filled only for events with two qualifying lepton jets, each a muon lepton jet
    with two or more muons or an electromagnetic one with two or more electrons
    (`lj_pair_evt_mask` in `sidm/definitions/hists.py`).  In that region's data 64 of the 7,447
    electromagnetic lepton jets hold two or more electrons, so few events qualify, while the
    signal's electron pair at m_Zd = 5.0 GeV is resolved far more often; in the four-muon
    validation region, where both lepton jets are muon lepton jets, the two counts differ by the
    factor 2.00 of two jets per event.  This is why the same region holds 0.3452 percent signal in
    one campaign and 10.6436 percent in the other.

    The effect on the two numbers quoted from the validation regions, computed in the same cell:
    subtracting each sample's expected entries at 1 fb from the four-muon validation region moves
    its data high-to-low ratio of 0.327 +- 0.018 by at most +0.013 (4Mu_1000GeV_1p2GeV_0p0096mm),
    within its statistical error; and in the decay-plane two-muon-two-electron validation region,
    where the data give A = -0.507 +- 0.150 on 86 events (section j), the shift f x (A_s - A) a signal of
    fraction f and coefficient A_s adds is at most +0.096 (2Mu2E_500GeV_5p0GeV_0p08mm,
    A_s = +0.394).  The signal pulls A towards positive values, so the negative value is not
    produced by signal; both shifts scale with the assumed cross section."""))
    cells.append(new_code_cell(f"""# signal expected in each data region at 1 fb, region by region and campaign by campaign
lib.configure_pair("{PAIR_INPUT}")
lib.configure_planes("{PLANES_INPUT}")
rows = lib.signal_contamination_table()"""))

    # ---------------------------------------------------------------- (h) verdict
    cells.append(new_markdown_cell("""## h. Verdict

The decay angle of the muon pair separates genuine from fake muon lepton jets against some
regions and not against others. Fakes in data fall steeply in |cos(theta*)| in the
single-lepton-jet control regions and the two-muon-two-electron validation region (density ratio
hi/lo 0.008 to 0.035) and only mildly in the four-muon validation region (0.33), while the
m_Zd = 1.2 and 5.0 GeV
signals are flat to rising, with the separation concentrated at |cos(theta*)| > 0.4. As signal
efficiency over the square root of the fake efficiency for a cut at 0.4, at m_XX = 500 and
1000 GeV (from the printed tables of sections f and j), the angle separates against the cleaned
single-lepton-jet control region (1.6 to 2.1) and against the two-muon-two-electron validation
region (1.2 to 1.6 for 2mu2e signal), less against the uncut single-lepton-jet control region (1.0
to 1.3), and not against the four-muon validation region (0.8 to 1.0 for 4mu signal) nor the
two-lepton-jet control region (0.8 to 1.1). Within each region it is larger at m_XX = 1000 GeV
than at 500 GeV.

The handle is real but narrower than a single number suggests. In the 2mu2e
channel a |cos(theta*)| > 0.4 requirement on the muon lepton jet keeps 55 to 74 percent of
signal pairs at m_XX = 500-1000 GeV while removing about 80 percent of validation-region fakes
(from the validation-region shapes, printed under the cut scan), but it keeps only 5 percent at
m_XX = 100 GeV, so it cannot be applied uniformly across the signal grid. In the 4mu
channel the per-jet numbers are 57 to 71 percent kept for 50 percent rejected, and a
per-jet cut acts on both lepton jets of an event, squaring the signal cost. At
m_Zd = 0.25 GeV the decay itself is nearly flat (alpha = 0.17 for beta* = 0.53, hi/lo 1.13) and
so are the reconstructed shapes; against the cleaned single-lepton-jet control region that still
separates (1.6 to 1.7, section j), but the angle as computed at this mass carries the
single-precision arithmetic of section b and needs a double-precision recomputation before use.

On correlation: the decay angle rises strongly with lepton-jet pT in every region, so a cut
would sculpt pT-correlated spectra, including the muon-lepton-jet-pair mass that the 4mu search
cuts into windows (section i).
Its dependence on displacement changes size and sign with the cleaning applied (flat with
rising tail uncut, falling after the cosmic veto, rising after the spread cleaning), which
means the fake composition, not the geometry, sets the shape; this study does not establish
that a fake shape measured in the inverted-displacement regions transfers to the displaced
signal regions.

Recommendation: treat |cos(theta*)| as a candidate purity handle and cross-check for the
2mu2e channel at m_XX of a few hundred GeV and above, and as a shape variable worth
monitoring in the 4mu channel; not as a drop-in axis or cut for the background estimate.
Before any use in the estimate itself it needs the displacement dependence measured in the
estimate's own regions, a closure test against the mass axis of the affected search, and
the per-mass acceptance map of this study repeated on the population that matters there.
Two structural caveats stand: the validation regions hold pairs containing
displaced-standalone muons (29 percent of the four-muon and 10 percent of the
two-muon-two-electron data pairs, caveat 2), whose shape differs from that of the
particle-flow pairs (table in section a), so a number quoted for a validation region averages
the two; and the background simulation is statistically too weak to arbitrate any of this, so
the data control and validation regions carry the conclusions."""))

    # ---------------------------------------------------------------- (i) the LJ pair
    cells.append(new_markdown_cell("""## i. The two muon lepton jets of a four-muon event

Sections b to h treat one muon lepton jet at a time.  A four-muon event holds two of them, and
two questions about the pair were left open: whether the two decay angles of one event are
correlated, and whether the angle still separates once the muon-lepton-jet-pair mass has been
used, that is, inside a mass window of the four-muon search.

The pair campaign answers them.  It fills two histograms: one carries both angles of an event,
ordered by the transverse momentum of the parent lepton jet, and one carries the angle of each
qualifying lepton jet against the muon-lepton-jet-pair invariant mass of its event, on a binning
fine enough to cut that mass into windows.  The mass is built exactly as the validation
region's event cut builds it.  Collision data was written in the four-muon validation region
only; simulated signal was written in the four-muon
channel, in its signal region, in a copy of the channel without the event-level muon momentum
cut, and in the same validation region.

The search uses bound-state masses of 200 GeV and above, so every number below is quoted for
m_XX = 500 and 1000 GeV; the lighter points are not loaded.  The three dark-photon
masses are kept apart, because sections d to f found the separation grows with that mass.  The
caveats of section g apply here unchanged, the raw-count one included."""))

    cells.append(new_code_cell(f"""# the pair campaign has its own merged outputs, so this is a second input directory
lib.configure_pair("{PAIR_INPUT}")
print("pair input     :", lib.PAIR_INPUT)
print("data samples   :", len(lib.pair_data_samples()))
print("signal points  :", " | ".join(f"m_XX = {{m:.0f}}, m_Zd = {{z}}"
                                     for m, z in lib.pair_points()))"""))

    cells.append(new_markdown_cell("""### Are the two angles of an event correlated

The figure divides the observed plane of the two angles by the product of its own two marginal
distributions.  A plane of ones is what independence looks like; a cell above one means that
combination occurs more often than two independent draws would give.  The independence
chi-square quoted below compares the observed plane with that product."""))

    cells.append(new_code_cell("""c_pair_plane = lib.fig_pair_plane()"""))
    cells.append(new_code_cell("""_ = lib.pair_correlation_table()"""))
    cells.append(new_code_cell("""factorisation_rows = lib.pair_factorisation_table()"""))
    cells.append(new_code_cell(
        """_ = lib.composition_excess_bound(observed=factorisation_rows)"""))
    cells.append(new_code_cell("""c_pair_cond = lib.fig_pair_conditionals()"""))

    cells.append(new_markdown_cell("""How to read these figures: in the validation-region data the cells along the diagonal are
enhanced, 1.20 where both lepton jets sit below 0.2 and 1.40 where both sit above 0.8, and the
two off-diagonal corners are depleted to 0.71 and 0.77.  The linear correlation coefficient is
+0.124 +- 0.019 on 2516 events, and the independence chi-square is 57 for 16 degrees of freedom,
a probability of 2 in a million.

Signal is consistent with independence, with two qualifications.  In the **inclusive** four-muon
channel, where each point carries between 119,000 and 612,000 events, the coefficient at
m_XX = 500 and 1000 GeV is at most +0.003 in absolute value and the independence chi-square runs
from 14 to 24 for the same 16 degrees of freedom.  In the **signal-region** channel, where the
samples are one to two orders of magnitude smaller, two points are less quiet: m_XX = 500 GeV
with m_Zd = 0.25 GeV gives +0.065 +- 0.030 on 1046 events, 2.2 standard deviations, and
m_XX = 1000 GeV with m_Zd = 1.2 GeV has a coefficient consistent with zero, +0.004 +- 0.004, but
an independence chi-square of 26 for 16 degrees of freedom, a probability of 0.059.  Neither is
significant among twelve sets and neither is a reason to distrust the signal side; both are
reasons not to carry "every signal point is consistent with independence, coefficient below
0.006" as a statement about the whole grid, because it is a statement about the inclusive channel
only.  The m_XX = 100 GeV points, below the masses the search uses, are correlated at the +0.09 to
+0.11 level in **both** channels.

One limit of scope, which sets what any of this can be carried to: the four-muon pair grid this
campaign ran holds m_XX = 100, 500 and 1000 GeV only.  The 200 to 250 GeV range at the low edge
of the search is not in it, and is bracketed by the 100 and 500 GeV points rather than measured.

Fake muon lepton jets in one event are therefore correlated with each other and signal ones are
not, which is what a cause shared by the whole event produces: the trigger that selected it, its
kinematics, or the source of its fake muons.

The second figure says the same thing without a model: it normalises the subleading jet's angle
separately in three slices of the leading jet's.  In signal the three curves lie on top of each
other.  In the data they do not: with the leading jet below 0.3 the subleading density falls from
1.57 to 0.99 across the first four bins, and with the leading jet above 0.7 it rises from 1.03 to
1.32 over the same range.  Both slices drop in the last bin, where the acceptance edge of
section g bites.

The practical consequence is in the printed table, and how large it is depends on what "the
per-lepton-jet rate" is taken to mean, so the table prints both.  Against **one inclusive
per-jet efficiency squared** (the usual shorthand, with both lepton jets of every event pooled
into one sample), a requirement applied to both jets keeps 1.029 +- 0.008 times as many
validation-region events at a threshold of 0.2, 1.102 +- 0.020 at 0.4, 1.185 +- 0.046 at 0.6 and
1.286 +- 0.221 at 0.8.  Against the **product of the two ordered marginals** (the leading jet's
rate times the subleading jet's, which at the tightest threshold are 0.116 and 0.064 and so are
not the same number), they are 1.029, 1.102, 1.187 and 1.405 +- 0.240.

In summary: about 10 percent at a threshold of 0.4, where the two
definitions agree to three decimals and the uncertainty is a fifth of the effect, and roughly 30
percent at 0.8, on the 26 events that pass it, where the inclusive definition gives 1.29 and the
ordered one 1.41, the two differ by half of either uncertainty, and neither is more than about
1.3 standard deviations from unity.  An event-level background rate obtained by squaring a
per-jet efficiency is optimistic by an amount that is measured at 0.4 and only indicated at 0.8,
and the two angles are not two independent axes for a factorised background estimate in this
region.  The composition table bounds what the mixture of pair types can contribute: if the two
jets of an event always shared a pair type, the mixture alone would give an excess of
8.7 +- 1.1 percent at 0.4 and 16 +- 4 percent at 0.8 over the square of the pooled per-jet
fraction, against the 10.2 +- 2.0 and 29 +- 22 percent observed."""))

    cells.append(new_markdown_cell("""### Does the angle still separate inside a mass window

The four-muon validation region cuts the muon-lepton-jet-pair mass at 150 GeV, and the four-muon
search reads that mass; the question here is what the angle adds once a window on it has been
taken. The table gives the shape of the angle window by window in the validation-region data, the
first figure draws it, and the second asks what a cut, or the whole shape, would buy inside the
window each signal point actually fills."""))

    cells.append(new_code_cell("""_ = lib.pair_window_table()"""))
    cells.append(new_code_cell("""c_window_shapes = lib.fig_window_shapes()"""))
    cells.append(new_code_cell("""gain_rows = lib.pair_gain_table()"""))
    cells.append(new_code_cell("""c_window_gain = lib.fig_window_gain(gain_rows)"""))
    cells.append(new_code_cell("""_ = lib.pair_plane_gain_table()"""))

    cells.append(new_markdown_cell("""How to read these: the validation-region data falls steeply in the pair mass.  Of its 5039
lepton-jet entries, 3009 sit in the 150 to 200 GeV window and 151 above 380 GeV.  The window used
for the m_XX = 500 GeV signal is 380 to 600 GeV, which that signal fills to 81 to 94 percent and
which holds 127 fake pairs; the window used for m_XX = 1000 GeV is everything above 600 GeV,
which that signal fills to 93 to 99 percent and which holds **24** fake pairs.  Those two numbers,
127 and 24, are what the table, both figures and every gain below are built on.

The fake shape is not the same in every window: the mean angle rises from 0.411 +- 0.005 at 150
to 200 GeV to 0.449 +- 0.027 at 380 to 500 GeV, and the lowest window alone contributes 16 to the
chi-square against the pooled shape.  A fake shape measured where the data is cannot be carried
unchanged to where the signal is.  The left panel of the first figure draws the five windows from
150 to 500 GeV.  The four windows above it (500 to 600, 600 to 780,
780 to 1000 and above 1000 GeV) hold 26, 20, 2 and 2 pairs, every one below the fifty-entry floor this study draws a shape
at, so they are named on the panel itself and read from the table above it instead of being drawn
as a picket fence of five bins holding four entries each.

Inside the window the signal fills, the separation is small.  At m_XX = 500 GeV the signal keeps
84 percent of its lepton jets above 0.2 while the data in the same window keeps 76 percent, a
signal over root background factor of 0.97: the best of the four thresholds does not improve the
sensitivity, and every tighter one makes it worse, down to 0.6 at a threshold of 0.8.  Using the
whole five-bin shape instead of one cut gives 1.060 +- 0.040 at m_XX = 500 GeV, m_Zd = 1.2 GeV
and 1.095 +- 0.161 at 1000 GeV.  That estimator is biased upwards when the background sample is
small, because noise in a background bin reads as separation; resampling puts the bias at +0.02
on the 127 fake pairs of the 500 GeV window and +0.10 on the 24 of the 1000 GeV one, so the
bias-corrected values are 1.040 +- 0.040 and 0.999 +- 0.161.  The right panel of the second
figure draws those corrected values as open markers beside the raw ones, because the raw points
on their own read as six numbers above one while the correction moves the three heavy ones onto
it.

So the answer at m_XX = 500 GeV is a corrected gain of 1.040 +- 0.040, one standard deviation from
unity: **consistent with no gain, and at most a few percent per lepton jet**.  At 1000 GeV the three
corrected values are 0.966, 0.999 and 0.987 with an uncertainty of 0.15 to 0.16 on 24 fake pairs,
which is no gain distinguishable from none.  Measured instead against the larger pooled fake
sample above 300 GeV, which assumes the mass independence the window table has just rejected,
the raw values are 1.016 to 1.049 at 500 GeV and 1.031 to 1.143 at 1000 GeV.

The last table repeats the exercise on the plane of both angles, where the data sample is large
enough to measure something: over all pair masses the plane reaches 1.244 +- 0.028 at
m_XX = 500 GeV, m_Zd = 1.2 GeV and 1.634 +- 0.076 at 1000 GeV, which is 0.99 and 0.95 of the
square of the corresponding one-jet number.  The correlation among fakes costs a few percent of
what two independent lepton jets would have given, no more.  That comparison is inclusive in the
pair mass, and the difference between it and the per-window numbers is the point.  The natural
reading of that difference is that most of the inclusive separation is the steeper angular shape
of the low-mass fakes and the mass window removes it.  That is an **inference** from the two
sets of numbers side by side, not a measurement: nothing here decomposes the inclusive gain into a
mass-shape part and an angle-shape part, and the study would need the gain recomputed window by
window on a fake sample large enough to support it before the reading could be called
established."""))

    cells.append(new_markdown_cell("""### What the pair says

Fake pairs are correlated and signal pairs are not.  In the four-muon validation region the
two decay angles of an event have a linear correlation of +0.124 +- 0.019
on 2516 events and an independence chi-square of 57 for 16 degrees of freedom, the excess sitting
on the diagonal.  In the inclusive four-muon channel every signal point that still matters is
consistent with independence at the +-0.003 level; in the smaller signal-region channel two of
the twelve sets sit at about two standard deviations, which chance alone produces among twelve
sets.  The
arithmetic consequence is that an event rate built by squaring a per-lepton-jet efficiency is
optimistic here by about 10 percent at a threshold of 0.4, where the measurement is solid, and by
roughly 30 percent at 0.8, where it rests on 26 events and the two reasonable definitions of the
per-jet rate differ by half their own uncertainty.  So the squaring shorthand of
section h fails for fakes; for signal, whose two jets are independent, it stands.

Inside a mass window the angle adds nothing this data can resolve.  The validation data falls
steeply in the pair mass and its angular shape is itself mass dependent, so the fake shape has to
be measured where the signal is, and there it rests on 127 pairs at m_XX = 500 GeV and 24 above
600 GeV.  No single threshold helps.  The whole five-bin shape gives a bias-corrected gain of
1.040 +- 0.040 per lepton jet at 500 GeV (consistent with no gain, at most a few percent) and
0.97 to 1.00 with an uncertainty of 0.16 at 1000 GeV.  The larger inclusive numbers, 1.24 and
1.63 for the two-angle plane over all pair masses, are most naturally read as the low-mass fakes'
steeper shape, which the window removes; that reading is an inference from the two sets of
numbers, not a decomposition this study performs.

Both readings inherit the study's caveats.  The validation region requires a muon with more
than two pixel hits in each muon lepton jet, which leaves 29 percent of its muon pairs with a
displaced-standalone muon (caveat 2).  That mixture matters for the factorisation test: the
particle-flow and mixed pairs of this region have unlike shapes (0.59 against 0.27 above 0.4),
and if the two jets of an event always shared a pair type the mixture alone would produce an
excess of 8.7 +- 1.1 percent at 0.4 and 16 +- 4 percent at 0.8 over the square of the pooled
per-jet fraction, against the 10.2 +- 2.0 and 29 +- 22 percent observed (against the product of
the two ordered per-jet fractions the observed excess at 0.8 is 40.5 +- 24.0 percent).  The
uncertainty on the bound resamples events, not jets, since the two jets of an event are
correlated.  The pair type of the two jets
is not recorded jointly in these histograms, so the study cannot say how much of the excess is
angular correlation at fixed pair type and how much is shared composition.  For an estimate that
squares an inclusive per-jet efficiency the consequence is the same.  Nothing here shows that a
fake shape measured in the validation region transfers to the signal region, and the grid stops
short of the 200 to 250 GeV bound-state masses at the low edge of the search.  The
recommendation keeps section h's direction and sharpens its size: do not add the decay angle to
the four-muon selection or to its background estimate, and do
keep the pair correlation as a measured fact about fakes: the two muon lepton jets of a fake
four-muon event are not independent, which any factorised estimate in that channel must face."""))

    # ---------------------------------------------------------------- (j) the decay planes
    cells.append(new_markdown_cell("""## j. The lepton jets as a decay-plane system

Sections b to i take the decay angle of one muon lepton jet, and then the two of a four-muon
event.  The decay-plane campaign (`LjAnglePlanes_Run2018_v1`) completes the construction.  It adds three
things: the **signed** decay angle, so that the negatively charged lepton picks a direction and
not only an axis; the same pair of quantities for the two leading **electrons** of an
electromagnetic lepton jet; and the event-level observables of the two leading qualifying lepton
jets of either type (their two helicity angles in the rest frame of their sum, the azimuthal
angle between their two decay planes, their transverse balance, and the missing transverse
momentum with and without the displaced-standalone muons added back).

Why the plane angle is worth a section.  The signal as generated is **CP even**: the mediator
couples to the two dark photons through a field-strength operator, phi F F, which produces only
transverse dark photons.  Each dark photon decay then follows 1 + alpha cos^2(theta*) in its own
rest frame, with alpha = beta*^2 / (2 - beta*^2), and the azimuthal angle Phi between the two
decay planes follows 1 + A cos(2 Phi) with A = [beta*^2 / (3 - beta*^2)]^2, the product of the two
decays' analyzing powers.  The planes are preferentially parallel, with A = +1/4 for massless leptons, 0.228 for
muons at m_Zd = 1.2 GeV and 0.011 at 0.25 GeV, all for an unselected sample.  A CP-odd parent of
the same mass would give 1 - A cos(2 Phi), the same magnitude with the opposite sign.  That is
the one place in this whole study where the
angles carry information about what the parent *is* rather than about how clean the lepton jet
is.

What the reconstructed distribution can and cannot say about that is the point of the section,
and the answer is set by two facts stated here once.  First, the **reference is never the
analytic curve**: the production filter of the sample (four leptons above 5 GeV within
|eta| < 2.4) and then the channel's own selection both reshape the truth distribution, so every
reconstructed distribution below is compared with the **generator-level histogram from the same
files and the same channel**, and the analytic +1/4 is drawn only as the unsculpted
massless-lepton reference it is.  Second, in the lab frame the energy asymmetry of a pair is
(E1 - E2) / (E1 + E2) = beta beta* cos(theta*), with beta the dark photon's speed in the lab and
beta* the lepton's in the dark-photon rest frame, so as |cos(theta*)| approaches one, one lepton
goes soft and falls below threshold: the reconstructed angular shapes are acceptance sculpted and
are **not** a polarization measurement without an efficiency map.  Electrons are worse than muons,
because bremsstrahlung moves energy out of the electron and into photons the pair does not
carry.

Collision data was written in control and validation regions only; the loader refuses to open a
data file in any other channel rather than trusting the caller.  The search uses bound-state
masses of 200 GeV and above, so every number is led by m_XX = 500 and 1000 GeV.  This
campaign's grid holds 100, 500 and 1000 GeV only, so the 200 to 250 GeV range is bracketed by it,
not covered."""))

    cells.append(new_code_cell(f"""# the decay-plane campaign has its own merged outputs, so this is a third input directory
#
#   campaign   : LjAnglePlanes_Run2018_v1  (submitted 2026-09-02; 7,096 jobs, 154 samples)
#   merged     : /store/group/lpcmetx/SIDM/coffea_outputs/murtazas/lj_angle_planes
#                and the local copy below; merged and read on 2026-09-05
#   collections: lj_decay_angle, lj_pair_decay_planes, and lj_pair_decay_planes_gen on signal
lib.configure_planes("{PLANES_INPUT}")
lib.planes_inventory()"""))

    cells.append(new_markdown_cell("""### Where the decay angle of a muon lepton jet lives, mass by mass

The first figure puts the decay angle against the dimuon mass of the same pair, and normalises
every mass row on its own.  That is the comparison that does not depend on the mass spectra of
signal and fakes being wildly different: at a given dimuon mass, is the angle distributed
differently?"""))

    cells.append(new_code_cell("""c_mulj_plane = lib.fig_mulj_mass_angle(m_xx=500.0)"""))
    cells.append(new_code_cell("""c_mulj_overlay = lib.fig_mulj_angle_overlays(m_xx=500.0)"""))
    cells.append(new_code_cell("""mu_rows = lib.planes_mu_separation_table()"""))
    cells.append(new_code_cell(
        """_ = lib.pair_type_separation(threshold=0.4, m_xx=500.0, m_zd=1.2)"""))

    cells.append(new_markdown_cell("""How to read these: every mass row is normalised on its own, so a cell's colour says how the
angle is distributed **at that dimuon mass**, not how many pairs sit there; the colour scale is
capped at the 99th percentile so that one sparsely filled row cannot set it for all six panels.

Two features carry the figure.  In the signal panels the row at the dark photon's own mass is
spread nearly flat across the angle, which is the decay; the brightest cells sit elsewhere, at
high dimuon mass and high |cos(theta*)|, and they are the pairs whose two hardest muons are not
one dark photon's: a muon lepton jet may hold more than two muons and this study always takes
the two hardest, as section a says.  In the fake panels the population traces a **ridge**: the
angle a pair reconstructs climbs steadily with its dimuon mass, from a fraction of a GeV up to a
few GeV, and there is no comparable population anywhere else in the plane.  The same ridge is
visible in the signal panels at dimuon masses well below the dark photon's, where the pair is
again not a decay, so it is a property of accidental pairs rather than of the fake samples; that
reading follows from where the ridge appears and is not separately demonstrated here.

The one-dimensional overlays and the printed table put numbers on it, and the numbers say that
the size of the separation is set almost entirely by
**which fake sample it is measured against**.  A threshold at |cos(theta*)| > 0.4 keeps 0.66 to
0.71 of signal muon-lepton-jet pairs at m_XX = 500 and 1000 GeV with m_Zd = 1.2 or 5.0 GeV.  The
same threshold keeps 0.120 of the pairs in the cleaned single-lepton-jet control region, 0.434
of those in the two-lepton-jet control region, and 0.500 of those in the four-muon validation
region.  The resulting signal efficiency over root fake efficiency is 1.9
against the first and 1.0 against the second: against the control region whose event topology
actually resembles the signal region, two lepton jets rather than one, the handle is worth
nothing.

The right-hand panel is the same statement without a threshold: the curve against the
cleaned single-lepton-jet control region bows well above the diagonal, and the curve against the
two-lepton-jet control region lies almost on it.  This is the sharpest form of the warning
of sections e and h, that a fake shape measured in one region does not transfer, and here it is
measured rather than argued.  Against the cleaned single-lepton-jet control region the
m_Zd = 0.25 GeV signal column is below the heavier masses in four-muon (1.65 and 1.74 against
1.90 to 2.04 at m_XX = 500 and 1000 GeV) and level with them in two-muon-two-electron at
m_XX = 500 GeV (1.62 against 1.59 and 1.61): a nearly flat decay still separates against steeply
falling fakes, though the angle at that mass carries the single-precision arithmetic of
section b.  The simulated background is shown
with its cross-section weights and rests on far fewer effective entries than the data."""))

    cells.append(new_markdown_cell("""### The electron pair of an electromagnetic lepton jet

The same construction on the two leading electrons of an electromagnetic lepton jet.  Before any
shape is read, one number has to be quoted: **how often the two electrons are reconstructed
separately at all**.  A dark photon of low mass is highly boosted, its two electrons merge into
one supercluster plus photons, and the lepton jet then holds one electron, which no electron-pair
histogram is filled from."""))

    cells.append(new_code_cell("""egm_rows = lib.planes_egm_resolved_table()"""))
    cells.append(new_code_cell("""c_egm = lib.fig_egm_ee()"""))
    cells.append(new_code_cell("""_ = lib.planes_egm_shape_table()"""))

    cells.append(new_markdown_cell("""How to read these: the resolved fraction sets what the electron channel can do.  At
m_Zd = 5.0 GeV it is 29 percent of electromagnetic lepton jets at m_XX = 500 GeV and 10.5 percent
at 1000 GeV; at m_Zd = 1.2 GeV it falls to 1.6 percent and 0.17 percent; at m_Zd = 0.25 GeV it is
about one in ten thousand, 13 lepton jets of 110,000 at m_XX = 500 GeV.  The trend is exactly the
boost: heavier bound state, more boost, more merging, and lighter dark photon likewise.  In the
data the fraction is 0.44 percent in the electromagnetic-lepton-jet control region and 0.87
percent in the two-lepton-jet control region.

That fraction also limits how the shapes can be read.  The resolved pairs at
m_Zd = 1.2 GeV are not a representative tenth of the decays: they are the asymmetric tail that
happened to split, with a mean momentum ratio of 0.15 against 0.38 at m_Zd = 5.0 GeV and a mean
|cos(theta*)| of 0.76 against 0.50.  A cut placed on the electron-pair angle would therefore be
acting on a sample that is already selected by whether the pair resolved, and what the numbers
support is that the electron pair separates from the fake control regions strongly (mean
|cos(theta*)| of 0.50 to 0.87 in signal against 0.31 to 0.33 in data), **on the subset where it
exists at all**, which at the masses the search cares about is a few percent of electromagnetic
lepton jets or less.  The panels draw only the shapes with enough entries to draw; the data
control regions carry 257 and 575 resolved pairs and are rebinned to five bins for that
reason."""))

    cells.append(new_markdown_cell("""### The two helicity angles of the lepton-jet pair

Section i asked whether the two **absolute** decay angles of a four-muon event are correlated.
The pair histograms of this campaign carry the **signed** helicity angle of each of the two
leading qualifying lepton jets, measured in the rest frame of their sum, which is the angle the
decay-plane construction actually uses.  The figure is built exactly as section i's is: the
observed plane divided by the product of its own two marginals, so a plane of ones is
independence."""))

    cells.append(new_code_cell("""c_pair_helicity = lib.fig_pair_helicity_plane(point=(500.0, 1.2))"""))
    cells.append(new_code_cell("""pair_rows = lib.planes_pair_correlation_table()"""))

    cells.append(new_markdown_cell("""How to read these: signal is independent and fakes are not, which is section i's conclusion
again, but with the opposite **sign**, and that is worth stating because the two are different
measurements and not two views of one.  On the signed angle the two-lepton-jet control region
gives
r = -0.062 +- 0.011 on 8,445 events and the four-muon validation region -0.053 +- 0.021 on 2,516;
on the absolute angle of section i the same validation-region events gave +0.124 +- 0.019.  A
sample can be positively correlated in |cos| and negatively correlated in cos: the first is a
statement about how central the two pairs are together, the second about whether their two
negative leptons prefer the same hemisphere.  The simulated background has a coefficient
consistent with zero, -0.006 +- 0.007, so the linear effect the data shows is not reproduced by
the simulation; see the second caution below before reading that as independence.

Signal is flat in the four-muon channel on **both** tests: the coefficient is no larger than
0.006 in absolute value at m_XX = 500 and 1000 GeV, on 118,000 to 612,000 events, and the
independence chi-square runs from 9 to 17 for 16 degrees of freedom.  The generator-level planes
of the same samples agree, with one exception, m_XX = 1000 GeV and m_Zd = 0.25 GeV, where the
generated plane carries +0.020 +- 0.003.  Single-precision arithmetic does not produce it: in the
toy of section b the correlation of the two helicity angles at that point comes out the same in
single and double precision, r = +0.0001 +- 0.0022 in both (the last two columns of its Phi
table), so the arithmetic that distorts the generated |cos(theta*)| there does not correlate the
two angles.  What produces the coefficient is not established here.

Two cautions on reading the printed table, because the coefficient and the chi-square answer
different questions and here they answer them differently.

First, the **simulated background is not independent** even though its coefficient is: -0.006 +-
0.007 against a chi-square of 601 for 16 degrees of freedom on 21,924 entries.  It is not
linearly correlated and it is not independent either, so "the simulation does not reproduce the
data's correlation" is a statement about the coefficient alone and must not be carried further
than that.

Second, the **two-muon-two-electron signal points also carry large chi-squares** (392 for 16
degrees of freedom at m_XX = 500 GeV, m_Zd = 1.2 GeV, and 46 at m_Zd = 5.0 GeV) with
coefficients consistent with zero.  In that channel the two lepton jets are of different types,
one muon and one electromagnetic, and their acceptances in the helicity angle are not the same
function of it; a test against the product of the two marginals registers that difference as
dependence.  The four-muon points, where both lepton jets are of one type, do not do this, which
is the reason the section's statement about signal independence is made in the four-muon channel
and not across both."""))

    cells.append(new_markdown_cell("""### The angle between the two decay planes

The observable the section was built for.  `Phi` is the signed azimuthal angle between the two
lepton jets' decay planes in the rest frame of their sum, and `alpha` is the same angle taken
between unoriented planes, so that cos(2 alpha) = cos(2 Phi) and the two carry the same
coefficient.  The coefficient is quoted as **A in 1 + A cos(2 Phi)**, obtained as twice the mean
of cos(2 Phi) over the distribution, which uses every entry and needs no fit."""))

    cells.append(new_code_cell("""c_phi = lib.fig_pair_planes_phi(points=((500.0, 1.2), (500.0, 5.0)))"""))
    cells.append(new_code_cell("""phi_rows = lib.planes_phi_table()"""))

    cells.append(new_markdown_cell("""How to read these: the generated coefficient is **not** +1/4 at every mass point, and that is the
first thing the figure shows.  For an unselected sample A is the product of the two decays'
analyzing powers, A = [beta*^2 / (3 - beta*^2)]^2: +1/4 for massless leptons, 0.228 for muons at
m_Zd = 1.2 GeV and 0.249 at 5.0 GeV.  At m_XX = 500 GeV the generated value is +0.255 +- 0.002 at
m_Zd = 1.2 GeV and +0.296 +- 0.002 at 5.0 GeV; at m_XX = 100 GeV it rises to +0.355 and +0.404,
above +1/4, the production filter and the selection between them enhancing it; and at
m_Zd = 0.25 GeV it is +0.015 +- 0.003, against 0.011 expected for beta* = 0.53.  The
two-muon-two-electron points follow the same rule with the electron pair's analyzing power of
1/2: 0.052 expected at m_Zd = 0.25 GeV, +0.068 and +0.053 generated at m_XX = 500 and 1000 GeV.
The precision check of section b computes the generated plane angle exactly at every point,
0.25 GeV included, so these generated values are a sound reference, and the analytic curve is
only the limit for an unfiltered sample with massless leptons.

Reconstruction dilutes the coefficient but does not destroy it: the four-muon points retain 70 to
86 percent of their generated value (+0.193 reconstructed against +0.255 generated at
m_XX = 500 GeV, m_Zd = 1.2 GeV; +0.255 against +0.296 at 5.0 GeV; +0.150 against +0.213 and
+0.218 against +0.256 at 1000 GeV).  For the four-muon points the generated and reconstructed
histograms are filled from the **same events** (the two entry counts agree exactly at
m_Zd = 1.2 and 5.0 GeV and differ by at most 27 entries in 156,000 at m_Zd = 0.25 GeV, where a
few events have no negatively charged daughter in the generator record), so that ratio is a
dilution and not an acceptance difference.  For the two-muon-two-electron points they are not the
same events, because a reconstructed pair needs a resolved electron pair that the generated one
does not: at m_XX = 500 GeV, m_Zd = 5.0 GeV the generated histogram holds 183,000 events and the
reconstructed one 54,000, and the reconstructed coefficient there is **larger** than the
generated one, +0.393 +- 0.006 against +0.346 +- 0.003, for that reason alone rather than because
reconstruction improved anything.

What this can and cannot say about a CP-even (0+) against a CP-odd (0-) parent.  It **can** say that the sign of A survives
reconstruction with a dilution of 14 to 30 percent in the four-muon channel at the masses where
beta* is close to one, measured against a generated reference the precision check shows to be
sound there, so a sample of real signal events would show a positive A and a CP-odd parent would show a negative
one, and the two hypotheses are not degenerate after reconstruction.  It **cannot** be read as a
measurement of A: the value depends on the dark-photon mass through beta*, on the production
filter, and on the channel's own acceptance, so any statement about the parent's quantum numbers
needs the efficiency in (theta*, theta*, Phi) mapped out for the point being tested, which this
study does not do.  And the discriminating power against **fakes** is modest and must not be
overstated: the two-lepton-jet control region gives A = +0.062 +- 0.015 on 8,446 events and the
four-muon validation region +0.061 +- 0.028, both positive and both about a quarter of the
signal's.  The simulated background gives +0.094 +- 0.010 counting its unweighted entries and
+0.021 once the cross-section weights are applied; a factor-four swing between those two shows
how little the simulation can settle here, and the numbers below are taken from the data regions
for that reason.  Fakes are not flat in Phi.  The one outlier is the two-muon-two-electron
validation region, A = -0.51 +- 0.15 on 86 events; on that sample size and with that region's
mixed pair composition it is not a result, and more data would be needed to check it.  Caveat 10
of section g explains why that region holds so few events here and sizes the signal in it."""))

    cells.append(new_markdown_cell("""### Do the two lepton jets balance

A real pair comes from one parent, so the two lepton jets should balance in the transverse plane:
a small momentum asymmetry, a small vector sum, and no genuine missing momentum.  A fake pair need
not.  Particle-flow missing momentum does not include displaced-standalone muons, so the
meaningful quantity is the corrected one, the missing-momentum vector with the two leading lepton
jets' displaced-standalone constituents added back."""))

    cells.append(new_code_cell("""c_balance = lib.fig_pair_balance()"""))
    cells.append(new_code_cell("""balance_rows = lib.planes_balance_table()"""))

    cells.append(new_markdown_cell("""How to read these: **the momentum asymmetry separates and the missing momentum does not.**  A
requirement of ptAsym < 0.2 keeps 0.847 to 0.884 of four-muon signal pairs at m_XX = 500 and
1000 GeV, and 0.874 to 0.905 of the two-muon-two-electron pairs at m_Zd = 5.0 GeV, against 0.560
in the two-lepton-jet control region and 0.465 in the four-muon validation region: a signal
efficiency over root fake efficiency of 1.13 and 1.24.  The weighted simulated background keeps
0.610, more than either data region, so on that sample the requirement separates nothing at all;
its shape is carried by a handful of high-weight simulated events, so the data regions are the
reference.  It is the best single number this section produces, and it is still modest.

The vector sum does not separate at all.  Its mean is 54 to 58 GeV for signal at
m_XX = 500 GeV against 49 GeV in the two-lepton-jet control region and 54 GeV in the
four-muon validation region, and at m_XX = 1000 GeV it is 86 to 91 GeV, **larger** than either
fake sample, because a heavier resonance recoils harder and its 400 to 500 GeV lepton jets carry
a momentum resolution of tens of GeV.  The weighted simulated background is lower still, 43 GeV,
which points the same way and rests on the same handful of high-weight events.

One result to read carefully, because it is the opposite of what the correction was built to do.
The displaced-standalone correction should shrink the missing momentum of a signal pair, and in
every sample here it enlarges it: the mean goes from 51 to 64 GeV at m_XX = 500 GeV, m_Zd = 0.25
GeV, from 78 to 88 GeV at 1000 GeV, m_Zd = 5.0 GeV, from 35 to 41 GeV in the two-lepton-jet control
region and from 40 to 44 GeV in the weighted simulated background: a 10 to 25 percent increase
everywhere, signal and fakes alike. It is a **bulk shift, not a tail effect**: the median moves
from 30.4 to 35.2 GeV in the two-lepton-jet control region and from 35.7 to 42.8 GeV for the m_XX =
500 GeV, m_Zd = 0.25 GeV signal.

The formula is the intended one.  The function `lj_pair_met_observables` subtracts the vector sum of the
displaced-standalone constituents of the **two leading qualifying lepton jets** from the
missing-momentum vector, with the sign that should cancel them, so the premise fails rather than
the code: either those muons' momentum already enters particle-flow missing momentum through
their tracker tracks, or the displaced-standalone momentum resolution adds more than the
subtraction removes.  Histograms cannot separate the two, and **neither the corrected nor the
uncorrected missing momentum is usable until a per-event check is made**; the balance argument
rests on the momentum asymmetry alone.  The last panel makes the same point jointly: the fraction
of pairs with both the corrected missing momentum and the vector sum below a common threshold is
no larger for signal than for the control region at any threshold."""))

    cells.append(new_markdown_cell("""### Summary table"""))

    cells.append(new_code_cell("""_ = lib.planes_summary_table()"""))

    cells.append(new_markdown_cell("""### Conclusions of section j

1. The decay-plane angle Φ is the one observable in this study that probes the parent rather
   than lepton-jet quality, and it survives reconstruction with the sign the CP-even model
   predicts: in the four-muon channel the coefficient A of cos 2Φ is +0.15 to +0.26 reconstructed
   against +0.21 to +0.30 generated for m_Zd = 1.2 and 5 GeV at m_XX = 500 and 1000 GeV, a
   dilution of 14 to 30 percent,
   against generated values the precision check of section b reproduces exactly.  It is not a
   measurement of A.  For an unselected sample A is the product of the two decays' analyzing
   powers, A = [β*²/(3 − β*²)]², 0.228 at m_Zd = 1.2 GeV and 0.011 at 0.25 GeV (β* = 0.53), where
   the generated value is +0.015 and the planes carry no information; the generated value also
   depends on the production filter and on the channel, so a 0+ against 0- statement would need
   an efficiency map this study does not have.  Against fakes it is weak: the two-lepton-jet
   control region and the four-muon validation region give A = +0.062 +- 0.015 and
   +0.061 +- 0.028, a quarter of the signal value and not zero.

2. Among the regions of this campaign, the per-lepton-jet decay angle separates only against the
   cleaned single-lepton-jet control region (ε_s/√ε_b = 1.9 at |cos θ*| > 0.4) and not against
   the two-lepton-jet control region (1.0), whose topology is the closer match to the signal
   region.
   That region is 59 percent particle-flow pairs, 0.62 of them above the cut, and 41 percent
   mixed pairs, 0.16 above it, so its 0.43 averages two unlike shapes: with the signal
   efficiency unchanged the same cut gives 0.84 against the particle-flow pairs alone and 1.6
   against the mixed ones.  The fake shape depends on the region, which is what section h found,
   and on the pair type.

3. Two resolved electrons in an electromagnetic lepton jet occur in 29 percent of signal jets at
   m_Zd = 5 GeV (m_XX = 500 GeV) against under 1 percent of fakes: a clean but low-efficiency tag
   for that mass family alone.  At m_Zd <= 1.2 GeV the resolved pair exists in at most 1.6 percent
   of jets and cannot be used.

4. Transverse-momentum balance is the most usable new handle and it is modest: ptAsym < 0.2 keeps
   0.85 to 0.90 of signal pairs against 0.56 in the two-lepton-jet control region, ε_s/√ε_b of
   1.1 to 1.2.

5. Missing transverse momentum separates nothing, and the displaced-standalone correction
   increases it in every sample, in the bulk of events and not only in a tail (median 30 -> 35 GeV
   in the two-lepton-jet control region, 36 -> 43 GeV for signal at m_Zd = 0.25 GeV), where a
   signal pair should make it shrink.  The formula is the intended one, the missing-momentum vector
   minus the vector sum of the displaced-standalone constituents of the two leading lepton jets, so
   the premise fails rather than the code: either those muons' momentum already enters particle-flow
   missing momentum through their tracker tracks, or the displaced-standalone momentum resolution
   adds more than the subtraction removes.  Histograms cannot separate the two; neither version of
   the missing momentum is usable until a per-event check is made.

6. The two lepton jets of a fake event are correlated in their signed helicity angles
   (r = -0.062 +- 0.011 with χ² = 309/16 in the two-lepton-jet control region) while four-muon
   signal pairs are not (|r| <= 0.006).  With section i, no factorised estimate over these angles
   may assume independence in fake-dominated regions.  A pair type shared by the two jets cannot
   produce the negative coefficient: with the same angular shape for both jets it gives a
   covariance of zero or above.  Pair types that anti-correlate between the two jets could, and
   the stored histograms do not record both jets' types.  The chi-square part, the joint
   distribution against the product of its marginals, can be compositional even without them:
   the region mixes pair types with unlike shapes (conclusion 2), and two jets of one event that
   share a pair type fail the product test without any angular correlation at fixed type.  The
   study does not separate that part.

7. The simulated background cannot arbitrate any of this: cross-section weights concentrate it in
   a few QCD events and flip its fake shapes (ptAsym < 0.2 keeps 0.33 unweighted, 0.61 weighted).
   The data control regions are the reference throughout.

Net: nothing here changes the search selection.  What the section establishes is the fake-pair
dependence (angular or compositional, which the stored histograms cannot separate), the CP-even
sign of the reconstructed plane correlation with its suppression at low m_Zd, where
A = [β*²/(3 − β*²)]² falls to 0.011, and the histograms that now record these quantities."""))

    # ---------------------------------------------------------------- run record
    cells.append(new_markdown_cell("""## Run record

Everything the run could not draw, and why.  On a complete set of campaign inputs this list should
be short; on a reduced input set it records which panels were left empty."""))

    cells.append(new_code_cell("""messages = lib.notes()
print(f"{len(messages)} note(s) recorded during this run")
for message in messages:
    print(" -", message)"""))

    notebook = new_notebook(cells=cells, metadata={
        "kernelspec": {"display_name": "SIDM (LCG_107 Py3.11)", "language": "python",
                       "name": "sidm_venv"},
        "language_info": {"name": "python", "version": "3.11.9",
                          "mimetype": "text/x-python", "file_extension": ".py",
                          "nbconvert_exporter": "python", "pygments_lexer": "ipython3",
                          "codemirror_mode": {"name": "ipython", "version": 3}},
    })
    with open(OUTPUT, "w", encoding="utf-8") as handle:
        nbformat.write(notebook, handle, version=4)
    print(f"wrote {OUTPUT} with {len(cells)} cells "
          f"({sum(c.cell_type == 'code' for c in cells)} code)")


if __name__ == "__main__":
    build()
