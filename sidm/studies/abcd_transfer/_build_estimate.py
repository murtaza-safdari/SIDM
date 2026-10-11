"""Build estimate.ipynb.

Run from this directory on the LPC after merge_abcd8d.py has filled the local directory
(ABCD8D_LOCAL, default /uscms_data/d3/murtazas/abcd8d_local; ABCD6D_LOCAL still points at the 6D
copies used by the earlier notebooks):
    python _build_estimate.py
    jupyter nbconvert --to notebook --execute --inplace estimate.ipynb
"""
import nbformat as nbf


def md(t):
    return nbf.v4.new_markdown_cell(t)


def code(t):
    return nbf.v4.new_code_cell(t)


KERNEL = {"kernelspec": {"name": "sidm_venv",
                         "display_name": "SIDM (LCG_107 Py3.11)",
                         "language": "python"},
          "language_info": {"name": "python"}}

cells = []

cells.append(md(r"""# Background estimate of the 2μ2e and 4μ search regions

This notebook assembles the ABCD background estimate of the two search regions from the 2018 data
and states how well the method's assumption is tested. Throughout, a lepton jet (LJ) is a cluster
of two muons (a muon LJ) or of electrons and photons (an egamma LJ); the search regions pair a
muon LJ with an egamma LJ (2μ2e) or with a second muon LJ (4μ).

**The method in one paragraph.** Every event has two objects: the leading muon LJ (object 0) and
the second object (object 1: the egamma LJ in 2μ2e, the subleading muon LJ in 4μ). Each object is
*prompt* or *displaced* — a muon LJ is displaced when the maximum pixel-hit count over its muons
is ≤ 2, an egamma LJ when its minimum lost-hit count is ≥ 1 (photon-only LJs carry the sentinel
999 and are displaced by construction) — which gives four planes **PP, PD, DP, DD**, first letter
the muon LJ. The signal lives in DD. Event cuts everywhere: LJ-pair mass M ≥ 150 GeV (the high-mass
window; M < 150 GeV is the open low-mass window) and the second object isolated, iso1 < 0.5. Inside
each plane the ABCD grid uses two variables of the leading muon LJ, its isolation iso0 and its
azimuthal separation from the other object $|\Delta\phi|$:

| | iso0 < 0.25 | iso0 ≥ 0.25 |
|---|---|---|
| $|\Delta\phi| \ge 2.4$ | **A** | B |
| $|\Delta\phi| < 2.4$ | C | D |

The sidebands start at A's own edges: there is no buffer band between A and B or between A and C.
If iso0 and $|\Delta\phi|$ are independent for the background, A = B C/D; the measure of that is
κ = A D/(B C), measurable wherever A is open. A configuration label such as
`dphi 2.4/2.4 x iso0 0.25/0.25, iso1<0.5, M>=150` reads: the $|\Delta\phi|$ axis passes at ≥ 2.4
and fails below 2.4, the object-0 isolation passes below 0.25 and fails at ≥ 0.25, object 1 is
isolated below 0.5, and the LJ-pair mass is at least 150 GeV. `PD:dsa1+` and the like split a plane
by a stored variable (`mu*` pixel hits of object 0, `dsa*` its DSA-muon count, `e`/`g` the egamma
class, in 4μ `sub*` and `subdsa*` for object 1).

**The strategy.** The background in A of the DD plane is estimated four ways and the four are
compared: directly, as B C/D from DD's own sidebands, which assumes independence and nothing
else; and by borrowing a ratio from the PD plane (muon LJ prompt, egamma LJ displaced), B × (A/B),
C × (A/C) and D × (A/D), each ratio taken per class of the muon LJ (with or without a
displaced-standalone, DSA, muon) and summed, because the two classes have ratios a factor two
apart and the search plane's sidebands hold a far larger DSA share than PD. The independence
assumption is tested where A is open: in PP and PD above 150 GeV (large statistics, different
background mix), in DP above 150 GeV (the direct test of the muon-displacement step, small
statistics) and in DD itself below 150 GeV (the search plane's own correlation, coarse). The
systematic uncertainty of the estimate is the non-closure seen in those tests plus the spread
between the routes. The headline numbers are integrated over M ≥ 150 GeV, the one-bin search
region of the analysis; the windows 150–300 GeV and M ≥ 300 GeV (where the 500–1000 GeV signal
points sit) are carried alongside, because κ falls between the two windows in both planes. 4μ has too few
events in its DD sidebands for any ABCD and gets a count-limited estimate instead (section 5).

**What changed since the previous version of this notebook.** The earlier version kept a buffer
band between A and its sidebands (B at iso0 ≥ 0.5, C and D at $|\Delta\phi| < 2.0$), cut the muon
LJ's mass below 75 GeV, and used a proxy-muon region with a b-tagged jet as a further validation
region. The buffers are gone: the $|\Delta\phi|$ buffer was most of PP's non-closure (0.84 → 0.97
without it), the isolation buffer part of PD's (1.25 → 1.20), the signal in the two bands is
0.3–0.9% and 1.1–2.8% of the search-region yield, and the configuration scan of section 4 shows
that what remains is in the planes: PD stays 11–27% above one for every unbuffered boundary, and
PP closes at the search region's $|\Delta\phi|$ edge of 2.4 but not at 2.0 or 2.2. The mass cut is gone because it removed 7 of 1343 events in A of PD and
moved no ratio beyond its last digit — the Z → μμ muon LJ it targeted hardly occurs. The proxy
region is gone because its lone muon is not the search channel's object (its isolation pass/fail
ratio among back-to-back events was 6–23 against about one for a muon LJ) and its one subsample
with the search plane's ratios held 36 events; the b-tag requirement entered only through it. The hidden box is now the search
region itself.

**Blinding.** Data cubes are loaded only through `transfer_lib.summed_cubes`, which hides region
A — iso0 < 0.25, iso1 < 0.5, $|\Delta\phi| \ge 2.4$, M ≥ 150 GeV — of the DD plane in both
channels and, in 4μ, of the DP and PD planes as well: those two corners hold a quarter to a half
of the 4μ signal's search-region yield (section 5) against a background of tens of events, so they
are search regions in all but name, while their sidebands hold about 1% of that yield and are
open. A projection touching a hidden cell prints as "blind" or NaN. Signal and simulation cubes
are never hidden; signal shares of every region read from are tabulated.

**Inputs and normalisation.** The 2018 DoubleMuon data (59.83 fb⁻¹) after the certified
luminosity sections of `Cert_314472-325175_13TeV_Legacy2018_Collisions18_JSON`, the skimmed
LLPNanoAOD background samples of the shared `Backgrounds/2018_v2/Skims` area (Drell–Yan, QCD, top,
diboson) and the v10 signal samples, processed with the campaign code under `campaign/` (Dongyub
Lee's framework copy with a raw-event pre-filter added) into the eight-dimensional histograms
`abcd8d_<channel>`: the six axes of his `ABCD_6D_inclusive_eHEM_skimv1` campaign (the two
isolations, the two displacement variables, $|\Delta\phi|$, the LJ-pair mass) plus the leading muon
LJ's own mass and its DSA-muon count (in 4μ, the subleading muon LJ's DSA count as well).
Simulated events carry the generator weight and the HEM15/16 treatment of the `_eHEM` channels
(data events of runs ≥ 319077 with a jet or electron in the HEM15/16 region are removed; simulated
events with such an object are weighted by the unaffected luminosity fraction, 0.351). Each sample
is normalised to luminosity × cross section (`cross_sections.yaml`) by its summed generator weight
before the skim; the Condor chunks, each scaled with its own sum of weights, are merged as
Σ H_j S_j / Σ S_j (`merge_abcd8d.py`). No pileup reweighting, L1-prefiring weight or lepton scale
factor is applied: these are corrections of a few per cent to simulated yields, and simulation
enters this notebook only through composition tables and the signal-contamination estimates, never
the data-driven estimates themselves.

**Uncertainties.** 68.27% Clopper–Pearson intervals for ratios of disjoint counts, relative errors
added in quadrature for ratios of ratios and for sums over classes, the 68% upper limit of 1.84
events times the factor for a count of zero, $N_{\text{eff}}$ = (Σw)²/Σw² for simulated yields; a
"±" is printed where the two sides of an interval agree within 5%, otherwise "−m +p".

**One known difference to the 6D histograms.** The isolation axis of the 8D histograms has no
underflow bin, and an LJ's isolation can be slightly negative (the matched jet's energy fractions
can sum to more than one). The data and the QCD, top and diboson samples were filled with the
isolation clipped at zero; the Drell–Yan and signal samples were filled before that clip, so they
lack those events, which all lie in the isolation-pass regions. Section 1 measures the effect
(−9 of 130 simulated Drell–Yan events in A of PP)."""))

cells.append(code(r"""import dataclasses
import os
import re
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import abcd6d_io as io
import transfer_lib as tl
import transfer_plots as tp
import plane_search_lib as ps
import plane_search_plots as pp
import estimate_lib as el

pd.set_option("display.max_columns", 40)
pd.set_option("display.max_rows", 200)
pd.set_option("display.width", 250)
pd.set_option("display.max_colwidth", 200)
tp.set_style()

BASE6D = tl.local_dir()
BASE = os.environ.get("ABCD8D_LOCAL", "/uscms_data/d3/murtazas/abcd8d_local")
print("6D inputs:", BASE6D)
print("8D inputs:", BASE)
data6 = tl.summed_cubes(BASE6D, "data")
data = tl.summed_cubes(BASE, "data", prefix="abcd8d")
print("hidden box:", io.blind_signature(prefix="abcd8d"))
for name, cubes in (("6D", data6), ("8D", data)):
    for t, c in cubes.items():
        print(f"{name} {t}: {c.sumw.shape}, {int(np.isnan(c.sumw).sum())} hidden cells, {np.nansum(c.sumw):.0f} open events")
d2, d4 = data["2mu2e"], data["4mu"]
NOMINAL = ps.Config(fail0=0.25, dphi_fail=2.4)             # no buffer between A and its sidebands
BUFFERED = ps.Config()                                      # the previous version's boundaries, for comparison
LOW = dataclasses.replace(NOMINAL, mass=(None, 150.0))
WINDOWS = [("M >= 150", NOMINAL),
           ("150-300", dataclasses.replace(NOMINAL, mass=(150.0, 300.0))),
           ("M >= 300", dataclasses.replace(NOMINAL, mass=(300.0, None)))]
PLANES = ("PP", "DP", "PD", "DD")
SIGNAL_POINTS = {"2mu2e": ["2Mu2E_200GeV_0p25GeV_1p0mm", "2Mu2E_500GeV_0p25GeV_0p4mm", "2Mu2E_800GeV_0p25GeV_0p25mm",
                           "2Mu2E_1000GeV_0p25GeV_0p2mm", "2Mu2E_800GeV_1p2GeV_1p2mm", "2Mu2E_1000GeV_1p2GeV_0p96mm",
                           "2Mu2E_800GeV_0p25GeV_1p2mm"],
                 "4mu": ["4Mu_200GeV_0p25GeV_1p0mm", "4Mu_500GeV_0p25GeV_0p4mm", "4Mu_800GeV_0p25GeV_0p25mm",
                         "4Mu_1000GeV_0p25GeV_0p2mm", "4Mu_800GeV_1p2GeV_1p2mm", "4Mu_1000GeV_1p2GeV_0p96mm",
                         "4Mu_800GeV_0p25GeV_1p2mm"]}
print("nominal configuration:", NOMINAL.label())
print("previous boundaries:  ", BUFFERED.label())"""))

cells.append(md(r"""## 1. The histograms against the earlier campaign, and what the merge found

The 8D data cubes projected onto the 6D regions must reproduce the 6D counts exactly (same
events, same campaign skims, the clipped isolation lands in the first bin of the pass region
either way). The Drell–Yan sample was filled before the clip: its differences are the events it
lost. The comparison uses the previous boundaries, whose sidebands exist in both sets of cubes.

They do, to the event, in every 4μ region and in the pass regions of 2μ2e; the 2μ2e fail regions
lose a few events (B of PP 1021 → 1016, B and D of PD 886 → 885 and 163 → 162), and the open
totals 20 of 23,515 events in 2μ2e and 13 of 9,247 in 4μ. The known differences between the two
campaigns are the raw-event pre-filter of the new code, built to leave every channel's yield
unchanged, and the chunks the merge did not find (the manifest summary below: one data chunk of
12,771, and in the simulation 7 of 540 Drell–Yan chunks, 1 of 309 QCD chunks and one or two chunks
of three signal points, among them 2Mu2E_800GeV_0p25GeV_0p25mm with 3 of 4 — a loss of statistics,
not of normalisation, since every simulated chunk is scaled by its own sum of generator weights);
neither is shown here to account for the 20 events, and the pixel-hit axis of the new histograms,
which has no overflow bin, is not the cause (no muon LJ of the 6D cubes has 12 or more pixel hits).
The loss is 0.1%, sits in the regions with a non-isolated muon LJ, and is unexplained at that
level; it moves no ratio by more than 0.6% and nothing below depends on it. The Drell–Yan
differences are what the table says they are: −9 of 130 simulated events in A of PP from the
negative-isolation events the first code version did not clip, at most +3% elsewhere."""))

cells.append(code(r"""print("data, previous boundaries (planes PP, DP, PD, DD; the hidden regions stay hidden in both)")
for t in tl.TOPOLOGIES:
    print(t)
    display(el.consistency_table(data6[t], data[t], BUFFERED))
groups6 = tl.group_cubes(BASE6D, "bkg")
g2 = tl.group_cubes(BASE, "bkg", prefix="abcd8d", topologies=("2mu2e",))     # one topology at a time: a 4mu cube is 0.4 GB
print("process groups, 6D:", sorted(groups6), " 8D:", sorted(g2))
print("Drell-Yan, 2mu2e, all masses: 6D against 8D (the 8D cube lacks the negative-isolation events)")
display(el.consistency_table(groups6["DY"]["2mu2e"], g2["DY"]["2mu2e"], dataclasses.replace(BUFFERED, mass=(None, None))))
del groups6, data6
print("merge manifest: samples whose merged chunks fall short of the job list")
lines = open(os.path.join(BASE, "MANIFEST.txt")).read().splitlines()
short = []
for line in lines:
    m = re.match(r"(\S+) (\S+) chunks=(\d+)/(\d+)", line)
    if m and m.group(3) != m.group(4):
        short.append(f"{m.group(1)} {m.group(2)} {m.group(3)}/{m.group(4)}")
print(f"  {len(lines)} samples; incomplete: {', '.join(short) if short else 'none'}")"""))

cells.append(md(r"""## 2. 2μ2e: does the method's assumption hold where it can be tested?

### 2.1 The planes with A open

κ of every 2μ2e plane whose A is open, over the whole high-mass window and in its two parts, then
by DSA class of the muon LJ, then against the LJ-pair mass bin. The figure draws κ, A/B, B/D and
C/D of the two large planes against the mass. The first table repeats the measurement with the previous
boundaries so the effect of the buffers is visible.

Without the buffers **PP closes, 0.97 ± 0.10** (797/1545/202/379 events in A/B/C/D), and **PD does
not quite, 1.20 ± 0.10** (1343/1401/347/433), two standard deviations above one. DP, the plane with the displaced muon LJ and a prompt egamma LJ, gives 1.02 −0.41 +0.48 on
44/75/15/26 events, consistent with both. The previous boundaries gave PP 0.84 ± 0.12 and PD
1.25 ± 0.15: the $|\Delta\phi|$ buffer, which removed the 2.0–2.4 band from C and D, is what pulled
PP down (dropping it alone moves PP to 0.99 and leaves PD at 1.26), and the isolation buffer, which
removed the 0.25–0.5 band from B and D, is what pushed PD up (dropping it alone moves PD to 1.16
and PP to 0.78). PD's residual 20% is a property of the plane, not of a class: the PF-only muon
LJs give 1.24 ± 0.11 and the DSA-containing ones 0.73 −0.17 +0.19 (PP: 1.02 ± 0.11 and
0.59 −0.18 +0.20), so the two classes miss in opposite directions and a veto on the DSA class
would not repair it.

κ falls between the two windows, in both planes and by about two standard deviations each:
150–300 GeV gives PP 1.08 ± 0.13 and PD 1.31 ± 0.12, and M ≥ 300 GeV, where the 500–1000 GeV
signal points sit, PP 0.67 ± 0.15 and PD 0.87 ± 0.17, below one by 2.2 and 0.8 standard deviations.
Against a flat κ the five mass bins give χ² = 3.9 for 4 degrees of freedom in each plane, so the
data do not establish a mass dependence; the two windows differ enough that the per-window numbers
are carried. The integrated values are dominated by 150–300 GeV (1068 of PD's 1343 A events), so "closes integrated" is mostly "closes at
150–300 GeV". With the previous boundaries PP fell to 0.30 above 300 GeV while PD stayed at
1.05 ± 0.30 — the two planes did not share a trend; now they do, and it is mild."""))

cells.append(code(r"""print("the planes with A open, nominal boundaries, then the previous boundaries")
display(el.closure_table([(f"{label}, {p}", d2, cfg, p) for label, cfg in WINDOWS for p in ("PP", "PD", "DP")]))
display(el.closure_table([(f"{label} (previous boundaries), {p}", d2, dataclasses.replace(BUFFERED, mass=cfg.mass), p)
                          for label, cfg in WINDOWS for p in ("PP", "PD")]))
print("one buffer at a time, M >= 150 GeV")
display(el.closure_table([("isolation buffer only (dphi 2.4/2.4 x iso0 0.25/0.5), " + p, d2, dataclasses.replace(NOMINAL, fail0=0.5), p) for p in ("PP", "PD")]
                         + [("dphi buffer only (dphi 2.4/2 x iso0 0.25/0.25), " + p, d2, dataclasses.replace(NOMINAL, dphi_fail=2.0), p) for p in ("PP", "PD")]))
print("by DSA class of the muon LJ, M >= 150 GeV")
DSA_SPLITS = (None, "dsa0", "dsa1+", "dsa2+", "mu0", "mu1-2", "mu3+")
display(el.class_grid_table(d2, NOMINAL, ("PP", "PD", "DP"), DSA_SPLITS))
print("against the LJ-pair mass")
display(ps.mass_table(d2, NOMINAL, planes=("PP", "PD")))
fig = pp.mass_figure(d2, planes=("PP", "PD"), cfg=NOMINAL)
plt.show()"""))

cells.append(md(r"""### 2.2 The search plane below 150 GeV, and why the prompt planes fail there

Below 150 GeV every corner is open, including A of the DD plane. The DD plane is the one region
with the search plane's own objects and an open A; PP and PD in the same window are shown for
completeness, not as validation regions.

DD below 150 GeV gives **κ = 1.26 ± 0.44** on 41/32/53/52 events: consistent with one, and with
the 1.20 of PD above 150 GeV, at 35% precision. PP and PD fail there by a factor 1.8 (1.79 ± 0.16
and 1.86 ± 0.13): low-mass LJ pairs are mostly not back-to-back (C + D far exceeds A + B in both
planes, the reverse of the high-mass window), a different population in which the isolation and
the angle are correlated. Low mass is therefore not a validation window for the prompt planes;
its value is the DD line."""))

cells.append(code(r"""display(el.closure_table([(f"M < 150, {p}", d2, LOW, p) for p in PLANES]))
display(el.class_grid_table(d2, LOW, PLANES, (None, "dsa0", "dsa1+")))"""))

cells.append(md(r"""### 2.3 The classes of the muon LJ

A muon LJ is prompt when its best-tracked muon has at least three pixel hits; a DSA muon has none,
so a prompt LJ with a DSA partner is a well-tracked PF muon clustered with a muon-system-only
track. The two classes have different isolation pass/fail ratios in every plane, and the search
plane's sidebands hold a far larger DSA share than PD's A: the ratios borrowed from PD must
therefore be taken per class. The figure shows A/B and κ against the muon LJ's pixel-hit class in
the three populations where the classes can be compared.

In PD at M ≥ 150 GeV, A/B is 1.04 ± 0.04 for PF-only muon LJs (1253/1203) and 0.45 −0.06 +0.07
for DSA-containing ones (90/198); A/C is 4.2 and 2.0, A/D 3.5 and 1.2. Seven per cent of PD's A
and 13–17% of its sidebands carry a DSA muon. In the signal's search region 46–73% of the events
do (section 3.3), and in the search plane's sidebands 44% of B and 45% of D (section 3.1) — so an
inclusive ratio from PD, weighted by PD's 7–14%, would be the wrong mixture for DD."""))

cells.append(code(r"""fig = pp.pixel_class_figure(d2, NOMINAL)
plt.show()
print("the borrowed factors per class and window, and C/D in PD")
display(el.class_factors_table(d2, WINDOWS)[["window", "class", "PD A/B/C/D", "A/B PD", "A/C PD", "A/D PD", "C/D PD"]])"""))

cells.append(md(r"""## 3. The 2μ2e estimate

### 3.1 Four routes to A of the search plane

The direct estimate B C/D from DD's own sidebands, and the three borrowed ones — B × (A/B),
C × (A/C), D × (A/D), with the ratio from PD per DSA class and summed — per mass window, the
inclusive borrowed forms beside them for comparison. Then the same four routes from PP into DP,
the plane with a prompt egamma LJ, where A is observed: the open analog of the estimate. Then the
per-class factors with the sideband-only test of the borrowing, C/D of DD against C/D of PD per
class. Last, the search plane by class.

Over the whole high-mass window the search plane's sidebands hold 78, 17 and 29 events (B, C,
D), 44% of B and 45% of D with a DSA muon in the muon LJ. The four routes give **46 ± 15** (direct,
B C/D), **61 ± 8** (B × A/B), **60 ± 16** (C × A/C) and **72 ± 15** (D × A/D) events. Taken
inclusively the borrowed routes would read 75, 66 and 90, 10–25% higher, because the sidebands'
DSA share is three times PD's and the DSA class has the smaller ratios.

The four routes are not independent measurements. All of them use the search plane's own
sidebands, and each borrowed route equals the direct one times an implied κ of the search plane:
1.34 (B route), 1.31 (C) and 1.57 (D), against the 1.00 the direct route assumes. The spread between
them is mostly the Poisson scatter of C (17 events) and D (29), which enter the direct route at
± 33%; it is a consistency check, not a systematic uncertainty, and an implied κ$_{DD}$ of 1.3 ± 0.5
is consistent with the 0.97, 1.20 and 1.02 of the open planes. What the comparison does say is that
the direct and the borrowed estimates agree within their statistics.

The B route's own assumption — that the muon LJ's isolation pass/fail ratio among back-to-back
events, per DSA class, does not change when the muon LJ goes from prompt to displaced, the egamma
LJ displaced in both planes — has two open tests. The second table applies the same four routes
from PP into DP, the same step with a prompt egamma LJ: over the whole window the B route gives
33.2 ± 4.5 against 44 observed (observed over predicted 1.32 ± 0.27, and above one in each window:
1.27 ± 0.33 at 150–300 GeV, 1.5 ± 0.5 above 300 GeV, 1.2 ± 0.4 below 150 GeV), its inclusive form
38.7 ± 4.8, the C and D routes 45.0 ± 13.8 and 45.8 ± 10.2 and the direct route 43.3 ± 14.9. Below
150 GeV, where the search plane's A is open (41 events), the routes give 33 ± 9 (direct) and
41 ± 8, 45 ± 7 and 36 ± 6 (per class), the inclusive forms 43, 58 and 42: the B route reads
1.00 ± 0.25 there. The two tests average 1.15 ± 0.18, consistent with one; their non-closure and
precision added in quadrature give 23%, and section 6 assigns 25%. Were the search plane to follow
DP rather than its own low-mass window, the estimate would read 80, just outside that band. The
same tests cannot tell the per-class form from the inclusive one, which reads 1.14 ± 0.22 and
0.95 ± 0.23 in them; the per-class form is kept as the one that follows from the different class
mixtures, and the inclusive form (75) lies inside the band.

The sideband test in the third table compares C/D of the search plane with C/D of PD per class:
0.59 −0.17 +0.24 against 0.80 ± 0.06, a ratio of 0.73 −0.22 +0.31 (per class 0.89 −0.34 +0.51 and
0.62 −0.34 +0.57; 0.54 −0.19 +0.28 at 150–300 GeV, 1.6 standard deviations low). Equal C/D ratios
are what the B route needs if the search plane shares PD's κ, and the test resolves that at 30%: it
is consistent with one and with a 30% deficit. The direct route and the inclusive B route differ by
exactly this ratio divided by κ$_{PD}$, 0.73/1.20 = 0.61.

Per window the picture is thinner. At 150–300 GeV the routes spread from 22 ± 9 (direct) through
41 ± 7 and 39 ± 13 to 65 ± 14 (D route); above 300 GeV, where the search plane holds 26/6/3
events, the direct route is 52 ± 38, the B and C routes 19 ± 4 and 21 ± 10 and the D route 6 ± 5.
With three D events the routes that divide by D or multiply it say nothing there, and the B route
carries that window; its one test above 300 GeV is the PP → DP analog, 15 observed against
10.2 ± 2.4 predicted."""))

cells.append(code(r"""display(el.routes_table(d2, WINDOWS + [("M < 150 (A open)", LOW)]))
print("the open analog: the same four routes from PP into DP (egamma LJ prompt), where A of DP is observed")
display(el.routes_table(d2, WINDOWS + [("M < 150", LOW)], ref_plane="PP", search="DP"))
print("the factors per class; the last column tests the borrowing on the open sidebands")
display(el.class_factors_table(d2, WINDOWS + [("M < 150 (A open)", LOW)]))
print("the search plane by class of the muon LJ")
display(el.class_grid_table(d2, NOMINAL, ("DD",), DSA_SPLITS))"""))

cells.append(md(r"""### 3.2 What the simulation says, and what it cannot

The simulated background per process group and region of the search plane and of PD, with the
effective counts. The simulation is not used in the estimate; this is the record of why.

The simulation puts 32 ± 15 events in A of the search plane (effective count 4.8, all QCD) and
157 ± 119 in B — one QCD event of weight ninety against 78 observed — and under-predicts A of PD by
a factor 2.8 (478 ± 145 against 1343 observed, of which Drell–Yan is 134). With one to seven
effective events per region of the search plane it neither confirms nor contradicts an estimate
of sixty, and it is not used."""))

cells.append(code(r"""print("simulation per process group, M >= 150 GeV (luminosity-scaled; N_eff = (sum w)^2 / sum w^2)")
display(ps.mc_composition_table(g2, d2, "2mu2e", NOMINAL, planes=("DD", "PD")))
mc2 = io.add_cubes([c["2mu2e"] for c in g2.values()])
display(el.mc_region_table(mc2, WINDOWS, ("DD", "PD")))
del mc2, g2"""))

cells.append(md(r"""### 3.3 Signal in the regions read from

Each signal point's yield in every region the estimate and the validation read, as a fraction of
its own yield in the search region A of DD (M ≥ 150 GeV), with the yield at the stored cross
section in brackets; then the signal's DSA-class composition in the search region and its yield
per mass window.

The sidebands of the search plane are signal-poor: B holds 0.4–1.1% of A's yield, C 1.6–4.4% and D
nothing (with the previous boundaries B held 0.1–0.2% and C 0.5–1.6%, so the two former buffer
bands, iso0 0.25–0.5 at $|\Delta\phi| \ge 2.4$ and $|\Delta\phi|$ 2.0–2.4 at iso0 < 0.25, account
for 0.3–0.9% and 1.1–2.8%). At the stored cross sections C holds up to 1.4 signal events of its
17, and at an exclusion-level signal of about 30 events in A the former $|\Delta\phi|$ band alone
would hold 0.3–0.8, C 0.5–1.3 (up to 8% of its 17 events, which enter the direct and the C route),
B 0.1–0.3 of its 78 and D none. The validation planes take more: A of PD holds 18–41% of the search region's yield — signal events
whose muon LJ was caught with three or more pixel hits — which is at most 1.7% of PD's 1343 data
events at the stored cross sections; A of DP holds 7–17%, up to 9.7 events against 44 observed; A
of PP 1–7%, up to 4 events against 797. The B route reads A and B of PD per class: A of PD:dsa0 holds 14–32% of the search region's
signal (up to 19 events against 1253 observed), A of PD:dsa1+ 2.5–9% (up to 4.7 against 90), and
the B cells 0.3% or less; at an exclusion-level signal of 30 events that is 0.8% of A in PD:dsa0
and 3% in PD:dsa1+, which together move the B route by less than one event. Of the signal in the search region 46–73% has a DSA muon
in the muon LJ (27–54% none, 29–53% one, 8–30% two or more), which is why the DSA class cannot be
vetoed and why the borrowed factors are taken per class. For the 500–1000 GeV points more than 99.8% of the yield lies above
300 GeV; the 200 GeV point lies entirely below it."""))

cells.append(code(r"""display(el.signal_cells_table(BASE, SIGNAL_POINTS["2mu2e"], "2mu2e", [
    ("A DD, M >= 150 (search region)", NOMINAL, "DD", "A"),
    ("B DD, M >= 150", NOMINAL, "DD", "B"), ("C DD, M >= 150", NOMINAL, "DD", "C"), ("D DD, M >= 150", NOMINAL, "DD", "D"),
    ("B DD with the previous boundaries (iso0 >= 0.5)", BUFFERED, "DD", "B"),
    ("C DD with the previous boundaries (|dphi| < 2.0)", BUFFERED, "DD", "C"),
    ("A PD, M >= 150", NOMINAL, "PD", "A"), ("B PD, M >= 150", NOMINAL, "PD", "B"),
    ("A PD:dsa0, M >= 150", NOMINAL, "PD:dsa0", "A"), ("B PD:dsa0, M >= 150", NOMINAL, "PD:dsa0", "B"),
    ("A PD:dsa1+, M >= 150", NOMINAL, "PD:dsa1+", "A"), ("B PD:dsa1+, M >= 150", NOMINAL, "PD:dsa1+", "B"),
    ("A PP, M >= 150", NOMINAL, "PP", "A"), ("A DP, M >= 150", NOMINAL, "DP", "A"),
    ("A DD, M < 150", LOW, "DD", "A")]))
print("the DSA class of the muon LJ in the search region (share of A), and the yield per mass window")
display(el.signal_cells_table(BASE, SIGNAL_POINTS["2mu2e"], "2mu2e", [
    ("A DD, M >= 150 (search region)", NOMINAL, "DD", "A"),
    ("no DSA muon", NOMINAL, "DD:dsa0", "A"), ("one DSA muon", NOMINAL, "DD:dsa1", "A"), ("two or more", NOMINAL, "DD:dsa2+", "A")]))
display(el.signal_a_table(BASE, SIGNAL_POINTS["2mu2e"], "2mu2e", WINDOWS, ("DD",), prefix="abcd8d"))"""))

cells.append(md(r"""## 4. The boundaries: the configuration scan

The nominal boundaries are one point of a family — the isolation working point and its fail
boundary, the $|\Delta\phi|$ pass and fail boundaries, the second object's isolation cut, the choice
of axes — that `plane_search_lib.config_grid` enumerates: 405 configurations on the
$|\Delta\phi|$ × iso0, iso0 × iso1 and $|\Delta\phi|$ × iso1 planes. Each is scored by its worst
|κ − 1| over the regions that *select* it (the search channel's PP) and checked on the regions that
*confirm* it (PD and DP), so that the choice is not tuned to the fluctuations of the region that
justifies it. A region whose κ is less precise than the 15% target cannot test it and is left out
of the worst deviation (counted as untestable). A configuration is only usable if the search
plane's B, C and D lie outside the hidden box ("DD in-plane estimate exists"); since the box is
A, that holds for every configuration whose sidebands start at or beyond A's edges. The reduced
cubes hold the four ABCD axes of one plane, which makes the scan take seconds. The last table
evaluates the estimate itself for the usable configurations of the $|\Delta\phi|$ × iso0 family
without buffers, to show how much the estimate depends on where the lines are drawn.

Of the 405 configurations, 291 keep the search plane's sidebands outside the hidden box. Among
them the unbuffered $|\Delta\phi|$ × iso0 configurations (18) give κ$_{PD}$ 1.11–1.27 (median 1.16)
and the buffered ones (162) 1.12–1.44 (median 1.24); the iso0 × iso1 family gives 1.31–1.38
unbuffered and 1.30–1.89 buffered, and the 21 configurations that use the egamma LJ's isolation as
the axis fail the other way, 0.45–0.79. For PP the unbuffered $|\Delta\phi|$ × iso0 configurations
give 0.58–1.01 and the buffered ones 0.56–1.08, and the iso0 × iso1 family closes PP (0.96–1.31,
58 of 90 within 15%) while failing PD. The 38 usable configurations with both planes within 20% of
one are all $|\Delta\phi|$ × iso0, and the ones with both closest to one are the unbuffered
boundaries at $|\Delta\phi|$ = 2.4 with iso0 at 0.25 or 0.3 (PP 0.97–1.01, PD 1.14–1.20), the
nominal among them; its $|\Delta\phi|$ edge is the search region's. PP's κ depends on where that edge
is drawn: among the unbuffered configurations with iso1 < 0.5 it is 0.77–0.81 at 2.0, 0.82–0.88 at
2.2 and 0.97–1.01 at 2.4 (1.00 at 2.6, a boundary whose C and D would reach into the hidden box),
while PD moves only between 1.12 and 1.20: the isolation and the angle of the leading muon LJ are
not quite independent in PP below $|\Delta\phi| \approx 2.4$, and PP's closure at the nominal
boundary is a closure at that boundary. The scan's loosest working point for the second object is
iso1 < 1; it is a cut, not its absence — the isolation axis runs to 5, and 29% of the events in
PP's isolated back-to-back corner have iso1 ≥ 1 (printed below the scan) — and the configurations
that use it change the search region, so it is not adopted.

The last table, the estimate over the six unbuffered $|\Delta\phi|$ × iso0 configurations whose
sidebands stay open: the direct route moves between 38 and 46, the B route between 61 and 73, the
C route between 54 and 73 and the D route between 72 and 141: the D route moves most with the
$|\Delta\phi|$ boundary and the B route by 20%. The configurations with iso0 at 0.3 or $|\Delta\phi|$
below 2.4 define a larger A than the nominal one (by the open bands next to it, 5–6 events each,
visible as the change of B and C), so part of their movement is the region, not the method."""))

cells.append(code(r"""CFGS = ps.config_grid("dphi x iso0") + ps.config_grid("iso0 x iso1") + ps.config_grid("dphi x iso1")
R2 = {f"search {p}": el.reduce_plane(d2, p) for p in ("PP", "DP", "PD")}
scan2 = el.closure_scan(R2, CFGS, select=["search PP"], confirm=["search PD", "search DP"])
usable = scan2[scan2["DD in-plane estimate exists"]]
print(f"{len(scan2)} configurations, {len(usable)} keep the search plane's B, C and D outside the hidden box")
print("best 15 by the worst deviation of the selecting region, then the nominal configuration and the previous boundaries")
display(pd.concat([scan2.head(15), scan2[scan2["config"].isin([NOMINAL.label(), BUFFERED.label()])]]).round(3))
print("kappa of the search channel's PD among the usable configurations, by plane family and buffer")
def has_buffer(c):
    iso0, iso1, dphi = c.fail0 > c.wp0, c.fail1 > c.wp1, c.dphi_fail < c.dphi_pass
    return {"dphi x iso0": iso0 or dphi, "iso0 x iso1": iso0 or iso1, "dphi x iso1": iso1 or dphi}[c.kind]

by_label = {c.label(): c for c in CFGS}
fam = usable.copy()
fam["buffered"] = [has_buffer(by_label[label]) for label in fam["config"]]
for buffered, part in fam.groupby("buffered"):
    print("buffered" if buffered else "no buffer")
    display(el.kappa_by_family(part, "kappa search PD"))
print("the same for the search channel's PP")
for buffered, part in fam.groupby("buffered"):
    print("buffered" if buffered else "no buffer")
    display(el.kappa_by_family(part, "kappa search PP"))
k_pd = pd.DataFrame([el.parse_fmt(t) for t in scan2["kappa search PD"]], columns=["kappa", "minus", "plus"], index=scan2.index)
k_pp = pd.DataFrame([el.parse_fmt(t) for t in scan2["kappa search PP"]], columns=["kappa", "minus", "plus"], index=scan2.index)
close_pd = scan2[((k_pd["kappa"] - 1).abs() <= 0.15) & (k_pd[["minus", "plus"]].max(axis=1) / k_pd["kappa"] <= 0.15)]
print(f"the {len(close_pd)} configurations in which the search channel's PD is measured to 15% and within 15% of one")
display(close_pd[["config", "kappa search PD", "kappa search PP", "DD in-plane estimate exists"]])
both = scan2[((k_pd["kappa"] - 1).abs() <= 0.2) & ((k_pp["kappa"] - 1).abs() <= 0.2) & scan2["DD in-plane estimate exists"]]
print(f"the {len(both)} usable configurations with both PP and PD within 20% of one")
display(both[["config", "kappa search PP", "kappa search PD"]])
a_all = ps.regions(d2, dataclasses.replace(NOMINAL, wp1=None), "PP")["A"][0]
a_1 = ps.regions(d2, dataclasses.replace(NOMINAL, wp1=1.0), "PP")["A"][0]
print(f"PP, A without any iso1 cut: {a_all:.0f} events; with iso1 < 1: {a_1:.0f}; share at iso1 >= 1: {1 - a_1 / a_all:.0%}")
print("the estimate for the usable, unbuffered dphi x iso0 configurations with iso1 < 0.5, M >= 150 GeV")
stable = [c for c in ps.config_grid("dphi x iso0") if c.dd_open() and c.fail0 == c.wp0 and c.dphi_fail == c.dphi_pass and c.wp1 == 0.5]
rows = []
for c in stable:
    r = el.routes_table(d2, [("M >= 150", c)]).iloc[0]
    rows.append({"config": c.label(), "kappa PP": tl.fmt(ps.kappa(ps.regions(d2, c, "PP")), 2), "kappa PD": tl.fmt(ps.kappa(ps.regions(d2, c, "PD")), 2),
                 "DD B/C/D": r["DD B/C/D"], "direct B C/D": r["direct B C/D"], "B x A/B per class": r["B x A/B per class"],
                 "C x A/C per class": r["C x A/C per class"], "D x A/D per class": r["D x A/D per class"]})
display(pd.DataFrame(rows))"""))

cells.append(md(r"""## 5. 4μ

### 5.1 The planes

In 4μ both objects are muon LJs. The DSA-muon counts of both split every plane; A of DD, DP and
PD is hidden above 150 GeV, and PP is the one plane with an open A there.

PP is consistent with one above 150 GeV, **κ = 1.10 ± 0.23** (264/348/58/84), at a precision
(21%) that cannot test the 15% target, and 16% above one below 150 GeV, 1.16 −0.13 +0.14
(332/725/163/412); with the previous boundaries it gave 1.62 ± 0.55 above 150 GeV. Below
150 GeV, where every corner is open, DP gives 0.79 ± 0.33 (30/39/30/31) and PD 0.48 −0.21 +0.24
(15/41/26/34); the DD plane holds 2/1/3/3 events. The 4μ planes are mixtures of populations with
different ratios — the DSA class of the subleading LJ moves κ of PP by a factor three, 0.80 −0.20
+0.22 without a DSA muon in it and 2.37 ± 0.87 with one — and pairs of DSA-only muon LJs, the
population `plane_search.ipynb` pointed to, do not occur under this selection (which carries the dxy-spread and cos α cosmic-muon veto): no open ABCD region of any
plane in either mass window holds an event with two or more DSA muons in each muon LJ. Above
150 GeV the search plane's sidebands hold 3, 0 and 0 events, DP's 34, 9 and 2 and PD's 25, 9 and
14. The simulation has one to fifteen effective events per region above 150 GeV and is of no use
here."""))

cells.append(code(r"""g4 = tl.group_cubes(BASE, "bkg", prefix="abcd8d", topologies=("4mu",))
SPLITS4 = (None, "dsa0", "dsa1+", "subdsa0", "subdsa1+", "dsa0:subdsa0", "dsa1+:subdsa1+", "dsa2+:subdsa2+",
           "mu0", "mu1-2", "sub0", "sub1-2")
display(el.closure_table([("M >= 150, PP", d4, NOMINAL, "PP"), ("M >= 150 (previous boundaries), PP", d4, BUFFERED, "PP")]
                         + [(f"M < 150, {p}", d4, LOW, p) for p in PLANES]))
print("M >= 150 GeV (A of DP, PD and DD hidden)")
display(el.class_grid_table(d4, NOMINAL, PLANES, SPLITS4))
print("M < 150 GeV (all open)")
display(el.class_grid_table(d4, LOW, PLANES, SPLITS4))
print("the planes against the LJ-pair mass")
display(ps.mass_table(d4, NOMINAL, planes=("PP", "PD", "DP", "DD")))
print("simulation per process group, M >= 150 GeV")
display(ps.mc_composition_table(g4, d4, "4mu", NOMINAL))
mc4 = io.add_cubes([c["4mu"] for c in g4.values()])      # the summed simulation, for section 5.2
del g4"""))

cells.append(md(r"""### 5.2 A count-limited estimate for the 4μ search region

The search plane's sidebands above 150 GeV hold three events, all in B, so no κ of this plane can
be measured and no in-plane B C/D exists. What the open regions give instead is an expectation for the search region of the kind a
fit on one or two events needs — its order of magnitude and a factor-two-to-three uncertainty —
by two routes that use different open regions:

1. **Along the isolation axis.** B of the search plane times A/B of a plane whose A is open: PP
   above 150 GeV (the one open plane there), and PP, DP and PD below 150 GeV.
2. **Along the mass axis.** A of the DD plane below 150 GeV times the ratio of the counts above
   to below 150 GeV in the open regions next to it: A of PP, B of DP and B of PD.

Route 1 gives 1.1 −0.7 +0.8 (PD's A/B), 1.4 ± 0.8 (PP's) and 2.3 ± 1.5 (DP's) events; with PP's
own A/B above 150 GeV, 0.76, it gives 2.3. Route 2 gives 1.2 ± 1.0, 1.6 ± 1.1 and 1.7 ± 1.3. The six
central values lie between 1.1 and 2.3, but each reading rests on one small count — B of the search
plane, 3 events, or its A below 150 GeV, 2 — and carries a 60–100% statistical error, and route 1
borrows A/B from below 150 GeV, where PP's A/B is 0.46 against 0.76 above; scaled by that factor
the DP and PD readings become 3.8 and 1.8. The 4μ search region therefore holds about 1.7
background events, uncertain by a factor two to three (the previous version read 0.5–2.3 from a
sideband of one event; the opened sideband holds three). The last table shows what the uncertainty
costs: over an expected background of 0.5 to 4 events the 95% CL upper limit on a signal count for
one observed event moves from 4.4 to 3.5 (a single-bin counting experiment with a known background
and no systematic uncertainty), for two observed events from 5.8 to 4.2, for none not at all. The
fit will carry the factor as the uncertainty of the 4μ background.

The regions the estimate reads from are signal-poor, except A of PP above 150 GeV: that corner
holds 5–25% of the search-region yield (most for the shortest lifetimes, whose muon LJs are prompt
by pixel count) over 264 observed events — at most 12 events, 4.5% of the data, at the stored cross
sections — which is why PP's κ is a validation number and its A/B one factor among several. The
corners of DP and PD hold 23–52% of the search-region yield against a background of tens of events
and stay hidden; their sidebands hold at most 1.9%."""))

cells.append(code(r"""HIGH4 = ("M >= 150", NOMINAL)
LOW4 = ("M < 150", LOW)
print("the DD plane by the DSA-muon counts of the two muon LJs (events per region; blind where hidden)")
display(el.count_grid(d4, [LOW4, HIGH4], "DD", [f"dsa{a}:subdsa{b}" for a in ("0", "1", "2+") for b in ("0", "1", "2+")]))
print("the in-plane estimate where its sidebands are populated")
display(el.final_table(d4, WINDOWS, {"1": (1.0, 0.0, 0.0)}))
print("route 1: B of the search plane above 150 GeV times A/B of the planes whose A is open")
display(el.sideband_transfer_table(d4, HIGH4, LOW4))
print("route 2: A of the DD plane below 150 GeV times the above/below-150 GeV ratio of the open neighbouring regions")
display(el.mass_transfer_table(d4, HIGH4, LOW4))
print("simulation, luminosity-scaled, per region")
display(el.mc_region_table(mc4, [HIGH4, LOW4], ("DD", "DP", "PD", "PP")))
del mc4
print("signal per region as a fraction of its own search-region yield [yield at the stored cross section]")
display(el.signal_cells_table(BASE, SIGNAL_POINTS["4mu"], "4mu", [
    ("A DD, M >= 150 (search region)", NOMINAL, "DD", "A"),
    ("B DD, M >= 150", NOMINAL, "DD", "B"), ("C DD, M >= 150", NOMINAL, "DD", "C"), ("D DD, M >= 150", NOMINAL, "DD", "D"),
    ("A DP, M >= 150 (hidden)", NOMINAL, "DP", "A"), ("A PD, M >= 150 (hidden)", NOMINAL, "PD", "A"),
    ("B DP, M >= 150", NOMINAL, "DP", "B"), ("C DP, M >= 150", NOMINAL, "DP", "C"),
    ("B PD, M >= 150", NOMINAL, "PD", "B"), ("C PD, M >= 150", NOMINAL, "PD", "C"),
    ("A DD, M < 150", LOW, "DD", "A"), ("A DP, M < 150", LOW, "DP", "A"), ("A PD, M < 150", LOW, "PD", "A"),
    ("B DP, M < 150", LOW, "DP", "B"), ("B PD, M < 150", LOW, "PD", "B"),
    ("A PP, M >= 150", NOMINAL, "PP", "A"), ("B PP, M >= 150", NOMINAL, "PP", "B")]))
display(el.signal_a_table(BASE, SIGNAL_POINTS["4mu"], "4mu", WINDOWS, ("DD",), prefix="abcd8d"))
print("95% CL upper limit on a signal count for n observed events and an expected background b "
      "(single-bin counting experiment, background known, no systematic uncertainty)")
display(el.counting_limit_table((0, 1, 2), (0.0, 0.5, 1.1, 2.3, 4.0)))"""))

cells.append(md(r"""## 6. The estimates and their uncertainties

The closure of every validation region against the 15% target, for the nominal boundaries: κ,
its deviation from one with the 68% interval, the statistical precision of the test, and a
verdict assigned in this order — *imprecise* (the precision of κ is worse than the target, so the
region cannot test it), *closes* (|κ − 1| within the target), *closes within errors* (the central
value is further from one than the target but the 68% interval reaches one), *fails*. Then the
2μ2e estimate: the direct B C/D per window, the same multiplied by κ$_{PD}$ of the window (the
correction a reader may prefer to apply) and by one, and the borrowed routes from section 3.1.

Of the twelve 2μ2e validation regions, two close (PP over the whole window and at 150–300 GeV),
PD fails at 20% and 31% (measured to 8–9%), the low-mass prompt planes fail at 80%, and the
remaining six are too imprecise to test the target. The direct estimate scaled by κ$_{PD}$ of the
same window reads 55 ± 18 over the whole window, 29 ± 11 at 150–300 GeV and 45 ± 34 above
300 GeV — the estimate a reader gets by assuming the search plane shares PD's correlation, for
which the low-mass window gives κ$_{DD}$ = 1.26 ± 0.44.

**The 2μ2e estimate.** Over the one-bin search region M ≥ 150 GeV: **61 ± 8 (stat.) ± 15 (syst.)
events**, the B × (A/B)$_{PD}$ route per DSA class. The systematic uncertainty, 25%, is set by the
two open tests of the route's own assumption in section 3.1 — the PP → DP analog (observed over
predicted 1.32 ± 0.27) and the low-mass window of the search plane (1.00 ± 0.25), which average
1.15 ± 0.18: their non-closure and precision in quadrature, 23%, rounded up; the band also holds
the inclusive form of the route (75). It is not the spread of the four routes, which share the
search plane's sidebands. Per window, with the same
25%: 41 ± 7 ± 10 at 150–300 GeV and 19 ± 4 ± 5 above 300 GeV; above 300 GeV the PP → DP analog
reads 1.5 ± 0.5 and the validation planes 0.67 ± 0.15 and 0.87 ± 0.17, so the per-window numbers
are weaker than the integrated one. The 15% target is not met: PD misses it at 20 ± 10%, the
transfer is tested to 18%, and the estimate carries 25%."""))

cells.append(code(r"""def kappa_of(cube, cfg, plane):
    return ps.kappa(ps.regions(cube, cfg, plane))

cases = ([(f"2mu2e {label}, {p}", d2, cfg, p) for label, cfg in WINDOWS for p in ("PP", "PD", "DP")]
         + [("2mu2e M < 150, DD", d2, LOW, "DD"), ("2mu2e M < 150, PD", d2, LOW, "PD"), ("2mu2e M < 150, PP", d2, LOW, "PP")]
         + [("4mu M >= 150, PP", d4, NOMINAL, "PP"), ("4mu M < 150, PP", d4, LOW, "PP"), ("4mu M < 150, DP", d4, LOW, "DP"),
            ("4mu M < 150, PD", d4, LOW, "PD"), ("4mu M < 150, DD", d4, LOW, "DD")])
display(el.closure_table(cases))
print("2mu2e: the direct estimate per window, as it is and scaled by kappa of PD in the same window")
for label, cfg in WINDOWS:
    k = {"PD, same window": kappa_of(d2, cfg, "PD"), "1": (1.0, 0.0, 0.0)}
    print(label, "kappa PD:", tl.fmt(k["PD, same window"], 2))
    display(el.final_table(d2, [(label, cfg)], k))
print("2mu2e: all four routes")
display(el.routes_table(d2, WINDOWS))"""))

cells.append(md(r"""## 7. Conclusions

1. **The $|\Delta\phi|$ buffer was most of PP's non-closure; PD's 20% is the plane's.** With the
   sidebands starting at the search region's own edges, PP closes (0.97 ± 0.10; 0.84 before) and PD
   moves from 1.25 to 1.20 ± 0.10. PD stays 11–27% above one for every unbuffered boundary of the
   scan, and PP closes at the search region's $|\Delta\phi|$ edge of 2.4 but reads 0.8 at 2.0–2.2.
   The muon LJ's mass cut and the proxy region are gone without loss. The hidden box is the search
   region itself, and in 4μ the same corner of the two one-LJ-displaced planes.

2. **2μ2e: 61 ± 8 (stat.) ± 15 (syst.) background events in the search region M ≥ 150 GeV**,
   from B × (A/B)$_{PD}$ per DSA class of the muon LJ, the 25% from the two open tests of that
   transfer (1.32 ± 0.27 in PP → DP, 1.00 ± 0.25 in the search plane below 150 GeV; non-closure
   and precision of their mean in quadrature). The four routes read 46–72; they share the search
   plane's sidebands, so their agreement is a consistency check (an implied κ$_{DD}$ of 1.3 ± 0.5),
   not a second measurement. The per-class form follows from the sidebands' DSA share, three times
   PD's; the open tests cannot tell it from the inclusive form, which reads 10–25% higher and sits
   inside the systematic. Per window, 41 ± 7 ± 10 at 150–300 GeV and 19 ± 4 ± 5 above 300 GeV.

3. **What is validated, and to what level.** PP closes within 15% and PD within 20% over the whole
   window; both fall by about two standard deviations between the two windows, to 0.67 ± 0.15 and
   0.87 ± 0.17 above 300 GeV where the heavy points sit, though the five mass bins are consistent
   with a flat κ (χ² 3.9/4 in each plane). DP (1.02 −0.41 +0.48) and the search plane below 150 GeV
   (1.26 ± 0.44) are consistent with one at their precision. The 15% target is not met; the
   transfer the estimate rests on is tested to 18%, and the estimate carries 25%.

4. **4μ: about 1.7 background events in the search region, uncertain by a factor two to three.**
   The search plane's sidebands hold three events, so no in-plane estimate exists; six count-limited
   readings from the open neighbouring regions give central values of 1.1–2.3, each with a 60–100%
   statistical error, and the low-mass A/B ratios that route 1 borrows may be low by the factor
   PP's A/B rises across 150 GeV (0.46 → 0.76), which would lift the DP and PD readings to 3.8 and
   1.8. PP (1.10 ± 0.23 above 150 GeV, 1.16 −0.13 +0.14 below) is the one 4μ validation region.
   Over an expected background of 0.5–4 events the 95% CL upper limit for one observed event moves
   from 4.4 to 3.5.

5. **What goes to the fit.** For 2μ2e a one-bin background of 61 ± 17 (statistical and systematic
   in quadrature, 28%), or per window 41 ± 12 and 19 ± 6. For 4μ a background of 1.7 with a
   log-normal uncertainty of a factor two to three.

## Files

`estimate_lib.py` (the tables of this notebook), `test_estimate_lib.py`, `plane_search_plots.py`
(its two figures), `_build_estimate.py` (writes this notebook); `merge_abcd8d.py` merges the campaign outputs into the
local cubes, with `test_merge_abcd8d.py`; `campaign/` holds the histogram producers and the patches
applied to the campaign's copy of the framework (`abcd8d.py`, `abcd8d_proxy.py`, `patch_proxy.py`,
`patch_prefilter.py`; the proxy-region histograms are produced but not used here)."""))

nb = nbf.v4.new_notebook(cells=cells, metadata=KERNEL)
nbf.write(nb, "estimate.ipynb")
