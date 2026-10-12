"""Build binning.ipynb.

Run from this directory on the LPC after merge_abcd8d.py has filled the local directory
(ABCD8D_LOCAL, default /uscms_data/d3/murtazas/abcd8d_local):
    python _build_binning.py
    jupyter nbconvert --to notebook --execute --inplace binning.ipynb
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

cells.append(md(r"""# The 2μ2e background estimate per bin of the LJ-pair mass

This notebook evaluates the 2μ2e background estimate of `estimate.ipynb` per bin of the LJ-pair
mass M instead of once over M ≥ 150 GeV, compares the sum over the bins with the one-bin estimate,
and measures with a toy expected limit what a binned search region would gain. The vocabulary is
that of `estimate.ipynb`: object 0 is the leading muon LJ, object 1 the egamma LJ (2μ2e) or the
subleading muon LJ (4μ); the planes **PP, PD, DP, DD** say whether the muon LJ (first letter) and
the other LJ (second letter) are prompt or displaced; inside each plane **A** (iso0 < 0.25 and
$|\Delta\phi| \ge 2.4$, with iso1 < 0.5) is the search region, **B** fails the isolation, **C**
fails $|\Delta\phi|$ and **D** fails both; **κ** = A D / (B C) of a plane is one when its two axes
are independent; `dsa0` and `dsa1+` split the muon LJs by their number of displaced-standalone
(DSA) muons. The histograms (the `abcd8d` campaign, "8D inputs" below) store the LJ-pair mass in
the bins 150–200, 200–300, 300–400, 400–600, 600–800, 800–1000 and ≥ 1000 GeV above the 150 GeV
edge; "cube bins" names the binning made of them. Signal yields are "at the stored cross section":
the cross section the samples were normalised to when the histograms were filled
(`cross_sections.yaml` of the repository), the signal strength μ = 1 of the toy limits.

**The one-bin estimate.** A of the search plane DD over M ≥ 150 GeV is estimated by the B route:
B of DD times A/B of PD, per DSA class of the muon LJ, summed over the two classes. It gives
**61.3 ± 8.0 (stat.) events with a 33% systematic uncertainty**, set by the two open tests of the
route's assumption (PP → DP, observed over predicted 1.32 ± 0.27; DD below 150 GeV,
1.00 ± 0.25).

**Why bin in mass.** The one-bin estimate averages a transfer factor that falls by about 40%
between 150 and 600 GeV over a window where about 70% of the background lies (41 of the 61 events
at 150–300 GeV) but none of the heavy signal does (the 500–1000 GeV points have more than 99.8% of
their yield above 300 GeV). Binning in mass borrows each bin's own factor, at the masses of the
search plane's B events, and moves most of the background out of the heavy points' bins. The price
is statistics — fewer B events of DD and fewer PD events per bin — and a weaker validation per bin:
the 33% comes from tests over the whole window, and above 300 GeV only the PP → DP test exists
(1.5 ± 0.5).

**The factor in a bin.** Per bin and DSA class, A/B of PD is measured in the bin. Where PD
measures it to 20% or better (the larger side of its 68% interval over its value) the measurement
is used; elsewhere the factor comes from a fit of A/B against M over the seven mass bins of the
histograms above 150 GeV, evaluated at the mean mass of the search plane's B events in the bin,
and |measured − fitted| enters as a shape systematic. The fit is a binomial likelihood: in every
bin the A count is binomial among the A + B events with pass probability r/(1 + r), r the form's
A/B at the bin's mass, so that a bin without an A event (an upper limit on A/B) or without a B
event (a lower limit) enters like any other. Two forms: the exponential A/B = a exp(−M/M₀), which
stays positive at every mass, and the straight line A/B = a + b M. The design note written before
this study listed the line first and the exponential as the alternative; this notebook was written
with the exponential as the nominal form, for its positivity, before any fit was run, and reports
the two side by side throughout. The two forms give the same estimates to 1%, but they do not lead
to the same choice of binning: the ≥ 600 GeV bin's estimate falls on either side of the rule's
3-event threshold (section 5, section 7). An earlier version of the notebook fitted the measured
ratios by least squares, which left the bins without an A or a B event out; the likelihood fit
replaced it after the first run, and section 3 says what that changed.

**The decision rule**, written down in the design note before any result, quoted as written
there:

> choose the binning that maximises the expected limit for the 500 and 800 GeV points, subject to: every search bin's B route has >= 5 B events in DD (stat <= 45%) or a background below 3 events (Poisson floor), per-bin A/B used where PD's A/B error <= 20% else the fitted value with |per-bin - fit| as a shape systematic; 150-300 stays a SEARCH bin (m_B=200 is a signal hypothesis until the group drops it); never look at A of DD; no selection on kappa or on closure outcome.

How it is applied (`binning_lib.rule_table`):

- *Maximises the expected limit* means the strongest expected limit: the lowest expected upper
  limit on the signal, taken relative to the one-bin estimate.
- The candidates are the four binnings of `binning_lib.BINNINGS`, fixed in advance: one bin
  (M ≥ 150 GeV); two windows (150–300, ≥ 300 GeV); three bins (150–300, 300–600, ≥ 600 GeV); and
  the histograms' own bins up to 600 GeV (150–200, 200–300, 300–400, 400–600, ≥ 600 GeV).
  A fifth binning, a split at 400 GeV (150–400, ≥ 400 GeV; `binning_lib.LATER_BINNINGS`), was
  added after the result of the four was known, because the ≥ 600 GeV bin of the rule's choice
  rests on four sideband events (section 7): it is evaluated beside the four in every table and
  figure, and it is not a candidate of the rule.
- The decision points are the 500 GeV point and the three 800 GeV points of the signal list. Each
  mass counts once: the relative limits are averaged over the points of each mass, then over the
  two masses.
- The limits are those of the selected factor (the rule's second clause), with the 33% systematic
  of `estimate.ipynb` as one nuisance common to all bins; section 5 shows two other readings of
  the systematic beside it, independent per bin and sized per bin by the open test.
- *A background below 3 events* is read on the selected estimate's central value, as the rule
  says; the parentheses of the rule explain its two thresholds (five B events give a 45% Poisson
  error; three events is where a limit stops depending on the background) and add no condition.
- *150–300 GeV stays a search bin* is read as: the range stays in the search region, as one bin or
  split, and is never made a control region. Every candidate satisfies it.
- *A of DD is never looked at*: the data cubes hide it in every mass bin above 150 GeV, no function
  of this notebook reads it, and the tables print "blind" where it would stand.
- *Nothing is selected on kappa or on closure*: the binnings are fixed in advance, the choice of
  factor depends only on PD's statistical precision, and the choice among binnings only on the toy
  limits and the count requirement. κ of the open planes is printed as context only.

**Blinding.** As in `estimate.ipynb`: data cubes are loaded only through
`transfer_lib.summed_cubes`, which hides A of DD above 150 GeV in both channels and, in 4μ, A of DP
and PD as well. Every mass sub-range of a hidden region projects to NaN and prints as "blind".
Signal cubes are never hidden.

**Uncertainties.** As in `estimate.ipynb`: 68.27% Clopper–Pearson intervals for ratios of disjoint
counts, relative errors added in quadrature for products and ratios of ratios, absolute errors
added in quadrature for sums over classes and bins, the 68% upper limit of 1.84 events times the
factor for a count of zero; a "±" is printed where the two sides of an interval agree within 5%,
otherwise "−m +p"."""))

cells.append(code(r"""import dataclasses
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import abcd6d_io as io
import transfer_lib as tl
import transfer_plots as tp
import plane_search_lib as ps
import plane_search_plots as pp
import estimate_lib as el
import binning_lib as bl
import binning_plots as bp

pd.set_option("display.max_columns", 40)
pd.set_option("display.max_rows", 200)
pd.set_option("display.width", 250)
pd.set_option("display.max_colwidth", 200)
tp.set_style()

BASE = os.environ.get("ABCD8D_LOCAL", "/uscms_data/d3/murtazas/abcd8d_local")
print("8D inputs:", BASE)
data = tl.summed_cubes(BASE, "data", prefix="abcd8d")
print("hidden box:", io.blind_signature(prefix="abcd8d"))
for t, c in data.items():
    print(f"{t}: {c.sumw.shape}, {int(np.isnan(c.sumw).sum())} hidden cells, {np.nansum(c.sumw):.0f} open events")
d2, d4 = data["2mu2e"], data["4mu"]
NOMINAL = ps.Config(fail0=0.25, dphi_fail=2.4)             # no buffer between A and its sidebands
WINDOWS = [("M >= 150", NOMINAL),
           ("150-300", dataclasses.replace(NOMINAL, mass=(150.0, 300.0))),
           ("M >= 300", dataclasses.replace(NOMINAL, mass=(300.0, None)))]
SIGNAL_POINTS = bl.SIGNAL_POINTS                            # the points of estimate.ipynb
DECISION_POINTS = bl.DECISION_POINTS                        # the 500 and 800 GeV points carry the choice
FORM = "exponential"                                        # the nominal form of the A/B(M) fit; linear is the cross-check
SYST = 0.33                                                 # the systematic uncertainty of the estimate (estimate.ipynb)
print("nominal configuration:", NOMINAL.label())
for name, bins in bl.BINNINGS.items():
    print(f"{name}: {', '.join(label for label, _ in bins)}")
print("decision points:", ", ".join(DECISION_POINTS["2mu2e"]))"""))

cells.append(md(r"""## 1. The planes and the per-bin route

The first figure draws the four 2μ2e planes as a 2 × 2 grid of squares, with the data counts over
M ≥ 150 GeV in each cell.

*How to read it.* The squares are arranged by displacement: the muon LJ is prompt in the top row
and displaced in the bottom row, the egamma LJ prompt in the left column and displaced in the right
one. Inside each square the ABCD grid has the muon LJ's isolation running from left (pass,
iso0 < 0.25) to right (fail) and $|\Delta\phi|$ from bottom (fail, below 2.4) to top (pass): A is
the top-left cell, B top right, C bottom left, D bottom right. The hatched cell is A of DD, the
search region, hidden. The solid arrow is the estimate: A/B of PD (the top row of PD's square),
taken per DSA class, applied to B of DD. The dashed arrow is the same step — the muon LJ going from
prompt to displaced — with the egamma LJ prompt, from PP to DP, where A is open: the open test of
the transfer.

The second figure is the estimate of one mass bin as a diagram.

*How to read it.* The axis at the bottom carries the bins of one binning, here three bins; the
highlighted bin is the one whose estimate the boxes spell out: B of DD in the bin for each DSA
class, times A/B of PD of the same class, summed over the two classes. The note above the boxes
says where the factor comes from. The diagram is schematic and carries no numbers; section 4 has
them."""))

cells.append(code(r"""fig = bp.planes_cartoon(d2, NOMINAL, "2mu2e")
plt.show()
fig = bp.route_cartoon()
plt.show()"""))

cells.append(md(r"""## 2. Signal, sidebands and the factor against the mass

The first figure shows where each part of the problem sits in mass. Top: the mass distribution of
each signal point in the search region (A of DD, from the signal samples, which are not hidden),
normalised to unit area above 150 GeV, with the inner edges of the candidate binnings as dotted
lines. Bottom: B, C and D of DD in data per mass bin of the histograms. A binning separates the
heavy points from the background only as far as the B events of DD allow, since each bin's estimate
is B of DD in the bin times a factor.

The tables give A, B, C and D, A/B, C/D and κ per mass bin above 150 GeV for DD, PD and PP,
inclusive and per DSA class; A of DD prints "blind" in every bin, and so do its A/B and κ. The last
figure is the one of `estimate.ipynb` section 2.1 — κ, A/B, B/D and C/D of PP and PD against the
mass — here for the trend of A/B."""))

cells.append(code(r"""PROFILES2 = bl.signal_profiles(BASE, SIGNAL_POINTS["2mu2e"], "2mu2e", NOMINAL)     # A of DD, every mass
EDGES = PROFILES2[0]
DD_SIDEBANDS = {r: el.spectrum(d2, "ljlj_mass", NOMINAL, "DD", r)[1] for r in "BCD"}    # the open regions of DD
fig = bp.mass_shapes_figure(PROFILES2[1], DD_SIDEBANDS, EDGES)
plt.show()
for plane in ("DD", "PD", "PP"):
    print(f"2mu2e, {plane}: the regions per mass bin and DSA class of the muon LJ")
    display(bl.counts_table(d2, NOMINAL, bl.FIT_BINS, planes=(plane,)))
fig = pp.mass_figure(d2, planes=("PP", "PD"), cfg=NOMINAL)
plt.show()"""))

cells.append(md(r"""## 3. A/B against the mass

The points of the fit are A/B per mass bin of the histograms above 150 GeV, per plane and DSA
class, at the mean mass of the plane's B events in the bin. The histograms store no mass inside a
bin, so each bin's events count at its centre and the overflow above 2500 GeV at 2500 GeV; the mass
of a point is known to about the width of its bin, and the last bin (≥ 1000 GeV) to a factor two.
Both forms are binomial-likelihood fits over all seven bins (introduction): the quality of a fit is
its deviance, twice the log-likelihood ratio of the data to the fit, which plays the part of χ² and
is distributed like it for large counts; a bin's residual is the signed square root of its share of
the deviance, positive where the measured A/B lies above the fit, and is defined also for a bin with
no A or no B event. The exponential is the nominal form, as fixed in the introduction.

The table gives per plane, class and form the parameters (the slope of the line per TeV, the scale
M₀ of the exponential) and the deviance over ndof, and per bin the measured A/B (a lower limit,
A/1.84, where B = 0), the fitted A/B and the residual. The figure draws the points and both curves
for PD and PP per class, with the 68% band of the nominal fit; an upward arrow is a bin without a
B event, drawn at its lower limit. Below it, the ratio of PP's A/B to PD's per bin and class with
its weighted mean: a flat ratio says that the trend does not change with the egamma LJ's state.
Over the whole window the ratio is 0.54 (797/1545 in PP against 1343/1401 in PD, all classes).
Whether the trend also holds for a displaced muon LJ is the B route's own assumption, which the
open tests of `estimate.ipynb` check over the whole window and section 4 checks per bin. The last
line gives the fall of PD's A/B between the first bin and the 400–600 GeV bin.

What the likelihood fit changed against the least-squares fit of the first version: the
least-squares fit used the five or six bins with both counts and read the exponential for PD's
dsa0 class as M₀ = 910 ± 344 GeV (χ²/ndof 5.5/4), 0.64 ± 0.12 at 700 GeV; the ≥ 1000 GeV bin, 4 A
and 0 B events, was not in it. The numbers below replace those."""))

cells.append(code(r"""POINTS = {(p, c): bl.ab_points(d2, NOMINAL, bl.FIT_BINS, p, c) for p in ("PD", "PP") for c in ("dsa0", "dsa1+", None)}
FITS = {(p, c, f): bl.fit_ab(POINTS[(p, c)], f) for (p, c) in POINTS for f in bl.FORMS}
print("A/B against the mass: the fit per plane, class and form, and the points against it")
display(bl.ab_fit_table(d2, NOMINAL, bl.FIT_BINS))
fig = bp.ab_mass_figure(POINTS, FITS, band_form=FORM)
plt.show()
print("A/B of PP over A/B of PD per mass bin")
display(bl.plane_ratio_table(POINTS))
first, mid = (tuple(POINTS[("PD", None)].set_index("bin").loc[b, ["A/B", "minus", "plus"]]) for b in ("150-200", "400-600"))
print("PD, all classes: A/B at 400-600 GeV over A/B at 150-200 GeV:", tl.fmt(tl.ratio_of_ratios(mid, first), 2))"""))

cells.append(md(r"""## 4. The estimate per bin

For each binning, the B route per bin and DSA class: B of DD and the mean mass of those events;
A/B of PD measured in the bin with its precision (the larger side of the 68% interval over the
value); the fitted A/B at the mean mass (nominal form); the estimate with the measured and with the
fitted factor; the factor the rule selects and the estimate with it; and the shape systematic,
|measured − fitted| over the selected estimate, with the part of it the selected estimate carries
(zero where the measured factor is taken). The "sum" rows add the two classes; the last row adds
the bins, in quadrature, which is exact for the measured factors and understates the error of
fitted ones (the bins share the fit's parameters).

The one-bin binning with the measured factor is the estimate of `estimate.ipynb`, 61.3 ± 8.0; the
two windows with the measured factors reproduce its per-window B route, 41 ± 7 at 150–300 GeV and
19 ± 4 above 300 GeV, which `estimate_lib.routes_table` repeats below the binned tables as the
check.

Then the sums over the bins against the one-bin estimate. With a factor that is the same at every
mass they would agree exactly. If A/B falls with the mass and the search plane's B events sit at
other masses than PD's, the binned sum moves away from the one-bin value: the difference is the
error the one-bin form makes by weighting PD's factor with PD's own mass distribution. The same
sums with the linear fit are the cross-check of the form.

The figure draws the selected background per bin of each binning with its statistical error
added in quadrature with the shape systematic it carries, the 33% systematic as a log-normal band
(the bin's value times and divided by 1.33, the toy's convention), and the yields per bin of the
points 500 GeV, 0.25 GeV, 0.4 mm and 800 GeV, 0.25 GeV, 0.25 mm at their stored cross sections;
the tables below it give every signal point's yield per bin.

**The open test per bin.** The same route can be run where its answer is observed: from PP into
DP, the muon LJ going from prompt to displaced with the egamma LJ prompt, the analog `estimate.ipynb`
section 3.1 uses over the whole window (observed over predicted 1.32 ± 0.27; 1.27 ± 0.33 at
150–300 GeV and 1.5 ± 0.5 above). The tables repeat it per bin of the cube bins and of the two
windows and of the split at 400 GeV: B of DP, the predicted A of DP with the factor the rule selects (A/B of PP per class,
measured where PP measures it to 20%, fitted otherwise), the observed A of DP with its 68% Poisson
interval, and their ratio. The last column is the systematic the recipe of `estimate.ipynb` would
assign to the bin from this test alone, |ratio − 1| plus the precision of the ratio; section 5 uses
it as the third reading of the systematic. DP's A holds signal too (the next table), so a ratio
above one is not only non-closure.

**Signal in the regions the estimate reads from.** Per bin of the cube bins, the yield of each
signal point at μ = 1 in A and B of PD per DSA class and in B of DD, next to the data counts: where
the signal's share of a region is large, the factor measured there — and the shape systematic built
from it — would move if the signal exists at that strength."""))

cells.append(code(r"""FIT = {("PD", c): FITS[("PD", c, FORM)] for c in bl.CLASSES}
BINNINGS = {**bl.BINNINGS, **bl.LATER_BINNINGS}                 # the four candidates of the rule, then the split at 400 GeV
ROUTES = {name: bl.binned_routes(d2, NOMINAL, bins, fits=FIT) for name, bins in BINNINGS.items()}
for name, routes in ROUTES.items():
    print(f"{name} ({FORM} fit)" + ("; added after the result of the four candidates" if name in bl.LATER_BINNINGS else ""))
    display(bl.routes_view(routes))
print("the B route per window of estimate.ipynb")
display(el.routes_table(d2, WINDOWS)[["window", "DD B/C/D", "B x A/B per class"]])
print(f"the sums over the bins ({FORM} fit) against the one-bin estimate with the measured A/B")
display(bl.sums_table(ROUTES))
FIT_LINEAR = {("PD", c): FITS[("PD", c, "linear")] for c in bl.CLASSES}
ROUTES_LINEAR = {name: bl.binned_routes(d2, NOMINAL, bins, fits=FIT_LINEAR) for name, bins in BINNINGS.items()}
print("the same with the linear fit")
display(bl.sums_table(ROUTES_LINEAR))
YIELDS2 = {name: bl.signal_yields(BASE, SIGNAL_POINTS["2mu2e"], "2mu2e", NOMINAL, bins, profiles=PROFILES2)
           for name, bins in BINNINGS.items()}
fig = bp.binning_figure(ROUTES, YIELDS2, syst_rel=SYST)
plt.show()
for name, yields in YIELDS2.items():
    print(f"signal in A of DD per bin at the stored cross section, {name}")
    display(bl.yields_view(yields))
FIT_PP = {("PP", c): FITS[("PP", c, FORM)] for c in bl.CLASSES}
OPEN = {name: bl.open_test_table(d2, NOMINAL, bins, fits=FIT_PP) for name, bins in BINNINGS.items()}
for name in ("cube bins", "two windows", "split at 400"):
    print(f"the open test per bin: the B route from PP into DP against the observed A of DP, {name}")
    display(OPEN[name].drop(columns=["ratio", "ratio minus", "ratio plus", "syst"]))
for name in ("cube bins", "split at 400"):
    print(f"signal at mu = 1 in the regions the estimate reads from, per bin, {name}, next to the data")
    display(bl.signal_regions_table(BASE, SIGNAL_POINTS["2mu2e"], "2mu2e", NOMINAL, BINNINGS[name], data=d2))"""))

cells.append(md(r"""### 4.1 The validation per bin

The 33% systematic of the estimate is the non-closure plus the precision of two tests of the
route's assumption over the whole window. Per bin the same assumption can be checked three ways in
the open, none of which reaches A of DD above 150 GeV:

1. **The B route where its answer is visible**, per bin: the PP → DP analog of section 4 (the
   muon LJ from prompt to displaced with the egamma LJ prompt), and the search plane itself below
   150 GeV, where A of DD is open, in the histograms' two bins there (< 100 and 100–150 GeV), each
   with PD's factor measured in the same bin. Observed over predicted per bin, against the 33%
   band. Below 150 GeV all four regions of DD are open, so the inclusive route's closure there can
   be split into its two halves, [κ$_{DD}$ / κ$_{PD}$] × [C/D of DD over C/D of PD]
   (`binning_lib.closure_decomposition`): whether the route closes because each half is one, or
   because the two compensate.
2. **The sidebands of the search plane against PD's**, per bin and DSA class: C/D of DD over C/D
   of PD. The B route A = B × (A/B)$_{PD}$ is exact when κ and C/D are the same in the two planes;
   C and D of DD are open above 150 GeV, so this half of the condition is tested where A is not.
   Over the whole window it reads 0.73 −0.22 +0.31 (`estimate.ipynb` section 3.1). A bin without
   a D event has no ratio and is not drawn.
3. **κ per bin in the open planes** PP, PD and DP: whether the correlation of the two ABCD axes
   changes with the mass where it can be seen, with the χ²/ndof of the bins against each plane's
   κ over the union of the bins. DP is the only open plane with the muon LJ displaced, so it is
   the only one that can say whether DD's κ follows PD's; above 300 GeV it has 3 finite bins at
   80–175% precision and no C event above 400 GeV.

The tables and the figure are made for the cube bins and for the split at 400 GeV. What these
tests can and cannot do: a bin whose ratio sits far from one at several standard deviations would
show a transfer that fails there; a bin whose interval covers both one and 1.5 says nothing either
way. The counts per bin above 300 GeV are those of DP (28 B events above 300 GeV, 3 above 600) and
of DD's own sidebands (6 C and 3 D events above 300 GeV, none above 600), so above 300 GeV the
only test with a result is PP → DP, at 70–130%, nothing above 600 GeV, and no per-bin test can
tighten the 33% per bin."""))

cells.append(code(r"""LOW = bl.open_test_table(d2, NOMINAL, bl.LOW_BINS, ref_plane="PD", search="DD", max_rel_err=np.inf)
print("the B route into the search plane below 150 GeV, where A of DD is open, with PD's factor measured per bin")
display(LOW.drop(columns=["ratio", "ratio minus", "ratio plus", "syst"]))
DECOMP = bl.closure_decomposition(d2, NOMINAL, bl.LOW_BINS)      # all four regions of DD are open below 150 GeV
print("the inclusive B route's closure below 150 GeV as the product of its two halves, kappa ratio x C/D ratio")
display(DECOMP.drop(columns=[c for c in DECOMP.columns if c.endswith(" value")]))
SIDEBAND_ALL = bl.sideband_test_table(d2, NOMINAL, bl.BINNINGS["one bin"])
REFERENCE = SIDEBAND_ALL.loc[SIDEBAND_ALL["class"] == "all"].iloc[0]
print("the sideband test over the whole window, for reference")
display(SIDEBAND_ALL.drop(columns=["ratio", "ratio minus", "ratio plus"]))
SIDEBAND, KAPPA = {}, {}
for name in ("cube bins", "split at 400"):
    SIDEBAND[name] = bl.sideband_test_table(d2, NOMINAL, BINNINGS[name])
    print(f"the sideband test per bin, {name}: C/D of DD over C/D of PD")
    display(SIDEBAND[name].drop(columns=["ratio", "ratio minus", "ratio plus"]))
    KAPPA[name] = bl.kappa_table(d2, NOMINAL, BINNINGS[name])
    print(f"kappa per bin in the open planes, {name}")
    display(KAPPA[name].drop(columns=["value", "minus", "plus"]))
    fig = bp.validation_figure(OPEN[name], LOW, SIDEBAND[name], KAPPA[name], BINNINGS[name], syst_rel=SYST,
                               sideband_reference=(REFERENCE["ratio"], f"whole window, all classes: {REFERENCE['DD over PD']}"),
                               name=name)
    plt.show()"""))

cells.append(md(r"""### 4.2 What the estimate absorbs of a signal in the control region

The last table of section 4 shows that A of PD, where the factor is measured, is not signal-blind at
high mass: at μ = 1 an 800 GeV point puts up to 19 events among the 25 dsa0 events above 600 GeV. A
signal that exists therefore raises the measured factor and with it the estimate under itself. The
size of that is measured here by injection (`binning_lib.injection_table`): μ times each signal
point is added to every region of the data cube as counts (the hidden box stays hidden), the
A/B(M) fit is redone on the injected cube, and the B route per bin is compared with the route on
the data alone. Per bin and factor the table gives the estimate before and after, the difference
(the leak) and the leak as a share of the injected signal in A of DD, which is the part of an
excess the estimate would absorb; with the measured factor and no signal in the sidebands the
leak is exactly B of DD × signal in A of PD / B of PD per class (`leak expected`), and it is
linear in μ. The fitted factor pools the seven bins and absorbs less in the bin that holds the
signal, but the injected points move the fit, and with it the estimate of bins without signal,
either way. `choice` shows where the injected signal changes the rule's choice of factor: a
signal in A of PD makes the measured factor more precise. The second table repeats the 800 GeV,
0.25 mm point at μ = 0.2, 1 and 5, for the bins above 400 GeV and the sum over bins: with the
measured factor the leak is linear in μ up to the signal in the sidebands, with the fitted factor
it is not, since the fit moves (the dsa1+ fit most, with no event above 600 GeV in A of PD: at
μ = 1 it raises the selected estimate of the 400–600 GeV bin by 0.7 events, half of the 1.4 signal
events there, while the measured route moves by 0.04). Section 5.1 removes the bias by putting
A of PD into the likelihood."""))

cells.append(code(r"""INJECTION = bl.injection_table(BASE, SIGNAL_POINTS["2mu2e"], "2mu2e", d2, NOMINAL,
                               {name: BINNINGS[name] for name in ("one bin", "cube bins", "split at 400")}, form=FORM)
print("the B route on the data with mu = 1 of each signal point added to every region, against the route on the data alone")
display(bl.injection_view(INJECTION))
INJECTION_MU = bl.injection_table(BASE, ["2Mu2E_800GeV_0p25GeV_0p25mm"], "2mu2e", d2, NOMINAL,
                                  {name: BINNINGS[name] for name in ("cube bins", "split at 400")}, form=FORM, mus=(0.2, 1.0, 5.0))
print("the 800 GeV, 0.25 mm point at mu = 0.2, 1 and 5: the bins above 400 GeV and the sum over bins")
display(bl.injection_view(INJECTION_MU[(INJECTION_MU["lo"] >= 400) | (INJECTION_MU["bin"] == bl.TOTAL)]))"""))

cells.append(md(r"""## 5. The toy expected limit and the choice of binning

The toy is a counting experiment per bin. The background is the estimate of the bin (selected,
measured or fitted factor), with its statistical error and the shape systematic it carries as an
independent log-normal nuisance per bin and the 33% systematic as one log-normal nuisance common to
all bins; the signal per bin is the point's yield in A of DD at its stored cross section. The limit
is the median expected 95% CL CLs upper limit in the asymptotic approximation (Cowan, Cranmer,
Gross, Vitells, Eur. Phys. J. C 71 (2011) 1554; what combine's AsymptoticLimits gives for such a
datacard). The tables give per signal point the one-bin limit in signal events (measured factor)
and each binning's limit relative to it; below one is a gain.

**Caveat.** The toy is an asymptotic counting approximation with no signal systematics, and it is
used only for relative comparisons between binnings; its absolute limits are not the analysis's. It
is optimistic for bins with less than about one background event (with no background it gives 1.9
signal events against the exact 3.0), and it treats the errors of fitted estimates as independent
per bin although the bins share the fit's parameters.

**The common systematic.** With one systematic nuisance for all bins, a bin without signal —
150–300 GeV for the heavy points — constrains it for the others; a datacard with one log-normal
line across the bins does the same. That is right only if the non-closure is the same in every bin.
κ of PP and PD fall by about two standard deviations between 150–300 GeV and M ≥ 300 GeV in
`estimate.ipynb`, so the limits are repeated with an independent 33% per bin, and once more with an
independent systematic per bin sized by the open test of that bin (section 4: |observed/predicted
− 1| plus the precision of the ratio, the recipe of `estimate.ipynb` applied bin by bin, which is
mostly the precision of a few DP events). The rule is applied to the common form, the toy as
defined; the two other forms show how much of the gain rests on that constraint and on the size of
the systematic.

The decision table checks each binning against the rule's count requirement — every bin with at
least 5 B events of DD or an estimate below 3 events — and lists where the fitted factor is taken
and the largest shape systematic. The rule table applies the rule: the eligible binnings, the
relative limit of the decision points per mass, their mean, and the binning chosen. It is repeated
with the independent systematic, with the per-bin systematic from the open test, and with the
linear fit. The rule is applied to its four candidates; the split at 400 GeV is shown in the limit
tables and figure beside them and listed under the rule tables for information, never chosen."""))

cells.append(code(r"""LIMITS = bl.limit_table(ROUTES, YIELDS2, syst_rel=SYST)
for factor in bl.FACTORS:
    print(f"expected limit relative to one bin, {factor} A/B, systematic common to all bins")
    display(bl.limit_view(LIMITS, factor))
fig = bp.limit_figure(LIMITS, "selected")
plt.show()
LIMITS_SEPARATE = bl.limit_table(ROUTES, YIELDS2, syst_rel=SYST, common=False)
print("the same, selected A/B, with an independent systematic per bin")
display(bl.limit_view(LIMITS_SEPARATE, "selected"))
OPEN_SYST = bl.open_test_syst(OPEN)
print("the per-bin systematic from the open test, per binning:", {k: [f"{v:.0%}" for v in a] for k, a in OPEN_SYST.items()})
LIMITS_OPEN = bl.limit_table(ROUTES, YIELDS2, syst_rel=OPEN_SYST, common=False)
print("the same, selected A/B, with an independent systematic per bin sized by the open test of the bin")
display(bl.limit_view(LIMITS_OPEN, "selected"))
DECISION = bl.decision_table(ROUTES)
print("the count requirement of the rule per binning")
display(DECISION)
CANDIDATES = {name: ROUTES[name] for name in bl.BINNINGS}       # the rule's four candidates
RULE = bl.rule_table(LIMITS, bl.decision_table(CANDIDATES))
print("the rule: the decision points' limits relative to one bin (selected A/B, common systematic), each mass counted once")
display(RULE.round(3))
print("chosen:", ", ".join(RULE.loc[RULE["chosen"], "binning"]) or "none")
print("the same with an independent systematic per bin")
display(bl.rule_table(LIMITS_SEPARATE, bl.decision_table(CANDIDATES)).round(3))
print("the same with the per-bin systematic from the open test")
display(bl.rule_table(LIMITS_OPEN, bl.decision_table(CANDIDATES)).round(3))
print("the same with the linear fit")
LINEAR = {name: ROUTES_LINEAR[name] for name in bl.BINNINGS}
display(bl.rule_table(bl.limit_table(LINEAR, YIELDS2, syst_rel=SYST), bl.decision_table(LINEAR)).round(3))
print("for information, the split at 400 GeV beside the candidates (added after the result; not a candidate of the rule)")
display(bl.rule_table(LIMITS, DECISION).drop(columns=["chosen"]).round(3))
print("the same with the linear fit")
LIMITS_LINEAR = bl.limit_table(ROUTES_LINEAR, YIELDS2, syst_rel=SYST)
display(bl.rule_table(LIMITS_LINEAR, bl.decision_table(ROUTES_LINEAR)).drop(columns=["chosen"]).round(3))"""))

cells.append(md(r"""### 5.1 The same counting experiments in combine

The toy of section 5 is checked against combine (`AsymptoticLimits`, blinded, the a-priori
expected limit), the machinery of `sidm/studies/limit_plotting` and `sidm/scripts/run_combine_limits.py`.
`binned_cards.py` writes one datacard per signal point and binning from the frames of section 4,
in three models of the background (`binning_cards/`, 105 cards): **lnN**, one background process
per bin with the selected estimate and an asymmetric log-normal from its 68% interval, the model
of the toy with the asymmetry kept; **gmN**, one process per bin and DSA class, its rate the B
count of DD times the selected factor, with the count a `gmN` nuisance (combine profiles it with
a Gamma distribution, as the counting cards of `limit_plotting` do for their simulated counts)
and the factor an asymmetric log-normal; and **cr**, the gmN model with the control region in the
likelihood: per bin and class a second channel whose observation is A of PD (an open region),
whose background is B of PD times the factor and whose signal is the point's yield there
(section 4.2), with the factor a free parameter (`rateParam`) shared by the search bin and its
control channel. In the cr model the factor is a free parameter in every bin and class,
constrained by the bin's own control counts — above 600 GeV by the 25 A and 16 B events of PD's
dsa0 class, where the rule found the measured factor too imprecise to use (45% by its measure,
the larger side of the interval) and took the fit — so the model carries no shape systematic; the
A/B(M) fit enters only as the starting value and the Asimov expectation of the one class without
an A event in PD (dsa1+ above 600 GeV, 0.95 expected events in its control channel against 0
observed, a channel the signal dominates at the expected limit). A signal in the control region is
part of the signal hypothesis instead of a bias of the estimate; the signal in the B sidebands
(0.3–0.5 events at μ = 1 for the 800 GeV points) is not modelled. The cr model was built after the
binning result as a cross-check: it is not a candidate of the rule, and it replaces the rule's
factor clause (measured where precise to 20%, else fitted) by the bin's own counts everywhere;
section 7 says what follows from that. The lnN and gmN
models carry the shape systematic where the factor is fitted; all three carry the 33% as one
log-normal common to the search bins, the signal per bin at the stored cross section with its
simulation-statistics log-normal, and the 2.5% luminosity uncertainty on the signal. The
`observation` line of a search bin holds the background; the expected limit never reads it. The
results (`binning_limits/limits.csv`, with the provenance sidecar) are r, the limit on μ with
μ = 1 the stored cross section (the same normalisation as the toy: `cross_sections.yaml`, without
Z$_d$ branching fractions), per binning and model, and the same relative to one bin of the same
model, next to the toy's relative limit. The runner's automatic `rMax` reads the first bin's
signal of a multi-bin card and overshoots by orders of magnitude for the heavy points (the fit
returns NaN); every card here was run with a fixed `--rMax 300`."""))

cells.append(code(r"""import binned_cards as bc
COMBINE = bc.read_limits("binning_limits/limits.csv")
print(f"{len(COMBINE)} expected limits from combine ({', '.join(dict.fromkeys(COMBINE['model']))} models)")
COMPARISON = bc.comparison(COMBINE, LIMITS)
print("r per binning and model (mu = 1 the stored cross section), each relative to one bin of the same model, and the toy")
display(bc.comparison_view(COMPARISON))
print("the toy's relative limit over combine's (lnN), and the cr model's limit over the gmN model's, decision points")
part = COMPARISON[COMPARISON["point"].isin(DECISION_POINTS["2mu2e"]) & (COMPARISON["binning"] != "one bin")]
display(pd.DataFrame({"toy / combine (lnN)": (part["relative toy"] / part["relative lnN"]).describe().round(3),
                      "r cr / r gmN": (part["r cr"] / part["r gmN"]).describe().round(3)}).T)
print("r cr / r gmN per binning, decision points")
display((part["r cr"] / part["r gmN"]).groupby(part["binning"]).agg(["min", "mean", "max"]).round(3))"""))

cells.append(md(r"""Combine itself separates the cost of the cr model into its two parts. The cr cards of the cube bins
and the split at 400 GeV were run once more with the control channels' signal set to zero
(`control_signal_off.py`, `binning_limits/control_signal_off/`): the same likelihood without a signal
in the control region. Its limit over the gmN limit is the price of the free factor, the bin's own
counts in every class with the constraint those counts give, in place of the selected factor with
its error and shape systematic; the cr limit over it is the price of the signal in the control
region, which under the signal hypothesis lowers the factor fitted to A of PD and with it the
background the search bin has to show an excess over. The two multiply to r cr / r gmN exactly.
Beside them, two toy estimates of the same parts: the toy's limit with the measured factor over
the toy's with the selected factor (section 5) for the first, and 1 / (1 − leak / signal) with the
measured leak of section 4.2 for the second. The table gives the means per binning and mass over
the decision points. The last table applies the rule's own metric, the relative limit averaged
over the two masses, to the limits of each model."""))

cells.append(code(r"""TOY_FACTOR = (LIMITS[LIMITS["factor"] == "measured"].set_index(["point", "binning"])["limit on mu"]
              / LIMITS[LIMITS["factor"] == "selected"].set_index(["point", "binning"])["limit on mu"])
LEAK = INJECTION[(INJECTION["bin"] == bl.TOTAL) & (INJECTION["mu"] == 1.0)].set_index(["point", "binning"])["measured leak / signal"]
CR_OFF = bc.read_limits("binning_limits/control_signal_off/limits.csv").set_index(["point", "binning"])["r"]
part = COMPARISON[COMPARISON["point"].isin(DECISION_POINTS["2mu2e"]) & COMPARISON["binning"].isin(["cube bins", "split at 400"])].copy()
keys = list(zip(part["point"], part["binning"]))
part["r cr, control signal off"] = [CR_OFF.get(k, np.nan) for k in keys]
part["r cr / r gmN"] = part["r cr"] / part["r gmN"]
part["free factor: r cr off / r gmN"] = part["r cr, control signal off"] / part["r gmN"]
part["control signal: r cr / r cr off"] = part["r cr"] / part["r cr, control signal off"]
part["toy: measured / selected"] = [TOY_FACTOR.get(k, np.nan) for k in keys]
part["toy: 1 / (1 - leak / signal)"] = [1 / (1 - LEAK[k]) if k in LEAK.index else np.nan for k in keys]
part["points"] = [f"{bl.mediator_mass(p):g} GeV" for p in part["point"]]
print("the cost of the cr model and its two parts measured in combine, with the toy's estimates of them; mean over the decision points of each mass")
display(part.groupby(["binning", "points"], sort=False)[["r cr / r gmN", "free factor: r cr off / r gmN", "control signal: r cr / r cr off",
                                                         "toy: measured / selected", "toy: 1 / (1 - leak / signal)"]].mean().round(3))
rel = COMPARISON[COMPARISON["point"].isin(DECISION_POINTS["2mu2e"])].copy()
rel["points"] = [f"{bl.mediator_mass(p):g} GeV" for p in rel["point"]]
METRIC = pd.DataFrame({model: rel.groupby(["binning", "points"], sort=False)[f"relative {model}"].mean().unstack("points").mean(axis=1)
                       for model in ("lnN", "gmN", "cr")})
print("the rule's metric (the relative limit averaged over the two masses) on the limits of each model; the count requirement is not applied here")
display(METRIC.round(3))"""))

cells.append(md(r"""## 6. 4μ

In 4μ both objects are muon LJs, and A of DD, DP and PD is hidden above 150 GeV (the figure below
is the cartoon of section 1 for 4μ). The search plane's sidebands hold three events above 150 GeV,
all in B, so its estimate is count-limited: about 1.7 events, uncertain by a factor two to three
(`estimate.ipynb` section 5.2, from B of DD times A/B of the open planes). That sits at the Poisson
floor of a limit: with no background at all a 95% CL limit needs 3 signal events, and over an
expected background of 0.5 to 4 events the limit for one observed event moves only between 4.4 and
3.5.

Per mass bin the picture is sharper than in 2μ2e but gains less. The tables below give the counts
per mass bin of the four planes and the signal yields per bin of each binning. The three B events
of DD sit at 200–300 GeV (two) and 400–600 GeV (one); none lies above 600 GeV, where the 800 and
1000 GeV points have 83–95% of their yield, and the one event at 400–600 GeV shares its bin with
81% of the 500 GeV point's. The only open A/B above 150 GeV is PP's (0.81, 0.71, 0.47 and
0.17 −0.14 +0.50 in the four bins up to 600 GeV, 2/1 events above), so a per-bin route reads
1.4 events at 200–300 GeV, 0.2 at 400–600 GeV and none above 600 GeV, on two, one and no B
events. The heavy points' bins are therefore empty of background up to a 68% upper limit of
1.84 events times a factor that three PP events cannot fix, and binning gains 4μ at most the step
from a background of 1.7 to none — from roughly 4 to 3 signal events — a factor 1.3 against the
factor 2.6 to 6 of 2μ2e's heavy points (section 7). The toy of section 5 is not run for 4μ: below
one background event it is 1.9 signal events against the exact 3.0, and no 4μ bin holding the
heavy points' signal reaches one background event."""))

cells.append(code(r"""fig = bp.planes_cartoon(d4, NOMINAL, "4mu")
plt.show()
print("4mu: the regions per mass bin and plane (A of DD, DP and PD hidden)")
display(bl.counts_table(d4, NOMINAL, bl.FIT_BINS, planes=("DD", "DP", "PD", "PP"), classes=(None,)))
PROFILES4 = bl.signal_profiles(BASE, SIGNAL_POINTS["4mu"], "4mu", NOMINAL)
for name, bins in bl.BINNINGS.items():
    print(f"4mu signal in A of DD per bin at the stored cross section, {name}")
    display(bl.yields_view(bl.signal_yields(BASE, SIGNAL_POINTS["4mu"], "4mu", NOMINAL, bins, profiles=PROFILES4)))"""))

cells.append(md(r"""## 7. Comparison with the one-bin estimate, and conclusions

- **The factor against the mass.** A/B of PD (all classes) at 400–600 GeV is 0.61 −0.10 +0.11
  times its value at 150–200 GeV; per bin, the dsa0 class reads 1.18, 1.02, 0.89 and 0.70 (errors
  0.07 to 0.13) from 150 to 600 GeV, then 1.64 −0.59 +0.95 at 600–800 GeV, 0.60 −0.38 +0.96 at
  800–1000 GeV and 4 A events against no B event above 1000 GeV (A/B ≥ 2.2 at 68%). No one-scale
  form describes the seven bins: for dsa0 the exponential has deviance 18.9/5 (M₀ = 3027 ± 2831 GeV,
  a flat line within its error) and the straight line 19.5/5 (slope −0.16 ± 0.27 per TeV), for all
  classes 18.5/5 and 19.6/5. The three bins above 600 GeV pull both forms flat (residuals +1.6,
  −0.5 and +2.9 against the exponential) while the four below fall (+1.5, −0.4, −1.1 and −2.1); PP's
  dsa0 class does the same (15.1/5 for both forms, flat at 0.54; 6 A events against 1 B event above
  1000 GeV, residual +2.8). The dsa1+ class is described by both forms (PD: M₀ = 507 ± 330 GeV at
  3.0/4, the line 2.3/4; PP: 7.0/5 and 7.3/5); the line for PD's dsa1+ reaches zero at 900 GeV and
  is held there by the fit's positivity floor, which is what the exponential was fixed to avoid. At
  700 GeV, the mean mass of DD's B events above 600 GeV, the two forms read 0.90 ± 0.13 and
  0.97 ± 0.12 for dsa0 and 0.19 ± 0.11 and 0.14 ± 0.02 for dsa1+. Against the least-squares fit of
  the first version (M₀ = 910 ± 344 GeV for dsa0 on the five bins with both counts, 0.64 ± 0.12 at
  700 GeV) the fitted dsa0 factor at 700 GeV moved to 0.90 once the bins without a B event entered:
  above 600 GeV the fit is the pooled A/B of all seven bins, not an extrapolation of the fall below
  600 GeV. A/B of PP over PD per bin: weighted mean 0.52 ± 0.03 (dsa0), 0.53 ± 0.15 (dsa1+),
  0.54 ± 0.03 (all), χ²/ndof 1.8/5, 4.7/3 and 0.6/5 against it: flat. The mass dependence of A/B is
  the same whether the egamma LJ is prompt or displaced, which is the first thing the B route's
  assumption needs per bin.
- **The estimate per bin.** One bin: 61.3 ± 8.0 (`estimate.ipynb`). Two windows: 41.2 ± 6.6
  (150–300 GeV) + 18.5 ± 4.2 (≥ 300 GeV) = 59.7 ± 7.8. Three bins: 41.2 ± 6.6 + 15.1 ± 3.7 +
  2.9 ± 1.6 = 59.2 ± 7.8. Cube bins: 24.7 ± 5.2, 16.7 ± 3.8, 10.1 ± 3.1, 5.1 ± 2.0 and 2.9 ± 1.6,
  59.4 ± 7.6 in all and 18.1 above 300 GeV. With the measured factor in every bin the sums
  reproduce the one-bin estimate to 3% (59.8, 61.5 and 61.1 against 61.3): the one-bin form makes
  no visible error from the mass dependence of the factor, although DD's B events sit at higher
  masses than PD's (33% above 300 GeV against 25%), because two thirds of them sit at 150–300 GeV,
  where the factor is within 6% (dsa0) and 13% (dsa1+) of its one-bin value, and the split by DSA
  class carries the larger variation (a factor two between the classes, with DD's B events 44%
  dsa1+ against PD's 14%). The selected sums are 3% lower (0.975, 0.969 and 0.967 of one bin)
  where a fitted factor replaces a measurement, most of it in the ≥ 600 GeV bin (4.7 −3.1 +3.5
  measured against 2.9 ± 1.6 fitted). The fitted factor is taken for dsa1+ in every bin that PD
  measures to worse than 20% — above 300 GeV in the two windows and the three bins, every bin of
  the cube bins — and for dsa0 only at ≥ 600 GeV; the shape systematic it carries is 2–6% of the
  bin's estimate below 600 GeV (11–33% of its dsa1+ part) and 69% in the ≥ 600 GeV bin, where PD
  measures 1.56 −0.48 +0.70 for dsa0 (25 A and 16 B events above 600 GeV) against the fitted
  0.90 ± 0.13. The linear form moves the selected sums by less than 1% (0.976, 0.972 and 0.971 of
  one bin) and the ≥ 600 GeV bin from 2.9 to 3.05 events.
- **The toy limits** (selected factor, common systematic, relative to one bin; below one is a
  gain). 500 GeV: 0.38 (two windows), 0.36 (three bins), 0.21 (cube bins). 800 GeV, mean of the
  three points: 0.38, 0.17, 0.17. With an independent 33% per bin: 0.42, 0.39, 0.22 and 0.42, 0.17,
  0.17; with the per-bin systematic from the open test (56–130% below 600 GeV, about 1000% above):
  0.44, 0.37, 0.19 and 0.44, 0.17, 0.17. The 200 GeV point: 0.67, 0.67, 0.64 — its signal sits at
  150–300 GeV, and with the 33% systematic dominant the limit follows the background of its bin,
  41 against 61 events. The gain of the heavy points is a factor 2.6 from the first split at
  300 GeV (18.5 instead of 61 events under all but 0.1% of their yield) and a factor 6 from a
  ≥ 600 GeV bin (2.9 events under 97% of it); with the measured factor in that bin instead,
  4.7 events, the factor is 4.9, and with the 1000% systematic of the open test on it 5.9: against
  21–61 signal events at μ = 1, a bin of three to five background events is limited by the signal
  count, not by the background or its error. The 500 GeV point gains its factor 4.7 from a
  400–600 GeV bin of its own (5.1 events under 91% of its yield) against 2.8 from 300–600 GeV
  (15.1 events).
- **The choice.** The rule chooses the cube bins: mean relative limit of the decision points
  0.190, against 0.263 for the three bins and 0.379 for the two windows. Every candidate passes the
  count requirement, but the three bins and the cube bins pass it at ≥ 600 GeV only through the
  fitted factor: the bin has 4 B events, below the 5 the rule asks for, and its fitted estimate,
  2.9, is below 3 where the measured 4.7 is not. With the linear form the same bin reads 3.05
  events, both binnings fail the requirement, and the rule leaves the two windows (0.379). The
  verdict therefore turns on which side of 3 events an estimate of 2.9 ± 1.6 (stat.) ± 2.0 (shape)
  falls, not on anything the limit is sensitive to: the heavy points' relative limit with that bin
  is 0.17 with the fitted factor, 0.21 with the measured one and 0.17 with a 1000% systematic on
  it. The choice between the cube bins and the three bins is made by the 500 GeV point alone (0.21
  against 0.36); for the 800 GeV points the two are the same binning above 600 GeV. With an
  independent systematic per bin the rule chooses the cube bins as well (0.193 against 0.277), and
  with the per-bin systematic of the open test too (0.181 against 0.271). Two of the five bins,
  150–200 and 200–300 GeV, carry no decision signal; merging them would change the decision points'
  limits only through the common nuisance, and the binning is kept as the rule chose it. What is
  left for the group is whether a bin above 600 GeV is admitted at all. The factor 2 it adds at
  800 GeV (0.17 against 0.38) rests on four sideband events and a factor that the PD data above
  600 GeV put at 1.56 −0.48 +0.70, the pooled fit at 0.90 ± 0.13 and the first version's
  least-squares fit at 0.64 ± 0.12; and the 25 A events of PD it is measured from would hold, if an
  800 GeV signal exists at its stored cross section, 3 to 19 signal events (section 4, last table),
  so that the measured factor there is not signal-blind (the estimate would absorb 4–8% of that
  signal, section 4.2), and the fit, which includes those bins, is only partly so.
- **The split at 400 GeV** (added after the result of the four, not a candidate of the rule).
  150–400 GeV: 51.6 ± 7.3 on 65 B events of DD, both classes measured in the bin (dsa0
  1.07 ± 0.05, dsa1+ 0.47 −0.06 +0.07). ≥ 400 GeV: 8.4 ± 2.8 on 13 B events, the dsa0 part (9 B
  events) measured in the bin, 0.82 −0.11 +0.13 (16%), giving 7.3 ± 2.7, and the dsa1+ part (4 B
  events) fitted, 0.26 ± 0.10 against the measured 0.29 −0.13 +0.21, giving 1.0 ± 0.7 with a
  shape systematic of 1% of the bin. The sum is 59.9 ± 7.9, 0.978 of one bin under either fit
  form. 400 GeV is the highest edge of the histograms at which PD still measures the dsa0 factor
  to the 20% the rule asks for (9% above 300 GeV, 16% above 400, not above 600, where it is
  fitted), and the bin above it holds 98% of the 500 GeV point's yield and 99.6–99.8% of the
  800 GeV points'. The toy limit is 0.254 at 500 GeV and 0.248 at 800 GeV, a factor four at both
  decision masses, between the two windows (0.38) and the cube bins (0.21 and 0.17). What
  separates it from the cube bins at 800 GeV, 0.25 against 0.17, is the ≥ 600 GeV bin; what
  separates it from the two windows is a factor the control plane measures in the bin to 16%
  from 13 B events, with no threshold of the rule inside the bin's error (13 events against the
  5 required, 8.4 against the 3) and no fitted factor on 87% of its estimate. The PP → DP test
  reads 1.35 −0.31 +0.34 in 150–400 GeV and 1.38 −0.62 +0.75 above (9 observed against 6.5 ± 2.0
  predicted). Two caveats carry over from the ≥ 600 GeV bin, diluted. The control region the dsa0
  factor is measured from, A of PD above 400 GeV, has 98 events, and at the stored cross section
  an 800 GeV signal would add 3 to 19 of them (3.7 for the 500 GeV point; section 4, last table):
  a 3–20% pull on the factor if that signal exists, against 12–76% on the 25 events above 600 GeV
  (the estimate would absorb 2–4% of the signal, against 4–8% in a ≥ 600 GeV bin, section 4.2);
  the dsa1+ control region there (6 events) would hold up to 4.7, which is one reason that part
  is fitted. And the measured 0.82 is PD's average weighted by its own B events, 104 of the 120
  at 400–600 GeV, while DD's 9 B events sit higher (3 above 600 GeV, mean 567 GeV), where the fit
  reads 0.94: the measured and the fitted estimates of the dsa0 part differ by 15% (14% of the
  bin), which the rule does not carry as a systematic because the factor is measured. The cost
  is the 200 GeV point, whose signal sits under 52 events instead of 25: 0.82 of one bin against
  0.64 with the cube bins; it is dead at its stored cross section either way (0.56 signal events).
  It would have passed the count requirement and placed second among the candidates had it been
  one (0.251 against 0.263 for the three bins) under the exponential form, and first under the
  linear form, where the cube bins and the three bins fail the count requirement and the rule
  leaves the two windows; since it was added once the result was known, it is reported beside
  the rule's choice and not in its place.
- **What goes to the fit.** For 2μ2e, per bin of the cube bins: 24.7 ± 5.2, 16.7 ± 3.8,
  10.1 ± 3.1, 5.1 ± 2.0 and 2.9 ± 1.6 events (statistical), the 33% systematic of
  `estimate.ipynb` on each bin — common to the bins or independent makes no difference to the
  decision points (0.190 against 0.193) — and the shape systematic of the fitted factors, 2–6% per
  bin below 600 GeV and 69% at ≥ 600 GeV. The 33% is the non-closure plus the precision of two
  tests over the whole window; per bin it is assumed, not measured. The PP → DP test per bin
  (section 4) reads 1.13 −0.40 +0.48, 1.50 −0.50 +0.58, 1.60 −0.85 +1.11 and 1.01 −0.51 +0.67 up
  to 600 GeV, and above 600 GeV 3 observed events against 0.3 −0.3 +1.1 predicted from DP's three
  B events with both factors fitted (PP measures neither class there): a ratio of 10 whose 68%
  interval reaches from 0 to 23, which says that the bin's transfer is untested rather than wrong.
  Everything here is relative: the toy reproduces the structure of a datacard, not the analysis's
  limits, and the 800 GeV ratios rest on a bin with about three expected events, near the edge of
  the asymptotic approximation; the ordering of the binnings does not depend on it.
- **The validation per bin** (section 4.1) finds no bin that fails and no bin it can clear
  below the 33%. The search plane below 150 GeV, where A of DD is open: 5 observed against
  3.6 −1.9 +2.8 predicted from 4 B events below 100 GeV (1.40 ± 1.24), and 36 against 37.0 ± 7.8
  from 28 B events at 100–150 GeV (0.97 −0.26 +0.28), the one per-bin closure test that reaches
  30%. Split into its two halves, the closure at 100–150 GeV is κ$_{DD}$ / κ$_{PD}$ = 0.68 ± 0.35
  (1.21 ± 0.62 against 1.79 ± 0.17) times C/D of DD over C/D of PD = 1.32 −0.44 +0.65 (1.06
  against 0.80), a product of 0.90 −0.23 +0.30: the route closes there because the two halves
  compensate, not because each is one (below 100 GeV, 0.93 −0.93 +0.86 times 1.47 −0.36 +0.46,
  with the κ half unresolved). Above 150 GeV only the second half is visible. The sidebands of
  the search plane against PD's read 0.44 −0.20 +0.34 at 150–200 GeV (6/16 against 171/199),
  0.74 −0.39 +0.72 at 200–300 GeV (5/10 against 105/155), 4.9 −3.9 +28.6 at 300–400 GeV (5/1
  against 43/42), 0.9 −0.8 +4.3 at 400–600 GeV (1/2 against 19/33) and nothing above 600 GeV
  (0/0 against 9/4); on the split's bins 0.74 −0.23 +0.33 and 0.66 −0.62 +3.26; over the whole
  window 0.73 −0.22 +0.31. Were κ the same in DD and PD above 150 GeV, these ratios would be the
  route's closure per bin: the 150–200 GeV bin over-predicted by up to a factor 2.3 and the whole
  window by 1.4, the deficit `estimate.ipynb` section 3.1 reports as consistent with one and with
  a 30% deficit; were the halves to compensate as they do below 150 GeV, it would close. With A
  of DD hidden the data cannot tell these apart, and no decision point feels it: the 200 GeV
  point has 0.27 of its 0.56 events at 150–200 GeV. κ per bin: PP 1.18 −0.18 +0.20,
  0.99 −0.17 +0.18, 0.67 −0.19 +0.20, 0.63 −0.25 +0.29 and 0.74 −0.74 +1.02 against 0.97 ± 0.10
  over all bins (χ²/ndof 3.9/4); PD 1.26 ± 0.16, 1.41 ± 0.21, 0.79 ± 0.21, 1.14 −0.40 +0.44 and
  0.53 −0.53 +0.41 against 1.20 ± 0.10 (3.9/4); DP 0.66 −0.49 +0.55, 1.28 −1.03 +1.48 and
  0.89 −0.89 +1.55, then no C event (1.02 −0.41 +0.48, 0.2/2); on the split's bins PP 1.00 ± 0.11
  and 0.67 −0.24 +0.27, PD 1.22 ± 0.11 and 0.97 −0.30 +0.31, DP 0.88 −0.39 +0.44 and no C event.
  In PP and PD alike the two bins below 300 GeV sit above the plane's κ and the three above it
  below: between the two windows of `estimate.ipynb` κ falls from 1.08 ± 0.13 to 0.67 ± 0.15 in
  PP and from 1.31 ± 0.12 to 0.87 ± 0.17 in PD, two standard deviations in each plane, while
  against a flat κ the five bins give χ² 3.9/4. The fall is the same in the two planes, and what
  the route carries from PD to DD per bin is A/B, which includes it; that it carries across the
  muon displacement as well is assumed, since DP, the only open plane with the muon LJ
  displaced, has three bins at 80–175% precision and none above 400 GeV. Of the
  observed-over-predicted ratios of the first two tests, the sideband ratio at 150–200 GeV is
  1.6 standard deviations below one, the deficit `estimate.ipynb` reports at 150–300 GeV
  (0.54 −0.19 +0.28) seen per bin; the others lie within 1.2 standard deviations of one, five of
  them between 0.9 and 1.2 (PP → DP at 200–300 GeV, above 600 GeV and at 150–400 GeV for the
  split, the sidebands at 300–400 GeV and the dsa1+ sideband at 150–200 GeV), and every PP → DP
  ratio is at or above one, as the whole window's 1.32 is. Taken together: the per-bin tests
  reach 30% at 100–150 GeV, 55–130% from 150 to 600 GeV (above 300 GeV from PP → DP alone, at
  70–130%) and nothing above 600 GeV; the 33% per bin stays assumed, contradicted by nothing at
  the level the counts allow.
- **Signal in the control region** (section 4.2). The two 800 GeV points with 59–63 signal events
  in A of DD have 21–23 events in A of PD at μ = 1, 33–38% of that yield, all but 0.2 of them above
  600 GeV, where the data have 25 (the 0.25 mm point: 17.8 dsa0 and 4.6 dsa1+ against 25 and 0; the
  1.2 GeV point 19.1 dsa0); the 0.25 GeV, 1.2 mm point has 3.9 (18%), the 500 GeV point 4.3 at
  400–600 GeV among 79. Added to the data as
  counts, the 0.25 mm point raises the one-bin estimate
  from 61.3 to 62.9, 3% of its 58.8 signal events in A of DD; the ≥ 600 GeV bin of the cube bins
  with the measured factor from 4.7 to 9.2, 4.5 events or 8% of the 57.3 signal events there (4.2
  from B of DD × signal in A of PD / B of PD per class, the rest from the 0.3 signal events in the
  B sidebands), with the fitted factor from 2.9 to 4.0 (2%), the fit pooling the seven bins and
  spreading the rest over the bins without signal (150–200 GeV goes from 24.7 to 23.7); the ≥ 400 GeV
  bin of the split from 8.5 to 10.9 (4%) measured and 8.4 to 11.0 (4%) selected. The other heavy
  points: 7% and 3% (800 GeV, 1.2 GeV, 1.2 mm), 4% and 2% (800 GeV, 0.25 GeV, 1.2 mm, 21.5 signal
  events), 7–8% and 3–4% of the 1000 GeV points' 1.5–2.1 events; the 500 GeV point 3% in either
  binning (0.4 events). With the measured factor the leak is linear in μ up to the signal in the
  sidebands (the 0.25 mm 800 GeV point at μ = 0.2, 1 and 5 leaks 8, 8 and 9% of its signal into
  the ≥ 600 GeV bin, 4% at each μ into the ≥ 400 GeV bin of the split), so at the expected limit it
  is the same share of a signal of that size; with the fitted factor it is not, since the fit moves
  (2% of the signal at each μ in the ≥ 600 GeV bin, but 0.7 events into the 400–600 GeV bin at
  μ = 1 from the dsa1+ fit alone). No injected point moves a factor
  across the rule's 20% precision threshold. That is the size of the bias: an excess of the 0.25 mm
  800 GeV signal would appear up to 8% smaller than it is in the cube bins with the measured factor,
  2% with the fitted one, 4% in the split at 400 GeV and 3% in one bin. The lnN and gmN cards, and
  the toy of the rule, do not carry it, because their model has no signal in the control region;
  the cr model of section 5.1 puts A of PD into the likelihood so that the signal there is part of
  the hypothesis, and its expected limit pays for it (section 5.1 measures the price with combine).
- **Combine** (section 5.1) reproduces the toy. Relative to one bin, the lnN model (the toy's,
  with the asymmetric intervals) gives 0.22 (cube bins), 0.37 (three bins), 0.39 (two windows)
  and 0.27 (split at 400) at 500 GeV, and 0.18, 0.18, 0.39 and 0.26 for the 800 GeV points,
  against the toy's 0.21, 0.36, 0.38, 0.25 and 0.17, 0.17, 0.38, 0.25: the toy is 3–5% more
  optimistic and orders the binnings the same way. The gmN model, which profiles the B counts of
  DD with Gamma distributions, is less favourable to the binnings with a bin on few events: by
  10–19% for the cube bins and the three bins (0.25 at 500 GeV and 0.21 at 800 GeV for the cube
  bins; their ≥ 600 GeV bin rests on 4 events), 7% for the split at 400 (0.29 and 0.28) and 2–4%
  for the two windows (0.40): a log-normal on an estimate resting on 4 to 13 events understates
  its uncertainty, and the finer the binning the more of the estimate rests on such counts. In
  absolute terms, with μ = 1 the stored cross section and no branching fractions, one bin gives
  r = 0.57 and 0.53 for the two 800 GeV points with 59–63 signal events, 1.6 for the 21-event one
  and 2.3 for the 500 GeV point; the cube bins 0.10, 0.09, 0.28 and 0.51 (gmN 0.12, 0.12, 0.34,
  0.58), the split at 400 0.15, 0.14, 0.41 and 0.61 (gmN 0.17, 0.16, 0.46, 0.68); the 1000 GeV
  points stay above r = 2.9 in every binning (16–23 for one bin), the 200 GeV point above 39. The
  cr model, the gmN model with the control region in the likelihood and the factor a free parameter
  per bin and class, costs the binnings with a ≥ 600 GeV bin 35% at 800 GeV (r 0.17, 0.16 and 0.46
  for the cube bins against gmN's 0.12, 0.12 and 0.34; the three bins the same), and at 500 GeV 4%
  for the cube bins and 1% for the three bins; the split at 400 GeV 2%, the two windows and one bin
  1%. Combine separates the 35% into two parts (the cr cards run once more with the control
  channels' signal set to zero): without the signal in the control region the cr cards give 1.30
  times the gmN limit for the cube bins at 800 GeV, the price of a factor that is the bin's own
  counts in every class, 1.56 for dsa0 above 600 GeV against the fitted 0.90, with the constraint
  those counts give in place of the fitted factor's error and shape systematic (the toy puts 1.24 of
  it on the factor's value alone); the signal in the control region is the remaining 1.03–1.05 at
  800 GeV (1.05 for the 0.25 mm point, 1.04 on average), 1.02 at 500 GeV, because under the signal
  hypothesis the factor fitted to A of PD falls
  and with it the background the search bin has to show an excess over (the toy's 1 / (1 − leak /
  signal), 1.07, overstates it: the dsa1+ class above 600 GeV has a fitted expectation in the cr
  cards where the measured estimate of section 4.2 has none). For the split at 400 GeV the cr cards
  without the control-region signal give the gmN limit to four digits: its 2–3% is the signal in
  the control region alone. Relative to one bin in the cr model the cube bins stand at 0.28 for the
  800 GeV points, the three bins at 0.28, the split at 400 at 0.28 and the two windows at 0.40; at
  500 GeV 0.25, 0.39, 0.29 and 0.40. The gain of the ≥ 600 GeV bin over the split at 400 GeV, a
  factor 1.5 in the lnN model and 1.34 in gmN, rests on the fitted factor there and is gone once the
  bin's factor is what the bin's own counts say and the signal in those counts is part of the
  hypothesis. The rule's own metric, the relative limit averaged over the two masses, applied to the
  cr limits with the count requirement set aside, still ranks the cube bins first, 0.27 against 0.29
  for the split and 0.34 for the three bins, on the 500 GeV point alone.
- **4μ.** Count-limited: 3 B events of DD above 150 GeV, at 200–300 GeV (two) and 400–600 GeV
  (one), against 98–99.7% of the 500–800 GeV signal yield above 300 GeV and 83–95% of the 800 and
  1000 GeV points' above 600 GeV, where the search plane has no B event. The one-bin estimate of
  about 1.7 events, uncertain by a factor two to three, stays; per bin it becomes 0, 1.4, 0, 0.2
  and 0 events on no, two, no, one and no B events, and the heavy points' bins are background-free
  up to what three sideband events can say. Binning gains 4μ at most the step from a background of
  1.7 to none, roughly 4 to 3 signal events.
- **Against the one-bin estimate.** The binned estimate is the same estimate — the same B events,
  the same plane, the same factor read per bin or fitted — and sums to the same number within 3%;
  what binning changes is not the background but where it sits relative to the signal. For the
  heavy points of 2μ2e that is a factor 2.6 in the expected limit from the split at 300 GeV, which
  rests on 26 sideband events and a factor PD measures to 9%, and a factor 6 with a bin above
  600 GeV, which rests on 4 sideband events and a factor the data do not pin down; in 4μ it is
  little. The price of the five bins is five estimates resting on 28, 24, 13, 9 and 4 sideband
  events in place of one on 78, a factor taken from a fit in the bin that matters most, and a
  systematic measured over the whole window and assumed per bin. The split at 400 GeV, which was
  not among the candidates, buys a factor four at both decision masses from 13 B events and a
  factor measured in the bin to 16%, with no fitted dsa0 factor and no rule threshold inside the
  bin's error, at the price of the 200 GeV point; it is the binning to propose if the group
  prefers not to lean on four events, and the cube bins remain the rule's choice. Combine agrees
  with the toy on all of this to 5%, and says that a Gamma treatment of the B counts costs the
  binnings with a 4-event bin another 10–19%, the split at 400 7%. The control region of the
  estimate is not signal-blind above 400 GeV, and least of all above 600 GeV: an 800 GeV signal at
  its stored cross section would raise the estimate under itself by up to 8% of its size in a
  ≥ 600 GeV bin with the measured factor (2% with the fitted one), 4% in the split at 400 GeV, 3% in
  one bin. With the control region in the likelihood (the cr model) that bias is part of the fit,
  and the comparison changes: the cube bins and the split at 400 GeV both stand at 0.28 of one bin
  for the 800 GeV points, since what separated them, a factor 1.34–1.5 from the ≥ 600 GeV bin, was
  the fitted factor 0.90 in a bin whose own 25 and 16 dsa0 control events say 1.56 −0.48 +0.70, 1.3
  standard deviations above the fit, and would hold up to 19 signal events. The cr model is a
  cross-check built after the result, not a candidate of the rule, and it departs from the rule in
  two places: its factor is the bin's own counts in every class, where the rule took the fit when
  the counts were imprecise, and with that factor the ≥ 600 GeV bin's estimate is 4.9 events on 4
  sideband events, which fails the rule's count requirement, so that by the rule's letter only one
  bin and the two windows would stay eligible and the two windows (0.40) be chosen; the split at
  400 GeV was never a candidate. The rule's own metric on the cr limits, the count requirement set
  aside, still ranks the cube bins first (0.27 against 0.29 and 0.34), through the 500 GeV point.
  What this notebook puts to the group, beside the rule's choice, is to carry the control region in
  the likelihood and, with it there, to prefer the split at 400 GeV: the same limit at 800 GeV from
  13 sideband events where the cube bins rest on 4, at the price of 0.29 against 0.25 at 500 GeV.
  The decision is the group's.

## Files

`binning_lib.py` (the tables, the fits, the signal injection, the toy limit and the decision rule
of this notebook), `test_binning_lib.py`, `binning_plots.py` (its figures and cartoons),
`binned_cards.py` (the combine datacards of section 5.1 in the lnN, gmN and cr models and the
reading of their limits; `test_binned_cards.py`), `control_signal_off.py` (the cr cards with the
control channels' signal set to zero), `binning_cards/` (the 105 cards with their provenance
sidecar), `binning_limits/` (combine's
expected limits, `limits.csv`, with the sidecar of `run_combine_limits.py`; `control_signal_off/`
the same for the eight cards without the control-region signal), `_build_binning.py`
(writes this notebook). The estimate it bins is that of `estimate.ipynb` (`estimate_lib.py`,
`_build_estimate.py`)."""))

nb = nbf.v4.new_notebook(cells=cells, metadata=KERNEL)
nbf.write(nb, "binning.ipynb")
