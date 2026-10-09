"""Build transfer_factors.ipynb.

Run from this directory on the LPC, with ABCD6D_LOCAL pointing at the directory filled by
fetch_6d.sh (default /uscms_data/d3/murtazas/abcd6d_local):
    python _build_notebook.py
    jupyter nbconvert --to notebook --execute --inplace transfer_factors.ipynb
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

cells.append(md(r"""# Do ABCD factors measured with prompt lepton jets carry over to displaced ones?

The background estimate of the search is an ABCD in the plane of $|\Delta\phi|$ between the two
lepton jets (LJs) against the leading muon LJ's isolation, with the other LJ's isolation below 0.5
and the LJ-pair mass above 150 GeV required at the event level. The search region is the corner
of that plane with both LJs displaced: leading muon LJ isolation < 0.25 and $|\Delta\phi| \ge 2.4$.
Its closure, $\kappa = A\,D/(B\,C)$, can only be checked where the isolated, back-to-back corner A
is open, which so far has meant the plane with the displacement requirement inverted on both LJs
(the validation plane). There the 2018 data close to within 5–25% depending on the boundaries,
the plane with both LJs displaced has too few events in its open regions to test anything, and
the simulation has about one effective event in the search region. The question of this notebook
is whether the quantities the method needs can be *measured* with prompt LJs and *used* with
displaced ones, which would move the statistical weight of the estimate to where the data are.

The test is done per axis. The ABCD prediction $A = B\,C/D$ can be read as

$$A = C \times \frac{B}{D} \qquad\text{or}\qquad A = B \times \frac{C}{D},$$

where $B/D$ is the pass/fail ratio of one axis among events that fail the other axis, and $C/D$ the
same with the roles exchanged. A ratio may be taken from another plane only if it does not depend
on the displacement requirement that differs between the two planes. Two things are then kept
apart throughout:

- $B/D$ and $C/D$ are the ratios *among failures* of the other axis. They are open everywhere,
  including the search plane, and are what the fail slices can test.
- The search region itself needs the ratio *among passes* of the other axis, $A/C$ or $A/B$.
  That is open only where A is, so its transfer can be tested in the one-LJ-displaced planes
  and at low mass, never in the search plane itself. Writing the prediction as
  $C \times (B/D)_{\text{other plane}}$ does not avoid this: it equals the in-plane
  $B\,C/D$ times the ratio of the two $B/D$ values and assumes $\kappa = 1$ in the search plane
  just as the in-plane estimate does.

Two planes are examined. The **analysis plane** ($|\Delta\phi|$ × leading-muon-LJ isolation,
section 1) is the one the estimate uses. The **companion plane** of the two LJ isolations
(section 2) separates the two LJs' isolation responses, which is where the physical cause of a
non-transferring factor can be seen; it is also the plane in which the validation-plane closure
was first shown.

**Inputs.** Dongyub Lee's inclusive six-dimensional histograms (campaign
`ABCD_6D_inclusive_eHEM_skimv1`, 2018 data and simulation, read-only copies made by
`fetch_6d.sh`). Each event enters once with the leading muon LJ's isolation and maximum pixel
hits, the leading egamma LJ's isolation and minimum lost hits (2μ2e) or the subleading muon
LJ's isolation and pixel hits (4μ), $|\Delta\phi|$ between the two LJs and their pair mass.
Cosmic veto, DSA cross-cleaning, muon veto and the electron HEM veto are applied in the fill;
no isolation, displacement, $\Delta\phi$ or mass requirement is. Every region below is a
projection of that cube (`abcd6d_io.py`, `transfer_lib.py`).

**Definitions.**

| symbol | meaning |
|---|---|
| object 0, object 1 | 2μ2e: leading muon LJ, leading egamma LJ. 4μ: leading, subleading muon LJ |
| displaced | muon LJ: max pixel hits ≤ 2; egamma LJ: min lost hits ≥ 1 (photon-only LJs carry the lost-hits sentinel 999 and count as displaced). Prompt = the complement |
| plane PP, DP, PD, DD | displacement state of (object 0, object 1): PP both prompt (validation plane), DD both displaced (search plane), DP and PD one each |
| mass window | M ≥ 150 GeV (the analysis requirement) or M < 150 GeV. Nothing measured below 150 GeV is projected above it: the mass requirement changes the kinematics, so the low-mass rows only ask whether a plane factorises at all |
| isolation pass / fail | isolation < 0.25 / isolation ≥ 0.5. The 0.25–0.5 strip belongs to no region: it keeps the fail slices away from the search region and from the hidden box |
| analysis plane | object 0 isolation against $\lvert\Delta\phi\rvert$, object 1 isolation < 0.5 at the event level. $\Delta\phi$ pass: ≥ 2.4, fail: < 2.0 (the 2.0–2.4 strip belongs to no region). A = isolation pass & $\Delta\phi$ pass, B = fail & pass, C = pass & fail, D = fail & fail |
| companion plane | object 0 isolation against object 1 isolation, with $\lvert\Delta\phi\rvert \ge 2$: A = pass/pass, B = fail/pass, C = pass/fail, D = fail/fail (first letter = object 0) |
| r1 = B/D, A/C | pass/fail ratio of the second axis among failures / passes of the first |
| r0 = C/D, A/B | pass/fail ratio of the first axis among failures / passes of the second |
| κ | A·D/(B·C), the in-plane non-closure; the ABCD estimate assumes κ = 1 |

**Blinding.** The search region is A of the analysis plane in the DD plane at M ≥ 150 GeV. The
data cubes are loaded with that region plus a margin on both of its axes set to NaN: DD plane,
M ≥ 150 GeV, object 0 isolation < 0.5 (the search region ends at 0.25), $|\Delta\phi| \ge 2$ (the
search region starts at 2.4), object 1 isolation < 0.5 (the search region's own edge). Any
projection touching a hidden cell is NaN and prints as "blind", so the hidden count can be neither
printed nor recovered by subtraction. In 4μ the isolated corners of the two one-LJ-displaced
planes are hidden as well: they hold up to half of the signal's search-region yield at the stored
normalisation (section 4), comparable to the background the prompt-plane factors predict there
(section 1c). Everything else is open. What is open and used, with the signal it holds at the stored normalisation, is listed
in section 4; the regions closest to the search region are its fail slices in the DD plane
(B, C and D of both planes at M ≥ 150 GeV), the isolated $|\Delta\phi| < 2$ corner of the DD plane
(C of the analysis plane), the isolated DD corner at M < 150 GeV, and in 2μ2e the isolated corners
of the DP and PD planes. Looking at these is a change from the practice so far, which has used
data only in the validation plane; it should be agreed with the group before the numbers are
circulated. Simulation is never blinded.

**Uncertainties.** Data ratios of disjoint counts carry the 68.27% Clopper–Pearson interval of
the corresponding fraction mapped to the ratio; a prediction from a count of zero carries the 68%
Poisson upper limit (1.84 × factor) as its upward error; weighted simulation uses sumw²
propagation. Shape comparisons use the two-sample χ² with pooled expected contents, which keeps
its calibration down to a few tens of events in the smaller sample; with a handful of events there
(the 4μ rows with one event) the p-value is indicative only."""))

cells.append(code(r"""import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import abcd6d_io as io
import transfer_lib as tl
import transfer_plots as tp

pd.set_option("display.max_columns", 30)
pd.set_option("display.width", 220)
pd.set_option("display.max_colwidth", 200)
tp.set_style()

BASE = tl.local_dir()
print("inputs:", BASE)
for kind in ("data", "bkg", "signal"):
    print(f"  {kind}: {len(tl.sample_paths(BASE, kind))} files")

data = tl.summed_cubes(BASE, "data")       # always blinded; nothing of the hidden box is printed
print("hidden box:", io.blind_signature())
for t, c in data.items():
    print(f"{t}: {int(np.isnan(c.sumw).sum())} hidden cells")"""))

cells.append(md(r"""## 0. Positive control: the validation-plane closures of the status report

Before anything new, the projections are checked against the numbers Dongyub showed on
2026-10-08 for the validation plane (PP, M ≥ 150 GeV), with his region conventions (no buffers:
fail = not pass). Companion plane at $|\Delta\phi| \ge 2$: 2μ2e κ = 0.928 ± 0.059 at the 0.25/0.25
working point and 1.05 ± 0.056 at 0.5/0.5; 4μ κ = 5.6 ± 0.7 and 2.6 ± 0.24. Analysis plane with the
other LJ's isolation < 0.5, object 0 isolation boundary 0.25 and $|\Delta\phi|$ boundary 2.0:
2μ2e κ = 0.76 ± 0.10, 4μ κ = 1.38 ± 0.37. The projections here include the isolation underflow bin
(isolation < 0, a handful of events) that his region definitions leave out, and differences of one
or two events in A remain; they change no conclusion and were not traced further."""))

cells.append(code(r"""rows = []
for t in tl.TOPOLOGIES:
    for wp in (0.25, 0.5):
        reg = tl.regions(data[t], "PP", "M>=150", wp=wp, fail=wp)
        f = tl.factors(reg)
        rows.append({"topology": t, "plane": "companion (iso x iso)", "boundaries": f"iso {wp}/{wp}",
                     **{r: tl.fmt_count(reg[r]) for r in "ABCD"}, "B C / D": tl.fmt(f["pred"], 1), "kappa": tl.fmt(f["kappa"])})
    reg = tl.dphi_regions(data[t], "PP", "M>=150", wp=0.25, fail=0.25, dphi_pass=2.0, dphi_fail=2.0, other_iso=0.5)
    f = tl.factors(reg)
    rows.append({"topology": t, "plane": "analysis (dphi x iso0)", "boundaries": "iso0 0.25, dphi 2.0, iso1 < 0.5",
                 **{r: tl.fmt_count(reg[r]) for r in "ABCD"}, "B C / D": tl.fmt(f["pred"], 1), "kappa": tl.fmt(f["kappa"])})
pd.DataFrame(rows)"""))

cells.append(md(r"""## 1. The analysis plane: $|\Delta\phi|$ × leading-muon-LJ isolation

### 1a. The four planes side by side

Region counts, the four pass/fail ratios and the in-plane κ in each plane, with the buffered
regions used from here on. A of the DD plane at M ≥ 150 GeV is the search region and is blind,
so its κ, A/C and A/B are not available; in 4μ the same holds for the DP and PD planes. The
M < 150 GeV rows ask whether the plane factorises at low mass, where the DD corner is open.

How to read it: if displacement did not matter, every column would be constant down the four
planes. κ ≠ 1 in a plane means the two axes are correlated there; a ratio changing between
planes means the corresponding axis responds to displacement."""))

cells.append(code(r"""for t in tl.TOPOLOGIES:
    print(t, "— planes:", "; ".join(f"{p}: {tl.plane_label(t, p)}" for p in tl.PLANES))
    display(tl.dphi_table(data[t]))"""))

cells.append(md(r"""### 1b. Each ratio across each change of displacement

$|\Delta\phi|$ is an event quantity, so there is no a-priori "legitimate" pair here: each of the
four ratios is compared across each single change of displacement (object 0 prompt → displaced
with object 1 held, and the reverse). The last column is the displaced-over-prompt ratio; 1
within its uncertainty is what a transferable quantity looks like. The rows that end in the DD
plane are the ones a search-region prediction would rely on: for s = B/D and t = C/D both ends
are open; for A/C and A/B the DD end is the search region and the row is blind, so those
quantities can only be tested across the changes that end in DP or PD."""))

cells.append(code(r"""for t in tl.TOPOLOGIES:
    for mass in ("M>=150", "M<150"):
        print(f"{t}, {mass}")
        display(tl.dphi_transfer_table(data[t], mass))"""))

cells.append(md(r"""The integrated ratios are single numbers; the same comparison as a shape. Left: the
$|\Delta\phi|$ distribution of events whose muon LJ fails isolation (≥ 0.5), per plane, normalised
to unit area, with the ratio to PP below; right: the same among events whose muon LJ passes
(< 0.25). The DD plane is hidden at $|\Delta\phi| \ge 2$ in the pass slice and is not drawn there.
The region above 2.4 is where s and A/C are set; bins with no event in the compared plane are not
drawn in the ratio panel. The table gives the pooled two-sample χ² between planes."""))

cells.append(code(r"""for t in tl.TOPOLOGIES:
    fig = tp.dphi_shapes_figure(data[t], "M>=150")
    plt.show()
    display(tl.dphi_shape_table(data[t], "M>=150"))"""))

cells.append(md(r"""### 1c. Closure where the isolated corner is open, and what the recipes give for the search region

Every prediction of A that a one-displacement transfer permits, tested in the planes whose A is
open: the one-LJ-displaced planes at M ≥ 150 GeV (from PP) and the DD plane at M < 150 GeV (from
PD and from DP, each one displacement away). The rows labelled "B C / D in the plane" are the
ordinary in-plane closure (κ) for reference. Observed over predicted is the closure; its
uncertainty combines the Poisson error of A with the factor's interval.

The second table lists what each recipe gives for the search region; A itself is not touched.
Every row states the assumption it rests on. Recipes that disagree with each other beyond their
uncertainties cannot all be right, which is a statement about the method that needs no
unblinding."""))

cells.append(code(r"""for t in tl.TOPOLOGIES:
    print(t)
    display(tl.dphi_closure_table(data[t]))
    display(tl.dphi_sr_prediction_table(data[t]))"""))

cells.append(md(r"""## 2. The companion plane: the two LJ isolations

### 2a. The four planes

The same table for the plane of the two isolations at $|\Delta\phi| \ge 2$. Here the two axes
belong to two different objects, so a ratio measures one LJ's isolation response while the other
LJ's displacement is varied, and a change between planes points at a physical population."""))

cells.append(code(r"""for t in tl.TOPOLOGIES:
    print(t, "— planes:", "; ".join(f"{p}: {tl.plane_label(t, p)}" for p in tl.PLANES))
    display(tl.region_table(data[t]))"""))

cells.append(md(r"""### 2b. Each LJ's isolation ratio across the other LJ's displacement

The pairs that matter: the object-1 ratios (r1 = B/D among object-0 failures, A/C among its
passes) compared between PP and DP and between PD and DD, i.e. with object 1 held in one state
while object 0 switches from prompt to displaced; the object-0 ratios (r0 = C/D, A/B) the other
way round. The same ratio across its own object's displacement is a different population (prompt
fakes against conversions, heavy-flavour decays and standalone-only muons) and is not expected
to transfer; it is not tabulated, but section 2d scans it."""))

cells.append(code(r"""for t in tl.TOPOLOGIES:
    for mass in ("M>=150", "M<150"):
        print(f"{t}, {mass}")
        display(tl.transfer_table(data[t], mass))"""))

cells.append(md(r"""Each panel shows the isolation distribution of the measured LJ inside the other LJ's fail slice
(isolation ≥ 0.5), normalised to unit area, for the two planes that differ only in the other LJ's
displacement. Top row: object 1's isolation (r1 pairs); bottom row: object 0's (r0 pairs). The
ratio panel is displaced over prompt; a flat ratio at 1 is a transferable shape, and the bins
below 0.25 (the pass region) are the ones that set the factor. Bins with no event in the displaced
plane are not drawn in the ratio panel. The table gives the pooled two-sample χ²."""))

cells.append(code(r"""for t in tl.TOPOLOGIES:
    fig = tp.fail_slice_shapes_figure(data[t], "M>=150")
    plt.show()
    display(tl.shape_table(data[t], "M>=150"))"""))

cells.append(md(r"""### 2c. The ratios against the displacement variable itself

Instead of two states, the full displacement variable of the held object: left column, the
object-1 ratios in bins of object 0's pixel hits with object 1 held prompt (black) or displaced
(red), r1 among object-0 failures above and A/C among its passes below; right column, the
object-0 ratios in bins of object 1's lost hits (2μ2e; the last bin is the photon-only LJs, which
carry the sentinel 999) or pixel hits (4μ). Green bins are the displaced ones. The numbers under
each bin are the numerator and denominator counts ("blind" where the hidden box is touched). A
transferable ratio is flat across the green/white boundary; a ratio that changes inside the
displaced bins (for instance between zero pixel hits, where standalone-only muon LJs sit, and one
or two) says which population drives the change."""))

cells.append(code(r"""for t in tl.TOPOLOGIES:
    fig = tp.factor_scan_figure(data[t], "M>=150")
    plt.show()"""))

cells.append(md(r"""### 2d. Closure where the isolated corner is open, and the recipes for the isolated DD corner

As in section 1c, for the companion plane. The isolated DD corner of this plane at M ≥ 150 GeV
is not the search region (the analysis plane's A is), but it is inside the hidden box and its
recipes are listed with their assumptions in the same way."""))

cells.append(code(r"""for t in tl.TOPOLOGIES:
    print(t)
    display(tl.closure_table(data[t]))
    display(tl.sr_prediction_table(data[t]))"""))

cells.append(md(r"""## 3. Simulation: the same tables, and what populates each region

The 18 background samples (QCD, Drell–Yan, tt̄, diboson), scaled to the 2018 luminosity. The
simulation decides nothing here: its DD plane holds a few effective events, and its
validation-plane κ has uncertainties of 50–100%, so it neither confirms nor contradicts the data.
The composition tables say which process each open region is made of, which is the physical
reason a ratio does or does not move between planes."""))

cells.append(code(r"""bkg = tl.group_cubes(BASE, "bkg")     # summed per process group: QCD, DY, ttbar, diboson
bkg_sum = {t: io.add_cubes([g[t] for g in bkg.values()]) for t in tl.TOPOLOGIES}
for t in tl.TOPOLOGIES:
    print(t, "— simulation, analysis plane, buffered regions")
    display(tl.dphi_table(bkg_sum[t]))
    print(t, "— simulation, companion plane, buffered regions")
    display(tl.region_table(bkg_sum[t]))
    for plane in tl.PLANES:
        print(f"{t} {plane} ({tl.plane_label(t, plane)}), M>=150: composition, analysis plane")
        display(tl.composition_table(bkg, t, plane, kind="dphi x iso0"))
        print(f"{t} {plane} ({tl.plane_label(t, plane)}), M>=150: composition, companion plane")
        display(tl.composition_table(bkg, t, plane))
del bkg"""))

cells.append(md(r"""## 4. Signal in the open regions

Seven signal points per channel: the four reference points of the status report (m_A = 200,
500, 800, 1000 GeV with m_Zd = 0.25 GeV and a mean transverse decay length near 40 cm), two
m_Zd = 1.2 GeV points at the same decay length (800 and 1000 GeV) and the longer-lived
800 GeV, m_Zd = 0.25 GeV point near 190 cm. The stored normalisation is the production cross
section of the samples (not a limit-level one); the search-region yields at that normalisation
are listed first. For every region the table then gives the percentage of the signal's own
search-region yield that falls there, the largest signal count among the seven points and, where
the region is open, the data count and the largest signal-over-data count. The hidden regions
appear with their signal share and "blind" for the data: that share is what justifies hiding
them. A region used to measure a ratio is safe when the signal-over-data count it could hold is
small compared with the ratio's statistical precision. Signal in a region that enters a prediction
multiplicatively (C of the analysis plane in C × s, C of the companion plane in C × r1) is absorbed
into the background estimate as that percentage times the factor, at any cross section."""))

cells.append(code(r"""SIGNAL_POINTS = {"2mu2e": ["2Mu2E_200GeV_0p25GeV_1p0mm", "2Mu2E_500GeV_0p25GeV_0p4mm",
                           "2Mu2E_800GeV_0p25GeV_0p25mm", "2Mu2E_1000GeV_0p25GeV_0p2mm",
                           "2Mu2E_800GeV_1p2GeV_1p2mm", "2Mu2E_1000GeV_1p2GeV_0p96mm",
                           "2Mu2E_800GeV_0p25GeV_1p2mm"],
                 "4mu": ["4Mu_200GeV_0p25GeV_1p0mm", "4Mu_500GeV_0p25GeV_0p4mm",
                         "4Mu_800GeV_0p25GeV_0p25mm", "4Mu_1000GeV_0p25GeV_0p2mm",
                         "4Mu_800GeV_1p2GeV_1p2mm", "4Mu_1000GeV_1p2GeV_0p96mm",
                         "4Mu_800GeV_0p25GeV_1p2mm"]}
for t in tl.TOPOLOGIES:
    signals = tl.sample_cubes(BASE, "signal", SIGNAL_POINTS[t], topologies=(t,))
    short = {k.split("_", 1)[1]: v for k, v in signals.items()}
    print(t)
    display(tl.signal_sr_yields(short, t))
    display(tl.contamination_table(short, data[t], t))
    del signals, short"""))

cells.append(md(r"""## 5. The LJ-pair mass: what the 150 GeV requirement does, and bin-by-bin factors

M ≥ 150 GeV is an event-level requirement of the analysis, and the M < 150 GeV rows above only ask
whether a plane factorises; nothing measured below 150 GeV is projected above it. This section
shows why that separation is needed and what the mass axis offers:

- the $|\Delta\phi|$ distribution of the validation plane in the two mass windows, among muon-LJ
  isolation failures (left) and passes (right): the back-to-back fraction is a kinematic quantity
  and the mass window changes it;
- the two ratios among failures (B/D, C/D), the pass-side ratio A/C and κ of each plane against
  the mass bin, in the analysis plane and in the companion plane. A ratio that depends on mass
  means the integrated ABCD averages over populations whose mix differs between planes and
  windows, and a factor measured in one window does not describe another; κ changing with mass
  means the integrated closure is a statement about that mixture;
- the mass spectrum of each region of the analysis plane, in PP (all four regions) and in the DD
  plane (its open fail slices). This is also the first look at a bin-by-bin version of the estimate,
  a transfer factor per mass bin instead of one integrated factor: the DD fail slices at
  M ≥ 150 GeV hold a few tens of events in total, so per-bin factors cannot be measured in the
  search plane itself and could only be imported from the open planes, which is the transfer
  question of sections 1 and 2 asked bin by bin."""))

cells.append(code(r"""for t in tl.TOPOLOGIES:
    fig = tp.mass_window_dphi_figure(data[t], "PP")
    plt.show()
    fig = tp.mass_factor_figure(data[t], "dphi x iso0")
    plt.show()
    for plane in ("PP", "PD", "DD"):
        print(f"{t}, analysis plane, {plane} ({tl.plane_label(t, plane)}), by mass bin")
        display(tl.mass_scan(data[t], plane, "dphi x iso0"))
    fig = tp.mass_factor_figure(data[t], "iso x iso")
    plt.show()
    fig = tp.mass_shape_figure(data[t], "dphi x iso0")
    plt.show()"""))

cells.append(md(r"""## 6. Verdict

**Short form.** Across the muon LJ's displacement, the one that turns the PD plane into the search
plane, every ratio of the analysis plane is consistent with transferring, but the displaced-muon
planes hold 5–14 events in the regions that set them, so "consistent" means 20–110% uncertainties.
Across the egamma LJ's displacement, where both ends have hundreds of events per region, three
of the four ratios change: s = B/D by 0.69 ± 0.09, t = C/D by 1.29 ± 0.23 and A/B by 1.94 ± 0.13,
while A/C stays (1.04 ± 0.14). A factor measured with prompt LJs therefore does not in general
describe displaced ones, and the recipes for the search region differ by a factor two depending
on which assumption replaces κ = 1. The companion plane locates the egamma side of this in the
egamma LJ's isolation response, in particular in the photon-only LJs, which have no prompt
counterpart. The mass axis shows that κ itself depends on the LJ-pair mass inside the
M ≥ 150 GeV window. 4μ has statistics only in its PP plane; its few finite ratios point the same
way as 2μ2e for s and the opposite way for the companion-plane C/D.

### 6.1 The analysis plane (2μ2e, M ≥ 150 GeV)

*Muon LJ prompt → displaced* (PP → DP with the egamma LJ prompt, PD → DD with it displaced;
section 1b):

- s = B/D: 0.98 (−0.38/+0.63) and 0.66 (−0.20/+0.28); the fail-slice $|\Delta\phi|$ shapes agree
  (χ²/ndf 8.6/13 and 12.8/13).
- t = C/D: 0.77 (−0.42/+0.85) and 0.65 (−0.26/+0.41).
- A/C: 1.34 (−0.59/+1.11), PP → DP only (the PD → DD end is the search region). A/B: 1.04
  (−0.22/+0.27).
- Closure of the DP plane (A = 44 observed, section 1c): 1.12 ± 0.54 from C × s(PP), 1.34 ± 0.65
  from C × A/C(PP), 0.88 ± 0.22 from B × t(PP), 1.04 ± 0.22 from B × A/B(PP), 1.14 ± 0.71 in-plane.

All consistent with 1; the PD → DD values sit below it by 1.2σ (s) and 0.9σ (t). With C = 5 and
D = 7 in DP and 11 and 14 in DD this is the precision the data allow, and it does not distinguish
a transferring factor from one that changes by 30–40%.

*Egamma LJ prompt → displaced* (PP → PD, every region above 100 events):

- s = B/D: 0.69 ± 0.09. Events whose muon LJ fails isolation are less often back-to-back once the
  egamma LJ is displaced.
- t = C/D: 1.29 ± 0.23. A/B: 1.94 ± 0.13.
- A/C: 1.04 ± 0.14. The $|\Delta\phi|$ pass/fail ratio among events with an isolated muon LJ is
  the one quantity that stays.
- Hence κ_PP = 0.84 ± 0.11 and κ_PD = 1.26 ± 0.14 (κ = (A/C)/(B/D)), 2.3σ apart; the PP closure of
  the status report (0.76 ± 0.10) is the same number with its boundaries. The closure of the PD plane from PP factors reads the same
  way: 0.87 (−0.11/+0.10) from C × s(PP), 1.04 (−0.14/+0.13) from C × A/C(PP), 1.63 (−0.24/+0.22)
  from B × t(PP), 1.94 ± 0.13 from B × A/B(PP).

*What this leaves for the search region* (A of the DD plane; B = 50, C = 11, D = 14; section 1c):

- in-plane B C/D (κ = 1): 39 ± 17; B C/D × κ_PP: 33 ± 15; B C/D × κ_PD: 50 ± 22;
- C × A/C from PD or PP (A/C transfers): 75 ± 24 and 72 ± 23; B × A/B from PP: 39 ± 6, from PD:
  76 ± 11;
- C × s from PD: 60 ± 19, from PP: 86 ± 28. These are B C/D times the ratio of the two s values
  and still assume κ_DD = 1.

The 33–39 family assumes κ_DD = 1 or κ_DD = κ_PP; the 72–76 family assumes the search region's
A/C is the PD (or PP) value. Since C × (A/C)_X = (B C/D)_DD × κ_X × s_X/s_DD for any source plane
X, the second family is the in-plane estimate times κ_PD × s_PD/s_DD = 1.26 × 1.52 = 1.9 (from PD)
or κ_PP × s_PP/s_DD = 0.84 × 2.20 = 1.8 (from PP). The factor two between the families is the
change of s from PP to DD, 1.45 across the egamma displacement times 1.52 across the muon
displacement; the first of these is measured to ±13%, the second is the 1.2σ effect above. The
question it poses is whether the s of the DD fail slice, 3.6 (−1.0/+1.5), is the $|\Delta\phi|$
pass/fail ratio of the search region's own population (then the in-plane estimate holds up to
κ_DD) or whether that ratio is the 5.4–7.9 of the isolated muon LJs in PD and PP. No open region
at M ≥ 150 GeV decides this, and the DP-sourced recipes (36–97) carry 50–90% errors. The one
place where the A/C transfer across the muon displacement can be
tested with the egamma LJ displaced is the DD plane at M < 150 GeV: C × A/C(PD) predicts 65 ± 10
for 41 observed (0.63 ± 0.14), the in-plane B C/D 32 ± 9 (1.27 ± 0.42). Low mass has different
kinematics (section 5), so this is not a measurement of the M ≥ 150 GeV case; it is a
counter-example to "A/C of the isolated muon LJ does not depend on the other LJ's displacement"
as a general statement.

### 6.2 The companion plane

- r1 = B/D, the egamma LJ's isolation ratio among muon-LJ failures, is consistent with
  transferring across the muon LJ's displacement where the egamma LJ is prompt, 0.95
  (−0.18/+0.22) PP → DP, and 1.2σ above it where the egamma LJ is displaced, 1.40 (−0.39/+0.54)
  PD → DD; the shapes are compatible in both (χ²/ndf 9.6/13 and 9.6/11). The scan against pixel
  hits (section 2c) shows no step at the displacement boundary: with the egamma LJ prompt, r1 is
  0.31–0.49 in the displaced-muon bins against 0.41–0.54 in the prompt ones; with it displaced,
  1.5–2.2 against 1.5–1.8.
- A/C, the same ratio among muon-LJ passes, does not: 1.99 (−0.45/+0.57) PP → DP. The DP plane's
  κ is 2.01 ± 0.59 and its isolated corner (38 events) is predicted at 19–20 by every PP factor.
  The scan puts the change in the two-pixel-hit bin (31/36 = 0.86 against 0.37–0.52 in the prompt
  bins), which holds 31 of the 38 events; the zero- and one-hit bins hold 5 and 2.
- r0 = C/D and A/B, the muon LJ's ratios across the egamma LJ's displacement: 1.55 ± 0.13 and
  2.25 ± 0.18, with incompatible shapes (χ²/ndf 55.8/13). The scan against lost hits says where:
  with the muon LJ prompt, C/D is 0.72 for prompt egamma LJs (1056/1464), 0.71 for displaced
  electron-type ones (84/119, lost hits 1–9) and 1.28 for the photon-only LJs (399/312); A/B is
  0.69, 1.07–1.19 and 1.62 in the same three classes. The photon-only class, which the lost-hits
  sentinel puts in the displaced category, has no prompt counterpart and drives the integrated
  change; the electron-type displaced LJs transfer C/D but not A/B.
- Against mass (section 5), the PP plane factorises in every bin (κ 0.9–1.2 above 150 GeV) while
  the PD plane does not in the 150–300 GeV bins that hold most of its events (1.37 ± 0.17,
  1.81 ± 0.25) and is consistent with 1 above 300 GeV within 30% errors: the PD non-closure is a
  property of the displaced egamma LJ, not of the mass mixture.

The companion-plane recipes for the hidden isolated DD corner span 20–106 for the same reason
as in the analysis plane.

### 6.3 The mass axis

- The 150 GeV requirement changes what the ABCD axes sort on. In the validation plane s = B/D is
  0.57 ± 0.04 below 150 GeV and 7.9 ± 0.8 above, A/C 1.11 ± 0.09 and 6.6 ± 0.7: low-mass pairs
  are mostly not back-to-back, high-mass pairs mostly are. Nothing measured below 150 GeV
  describes the window above it.
- Inside M ≥ 150 GeV the factors keep moving. In PP, s rises from 4.5 (150–200 GeV) through 8.9,
  21.6 and 15.5 to 31 (600–1000 GeV) while A/C stays at 5.8–7.3, so κ_PP falls from 1.41 ± 0.29
  to 0.82 ± 0.19, 0.27 ± 0.11, 0.37 ± 0.21 and 0.22 ± 0.26. The integrated κ_PP = 0.84 averages
  over this: a validation-plane closure within 5–25% is a statement about the mass mixture of
  PP, not about the plane. In PD, s rises more slowly (4.5, 5.6, 7.2, 6.6, 17) and κ_PD stays at
  1.2–1.5 up to 600 GeV. PP and PD agree at 150–200 GeV (s 4.51 vs 4.46) and separate above (8.9
  vs 5.6, 21.6 vs 7.2, 15.5 vs 6.6): the integrated s change of 0.69 is a high-mass effect, and
  the mass spectra of the regions differ (the fail/fail region D is the softest).
- Bin-by-bin factors: in the DD plane at M ≥ 150 GeV the mass bins hold B/C/D = 14/5/6, 20/1/7,
  8/4/1, 5/1/0 and 3/0/0 events (150–200, 200–300, 300–400, 400–600, 600–1000 GeV). A per-bin
  factor cannot be measured in the search plane,
  and an imported one cannot be validated there; the PP and PD per-bin factors differ from each
  other above 200 GeV, so the choice of source plane would be the dominant systematic of such a
  scheme with 2018 data.

### 6.4 Signal and simulation

- The fail slices that set s and t hold at most 0.2% of the signal's search-region yield (B and D
  of every plane at M ≥ 150 GeV, all seven points); the C regions hold 0.5–1.6% in the analysis
  plane's DD plane and 3–8% in the companion plane's. The isolated corners of DP and PD, open in
  2μ2e, hold 7–17% (DP) and 18–42% (PD) of it, with signal over data at most 0.26 and 0.02 at the
  stored cross sections. A recipe of the form C × factor absorbs the C-region signal into the
  background estimate: 3–13% of the signal yield with the analysis-plane C (its share times s of
  5–8), 5–13% with the companion-plane C, at any cross section.
- The simulation's DD plane holds between zero and 150 weighted events per region, with 50–100%
  uncertainties; its validation-plane κ is 3.2 ± 2.3 (analysis plane) and 1.6 ± 0.7 (companion
  plane). It decides nothing.

### 6.5 4μ

The PP plane is the only one with statistics (264/198/23/28 at M ≥ 150 GeV, κ = 1.62 ± 0.48); the
DP and PD fail slices hold 24/4/1 and 14/7/8 events, the DD plane 1/0/0, so every ratio that
ends in DD is undefined and the PP → DP ratios rest on D = 1. The ratios with both ends
populated are PP → PD: s = 0.25 (−0.12/+0.19) and t = 1.07 (−0.65/+1.01) in the analysis plane,
with fail-slice $|\Delta\phi|$ shapes that differ (χ²/ndf 31.0/13 on 23 events), and
C/D = 0.59 (−0.16/+0.21) in the companion plane. The s change points the same way as in 2μ2e (s
drops when the second LJ is displaced); the companion-plane C/D moves the opposite way to 2μ2e's
1.55 ± 0.13. The isolated corners of DP and PD are hidden here because they hold up to half of
the search-region signal (section 4) against a predicted background of 20 and 12 events
(B × t(PP), section 1c). Beyond this the 2018 data say nothing about 4μ.

### 6.6 What this does and does not establish

Established:

1. The ratios among failures (B/D, C/D) are open in the search plane and the data test them
   directly; the ratios among passes (A/C, A/B) are not. Across the muon LJ's displacement all
   four are consistent with transferring at 20–110% precision; across the egamma LJ's
   displacement s, t and A/B change (0.69 ± 0.09, 1.29 ± 0.23, 1.94 ± 0.13) and A/C does not
   (1.04 ± 0.14).
2. A search-region prediction built from a prompt-plane ratio among failures is the in-plane
   B C/D times the ratio of the two factors and inherits the κ_DD = 1 assumption; one built from
   a ratio among passes assumes that ratio is the same in the search region. The sixteen recipes
   span 33–97 for a region whose in-plane estimate is 39 ± 17.
3. κ of the analysis plane depends on the LJ-pair mass inside the M ≥ 150 GeV window (PP:
   1.4 → 0.2) and on the egamma LJ's displacement (0.84 → 1.26); κ_PP = 1 within 5–25% is a
   property of the integrated PP sample.
4. The egamma-LJ side of the non-transfer is traceable to the photon-only LJs (lost-hits
   sentinel), which have no prompt analogue; the electron-type displaced LJs transfer C/D
   (0.71 vs 0.72) but not A/B (1.1 vs 0.69).

Not established:

- Whether κ_DD is closer to κ_PP (0.84), κ_PD (1.26) or 1. No open region at M ≥ 150 GeV
  constrains it; the low-mass DD plane gives 1.27 ± 0.42 in-plane and rejects the A/C transfer
  from PD (0.63 ± 0.14), with the kinematic caveat of section 5.
- Anything about the muon-LJ side of the DD plane beyond the 20–110% level: the displaced-muon
  fail slices need several times the 2018 statistics.
- Anything in 4μ.

For the group:

- A systematic on the DD estimate should cover at least the PP → PD variation of the factor used
  (30% on s, a factor two on A/B) and the κ spread 0.84–1.26; separating the egamma-LJ classes
  would address the egamma part of that variation, not the muon part.
- Separate the photon-only egamma LJs from the electron-type displaced ones in the ABCD, or at
  least show the estimate per class, and check the signal share per class at the same time.
- The mass dependence of κ inside the window needs either a per-bin treatment with an imported
  factor (and the PP–PD difference as its systematic) or a demonstration that the search region's
  mass mixture matches the plane the factor comes from.
- The regions opened here (the DD fail slices at M ≥ 150 GeV, the low-mass DD corner, the 2μ2e DP
  and PD isolated corners) should be agreed with the group before any number from them
  circulates."""))

cells.append(code(r"""summary = pd.concat([fn(data[t], mass) for t in tl.TOPOLOGIES for mass in ("M>=150", "M<150")
                     for fn in (tl.dphi_transfer_numbers, tl.transfer_numbers)], ignore_index=True)
summary.to_csv("transfer_summary.csv", index=False, float_format="%.4f")
summary"""))

nb = nbf.v4.new_notebook(cells=cells, metadata=KERNEL)
nbf.write(nb, "transfer_factors.ipynb")
print("wrote transfer_factors.ipynb with", len(cells), "cells")
