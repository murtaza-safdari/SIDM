"""Tables, shapes, scans and closure tests for the prompt-to-displaced transfer study.

Everything here is a projection of the 6D cubes (see ``abcd6d_io``).  Conventions:

* planes are named by the displacement state of (object 0, object 1): ``PP`` both prompt
  (the validation plane), ``DP`` object 0 displaced only, ``PD`` object 1 displaced only,
  ``DD`` both displaced (the search plane);
* every region carries |dphi| >= 2 and a mass window (``M>=150`` is the search window);
* isolation regions use the 0.25 working point for "pass" and a buffered "fail" at
  >= 0.5, so the 0.25-0.5 strips belong to no region: A = pass/pass, B = fail/pass,
  C = pass/fail, D = fail/fail (first letter: object 0);
* the per-object factors are pass/fail ratios (odds) of one object's isolation at a fixed
  state of the other: ``r1 = B/D`` (object 1, among object-0 failures), ``ac = A/C``
  (object 1, among object-0 passes), ``r0 = C/D`` and ``ab = A/B`` (object 0, among
  object-1 failures and passes).  The search region needs the odds among passes; the odds
  among failures are the ones the open fail slices can test.
"""
import glob
import hashlib
import os

import numpy as np
import pandas as pd
from scipy import stats

import abcd6d_io as io

TOPOLOGIES = ("2mu2e", "4mu")
PLANES = {"PP": ("prompt", "prompt"), "DP": ("displaced", "prompt"),
          "PD": ("prompt", "displaced"), "DD": ("displaced", "displaced")}
OBJECTS = {"2mu2e": ("mu-LJ", "EGM-LJ"), "4mu": ("leading mu-LJ", "subleading mu-LJ")}
WP = 0.25
FAIL = 0.5
DPHI_MIN = 2.0
MASS_MIN = 150.0
MASS_WINDOWS = {"M>=150": (MASS_MIN, None), "M<150": (None, MASS_MIN), "all M": (None, None)}
# Bins of the LJ-pair mass for the mass scans (merged from the stored edges); also usable as windows.
MASS_BINS = [("<100", (None, 100.0)), ("100-150", (100.0, 150.0)), ("150-200", (150.0, 200.0)),
             ("200-300", (200.0, 300.0)), ("300-400", (300.0, 400.0)), ("400-600", (400.0, 600.0)),
             ("600-1000", (600.0, 1000.0)), (">=1000", (1000.0, None))]
MASS_WINDOWS.update(dict(MASS_BINS))
LUMI_FB = 59.8

# Displacement bins of the factor scans: pixel hits of a muon LJ (displaced <= 2), lost
# hits of an egamma LJ (displaced >= 1; photon-only LJs carry the sentinel 999, the
# overflow bin).
DISP_BINS = {
    "mu": [("0", (None, 0.5)), ("1", (0.5, 1.5)), ("2", (1.5, 2.5)),
           ("3", (2.5, 3.5)), ("4", (3.5, 4.5)), (">=5", (4.5, None))],
    "egm": [("0", (None, 0.5)), ("1", (0.5, 1.5)), ("2-9", (1.5, 9.5)), ("photon-only", (9.5, None))],
}
DISP_PROMPT_BINS = {"mu": ("3", "4", ">=5"), "egm": ("0",)}

BKG_GROUPS = {"QCD": ("QCD_",), "DY": ("DYJetsToLL",), "ttbar": ("TTJets",), "diboson": ("WW", "WZ", "ZZ")}

# The analysis plane: |dphi| between the LJs against the leading muon LJ isolation, with the
# other LJ isolated at the event level (iso1 < OTHER_ISO).  |dphi| passes at >= DPHI_PASS and
# fails below DPHI_FAIL (the 2.0-2.4 strip belongs to no region); iso0 pass/fail as above.
# A = iso0 pass & dphi pass (the search region in the DD plane), B = iso0 fail & dphi pass,
# C = iso0 pass & dphi fail, D = iso0 fail & dphi fail, so that factors() returns
# r1 = B/D = s (dphi pass/fail ratio among iso0 failures), ac = A/C (same among iso0 passes),
# r0 = C/D = t (iso0 pass/fail ratio among dphi failures) and ab = A/B (same among dphi passes).
DPHI_PASS = 2.4
DPHI_FAIL = 2.0
OTHER_ISO = 0.5
DPHI_FACTORS = {"r1": ("B", "D", "|dphi| pass/fail ratio among iso0 failures", "s = B/D"),
                "ac": ("A", "C", "|dphi| pass/fail ratio among iso0 passes", "A/C"),
                "r0": ("C", "D", "iso0 pass/fail ratio among |dphi| failures", "t = C/D"),
                "ab": ("A", "B", "iso0 pass/fail ratio among |dphi| passes", "A/B")}
# Every displacement change between two planes: (prompt plane, displaced plane, object that changes).
DISPLACEMENT_CHANGES = [("PP", "DP", 0), ("PD", "DD", 0), ("PP", "PD", 1), ("DP", "DD", 1)]


def plane_label(topology, plane):
    o0, o1 = OBJECTS[topology]
    s0, s1 = PLANES[plane]
    return f"{o0} {s0}, {o1} {s1}"


# ----------------------------------------------------------------------------- inputs


def local_dir():
    return os.environ.get("ABCD6D_LOCAL", "/uscms_data/d3/murtazas/abcd6d_local")


def _manifest_hash(base):
    with open(os.path.join(base, "MANIFEST.txt"), "rb") as f:
        return hashlib.sha1(f.read()).hexdigest()


def sample_paths(base, kind, names=None):
    paths = sorted(glob.glob(os.path.join(base, kind, "*.coffea")))
    if names is not None:
        wanted = {f"{n}.coffea" for n in names}
        paths = [p for p in paths if os.path.basename(p) in wanted]
        missing = wanted - {os.path.basename(p) for p in paths}
        if missing:
            raise FileNotFoundError(f"{kind}: {sorted(missing)}")
    return paths


def summed_cubes(base, kind):
    """Sum of all files of one kind, per topology.

    Data is always blinded (:func:`abcd6d_io.blind`) before it is cached or returned; the
    cache is keyed on the file manifest and on the hidden box, so a changed box is never
    served from an older cache.
    """
    blinded = kind == "data"
    tag = _manifest_hash(base)
    box = io.blind_signature() if blinded else "none"
    cache = os.path.join(base, "summed", f"{kind}.npz")
    if os.path.exists(cache):
        z = np.load(cache)
        if "blind_box" in z.files and str(z["manifest"]) == tag and str(z["blind_box"]) == box:
            cubes = {t: io.Cube(t, z[f"{t}_sumw"], z[f"{t}_sumw2"]) for t in TOPOLOGIES}
            return {t: io.blind(c) for t, c in cubes.items()} if blinded else cubes     # re-hidden on every load
    total = {}
    for path in sample_paths(base, kind):          # running sum: one cube is ~110 MB
        for t, c in io.load_cubes(path).items():
            total[t] = c if t not in total else io.add_cubes([total[t], c])
    if blinded:
        total = {t: io.blind(c) for t, c in total.items()}
    os.makedirs(os.path.dirname(cache), exist_ok=True)
    arrays = {"manifest": tag, "blind_box": box}
    for t, c in total.items():
        arrays[f"{t}_sumw"], arrays[f"{t}_sumw2"] = c.sumw, c.sumw2
    np.savez(cache, **arrays)
    return total


def _refuse_data(kind):
    if kind == "data":
        raise ValueError("data is only available summed and blinded: use summed_cubes(base, 'data')")


def sample_cubes(base, kind, names=None, topologies=TOPOLOGIES):
    """``{sample: {topology: Cube}}`` for the files of one kind (MC only, never blinded)."""
    _refuse_data(kind)
    return {os.path.basename(p)[: -len(".coffea")]: io.load_cubes(p, topologies)
            for p in sample_paths(base, kind, names)}


def bkg_group(sample):
    for group, prefixes in BKG_GROUPS.items():
        if any(sample.startswith(p) for p in prefixes):
            return group
    return "other"


def group_cubes(base, kind="bkg"):
    """``{group: {topology: Cube}}``: the background samples summed per process group."""
    _refuse_data(kind)
    groups = {}
    for path in sample_paths(base, kind):
        g = bkg_group(os.path.basename(path)[: -len(".coffea")])
        for t, c in io.load_cubes(path).items():
            groups.setdefault(g, {})
            groups[g][t] = c if t not in groups[g] else io.add_cubes([groups[g][t], c])
    return groups


# ----------------------------------------------------------------------------- statistics


def _unit_weights(*pairs):
    return all(w >= 0 and np.isclose(w2, w) for w, w2 in pairs)


def clopper_pearson(k, n, cl=0.6827):
    a = (1 - cl) / 2
    lo = stats.beta.ppf(a, k, n - k + 1) if k > 0 else 0.0
    hi = stats.beta.ppf(1 - a, k + 1, n - k) if k < n else 1.0
    return lo, hi


def ratio(num, den):
    """num/den of two disjoint counts as (value, minus, plus) at 68.27%.

    Unit-weight counts (data) get the Clopper-Pearson interval of num/(num+den) mapped to
    the ratio; weighted counts get symmetric propagation of sumw2.
    """
    (n, n2), (d, d2) = num, den
    if np.isnan(n) or np.isnan(d) or d <= 0:
        return np.nan, np.nan, np.nan
    value = n / d
    if _unit_weights(num, den):
        lo, hi = clopper_pearson(n, n + d)
        return value, value - lo / (1 - lo), (hi / (1 - hi) if hi < 1 else np.inf) - value
    rel = np.sqrt((n2 / n**2 if n > 0 else 0.0) + d2 / d**2)
    return value, value * rel, value * rel


def ratio_of_ratios(r_num, r_den):
    """(value, minus, plus) of two independent ratio results, errors added in quadrature."""
    if not (np.isfinite(r_num[0]) and np.isfinite(r_den[0])) or r_den[0] <= 0:
        return np.nan, np.nan, np.nan
    v = r_num[0] / r_den[0]
    if r_num[0] <= 0:
        return 0.0, 0.0, r_num[2] / r_den[0]
    minus = v * np.hypot(r_num[1] / r_num[0], r_den[2] / r_den[0])
    plus = v * np.hypot(r_num[2] / r_num[0], r_den[1] / r_den[0])
    return v, min(minus, v), plus      # the quadrature sum can overshoot zero for wide intervals


def fmt(r, digits=3):
    v, m, p = r
    if np.isnan(v):
        return "n/a"
    if np.isinf(p):
        return f"{v:.{digits}f} -{m:.{digits}f} +inf"
    if np.isclose(m, p, rtol=0.05):
        return f"{v:.{digits}f} ± {max(m, p):.{digits}f}"
    return f"{v:.{digits}f} -{m:.{digits}f} +{p:.{digits}f}"


def fmt_count(pair):
    w, w2 = pair
    if np.isnan(w):
        return "blind"
    if np.isclose(w, w2):
        return f"{w:.0f}"
    return f"{w:.1f} ± {np.sqrt(w2):.1f}"


def n_eff(pair):
    w, w2 = pair
    return w**2 / w2 if w2 > 0 else 0.0


# ----------------------------------------------------------------------------- regions


def selection(topology, plane, mass="M>=150", dphi=True):
    sel = dict(io.plane(topology, *PLANES[plane]))
    sel["ljlj_mass"] = MASS_WINDOWS[mass]
    if dphi:
        sel["abs_dphi"] = (DPHI_MIN, None)
    return sel


def regions(cube, plane, mass="M>=150", wp=WP, fail=FAIL, **extra):
    sel = selection(cube.topology, plane, mass)
    sel.update(extra)
    return io.regions(cube, wp, wp, fail0=fail, fail1=fail, **sel)


# Pass/fail ratios: name -> (numerator, denominator, object measured, object held, held state)
FACTORS = {"r1": ("B", "D", 1, 0, "fail"), "ac": ("A", "C", 1, 0, "pass"),
           "r0": ("C", "D", 0, 1, "fail"), "ab": ("A", "B", 0, 1, "pass")}


def factor_label(topology, name):
    num, den, measured, held, state = FACTORS[name]
    return f"{name} = {num}/{den} ({OBJECTS[topology][measured]} pass/fail ratio, {OBJECTS[topology][held]} {state})"


def factors(reg):
    """The four pass/fail ratios, kappa = A D / (B C) and the in-plane prediction B C / D."""
    (k, ke), (pred, pe) = io.kappa(reg)
    out = {name: ratio(reg[num], reg[den]) for name, (num, den, *_) in FACTORS.items()}
    out.update(kappa=(k, ke, ke), pred=(pred, pe, pe))
    return out


def region_table(cube, masses=("M>=150", "M<150"), planes=PLANES, wp=WP, fail=FAIL):
    rows = []
    for mass in masses:
        for plane in planes:
            reg = regions(cube, plane, mass, wp, fail)
            f = factors(reg)
            rows.append({"mass": mass, "plane": plane, **{r: fmt_count(reg[r]) for r in "ABCD"},
                         "r1 = B/D": fmt(f["r1"]), "ac = A/C": fmt(f["ac"]),
                         "r0 = C/D": fmt(f["r0"]), "ab = A/B": fmt(f["ab"]),
                         "B C / D": fmt(f["pred"], 1), "kappa": fmt(f["kappa"]),
                         "min N_eff": f"{min(n_eff(reg[r]) for r in 'ABCD' if not np.isnan(reg[r][0])):.0f}"})
    return pd.DataFrame(rows)


# Each factor compared between the two planes that differ only in the displacement of the
# object it holds fixed: (factor, prompt plane, displaced plane).
TRANSFER_PAIRS = [("r1", "PP", "DP"), ("r1", "PD", "DD"), ("ac", "PP", "DP"), ("ac", "PD", "DD"),
                  ("r0", "PP", "PD"), ("r0", "DP", "DD"), ("ab", "PP", "PD"), ("ab", "DP", "DD")]


def transfer_table(cube, mass="M>=150", wp=WP, fail=FAIL):
    """Each pass/fail ratio compared between the two planes that differ only in the other object."""
    reg = {p: regions(cube, p, mass, wp, fail) for p in PLANES}
    f = {p: factors(reg[p]) for p in PLANES}
    rows = []
    for name, p_prompt, p_disp in TRANSFER_PAIRS:
        num, den, _, held, _ = FACTORS[name]
        rows.append({"factor": factor_label(cube.topology, name),
                     "varied": f"{OBJECTS[cube.topology][held]}: {p_prompt} -> {p_disp}",
                     "counts, prompt": f"{num}={fmt_count(reg[p_prompt][num])}, {den}={fmt_count(reg[p_prompt][den])}",
                     "value, prompt": fmt(f[p_prompt][name]),
                     "counts, displaced": f"{num}={fmt_count(reg[p_disp][num])}, {den}={fmt_count(reg[p_disp][den])}",
                     "value, displaced": fmt(f[p_disp][name]),
                     "displaced / prompt": fmt(ratio_of_ratios(f[p_disp][name], f[p_prompt][name]))})
    return pd.DataFrame(rows)


def transfer_numbers(cube, mass="M>=150", wp=WP, fail=FAIL):
    """Numeric form of :func:`transfer_table` (one row per factor and plane pair)."""
    reg = {p: regions(cube, p, mass, wp, fail) for p in PLANES}
    f = {p: factors(reg[p]) for p in PLANES}
    rows = []
    for name, p_prompt, p_disp in TRANSFER_PAIRS:
        rr = ratio_of_ratios(f[p_disp][name], f[p_prompt][name])
        rows.append({"topology": cube.topology, "plane kind": "iso x iso", "mass": mass, "factor": name,
                     "prompt_plane": p_prompt, "displaced_plane": p_disp, "prompt_value": f[p_prompt][name][0],
                     "prompt_minus": f[p_prompt][name][1], "prompt_plus": f[p_prompt][name][2],
                     "displaced_value": f[p_disp][name][0], "displaced_minus": f[p_disp][name][1],
                     "displaced_plus": f[p_disp][name][2], "ratio": rr[0], "ratio_minus": rr[1], "ratio_plus": rr[2]})
    return pd.DataFrame(rows)


# ----------------------------------------------------------------------------- shapes


def shape(cube, variable, **sel):
    """Normalised distribution of ``variable`` (flow bins included) under ``sel``."""
    sumw, sumw2 = io.profile(cube, variable, **sel)
    total = np.nansum(sumw)
    if not total > 0:
        return np.full_like(sumw, np.nan), np.full_like(sumw, np.nan), sumw
    return sumw / total, np.sqrt(sumw2) / total, sumw


def shape_chi2(n1, n2):
    """Two-sample chi2 of two unweighted histograms having the same shape.

    Pearson's statistic with the expected bin contents pooled from both samples,
    sum (N2 n1 - N1 n2)^2 / (N1 N2 (n1 + n2)) over the non-empty bins, with ndf = bins - 1.
    (Comparing unit-normalised shapes with per-sample errors is miscalibrated: a bin that is
    empty in the smaller sample then contributes its full count in the larger one.)
    """
    n1, n2 = np.asarray(n1, float), np.asarray(n2, float)
    if not (np.allclose(n1, np.round(n1)) and np.allclose(n2, np.round(n2))):
        raise ValueError("shape_chi2 is for unweighted counts")
    tot1, tot2 = n1.sum(), n2.sum()
    mask = (n1 + n2) > 0
    chi2 = float(np.sum((tot2 * n1[mask] - tot1 * n2[mask]) ** 2 / (tot1 * tot2 * (n1[mask] + n2[mask]))))
    return chi2, int(mask.sum()) - 1


def fail_slice_shapes(cube, factor, mass="M>=150", fail=FAIL):
    """Isolation shape of the measured object inside the other object's fail slice, per plane.

    ``r1``: iso1 shape where iso0 >= fail; ``r0``: iso0 shape where iso1 >= fail.
    """
    var, other = ("iso1", "iso0") if factor == "r1" else ("iso0", "iso1")
    out = {}
    for plane in PLANES:
        sel = selection(cube.topology, plane, mass)
        sel[other] = (fail, None)
        out[plane] = shape(cube, var, **sel)
    return var, out


def shape_table(cube, mass="M>=150"):
    rows = []
    for factor, pairs in (("r1", (("PP", "DP"), ("PD", "DD"))), ("r0", (("PP", "PD"), ("DP", "DD")))):
        var, shapes = fail_slice_shapes(cube, factor, mass)
        for a, b in pairs:
            chi2, ndf = shape_chi2(shapes[a][2], shapes[b][2])
            rows.append({"shape": f"{var} where {'iso0' if var == 'iso1' else 'iso1'} >= {FAIL}",
                         "planes": f"{a} vs {b}",
                         "events": f"{np.nansum(shapes[a][2]):.0f} vs {np.nansum(shapes[b][2]):.0f}",
                         "chi2/ndf": f"{chi2:.1f}/{ndf}", "p": f"{stats.chi2.sf(chi2, ndf):.3f}"})
    return pd.DataFrame(rows)


# ----------------------------------------------------------------------------- scans


def disp_kind(topology, obj):
    return "egm" if topology == "2mu2e" and obj == 1 else "mu"


def factor_scan(cube, factor, other_state, mass="M>=150", wp=WP, fail=FAIL):
    """A pass/fail ratio against the displacement bin of the object it holds fixed.

    ``r1`` and ``ac`` (object-1 ratios) are scanned in bins of ``disp0`` with object 1 held in
    ``other_state``; ``r0`` and ``ab`` in bins of ``disp1`` with object 0 held.
    """
    num, den, measured, obj, _ = FACTORS[factor]
    rows = []
    for label, rng in DISP_BINS[disp_kind(cube.topology, obj)]:
        sel = {f"disp{obj}": rng, f"disp{measured}": io.displacement(cube.topology, measured, other_state),
               "ljlj_mass": MASS_WINDOWS[mass], "abs_dphi": (DPHI_MIN, None)}
        reg = io.regions(cube, wp, wp, fail0=fail, fail1=fail, **sel)
        r = ratio(reg[num], reg[den])
        rows.append({"bin": label, num: reg[num][0], den: reg[den][0], "value": r[0], "minus": r[1], "plus": r[2]})
    return pd.DataFrame(rows)


# ----------------------------------------------------------------------------- closures


POISSON_UPPER_ZERO = 1.84   # upper edge of the central 68.27% interval for an observed 0


def _pred(count, factor):
    """count x factor with the count's Poisson error and the factor's interval.

    An observed count of zero gives 0 with the 68% upper limit (1.84 x factor) as the
    upward error.
    """
    (c, c2) = count
    v, m, p = factor
    if np.isnan(c) or np.isnan(v):
        return np.nan, np.nan, np.nan
    if c <= 0:
        return 0.0, 0.0, POISSON_UPPER_ZERO * v
    if v <= 0:
        return 0.0, 0.0, c * p
    rel_c = np.sqrt(c2) / c
    value = c * v
    return value, min(value * np.hypot(rel_c, m / v), value), value * np.hypot(rel_c, p / v)


def _recipe(topology, name, source):
    """Reviewer-facing label of a prediction 'count x factor(source plane)'."""
    num, den, measured, held, state = FACTORS[name]
    count = "C" if measured == 1 else "B"
    among = {"fail": "failures", "pass": "passes"}[state]
    return (f"{count} x {name}({source}): {OBJECTS[topology][measured]} pass/fail ratio among "
            f"{OBJECTS[topology][held]} {among}, taken across {OBJECTS[topology][held]} displacement")


def closure_table(cube, wp=WP, fail=FAIL):
    """Predictions of A where A is open, each from one factor taken from another plane.

    Per open plane: the odds among failures (``r1``/``r0``, what the fail slices test) and
    the odds among passes (``ac``/``ab``, what the search region needs), each taken from the
    plane that differs only in the held object's displacement, plus the in-plane B C / D.
    """
    t = cube.topology
    reg = {(p, m): regions(cube, p, m, wp, fail) for p in PLANES for m in ("M>=150", "M<150")}
    f = {k: factors(v) for k, v in reg.items()}
    tests = []
    for plane, mass, source in (("DP", "M>=150", "PP"), ("PD", "M>=150", "PP"), ("DD", "M<150", None)):
        names = ("r1", "ac") if plane == "DP" else ("r0", "ab") if plane == "PD" else ("r1", "ac", "r0", "ab")
        for name in names:
            src = source or ("PD" if name in ("r1", "ac") else "DP")
            tests.append((plane, mass, _recipe(t, name, src), "C" if name in ("r1", "ac") else "B", f[(src, mass)][name]))
        tests.append((plane, mass, "B C / D in the plane (kappa = 1)", None, None))
    tests.append(("DD", "M<150", "B C / D x kappa(PP, M<150)", "kappa", f[("PP", "M<150")]["kappa"]))
    rows = []
    for plane, mass, label, count, factor in tests:
        r = reg[(plane, mass)]
        if count is None:
            pred = f[(plane, mass)]["pred"]
        elif count == "kappa":
            pred = _pred((f[(plane, mass)]["pred"][0], f[(plane, mass)]["pred"][1] ** 2), factor)
        else:
            pred = _pred(r[count], factor)
        obs = r["A"]
        closure = ratio_of_ratios((obs[0], np.sqrt(obs[1]), np.sqrt(obs[1])), pred) if obs[0] > 0 else (np.nan,) * 3
        rows.append({"plane": plane, "mass": mass, "prediction of A": label, "A observed": fmt_count(obs),
                     "A predicted": fmt(pred, 1), "observed / predicted": fmt(closure, 2)})
    return pd.DataFrame(rows)


def sr_prediction_table(cube, wp=WP, fail=FAIL):
    """What each recipe gives for A of the DD plane at M >= 150; A itself stays hidden.

    Every row states its assumption.  ``C x r1(PD)`` equals the in-plane B C / D times
    r1(PD)/r1(DD) and so assumes kappa(DD) = 1 as well; the ``kappa`` rows show the spread
    from taking kappa of another plane instead of 1.
    """
    t = cube.topology
    reg = {(p, m): regions(cube, p, m, wp, fail) for p in PLANES for m in ("M>=150", "M<150")}
    f = {k: factors(v) for k, v in reg.items()}
    dd, fdd = reg[("DD", "M>=150")], f[("DD", "M>=150")]
    pred_pair = (fdd["pred"][0], fdd["pred"][1] ** 2)
    rows = [("B C / D in the plane; assumes kappa(DD) = 1", fdd["pred"])]
    for name in ("r1", "r0"):
        src = "PD" if name == "r1" else "DP"
        rows.append((f"{_recipe(t, name, src)}; = B C / D x {name}({src})/{name}(DD), assumes kappa(DD) = 1",
                     _pred(dd["C" if name == "r1" else "B"], f[(src, "M>=150")][name])))
    for name in ("ac", "ab"):
        src = "PD" if name == "ac" else "DP"
        rows.append((f"{_recipe(t, name, src)}; assumes {name} transfers across that displacement",
                     _pred(dd["C" if name == "ac" else "B"], f[(src, "M>=150")][name])))
    for plane in ("PP", "DP", "PD"):
        rows.append((f"B C / D x kappa({plane}); assumes kappa(DD) = kappa({plane})",
                     _pred(pred_pair, f[(plane, "M>=150")]["kappa"])))
    return pd.DataFrame([{"prediction of A (DD, M>=150)": k, "value": fmt(v, 1)} for k, v in rows])


# ----------------------------------------------------------------------------- dphi plane


def dphi_regions(cube, plane, mass="M>=150", wp=WP, fail=FAIL, dphi_pass=DPHI_PASS, dphi_fail=DPHI_FAIL,
                 other_iso=OTHER_ISO):
    """A/B/C/D of the |dphi| x iso0 plane (see the module constants for the region definitions)."""
    base = dict(io.plane(cube.topology, *PLANES[plane]))
    base["ljlj_mass"] = MASS_WINDOWS[mass]
    base["iso1"] = (None, other_iso)
    return {"A": io.project(cube, iso0=(None, wp), abs_dphi=(dphi_pass, None), **base),
            "B": io.project(cube, iso0=(fail, None), abs_dphi=(dphi_pass, None), **base),
            "C": io.project(cube, iso0=(None, wp), abs_dphi=(None, dphi_fail), **base),
            "D": io.project(cube, iso0=(fail, None), abs_dphi=(None, dphi_fail), **base)}


def dphi_table(cube, masses=("M>=150", "M<150")):
    rows = []
    for mass in masses:
        for plane in PLANES:
            reg = dphi_regions(cube, plane, mass)
            f = factors(reg)
            rows.append({"mass": mass, "plane": plane, **{r: fmt_count(reg[r]) for r in "ABCD"},
                         **{DPHI_FACTORS[n][3]: fmt(f[n]) for n in DPHI_FACTORS},
                         "B C / D": fmt(f["pred"], 1), "kappa": fmt(f["kappa"])})
    return pd.DataFrame(rows)


def dphi_transfer_table(cube, mass="M>=150"):
    """Each pass/fail ratio of the dphi plane across each single change of displacement."""
    t = cube.topology
    reg = {p: dphi_regions(cube, p, mass) for p in PLANES}
    f = {p: factors(reg[p]) for p in PLANES}
    rows = []
    for name, (num, den, meaning, short) in DPHI_FACTORS.items():
        for p_prompt, p_disp, obj in DISPLACEMENT_CHANGES:
            rows.append({"factor": f"{short}: {meaning}", "varied": f"{OBJECTS[t][obj]}: {p_prompt} -> {p_disp}",
                         "counts, prompt": f"{num}={fmt_count(reg[p_prompt][num])}, {den}={fmt_count(reg[p_prompt][den])}",
                         "value, prompt": fmt(f[p_prompt][name]),
                         "counts, displaced": f"{num}={fmt_count(reg[p_disp][num])}, {den}={fmt_count(reg[p_disp][den])}",
                         "value, displaced": fmt(f[p_disp][name]),
                         "displaced / prompt": fmt(ratio_of_ratios(f[p_disp][name], f[p_prompt][name]))})
    return pd.DataFrame(rows)


def dphi_transfer_numbers(cube, mass="M>=150"):
    """Numeric form of :func:`dphi_transfer_table`."""
    reg = {p: dphi_regions(cube, p, mass) for p in PLANES}
    f = {p: factors(reg[p]) for p in PLANES}
    rows = []
    for name in DPHI_FACTORS:
        for p_prompt, p_disp, _ in DISPLACEMENT_CHANGES:
            rr = ratio_of_ratios(f[p_disp][name], f[p_prompt][name])
            rows.append({"topology": cube.topology, "plane kind": "dphi x iso0", "mass": mass, "factor": name,
                         "prompt_plane": p_prompt, "displaced_plane": p_disp,
                         "prompt_value": f[p_prompt][name][0], "prompt_minus": f[p_prompt][name][1],
                         "prompt_plus": f[p_prompt][name][2], "displaced_value": f[p_disp][name][0],
                         "displaced_minus": f[p_disp][name][1], "displaced_plus": f[p_disp][name][2],
                         "ratio": rr[0], "ratio_minus": rr[1], "ratio_plus": rr[2]})
    return pd.DataFrame(rows)


def _dphi_recipe(name, source):
    _, _, meaning, short = DPHI_FACTORS[name]
    count = "C" if name in ("r1", "ac") else "B"
    return f"{count} x {short.split(' = ')[0]}({source}): {meaning}, taken from {source}"


def dphi_closure_table(cube):
    """Predictions of A of the dphi plane where A is open, from each factor of a plane one displacement away."""
    t = cube.topology
    reg = {(p, m): dphi_regions(cube, p, m) for p in PLANES for m in ("M>=150", "M<150")}
    f = {k: factors(v) for k, v in reg.items()}
    targets = [("DP", "M>=150", ("PP",)), ("PD", "M>=150", ("PP",)), ("DD", "M<150", ("PD", "DP"))]
    rows = []
    for plane, mass, sources in targets:
        r = reg[(plane, mass)]
        obs = r["A"]
        tests = []
        for src in sources:
            for name in DPHI_FACTORS:
                count = "C" if name in ("r1", "ac") else "B"
                tests.append((_dphi_recipe(name, src), _pred(r[count], f[(src, mass)][name])))
        tests.append(("B C / D in the plane (kappa = 1)", f[(plane, mass)]["pred"]))
        if plane == "DD":
            tests.append(("B C / D x kappa(PP, M<150)",
                          _pred((f[(plane, mass)]["pred"][0], f[(plane, mass)]["pred"][1] ** 2), f[("PP", mass)]["kappa"])))
        for label, pred in tests:
            closure = ratio_of_ratios((obs[0], np.sqrt(obs[1]), np.sqrt(obs[1])), pred) if obs[0] > 0 else (np.nan,) * 3
            rows.append({"plane": plane, "mass": mass, "prediction of A": label, "A observed": fmt_count(obs),
                         "A predicted": fmt(pred, 1), "observed / predicted": fmt(closure, 2)})
    return pd.DataFrame(rows)


def dphi_sr_prediction_table(cube):
    """What each recipe gives for the search region (A of the dphi plane, DD, M >= 150), with its assumption."""
    reg = {p: dphi_regions(cube, p, "M>=150") for p in PLANES}
    f = {p: factors(reg[p]) for p in PLANES}
    dd, fdd = reg["DD"], f["DD"]
    pred_pair = (fdd["pred"][0], fdd["pred"][1] ** 2)
    rows = [("B C / D in the plane; assumes kappa(DD) = 1", fdd["pred"])]
    for name in ("r1", "r0"):
        short = DPHI_FACTORS[name][3].split(" = ")[0]
        for src in ("PD", "DP", "PP"):
            rows.append((f"{_dphi_recipe(name, src)}; = B C / D x {short}({src})/{short}(DD), assumes kappa(DD) = 1",
                         _pred(dd["C" if name == "r1" else "B"], f[src][name])))
    for name in ("ac", "ab"):
        for src in ("PD", "DP", "PP"):
            rows.append((f"{_dphi_recipe(name, src)}; assumes {DPHI_FACTORS[name][3]} is the same in DD",
                         _pred(dd["C" if name == "ac" else "B"], f[src][name])))
    for src in ("PP", "DP", "PD"):
        rows.append((f"B C / D x kappa({src}); assumes kappa(DD) = kappa({src})", _pred(pred_pair, f[src]["kappa"])))
    return pd.DataFrame([{"prediction of the search region A": k, "value": fmt(v, 1)} for k, v in rows])


def dphi_shapes(cube, mass="M>=150", which="fail", wp=WP, fail=FAIL, other_iso=OTHER_ISO):
    """|dphi| shape per plane among iso0 failures (``which="fail"``) or iso0 passes (``"pass"``), iso1 < other_iso."""
    out = {}
    for plane in PLANES:
        sel = dict(io.plane(cube.topology, *PLANES[plane]))
        sel.update(ljlj_mass=MASS_WINDOWS[mass], iso1=(None, other_iso),
                   iso0=(fail, None) if which == "fail" else (None, wp))
        out[plane] = shape(cube, "abs_dphi", **sel)
    return out


def dphi_shape_table(cube, mass="M>=150"):
    rows = []
    for which, pairs in (("fail", (("PP", "DP"), ("PP", "PD"), ("PD", "DD"), ("DP", "DD"))),
                         ("pass", (("PP", "DP"), ("PP", "PD")))):
        shapes = dphi_shapes(cube, mass, which)
        for a, b in pairs:
            na, nb = shapes[a][2], shapes[b][2]
            if np.isnan(na).any() or np.isnan(nb).any():
                continue
            chi2, ndf = shape_chi2(na, nb)
            rows.append({"shape": f"|dphi| where iso0 {'>= ' + str(FAIL) if which == 'fail' else '< ' + str(WP)}, iso1 < {OTHER_ISO}",
                         "planes": f"{a} vs {b}", "events": f"{np.nansum(na):.0f} vs {np.nansum(nb):.0f}",
                         "chi2/ndf": f"{chi2:.1f}/{ndf}", "p": f"{stats.chi2.sf(chi2, ndf):.3f}"})
    return pd.DataFrame(rows)


# ----------------------------------------------------------------------------- LJ-pair mass


def mass_scan(cube, plane, kind="dphi x iso0", wp=WP, fail=FAIL):
    """Region counts, the four ratios and kappa of one plane in bins of the LJ-pair mass."""
    project_regions = regions if kind == "iso x iso" else dphi_regions
    names = DPHI_FACTORS if kind == "dphi x iso0" else {n: (None, None, None, factor_label(cube.topology, n).split(" (")[0])
                                                         for n in FACTORS}
    rows = []
    for label, _ in MASS_BINS:
        reg = project_regions(cube, plane, label, wp, fail)
        f = factors(reg)
        rows.append({"mass bin": label, **{r: fmt_count(reg[r]) for r in "ABCD"},
                     **{names[n][3]: fmt(f[n], 2) for n in names}, "B C / D": fmt(f["pred"], 1), "kappa": fmt(f["kappa"], 2)})
    return pd.DataFrame(rows)


def mass_factor_scan(cube, factor, plane, kind="dphi x iso0", wp=WP, fail=FAIL):
    """One ratio of one plane against the mass bin, as a numeric frame (value, minus, plus, counts)."""
    project_regions = regions if kind == "iso x iso" else dphi_regions
    num, den = ("A", "A") if factor == "kappa" else (DPHI_FACTORS if kind == "dphi x iso0" else FACTORS)[factor][:2]
    rows = []
    for label, _ in MASS_BINS:
        reg = project_regions(cube, plane, label, wp, fail)
        v, m, p = factors(reg)["kappa"] if factor == "kappa" else ratio(reg[num], reg[den])
        rows.append({"bin": label, "num": reg[num][0], "den": reg[den][0], "value": v, "minus": m, "plus": p})
    return pd.DataFrame(rows)


def mass_shapes(cube, plane, kind="dphi x iso0", mass="all M", wp=WP, fail=FAIL):
    """Normalised LJ-pair mass distribution of each region of one plane (NaN where hidden)."""
    t = cube.topology
    base = dict(io.plane(t, *PLANES[plane]))
    base["ljlj_mass"] = MASS_WINDOWS[mass]
    if kind == "dphi x iso0":
        base["iso1"] = (None, OTHER_ISO)
        sel = {"A": dict(iso0=(None, wp), abs_dphi=(DPHI_PASS, None)), "B": dict(iso0=(fail, None), abs_dphi=(DPHI_PASS, None)),
               "C": dict(iso0=(None, wp), abs_dphi=(None, DPHI_FAIL)), "D": dict(iso0=(fail, None), abs_dphi=(None, DPHI_FAIL))}
    else:
        base["abs_dphi"] = (DPHI_MIN, None)
        sel = {"A": dict(iso0=(None, wp), iso1=(None, wp)), "B": dict(iso0=(fail, None), iso1=(None, wp)),
               "C": dict(iso0=(None, wp), iso1=(fail, None)), "D": dict(iso0=(fail, None), iso1=(fail, None))}
    return {r: shape(cube, "ljlj_mass", **base, **s) for r, s in sel.items()}


# ----------------------------------------------------------------------------- MC and signal


def composition_table(groups, topology, plane, mass="M>=150", kind="iso x iso", wp=WP, fail=FAIL):
    """Weighted yield per background group and region of one plane, with the total's N_eff."""
    project_regions = regions if kind == "iso x iso" else dphi_regions
    rows = []
    totals = {r: [0.0, 0.0] for r in "ABCD"}
    for g in sorted(groups):
        reg = project_regions(groups[g][topology], plane, mass, wp, fail)
        for r in "ABCD":
            totals[r][0] += reg[r][0]
            totals[r][1] += reg[r][1]
        rows.append({"component": g, **{r: f"{reg[r][0]:.1f}" for r in "ABCD"}})
    rows.append({"component": "total", **{r: fmt_count(tuple(totals[r])) for r in "ABCD"}})
    rows.append({"component": "N_eff", **{r: f"{n_eff(tuple(totals[r])):.0f}" for r in "ABCD"}})
    return pd.DataFrame(rows)


def search_region(cube, wp=WP):
    """The search region: A of the dphi plane in the DD plane at M >= 150 (iso0 < wp, iso1 < 0.5, |dphi| >= 2.4)."""
    return dphi_regions(cube, "DD", "M>=150", wp)["A"]


def contamination_table(signals, data, topology, wp=WP, fail=FAIL):
    """Signal in every region, as a fraction of that signal's own search-region yield.

    Columns per signal: percentage of its search-region yield that falls in the region;
    ``data`` is the observed count ("blind" in the hidden regions, whose signal share is what
    justifies hiding them); ``max signal`` the largest signal count among the points at the
    stored normalisation and ``max S/N`` the largest signal over data count ("n/a" when the
    region is hidden or holds no data event).
    """
    rows = []
    sr = {name: search_region(c[topology], wp)[0] for name, c in signals.items()}
    cases = [("M>=150", plane, "dphi x iso0") for plane in PLANES]
    cases += [(mass, plane, "iso x iso") for mass in ("M>=150", "M<150") for plane in PLANES]
    for mass, plane, kind in cases:
        project_regions = regions if kind == "iso x iso" else dphi_regions
        dreg = project_regions(data, plane, mass, wp, fail)
        sreg = {name: project_regions(c[topology], plane, mass, wp, fail) for name, c in signals.items()}
        for r in "ABCD":
            row = {"plane kind": kind, "mass": mass, "plane": plane, "region": r, "data": fmt_count(dreg[r])}
            best_s = 0.0
            for name in signals:
                s = sreg[name][r][0]
                row[name] = f"{100 * s / sr[name]:.1f}" if sr[name] > 0 else "n/a"
                best_s = max(best_s, s)
            row["max signal"] = f"{best_s:.1f}"
            row["max S/N"] = f"{best_s / dreg[r][0]:.3f}" if dreg[r][0] > 0 else "n/a"     # NaN > 0 is False
            rows.append(row)
    return pd.DataFrame(rows)


def signal_sr_yields(signals, topology, wp=WP):
    rows = []
    for name, c in signals.items():
        a = search_region(c[topology], wp)
        rows.append({"signal": name, "search region (dphi plane A, DD, M>=150), stored normalisation": fmt_count(a),
                     "N_eff": f"{n_eff(a):.0f}"})
    return pd.DataFrame(rows)
