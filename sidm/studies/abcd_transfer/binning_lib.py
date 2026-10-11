"""The 2mu2e B-route estimate per bin of the LJ-pair mass, the fit of A/B against the mass it can
borrow from, and a toy expected limit to compare binnings.

The estimate of :mod:`estimate_lib` for A of the DD plane at M >= 150 GeV is the B route: B of DD
per DSA class of the muon LJ (``dsa0``, ``dsa1+``) times A/B of the PD plane of the same class,
summed over the classes.  A/B of PD falls with the LJ-pair mass, and the one-bin form weights it
with PD's mass distribution.  :func:`binned_routes` takes the same route per bin of a binning of
M >= 150 GeV (:data:`BINNINGS`): B of DD in the bin times A/B of PD measured in the bin, or A/B of
PD from a fit against the mass (:func:`fit_ab` on the points of :func:`ab_points`) evaluated at the
mean mass of DD's B events in the bin.  Per bin and class the decision rule takes the measured
factor when PD measures it to ``max_rel_err`` (20%) and the fitted one otherwise, with
|measured - fitted| as a shape systematic.

Blinding: A of DD above 150 GeV is NaN in the data cubes (``abcd6d_io.blind``), and so is every
mass sub-range of it.  The estimate reads B of DD and A and B of PD only; the tables print "blind"
where A of a hidden region would stand, and its A/B and kappa stay NaN.

:func:`expected_limit` is a toy for comparing binnings with each other: the median expected CLs
limit of a counting experiment in the asymptotic approximation, with the estimate as the
background.  :func:`decision_table` checks each binning against the count requirement of the
decision rule and :func:`rule_table` applies the rule to the toy limits.
"""
import dataclasses

import numpy as np
import pandas as pd
from scipy import optimize, stats

import abcd6d_io as io
import estimate_lib as el
import plane_search_lib as ps
import transfer_lib as tl

BINNINGS = {
    "one bin": [(">=150", (150.0, None))],
    "two windows": [("150-300", (150.0, 300.0)), (">=300", (300.0, None))],
    "cube bins": [("150-200", (150.0, 200.0)), ("200-300", (200.0, 300.0)), ("300-400", (300.0, 400.0)),
                  ("400-600", (400.0, 600.0)), (">=600", (600.0, None))],
    "three bins": [("150-300", (150.0, 300.0)), ("300-600", (300.0, 600.0)), (">=600", (600.0, None))],
}
# The bins A/B is measured and fitted in against the mass: the abcd8d cube's own LJ-pair mass bins
# above 150 GeV (the last one holds 1000-2500 GeV and the overflow).
FIT_BINS = [("150-200", (150.0, 200.0)), ("200-300", (200.0, 300.0)), ("300-400", (300.0, 400.0)),
            ("400-600", (400.0, 600.0)), ("600-800", (600.0, 800.0)), ("800-1000", (800.0, 1000.0)),
            (">=1000", (1000.0, None))]
CLASSES = ("dsa0", "dsa1+")
FORMS = ("linear", "exponential")
ZERO_UPPER = tl.POISSON_UPPER_ZERO    # 68% upper limit on a count of zero: a lower limit on A/B where B = 0
RATE_FLOOR = 1e-6       # the smallest A/B a fit form may take at a bin (a straight line can cross zero)
FACTORS = ("measured", "fitted", "selected")
MAX_REL_ERR = 0.20      # the measured A/B is used when the larger side of its 68% interval is <= 20% of it
MIN_B = 5               # a search bin's estimate rests on at least this many B events of DD ...
SMALL = 3.0             # ... or is below this many events
SUM = "sum"             # class label of the rows summed over the classes
TOTAL = "sum over bins"

# The signal points of estimate.ipynb.  The binning is chosen for the 500 and 800 GeV points
# (DECISION_POINTS); the 200 and 1000 GeV points are shown alongside.
SIGNAL_POINTS = {"2mu2e": ["2Mu2E_200GeV_0p25GeV_1p0mm", "2Mu2E_500GeV_0p25GeV_0p4mm", "2Mu2E_800GeV_0p25GeV_0p25mm",
                           "2Mu2E_1000GeV_0p25GeV_0p2mm", "2Mu2E_800GeV_1p2GeV_1p2mm", "2Mu2E_1000GeV_1p2GeV_0p96mm",
                           "2Mu2E_800GeV_0p25GeV_1p2mm"],
                 "4mu": ["4Mu_200GeV_0p25GeV_1p0mm", "4Mu_500GeV_0p25GeV_0p4mm", "4Mu_800GeV_0p25GeV_0p25mm",
                         "4Mu_1000GeV_0p25GeV_0p2mm", "4Mu_800GeV_1p2GeV_1p2mm", "4Mu_1000GeV_1p2GeV_0p96mm",
                         "4Mu_800GeV_0p25GeV_1p2mm"]}
DECISION_POINTS = {t: [n for n in names if "_500GeV_" in n or "_800GeV_" in n] for t, names in SIGNAL_POINTS.items()}


def bin_cfg(cfg, lo, hi):
    """``cfg`` restricted to the LJ-pair mass bin [lo, hi) (``hi=None``: open-ended, overflow included)."""
    return dataclasses.replace(cfg, mass=(lo, hi))


def _label(plane, cls):
    return plane if cls is None else f"{plane}:{cls}"


def _triple(row, name):
    return row[name], row[f"{name} minus"], row[f"{name} plus"]


def _pct(v):
    return f"{100 * v:.0f}%" if np.isfinite(v) else "n/a"


def _weighted(xs, ws):
    """Mean of ``xs`` weighted by ``ws``; the unweighted mean when the weights sum to zero."""
    xs, ws = np.asarray(xs, float), np.asarray(ws, float)
    return float(np.sum(ws * xs) / np.sum(ws)) if np.sum(ws) > 0 else float(np.mean(xs))


# ----------------------------------------------------------------------------- counts and mass


def counts_table(cube, cfg, bins, planes=("PP", "PD", "DD"), classes=(None, "dsa0", "dsa1+")):
    """A/B/C/D, A/B, C/D and kappa per mass bin, plane and DSA class of the muon LJ (``None``: all).
    A hidden A prints as "blind", and so do the A/B and kappa that would need it."""
    rows = []
    for label, (lo, hi) in bins:
        c = bin_cfg(cfg, lo, hi)
        for plane in planes:
            for cls in classes:
                reg = ps.regions(cube, c, _label(plane, cls))
                hidden = np.isnan(reg["A"][0])
                rows.append({"bin": label, "plane": plane, "class": "all" if cls is None else cls,
                             **{r: tl.fmt_count(reg[r]) for r in "ABCD"},
                             "A/B": "blind" if hidden else tl.fmt(tl.ratio(reg["A"], reg["B"]), 2),
                             "C/D": tl.fmt(tl.ratio(reg["C"], reg["D"]), 2),
                             "kappa": "blind" if hidden else tl.fmt(ps.kappa(reg), 2)})
    return pd.DataFrame(rows)


def _mass_axis(cube):
    """Edges and flow flags of the cube's LJ-pair mass axis, and the mass assigned to every entry of
    a profile along it: the centre of each stored bin, the outer edge for a flow bin."""
    i = cube.spec.index("ljlj_mass")
    edges, under, over = cube.spec.edges[i], cube.spec.underflow[i], cube.spec.overflow[i]
    centres = 0.5 * (edges[:-1] + edges[1:])
    x = np.r_[edges[:1] if under else [], centres, edges[-1:] if over else []]
    return edges, under, over, x


def mean_mass(cube, cfg, plane, lo, hi, region="B"):
    """The ``region``-count-weighted mean LJ-pair mass of ``plane`` in the bin [lo, hi), from the
    cube's own mass bins: each stored bin counts at its centre (the mean mass inside a stored bin is
    not known), the overflow bin above the last edge (2500 GeV) at that edge.  An empty region gives
    the unweighted mean of the bin centres; a hidden region gives NaN."""
    edges, under, over, x = _mass_axis(cube)
    sumw, _ = ps.profile(cube, "ljlj_mass", **ps.region_ranges(cfg)[region],
                         **ps.plane_selection(cube.topology, plane))
    sl = io.bin_slice(edges, lo, hi, under, over)
    w = sumw[sl]
    if np.isnan(w).any():
        return np.nan
    return _weighted(x[sl], w)


def ab_points(cube, cfg, bins, plane, cls=None):
    """A/B of ``plane`` (one DSA class of the muon LJ, or all) in each mass bin, the points of the
    A/B(M) fit: ``x`` the B-weighted mean mass of the plane in the bin (:func:`mean_mass`), ``A/B``
    with its 68% interval (``minus``, ``plus``; ``transfer_lib.ratio``), ``sigma`` the mean of the
    two sides (the symmetric error the linear fit uses), and the counts ``A`` and ``B``.  A hidden A
    gives NaN."""
    rows = []
    label_plane = _label(plane, cls)
    for label, (lo, hi) in bins:
        reg = ps.regions(cube, bin_cfg(cfg, lo, hi), label_plane)
        r = tl.ratio(reg["A"], reg["B"])
        a, b = reg["A"][0], reg["B"][0]
        rows.append({"bin": label, "lo": lo, "hi": hi, "x": mean_mass(cube, cfg, label_plane, lo, hi),
                     "A": a, "B": b, "A/B": r[0], "minus": r[1], "plus": r[2],
                     "sigma": 0.5 * (r[1] + r[2]),
                     "lower": a / ZERO_UPPER if np.isfinite(a) and a > 0 and b == 0 else np.nan})
    return pd.DataFrame(rows)


# ----------------------------------------------------------------------------- the A/B(M) fit


def _rate(form, params, x):
    """A/B at the masses ``x`` for the parameters of ``form`` (``linear``: a + b x; ``exponential``:
    exp(p0 + p1 x) = a exp(-x / M0)), and its gradient with respect to the parameters."""
    x = np.asarray(x, float)
    if form == "linear":
        r = params[0] + params[1] * x
        return r, np.stack([np.ones_like(x), x])
    if form == "exponential":
        r = np.exp(params[0] + params[1] * x)
        return r, np.stack([r, r * x])
    raise ValueError(f"unknown form {form!r}")


def _start(form, x, a, b):
    """Starting parameters for :func:`fit_ab`: a weighted least-squares line through the bins that
    hold both an A and a B event (through ln(A/B) for the exponential form), or a flat line at the
    pooled A/B when fewer than two bins do."""
    both = (a > 0) & (b > 0)
    pooled = np.log(max(a.sum(), 0.5) / max(b.sum(), 0.5))
    if both.sum() < 2:
        return np.array([pooled if form == "exponential" else np.exp(pooled), 0.0])
    xs, y = x[both], a[both] / b[both]
    w = y**2 * a[both] * b[both] / (a[both] + b[both])          # inverse variance of the ratio
    if form == "exponential":
        y, w = np.log(y), w / y**2 * np.ones_like(y)             # the variance of the logarithm
    design = np.vstack([np.ones_like(xs), xs]).T
    return np.linalg.solve(design.T @ (design * w[:, None]), design.T @ (w * y))


def fit_ab(points, form="exponential"):
    """Binomial-likelihood fit of A/B against the mass ``x`` of :func:`ab_points`.

    In every bin with A + B > 0 the A count is binomial among the A + B events with the pass
    probability r / (1 + r), r = A/B the form's value at the bin's mass: ``linear`` r = a + b x,
    ``exponential`` r = a exp(-x / M0).  Every bin enters, including one without an A event (an
    upper limit on r) or without a B event (a lower limit), which a fit to the ratios would have to
    leave out.  Bins with a hidden A (NaN) are left out.

    Returns a dict: ``form``; ``params``, the intercept and slope of the line in the fitted
    space ((a, b), or (ln a, -1/M0)); ``cov``, their covariance from the expected Fisher information
    at the minimum; ``deviance``, twice the log-likelihood ratio of the saturated model to the fit
    (chi2-distributed for large counts), with ``ndof`` = bins - 2 and ``n`` the bins used; ``x``,
    their masses; ``residuals``, per bin the signed square root of its deviance contribution
    (positive where the measured A/B lies above the fit; the pull of the bin, defined also where A
    or B is zero); and ``f``, a function of the mass returning the fitted A/B and its 1-sigma error
    from the covariance.  With fewer than two bins the parameters are NaN.
    """
    if form not in FORMS:
        raise ValueError(f"unknown form {form!r}")
    p = points[np.isfinite(points["A"]) & np.isfinite(points["B"]) & np.isfinite(points["x"])]
    p = p[(p["A"] + p["B"]) > 0]
    x, a, b = (p[c].to_numpy(float) for c in ("x", "A", "B"))
    n = a + b
    nan = {"form": form, "params": np.full(2, np.nan), "cov": np.full((2, 2), np.nan), "deviance": np.nan,
           "ndof": 0, "n": len(x), "x": x, "residuals": {}}
    if len(x) < 2:
        return {**nan, "f": lambda mass: (np.full_like(np.asarray(mass, float), np.nan),) * 2}

    def nll(params):
        r, _ = _rate(form, params, x)
        if not np.isfinite(r).all():
            return np.inf
        below = np.maximum(RATE_FLOOR - r, 0.0)             # a line below zero: penalised, not forbidden
        r = np.maximum(r, RATE_FLOOR)
        return float(np.sum(n * np.log1p(r) - a * np.log(r)) + 1e6 * np.sum(below**2))

    res = optimize.minimize(nll, _start(form, x, a, b), method="Nelder-Mead",
                            options={"xatol": 1e-10, "fatol": 1e-12, "maxiter": 20000, "maxfev": 40000})
    if not res.success or not np.isfinite(res.fun):
        raise RuntimeError(f"the {form} fit of A/B against the mass did not converge: {res.message}")
    params = res.x
    r, g = _rate(form, params, x)
    info = (g * (n / (r * (1 + r) ** 2))) @ g.T                  # expected Fisher information
    cov = np.linalg.inv(info)
    with np.errstate(divide="ignore", invalid="ignore"):
        saturated = np.where(a > 0, a * np.log(a / n), 0.0) + np.where(b > 0, b * np.log(b / n), 0.0)
    r = np.maximum(r, RATE_FLOOR)
    per_bin = np.maximum(2 * (saturated - (a * np.log(r / (1 + r)) + b * np.log(1 / (1 + r)))), 0.0)
    residuals = np.sign(a / n - r / (1 + r)) * np.sqrt(per_bin)

    def f(mass):
        value, grad = _rate(form, params, np.atleast_1d(np.asarray(mass, float)))
        err = np.sqrt(np.maximum(np.einsum("ij,ik,kj->j", grad, cov, grad), 0.0))
        return (float(value[0]), float(err[0])) if np.ndim(mass) == 0 else (value, err)

    return {**nan, "params": params, "cov": cov, "deviance": float(np.sum(per_bin)), "ndof": len(x) - 2,
            "residuals": dict(zip(p["bin"], residuals)), "f": f}


def fit_text(fit):
    """The parameters of a :func:`fit_ab` result as text (the linear slope per TeV)."""
    (p0, p1), cov = fit["params"], fit["cov"]
    if not np.isfinite(p0):
        return "fewer than two points"
    e0, e1 = np.sqrt(np.diag(cov))
    if fit["form"] == "linear":
        return f"a = {p0:.3f} ± {e0:.3f}, b = {1000 * p1:.3f} ± {1000 * e1:.3f} per TeV"
    a = np.exp(p0)
    m0 = -1.0 / p1 if p1 != 0 else np.inf
    return f"a = {a:.3f} ± {a * e0:.3f}, M0 = {m0:.0f} ± {e1 / p1**2:.0f} GeV"


def _residual(fit, point):
    """The bin's residual of a :func:`fit_ab` result (NaN for a bin the fit did not use)."""
    return fit["residuals"].get(point["bin"], np.nan)


def ab_fit_table(cube, cfg, bins, planes=("PD", "PP"), classes=("dsa0", "dsa1+", None), forms=FORMS):
    """Per plane, DSA class and fit form: a row with the parameters and the deviance over ndof,
    then one row per bin with its mass, A/B measured (a lower limit where B = 0), A/B fitted there
    and the bin's residual (:func:`fit_ab`).  A hidden A prints as "blind"."""
    rows = []
    for plane in planes:
        for cls in classes:
            points = ab_points(cube, cfg, bins, plane, cls)
            for form in forms:
                fit = fit_ab(points, form)
                head = {"plane": plane, "class": "all" if cls is None else cls, "form": form}
                rows.append({**head, "bin": "fit", "parameters": fit_text(fit),
                             "deviance / ndof": f"{fit['deviance']:.1f} / {fit['ndof']}" if np.isfinite(fit["deviance"]) else "n/a"})
                for _, pt in points.iterrows():
                    value, err = fit["f"](pt["x"])
                    residual = _residual(fit, pt)
                    if np.isnan(pt["A"]):
                        measured = "blind"
                    elif np.isfinite(pt["lower"]):
                        measured = f">= {pt['lower']:.2f} ({pt['A']:.0f}/0)"
                    else:
                        measured = tl.fmt((pt["A/B"], pt["minus"], pt["plus"]), 2)
                    rows.append({**head, "bin": pt["bin"], "M [GeV]": f"{pt['x']:.0f}", "A/B measured": measured,
                                 "A/B fitted": f"{value:.2f} ± {err:.2f}" if np.isfinite(value) else "n/a",
                                 "residual": f"{residual:+.1f}" if np.isfinite(residual) else "n/a"})
    return pd.DataFrame(rows).fillna("")


def plane_ratio_points(num, den):
    """Per bin: A/B of the points ``num`` over A/B of the points ``den`` (two :func:`ab_points`
    frames on the same bins; ``transfer_lib.ratio_of_ratios``), at the mean of their masses."""
    rows = []
    for (_, a), (_, b) in zip(num.iterrows(), den.iterrows()):
        r = tl.ratio_of_ratios((a["A/B"], a["minus"], a["plus"]), (b["A/B"], b["minus"], b["plus"]))
        rows.append({"bin": a["bin"], "x": 0.5 * (a["x"] + b["x"]), "ratio": r[0], "minus": r[1], "plus": r[2]})
    return pd.DataFrame(rows)


def weighted_mean(values, minus, plus):
    """Inverse-variance weighted mean of (value, minus, plus) results, each with the mean of its two
    sides as its error: (mean, error, chi2 of the values against the mean, ndof = n - 1), over the
    finite ones."""
    v, m, p = (np.asarray(a, float) for a in (values, minus, plus))
    s = 0.5 * (m + p)
    ok = np.isfinite(v) & np.isfinite(s) & (s > 0)
    if not ok.any():
        return np.nan, np.nan, np.nan, 0
    w = 1.0 / s[ok] ** 2
    mean = np.sum(w * v[ok]) / np.sum(w)
    return float(mean), float(1.0 / np.sqrt(np.sum(w))), float(np.sum(w * (v[ok] - mean) ** 2)), int(ok.sum()) - 1


def plane_ratio_table(points, num="PP", den="PD", classes=("dsa0", "dsa1+", None)):
    """Per DSA class: A/B of ``num`` over A/B of ``den`` in each bin, their weighted mean and the
    chi2 of the bins against it.  ``points`` maps (plane, class) to :func:`ab_points` frames."""
    rows = []
    for cls in classes:
        r = plane_ratio_points(points[(num, cls)], points[(den, cls)])
        mean, err, chi2, ndof = weighted_mean(r["ratio"], r["minus"], r["plus"])
        row = {"class": "all" if cls is None else cls}
        row.update({b: tl.fmt((v, m, p), 2) for b, v, m, p in zip(r["bin"], r["ratio"], r["minus"], r["plus"])})
        row["weighted mean"] = f"{mean:.3f} ± {err:.3f}" if np.isfinite(mean) else "n/a"
        row["chi2 / ndof against it"] = f"{chi2:.1f} / {ndof}" if ndof > 0 else "n/a"
        rows.append(row)
    return pd.DataFrame(rows)


# ----------------------------------------------------------------------------- the binned estimate


def _sum_rows(label, lo, hi, per):
    """The rows ``per`` summed: counts and estimates added (errors in quadrature), the mass
    weighted by B, the shape systematics in quadrature."""
    row = {"bin": label, "lo": lo, "hi": hi, "class": SUM, "B": float(sum(p["B"] for p in per)),
           "B err": float(np.sqrt(sum(p["B err"] ** 2 for p in per))),
           "x": _weighted([p["x"] for p in per], [p["B"] for p in per])}
    for name in FACTORS:
        total = (0.0, 0.0, 0.0)
        for p in per:
            total = el._add(total, _triple(p, name))
        row.update({name: total[0], f"{name} minus": total[1], f"{name} plus": total[2]})
    shapes = [p["shape"] for p in per if np.isfinite(p["shape"])]
    row["shape"] = float(np.sqrt(np.sum(np.square(shapes)))) if shapes else np.nan
    row["shape carried"] = float(np.sqrt(sum(p["shape carried"] ** 2 for p in per)))
    return row


def binned_routes(cube, cfg, binning, ref_plane="PD", search="DD", classes=CLASSES, fits=None,
                  max_rel_err=MAX_REL_ERR):
    """The B route per bin of ``binning`` and DSA class of the muon LJ, as a numeric frame.

    Per bin and class: ``B`` of the search plane (``B err`` its Poisson error), ``x`` the mean mass
    of those B events (:func:`mean_mass`), ``A/B`` of ``ref_plane`` measured in the bin with its 68%
    interval and ``A/B rel`` the larger side of it over the value (``estimate_lib.precision``),
    ``A/B fit`` and ``A/B fit err`` the fitted A/B at ``x`` (``fits[(ref_plane, class)]``, a
    :func:`fit_ab` result; NaN without one), and three estimates, each as value, ``minus`` and
    ``plus`` (``transfer_lib._pred``: Poisson error of B and the factor's error): ``measured`` and
    ``fitted`` with the two factors, and ``selected``, the one the decision rule (``choice``) takes:
    the measured factor when ``A/B rel`` <= ``max_rel_err``, the fitted one otherwise.  ``shape`` is
    |measured - fitted| (NaN unless both exist) and ``shape carried`` the part of it the selected
    estimate carries (zero where the measured factor is taken).

    A row per bin with class ``"sum"`` adds the classes (errors and shapes in quadrature; the
    choice names the class taking each factor when they differ), and a last row ``"sum over bins"``
    adds the bins in quadrature, which is exact for the measured factors and the counts (disjoint
    events) and understates the error of a sum of fitted estimates (the bins share the fit's
    parameters).  The search plane's A is never read.
    """
    rows = []
    sums = []
    for label, (lo, hi) in binning:
        c = bin_cfg(cfg, lo, hi)
        per = []
        for cls in classes:
            b = ps.regions(cube, c, _label(search, cls))["B"]
            ref = ps.regions(cube, c, _label(ref_plane, cls))
            meas = tl.ratio(ref["A"], ref["B"])
            rel = el.precision(meas)
            x = mean_mass(cube, cfg, _label(search, cls), lo, hi)
            fit = None if fits is None else fits.get((ref_plane, cls))
            value, err = fit["f"](x) if fit is not None else (np.nan, np.nan)
            fitted = (float(value), float(err), float(err))
            est = {"measured": tl._pred(b, meas), "fitted": tl._pred(b, fitted)}
            choice = "measured" if rel <= max_rel_err else "fitted"
            est["selected"] = est[choice]
            shape = abs(est["measured"][0] - est["fitted"][0])
            row = {"bin": label, "lo": lo, "hi": hi, "class": cls, "B": b[0], "B err": float(np.sqrt(b[1])), "x": x,
                   "A/B": meas[0], "A/B minus": meas[1], "A/B plus": meas[2], "A/B rel": rel,
                   "A/B fit": fitted[0], "A/B fit err": fitted[1], "choice": choice,
                   "shape": shape, "shape carried": shape if choice == "fitted" and np.isfinite(shape) else 0.0}
            for name, (v, m, p) in est.items():
                row.update({name: v, f"{name} minus": m, f"{name} plus": p})
            per.append(row)
        total = _sum_rows(label, lo, hi, per)
        choices = [p["choice"] for p in per]
        total["choice"] = choices[0] if len(set(choices)) == 1 else ", ".join(f"{p['class']} {p['choice']}" for p in per)
        rows.extend(per + [total])
        sums.append(total)
    grand = _sum_rows(TOTAL, sums[0]["lo"], sums[-1]["hi"], sums)
    grand.update({"choice": "", "shape": np.nan, "shape carried": np.nan})
    rows.append(grand)
    df = pd.DataFrame(rows)
    df.attrs.update(ref_plane=ref_plane, search=search)
    return df


def routes_view(df):
    """:func:`binned_routes` formatted for display."""
    ref, search = df.attrs.get("ref_plane", "PD"), df.attrs.get("search", "DD")
    rows = []
    for _, r in df.iterrows():
        per_class = r["class"] != SUM
        sel = r["selected"]
        row = {"bin": r["bin"], "class": r["class"], f"B {search}": tl.fmt_count((r["B"], r["B err"] ** 2)),
               f"mean M of B {search} [GeV]": f"{r['x']:.0f}" if np.isfinite(r["x"]) else "n/a",
               f"A/B {ref} measured (precision)": (f"{tl.fmt(_triple(r, 'A/B'), 2)} ({_pct(r['A/B rel'])})"
                                                   if per_class else ""),
               f"A/B {ref} fitted": tl.fmt((r["A/B fit"], r["A/B fit err"], r["A/B fit err"]), 2) if per_class else "",
               "estimate, measured A/B": tl.fmt(_triple(r, "measured"), 1),
               "estimate, fitted A/B": tl.fmt(_triple(r, "fitted"), 1),
               "choice": r["choice"], "estimate, selected": tl.fmt(_triple(r, "selected"), 1)}
        if r["bin"] == TOTAL:
            row["|measured - fitted| / selected"] = row["shape systematic carried"] = ""
        else:
            row["|measured - fitted| / selected"] = _pct(r["shape"] / sel) if sel > 0 else "n/a"
            row["shape systematic carried"] = _pct(r["shape carried"] / sel) if sel > 0 else "n/a"
        rows.append(row)
    return pd.DataFrame(rows)


def sums_table(routes_by_binning, reference="one bin"):
    """Per binning: the estimate summed over its bins with the measured, the fitted and the selected
    A/B (errors in quadrature, see :func:`binned_routes`), and the selected sum over the
    ``reference`` binning's estimate with the measured A/B (the one-bin estimate of estimate.ipynb)."""
    ref = routes_by_binning[reference].set_index("bin").loc[TOTAL, "measured"]
    rows = []
    for name, df in routes_by_binning.items():
        t = df.set_index("bin").loc[TOTAL]
        rows.append({"binning": name, "bins": int(((df["class"] == SUM) & (df["bin"] != TOTAL)).sum()),
                     "B of the search plane": f"{t['B']:.0f}",
                     **{f"sum, {f} A/B" if f != "selected" else "sum, selected": tl.fmt(_triple(t, f), 1) for f in FACTORS},
                     f"selected / {reference} (measured)": f"{t['selected'] / ref:.3f}" if ref > 0 else "n/a"})
    return pd.DataFrame(rows)


# ----------------------------------------------------------------------------- the open test per bin


def poisson_interval(n):
    """The central 68.27% interval of a Poisson count ``n`` (Garwood) as (n - lower, upper - n)."""
    lower = 0.5 * stats.chi2.ppf(0.15865, 2 * n) if n > 0 else 0.0
    upper = 0.5 * stats.chi2.ppf(0.84135, 2 * n + 2)
    return n - lower, upper - n


def open_test_table(cube, cfg, binning, fits=None, ref_plane="PP", search="DP", classes=CLASSES,
                    max_rel_err=MAX_REL_ERR):
    """The B route per bin between two open planes, against the observed A: by default from PP into
    DP, the muon LJ going from prompt to displaced with a prompt egamma LJ, the analog of the
    estimate where A is observed (estimate.ipynb section 3.1, here per bin).  Per bin: B of the
    search plane, the predicted A with the selected factor (the rule of :func:`binned_routes`, with
    ``fits`` of the reference plane), the observed A with its 68% Poisson interval, observed over
    predicted (errors in quadrature), and the systematic the recipe of estimate.ipynb assigns from
    this test alone: |observed / predicted - 1| plus the precision of the ratio (the larger side of
    its interval over its value).  The last row repeats it over all bins.  The numeric columns
    ``ratio``, ``ratio minus``, ``ratio plus`` and ``syst`` are kept beside the text for the limit
    toy; a hidden A gives NaN."""
    routes = binned_routes(cube, cfg, binning, ref_plane=ref_plane, search=search, classes=classes, fits=fits,
                           max_rel_err=max_rel_err)
    rows = []
    for _, r in routes[routes["class"] == SUM].iterrows():
        pred = _triple(r, "selected")
        hi = None if pd.isna(r["hi"]) else float(r["hi"])
        obs = ps.regions(cube, bin_cfg(cfg, float(r["lo"]), hi), search)["A"][0]
        observed = (obs, *poisson_interval(obs)) if np.isfinite(obs) else (np.nan, np.nan, np.nan)
        ratio = tl.ratio_of_ratios(observed, pred)
        precision = el.precision(ratio)
        syst = abs(ratio[0] - 1.0) + precision if np.isfinite(ratio[0]) and np.isfinite(precision) else np.nan
        rows.append({"bin": r["bin"], f"B {search}": f"{r['B']:.0f}", "choice": r["choice"],
                     f"predicted A {search} (B x A/B {ref_plane}, selected)": tl.fmt(pred, 1),
                     f"observed A {search}": tl.fmt_count((obs, obs)),
                     "observed / predicted": tl.fmt(ratio, 2),
                     "|ratio - 1| + precision": _pct(syst),
                     "ratio": ratio[0], "ratio minus": ratio[1], "ratio plus": ratio[2], "syst": syst})
    return pd.DataFrame(rows)


def open_test_syst(tables):
    """Per binning, the per-bin systematic of its :func:`open_test_table` (the ``syst`` column
    without the last row), as :func:`limit_table` takes it."""
    return {name: t["syst"].to_numpy(float)[:-1] for name, t in tables.items()}


# ----------------------------------------------------------------------------- signal


def _folded_slice(edges, lo, hi):
    """Slice of a flow-folded spectrum (``estimate_lib.spectrum``) covering [lo, hi)."""
    start = 0 if lo is None else io._edge_index(edges, lo)
    stop = len(edges) - 1 if hi is None else io._edge_index(edges, hi)
    return slice(start, stop)


def signal_profiles(base, names, topology, cfg, plane="DD", region="A", prefix="abcd8d"):
    """The LJ-pair mass distribution of each signal point in one region of a plane (the
    configuration's other cuts applied, every mass), on the cube's mass bins with the flow bins
    folded into the outer ones (``estimate_lib.spectrum``): ``(edges, {point: (sumw, sumw2)})`` at
    the stored cross section.  Signal cubes are never hidden; samples are loaded one at a time."""
    out, edges = {}, None
    for name in names:
        cube = tl.sample_cubes(base, "signal", [name], topologies=(topology,), prefix=prefix)[name][topology]
        edges, sumw, sumw2 = el.spectrum(cube, "ljlj_mass", cfg, plane, region)
        out[name] = (sumw, sumw2)
        del cube
    return edges, out


def signal_yields(base, names, topology, cfg, binning, prefix="abcd8d", profiles=None):
    """Yield of each signal point in A of the DD plane per bin of ``binning`` at the stored cross
    section: one row per point and bin (``point``, ``bin``, ``lo``, ``hi``, ``yield``, ``error``).
    Read from ``profiles`` when given (the :func:`signal_profiles` of A of DD for the same ``cfg``),
    otherwise from the samples, loaded one at a time."""
    if profiles is None:
        profiles = signal_profiles(base, names, topology, cfg, prefix=prefix)
    edges, prof = profiles
    rows = []
    for name in names:
        sumw, sumw2 = prof[name]
        for label, (lo, hi) in binning:
            sl = _folded_slice(edges, lo, hi)
            rows.append({"point": name, "bin": label, "lo": lo, "hi": hi, "yield": float(np.sum(sumw[sl])),
                         "error": float(np.sqrt(np.sum(sumw2[sl])))})
    return pd.DataFrame(rows)


def signal_regions_table(base, names, topology, cfg, binning, data=None,
                         regions=(("PD:dsa0", "A"), ("PD:dsa0", "B"), ("PD:dsa1+", "A"), ("PD:dsa1+", "B"), ("DD", "B")),
                         prefix="abcd8d"):
    """The yield of each signal point per bin of ``binning`` in the regions the estimate reads from
    (``regions``: (plane with an optional DSA class, region)), at the stored cross section, one
    row per point and region with a column per bin and the sum; with ``data`` (a blinded data
    cube) a first row per region with the data counts, "blind" where hidden.  Signal cubes are
    never hidden; samples are loaded one at a time."""
    rows = []
    if data is not None:
        for plane, region in regions:
            row = {"point": "data", "region": f"{region} {plane}"}
            for label, (lo, hi) in binning:
                row[label] = tl.fmt_count(ps.regions(data, bin_cfg(cfg, lo, hi), plane)[region])
            lo, hi = binning[0][1][0], binning[-1][1][1]
            row["sum"] = tl.fmt_count(ps.regions(data, bin_cfg(cfg, lo, hi), plane)[region])
            rows.append(row)
    for name in names:
        cube = tl.sample_cubes(base, "signal", [name], topologies=(topology,), prefix=prefix)[name][topology]
        for plane, region in regions:
            edges, sumw, _ = el.spectrum(cube, "ljlj_mass", cfg, plane, region)
            row = {"point": name, "region": f"{region} {plane}"}
            total = 0.0
            for label, (lo, hi) in binning:
                v = float(np.sum(sumw[_folded_slice(edges, lo, hi)]))
                row[label] = f"{v:.2f}"
                total += v
            row["sum"] = f"{total:.2f}"
            rows.append(row)
        del cube
    return pd.DataFrame(rows)


def yields_view(yields):
    """:func:`signal_yields` as one row per point and one column per bin."""
    table = yields.pivot(index="point", columns="bin", values="yield")
    table = table.reindex(index=list(dict.fromkeys(yields["point"])), columns=list(dict.fromkeys(yields["bin"])))
    table["sum"] = table.sum(axis=1)
    return pd.DataFrame({c: [f"{v:.2f}" for v in table[c]] for c in table.columns}, index=table.index).reset_index()


# ----------------------------------------------------------------------------- toy expected limit


def expected_limit(s, b, stat_rel, syst_rel, cl=0.95):
    """Median expected upper limit on the signal strength mu of a multi-bin counting experiment:
    CLs in the asymptotic approximation for the background-only expectation (Cowan, Cranmer, Gross,
    Vitells, Eur. Phys. J. C 71 (2011) 1554), as combine's AsymptoticLimits computes it.

    Bin i expects nu_i = mu s_i + b_i exp(theta_i ln(1 + stat_rel_i)) exp(theta_c ln(1 + syst_rel)):
    the signal s_i per unit mu and the background b_i with an independent log-normal nuisance
    theta_i of relative size stat_rel_i and one log-normal nuisance theta_c of relative size
    syst_rel common to all bins, each with a unit Gaussian constraint.  The Asimov data set of the
    background-only hypothesis is n_i = b_i, whose best fit is mu = 0 with every theta at zero; the
    profiled test statistic on it is

        q_A(mu) = min over theta of 2 sum_i [nu_i - n_i - n_i ln(nu_i / n_i)] + sum_i theta_i^2 + theta_c^2

    (a bin with n_i = 0 contributes 2 nu_i).  For data equal to the Asimov set 1 - p_b = 1/2 and
    CLs = 2 (1 - Phi(sqrt(q_A))), so the median expected limit at confidence ``cl`` solves
    q_A(mu) = Phi^-1(1 - (1 - cl) / 2)^2 = 1.96^2 at 95%.  The signal carries no uncertainty, and
    the approximation is optimistic for bins with less than about one background event (with no
    background it gives 1.92 signal events against the exact 3.0).

    Returns (mu_up, mu_up x sum(s)): the limit on mu and the same in signal events summed over the
    bins.
    """
    s, b, k = (np.atleast_1d(np.asarray(v, float)) for v in (s, b, stat_rel))
    if not s.shape == b.shape == k.shape:
        raise ValueError("s, b and stat_rel need one entry per bin")
    if not (np.isfinite(s).all() and np.isfinite(b).all() and np.isfinite(k).all() and np.isfinite(syst_rel)):
        raise ValueError("expected_limit needs finite inputs")
    if (s < 0).any() or (b < 0).any() or (k < 0).any() or syst_rel < 0:
        raise ValueError("expected_limit needs non-negative inputs")
    used = (s > 0) | (b > 0)
    s, n, ln_k = s[used], b[used], np.log1p(k[used])
    ln_c = np.log1p(syst_rel)
    if not s.sum() > 0:
        return np.nan, np.nan
    n_safe = np.where(n > 0, n, 1.0)
    target = stats.norm.ppf(1 - (1 - cl) / 2) ** 2

    def q(mu):
        if mu <= 0:
            return 0.0

        def nll(theta):
            bkg = n * np.exp(theta[:-1] * ln_k + theta[-1] * ln_c)
            nu = mu * s + bkg
            value = 2 * np.sum(nu - n - n * np.log(nu / n_safe)) + np.sum(theta**2)
            pull = 2 * (1 - n / nu) * bkg
            return value, np.r_[pull * ln_k, np.sum(pull) * ln_c] + 2 * theta

        res = optimize.minimize(nll, np.zeros(len(n) + 1), jac=True, method="L-BFGS-B",
                                bounds=[(-10.0, 10.0)] * (len(n) + 1))
        return float(res.fun)

    hi = 1.0 / s.sum()                  # one signal event
    for _ in range(60):
        if q(hi) > target:
            break
        hi *= 2.0
    mu_up = optimize.brentq(lambda mu: q(mu) - target, 0.0, hi, xtol=1e-7 * hi)
    return mu_up, mu_up * s.sum()


def _bin_inputs(routes, factor):
    """Bins, background and uncorrelated relative error of one factor choice of a
    :func:`binned_routes` frame: the statistical error (the mean of its two sides) in quadrature
    with the shape systematic the choice carries (none for the measured factor, |measured - fitted|
    for the fitted one, the carried part for the selected one)."""
    rows = routes[(routes["class"] == SUM) & (routes["bin"] != TOTAL)]
    b = rows[factor].to_numpy(float)
    err = 0.5 * (rows[f"{factor} minus"] + rows[f"{factor} plus"]).to_numpy(float)
    shape = {"measured": np.zeros(len(rows)), "fitted": rows["shape"].fillna(0.0).to_numpy(float),
             "selected": rows["shape carried"].to_numpy(float)}[factor]
    with np.errstate(divide="ignore", invalid="ignore"):
        rel = np.where(b > 0, np.hypot(err, shape) / b, 0.0)
    return list(rows["bin"]), b, rel


def limit_table(routes_by_binning, yields_by_binning, syst_rel=0.33, factors=FACTORS, reference=("one bin", "measured"),
                common=True):
    """The toy expected limit (:func:`expected_limit`) per signal point, binning and factor choice:
    the background of each bin from :func:`binned_routes` with its uncorrelated error
    (:func:`_bin_inputs`), the systematic ``syst_rel``, the signal per bin from
    :func:`signal_yields` (mu = 1 is the stored cross section).  Columns: the signal in the search
    region, the limit on mu, the limit in signal events, and the latter relative to the
    ``reference`` (the one-bin estimate with the measured A/B, the estimate of estimate.ipynb).  A
    background that is NaN in any bin gives NaN.

    ``common``: the systematic is one nuisance shared by all bins (one log-normal uncertainty on
    every bin of a datacard), so that a bin without signal constrains it for the others.  With
    ``common=False`` every bin carries its own, independent of the others, folded into the bin's
    uncorrelated nuisance (log-normal widths added in quadrature); with one bin the two are the
    same.  ``syst_rel`` is one relative size for every bin, or a dict per binning name of one size
    per bin (:func:`open_test_syst`), which needs ``common=False``."""
    rows = []
    decision = {n for names in DECISION_POINTS.values() for n in names}
    for name, routes in routes_by_binning.items():
        yields = yields_by_binning[name]
        for factor in factors:
            labels, b, rel = _bin_inputs(routes, factor)
            per_bin = np.broadcast_to(np.asarray(syst_rel[name] if isinstance(syst_rel, dict) else syst_rel, float), b.shape)
            if common:
                if not np.all(per_bin == per_bin[0]):
                    raise ValueError("a systematic that differs between the bins needs common=False")
                syst = float(per_bin[0])
            else:
                rel = np.expm1(np.hypot(np.log1p(rel), np.log1p(per_bin)))
                syst = 0.0
            for point, part in yields.groupby("point", sort=False):
                s = part.set_index("bin").loc[labels, "yield"].to_numpy(float)
                mu, events = expected_limit(s, b, rel, syst) if np.isfinite(b).all() and np.isfinite(rel).all() else (np.nan, np.nan)
                rows.append({"point": point, "decision point": point in decision, "binning": name, "factor": factor,
                             "signal in the search region": float(s.sum()), "limit on mu": mu, "limit [events]": events})
    df = pd.DataFrame(rows)
    ref = df[(df["binning"] == reference[0]) & (df["factor"] == reference[1])].set_index("point")["limit [events]"]
    df["relative to one bin"] = df["limit [events]"] / df["point"].map(ref)
    return df


def limit_view(limits, factor="selected"):
    """One row per signal point: the one-bin limit in signal events (measured A/B) and, per
    binning, the limit for ``factor`` relative to it."""
    part = limits[limits["factor"] == factor]
    table = part.pivot(index="point", columns="binning", values="relative to one bin")
    table = table.reindex(index=list(dict.fromkeys(part["point"])), columns=list(dict.fromkeys(part["binning"])))
    table = pd.DataFrame({c: [f"{v:.3f}" if np.isfinite(v) else "n/a" for v in table[c]] for c in table.columns},
                         index=table.index)
    ref = limits[(limits["binning"] == "one bin") & (limits["factor"] == "measured")].set_index("point")["limit [events]"]
    table.insert(0, "one bin, measured A/B [events]", [f"{ref[p]:.1f}" for p in table.index])
    return table.reset_index()


def decision_table(routes_by_binning, min_b=MIN_B, small=SMALL):
    """Per binning, the constraints of the decision rule: whether the bins cover M >= 150 GeV
    without a gap, B of the search plane and the selected estimate per bin, whether each bin rests
    on at least ``min_b`` B events or is below ``small`` events, where the fitted A/B is taken, the
    largest |measured - fitted| / selected over the bins and the largest shape systematic the
    selected estimate carries."""
    rows = []
    for name, routes in routes_by_binning.items():
        bins = routes[(routes["class"] == SUM) & (routes["bin"] != TOTAL)]
        lo, hi = list(bins["lo"]), [None if pd.isna(h) else h for h in bins["hi"]]
        covers = lo[0] == 150.0 and hi[-1] is None and all(h == l for h, l in zip(hi[:-1], lo[1:]))
        ok = [bool(r["B"] >= min_b or r["selected"] < small) for _, r in bins.iterrows()]
        per_class = routes[(routes["class"] != SUM)]
        fitted = [f"{r['bin']} {r['class']}" for _, r in per_class.iterrows() if r["choice"] == "fitted"]
        with np.errstate(divide="ignore", invalid="ignore"):
            shape = (bins["shape"] / bins["selected"]).to_numpy(float)
            carried = (bins["shape carried"] / bins["selected"]).to_numpy(float)
        rows.append({"binning": name, "bins": " | ".join(bins["bin"]), "bins cover M >= 150": covers,
                     "B of the search plane": " | ".join(f"{v:.0f}" for v in bins["B"]),
                     "selected estimate": " | ".join(f"{v:.1f}" for v in bins["selected"]),
                     f"B >= {min_b} or estimate < {small:g}": " | ".join("yes" if o else "no" for o in ok),
                     "every bin passes": all(ok), "fitted A/B taken in": ", ".join(fitted) or "none",
                     "largest |measured - fitted| / selected": _pct(np.nanmax(shape)) if np.isfinite(shape).any() else "n/a",
                     "largest shape systematic carried": _pct(np.nanmax(carried)) if np.isfinite(carried).any() else "n/a"})
    return pd.DataFrame(rows)


def mediator_mass(name):
    """The mediator mass in GeV of a signal point (``2Mu2E_500GeV_0p25GeV_0p4mm`` -> 500)."""
    return float(name.split("_")[1].removesuffix("GeV"))


def rule_table(limits, decision, factor="selected"):
    """The decision rule applied to :func:`limit_table` and :func:`decision_table`: per binning,
    whether it is eligible (every bin passes the count requirement and the bins cover M >= 150 GeV
    without a gap), the expected limit relative to one bin for ``factor`` averaged over the
    decision points of each mediator mass, and the mean over the masses, each mass counting once
    however many points it has.  ``chosen`` marks the eligible binning with the lowest mean."""
    part = limits[(limits["factor"] == factor) & limits["decision point"]]
    rows = []
    for _, d in decision.iterrows():
        sub = part[part["binning"] == d["binning"]]
        per_mass = sub.groupby(sub["point"].map(mediator_mass))["relative to one bin"].mean()
        row = {"binning": d["binning"], "eligible": bool(d["every bin passes"] and d["bins cover M >= 150"])}
        row.update({f"{m:g} GeV": v for m, v in per_mass.items()})
        row["mean over the masses"] = float(per_mass.mean()) if len(per_mass) else np.nan
        rows.append(row)
    df = pd.DataFrame(rows)
    ok = df["eligible"] & np.isfinite(df["mean over the masses"])
    df["chosen"] = False
    if ok.any():
        df.loc[df.loc[ok, "mean over the masses"].idxmin(), "chosen"] = True
    return df
