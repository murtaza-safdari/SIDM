"""Statistical tools for the ABCD plane-choice study.

Everything downstream of the histograms works on plain numpy arrays of per-bin sums of
weights and sums of squared weights, extracted once from the N-dimensional scan
histograms (see hists.py "ABCD plane-choice scan hists"). The pre-registered decision
rule these functions implement is documented in the study README.

Conventions
-----------
* A cut value is only ever applied at an exact bin edge (the scan hists are built that
  way); `edge_index` raises if a requested cut is not an edge.
* A region selection on an axis is ("lt", cut) or ("ge", cut) — pass-low or pass-high.
  The A (signal-like) region is the one passing BOTH plane axes.
* Isolation axes carry a sentinel first bin [-0.02, 0) holding LJs with no matched AK4
  jet.  `lo` bounds implement the quirk prescriptions: include the sentinel (lo=None)
  or exclude it (lo=0.0).
* Yields are (value, variance) pairs; variance is the sum of squared weights.
"""

import json
import numpy as np
from scipy import stats as sps


# ---------------------------------------------------------------------------
# histogram -> numpy extraction (needs `hist`, imported lazily so the numpy-only
# statistics below stay importable/testable without it)
# ---------------------------------------------------------------------------

def get_channel(h, channel):
    """Select one channel of a scan hist (drops the StrCategory axis)."""
    return h[{"channel": channel}]


def project_plane(h, xaxis, yaxis, sel=None):
    """Reduce an N-dim hist to the 2D (xaxis, yaxis) plane.

    sel: dict axis_name -> selection for every OTHER axis:
      "sum"            integrate the full axis
      ("lt", v)        integrate bins with upper edge <= v (v must be an edge)
      ("ge", v)        integrate bins with lower edge >= v
      ("bin", i)       take bin index i only
      ("bins", [i,..]) sum a list of bin indices
    Any axis not in sel is integrated over ("sum").
    """
    import hist  # noqa: F401  (UHI slicing)
    sel = dict(sel or {})
    out = h
    for ax in [a.name for a in h.axes]:
        if ax in (xaxis, yaxis):
            continue
        spec = sel.pop(ax, "sum")
        if spec == "sum":
            out = out[{ax: slice(None, None, sum)}]
        elif spec[0] == "lt":
            i = edge_index(np.asarray(out.axes[ax].edges), spec[1])
            out = out[{ax: slice(0, i, sum)}]
        elif spec[0] == "ge":
            i = edge_index(np.asarray(out.axes[ax].edges), spec[1])
            out = out[{ax: slice(i, len(out.axes[ax].edges) - 1, sum)}]
        elif spec[0] == "bin":
            out = out[{ax: slice(spec[1], spec[1] + 1, sum)}]
        elif spec[0] == "bins":
            import functools
            picked = None
            for i in spec[1]:
                piece = out[{ax: slice(i, i + 1, sum)}]
                picked = piece if picked is None else picked + piece
            out = picked
        else:
            raise ValueError(f"bad selection {spec} for axis {ax}")
    if sel:
        raise ValueError(f"selections for unknown axes: {list(sel)}")
    # order axes as (x, y)
    if [a.name for a in out.axes] == [yaxis, xaxis]:
        vals = out.view()["value"].T
        var = out.view()["variance"].T
        xe, ye = np.asarray(out.axes[xaxis].edges), np.asarray(out.axes[yaxis].edges)
    else:
        vals = out.view()["value"]
        var = out.view()["variance"]
        xe, ye = np.asarray(out.axes[xaxis].edges), np.asarray(out.axes[yaxis].edges)
    return np.asarray(vals, float), np.asarray(var, float), xe, ye


# ---------------------------------------------------------------------------
# region arithmetic (numpy only)
# ---------------------------------------------------------------------------

def edge_index(edges, cut, atol=1e-9):
    """Index of `cut` in `edges`; raises if the cut is not an exact bin edge."""
    i = int(np.argmin(np.abs(np.asarray(edges) - cut)))
    if abs(edges[i] - cut) > atol:
        raise ValueError(f"cut {cut} is not a bin edge of {edges}")
    return i


def _axis_split(edges, spec, lo=None, hi=None, guard=0):
    """Return (pass_slice, fail_slice) bin-index slices for one axis.

    spec = ("lt", cut): pass = bins below cut, fail = bins at/above cut.
    spec = ("ge", cut): pass = bins at/above cut, fail = bins below.
    lo/hi: optional outer bounds (both must be edges) restricting the used range,
    e.g. lo=0.0 excludes the isolation sentinel bin from BOTH regions.
    guard: number of bins skipped on each side of the boundary (guard band).
    """
    n = len(edges) - 1
    ilo = 0 if lo is None else edge_index(edges, lo)
    ihi = n if hi is None else edge_index(edges, hi)
    ic = edge_index(edges, spec[1])
    low, high = slice(ilo, max(ic - guard, ilo)), slice(min(ic + guard, ihi), ihi)
    if spec[0] == "lt":
        return low, high
    if spec[0] == "ge":
        return high, low
    raise ValueError(f"bad axis spec {spec}")


def region_sums(vals, var, xedges, yedges, xspec, yspec,
                xlo=None, xhi=None, ylo=None, yhi=None, xguard=0, yguard=0):
    """ABCD yields from a 2D (x, y) array.

    Returns {"A": (v, var), "B": ..., "C": ..., "D": ...} with
    A = pass-x & pass-y, B = fail-x & pass-y, C = pass-x & fail-y, D = fail-both.
    """
    xp, xf = _axis_split(xedges, xspec, xlo, xhi, xguard)
    yp, yf = _axis_split(yedges, yspec, ylo, yhi, yguard)
    out = {}
    for name, (sx, sy) in {"A": (xp, yp), "B": (xf, yp), "C": (xp, yf), "D": (xf, yf)}.items():
        out[name] = (float(vals[sx, sy].sum()), float(var[sx, sy].sum()))
    return out


def n_eff(v, var):
    """Kish effective count of a weighted yield."""
    return v * v / var if var > 0 else 0.0


def abcd_prediction(regions):
    """A_pred = B*C/D with first-order error propagation."""
    (b, vb), (c, vc), (d, vd) = regions["B"], regions["C"], regions["D"]
    if d <= 0:
        return np.nan, np.nan
    pred = b * c / d
    rel2 = (vb / b**2 if b > 0 else 0) + (vc / c**2 if c > 0 else 0) + vd / d**2
    return pred, pred**2 * rel2


def closure_ratio(regions):
    """R = A_obs / A_pred and its variance (regions statistically independent)."""
    a, va = regions["A"]
    pred, vpred = abcd_prediction(regions)
    if not np.isfinite(pred) or pred <= 0 or a <= 0:
        return np.nan, np.nan
    r = a / pred
    return r, r**2 * (va / a**2 + vpred / pred**2)


def kappa(regions):
    """kappa = A*D/(B*C), the factorization ratio; 1 = independent."""
    a, va = regions["A"]
    b, vb = regions["B"]
    c, vc = regions["C"]
    d, vd = regions["D"]
    if min(b, c) <= 0 or a <= 0 or d <= 0:
        return np.nan, np.nan
    k = a * d / (b * c)
    rel2 = va / a**2 + vb / b**2 + vc / c**2 + vd / d**2
    return k, k**2 * rel2


# ---------------------------------------------------------------------------
# independence: factorization fit on an adaptively rebinned grid
# ---------------------------------------------------------------------------

def adaptive_rebin(vals, var, n_eff_floor=10.0):
    """Greedily merge adjacent rows/columns until every occupied super-bin has
    n_eff >= floor (or the grid can shrink no further). Returns (vals, var,
    row_groups, col_groups)."""
    def groups_ok(v, w):
        occ = v > 0
        ne = np.where(w > 0, v * v / np.maximum(w, 1e-300), np.inf)
        return bool(np.all(ne[occ] >= n_eff_floor))

    rows = [[i] for i in range(vals.shape[0])]
    cols = [[j] for j in range(vals.shape[1])]

    def build(vv, ww, rgs, cgs):
        v = np.array([[vv[np.ix_(r, c)].sum() for c in cgs] for r in rgs])
        w = np.array([[ww[np.ix_(r, c)].sum() for c in cgs] for r in rgs])
        return v, w

    v, w = build(vals, var, rows, cols)
    # alternate merging the worst row / worst column until fine
    for _ in range(vals.shape[0] + vals.shape[1]):
        if groups_ok(v, w) or (len(rows) <= 2 and len(cols) <= 2):
            break
        ne = np.where(w > 0, v * v / np.maximum(w, 1e-300), np.inf)
        ne = np.where(v > 0, ne, np.inf)
        i, j = np.unravel_index(np.argmin(ne), ne.shape)
        # merge along the longer dimension first to keep the grid squarish
        if len(rows) >= len(cols) and len(rows) > 2:
            k = i if i < len(rows) - 1 else i - 1
            rows[k] = rows[k] + rows[k + 1]
            del rows[k + 1]
        elif len(cols) > 2:
            k = j if j < len(cols) - 1 else j - 1
            cols[k] = cols[k] + cols[k + 1]
            del cols[k + 1]
        else:
            break
        v, w = build(vals, var, rows, cols)
    return v, w, rows, cols


def factorization_fit(vals, var, n_eff_floor=10.0, max_iter=500, tol=1e-10):
    """Fit mu_ij = a_i * b_j to a (rebinned) plane by iterative proportional scaling,
    with chi2 computed against the per-bin variances.

    Returns dict with chi2, ndf, pvalue, a, b, expected, used (I, J).
    Bins with var == 0 and vals == 0 are excluded from chi2 (unoccupied).
    """
    v, w, rows, cols = adaptive_rebin(np.asarray(vals, float), np.asarray(var, float),
                                      n_eff_floor)
    mask = ~((v == 0) & (w == 0))
    m = mask.astype(float)
    a = np.maximum(v.sum(axis=1), 1e-12)
    b = np.maximum(v.sum(axis=0), 1e-12) / max(v.sum(), 1e-300)
    for _ in range(max_iter):
        # mask-aware iterative proportional fitting: Poisson MLE of mu_ij = a_i b_j
        # restricted to the used bins
        a_new = (v * m).sum(axis=1) / np.maximum((m * b[None, :]).sum(axis=1), 1e-300)
        b_new = (v * m).sum(axis=0) / np.maximum((m * a_new[:, None]).sum(axis=0), 1e-300)
        if np.allclose(a_new, a, rtol=tol) and np.allclose(b_new, b, rtol=tol):
            a, b = a_new, b_new
            break
        a, b = a_new, b_new
    expected = np.outer(a, b)
    use = mask & (w > 0)
    chi2 = float((((v - expected) ** 2 / np.where(use, w, 1))[use]).sum())
    ndf = int(use.sum() - (len(a) + len(b) - 1))
    p = float(sps.chi2.sf(chi2, ndf)) if ndf > 0 else np.nan
    return {"chi2": chi2, "ndf": ndf, "pvalue": p, "a": a, "b": b,
            "expected": expected, "rebinned": (v, w), "groups": (rows, cols)}


def weighted_correlation(vals, xedges, yedges):
    """Weighted Pearson correlation of the two plane variables from the 2D array
    (bin centers weighted by bin content). Descriptive only — the gate is the fit."""
    xc = 0.5 * (np.asarray(xedges[:-1]) + np.asarray(xedges[1:]))
    yc = 0.5 * (np.asarray(yedges[:-1]) + np.asarray(yedges[1:]))
    w = np.asarray(vals, float)
    tot = w.sum()
    if tot <= 0:
        return np.nan
    mx = (w.sum(axis=1) * xc).sum() / tot
    my = (w.sum(axis=0) * yc).sum() / tot
    cov = (w * np.outer(xc - mx, yc - my)).sum() / tot
    sx = np.sqrt((w.sum(axis=1) * (xc - mx) ** 2).sum() / tot)
    sy = np.sqrt((w.sum(axis=0) * (yc - my) ** 2).sum() / tot)
    return float(cov / (sx * sy)) if sx > 0 and sy > 0 else np.nan


# ---------------------------------------------------------------------------
# bootstrap covariance between (nested) scan points
# ---------------------------------------------------------------------------

def bootstrap_closures(vals, var, xedges, yedges, points, n_boot=500, seed=13):
    """Bootstrap the closure ratio R at several scan points from ONE plane.

    Each bin is resampled with a scaled-Poisson model: k ~ Poisson(n_eff_bin),
    new_val = val * k / n_eff_bin, preserving the per-bin relative fluctuation of a
    weighted count. `points` is a list of kwargs dicts for region_sums.
    Returns (R_central [npts], cov [npts, npts]); nan-safe.
    """
    rng = np.random.default_rng(seed)
    vals = np.asarray(vals, float)
    var = np.asarray(var, float)
    ne = np.where(var > 0, vals**2 / np.maximum(var, 1e-300), 0.0)
    scale = np.where(ne > 0, vals / np.maximum(ne, 1e-300), 0.0)
    central = np.array([closure_ratio(region_sums(vals, var, xedges, yedges, **p))[0]
                        for p in points])
    draws = np.empty((n_boot, len(points)))
    for t in range(n_boot):
        k = rng.poisson(ne)
        v = scale * k
        w = scale**2 * k  # resampled sumw2 consistent with the toy weights
        draws[t] = [closure_ratio(region_sums(v, np.maximum(w, 1e-300), xedges, yedges, **p))[0]
                    for p in points]
    good = np.all(np.isfinite(draws), axis=1)
    cov = np.cov(draws[good].T) if good.sum() > 2 else np.full((len(points),) * 2, np.nan)
    return central, np.atleast_2d(cov)


# ---------------------------------------------------------------------------
# sensitivity
# ---------------------------------------------------------------------------

def asimov_z(s, b, sigma_b=0.0):
    """Asimov discovery significance with background uncertainty (Cowan et al.)."""
    if s <= 0 or b <= 0:
        return 0.0
    if sigma_b <= 0:
        return float(np.sqrt(2 * ((s + b) * np.log(1 + s / b) - s)))
    sb2 = sigma_b**2
    t1 = (s + b) * np.log((s + b) * (b + sb2) / (b**2 + (s + b) * sb2))
    t2 = (b**2 / sb2) * np.log(1 + sb2 * s / (b * (b + sb2)))
    val = 2 * (t1 - t2)
    return float(np.sqrt(max(val, 0.0)))


def leakage_ratios(sig_regions):
    """Signal leakage into the sidebands relative to A: S_B/S_A, S_C/S_A, S_D/S_A."""
    sa = sig_regions["A"][0]
    if sa <= 0:
        return {k: np.nan for k in ("B", "C", "D")}
    return {k: sig_regions[k][0] / sa for k in ("B", "C", "D")}


def prediction_bias_vs_mu(bkg_regions, sig_regions, mus):
    """Relative bias of the ABCD prediction when signal of strength mu leaks into the
    sidebands: returns [(mu, (pred(mu) - pred(0)) / (A_bkg + mu*S_A))]."""
    out = []
    for mu in mus:
        tot = {k: (bkg_regions[k][0] + mu * sig_regions[k][0],
                   bkg_regions[k][1] + mu**2 * sig_regions[k][1]) for k in "ABCD"}
        pred_mu, _ = abcd_prediction(tot)
        pred_0, _ = abcd_prediction(bkg_regions)
        denom = bkg_regions["A"][0] + mu * sig_regions["A"][0]
        out.append((mu, (pred_mu - pred_0) / denom if denom > 0 else np.nan))
    return out


# ---------------------------------------------------------------------------
# normalization (offline, census-based — see study README)
# ---------------------------------------------------------------------------

TTJETS_XSEC_CAMPAIGN = 471.7   # value baked into the campaign hists (cross_sections.yaml)
TTJETS_XSEC_NNLO = 831.76      # NNLO+NNLL ttbar; adoption pending explicit user sign-off


def ttjets_xsec_rescale():
    """Factor moving TTJets campaign hists from the yaml xsec to the NNLO+NNLL value.

    The campaign was deliberately produced with the repo's existing 471.7 pb so the
    cross-section choice stays an explicit, reversible offline decision.
    """
    return TTJETS_XSEC_NNLO / TTJETS_XSEC_CAMPAIGN


def census_sumw_pre(summary_path):
    """{sample: pre-skim/pre-filter sum of gen weights} from a census summary JSON."""
    d = json.load(open(summary_path))
    out = {}
    for rec in d["samples"]:
        out[rec["sample"]] = rec.get("genEventSumw_reachable_raw") or 0.0
    return out


def offline_norm_factor(merged_metadata, sumw_pre):
    """Factor that takes a merged sample's hists from the campaign normalization
    (lumi*xs / scaled_sum_weights) to the honest one (lumi*xs / sumw_pre).

    Applies to HISTS only. Campaign cutflows are relative-only: the condor path
    hardcodes skim_factor = 1.0, so absolute cutflow yields are off by the skim /
    gen-filter fraction (up to ~7e3 for the QCD skims) — never quote them."""
    ssw = merged_metadata["scaled_sum_weights"]
    if sumw_pre <= 0:
        raise ValueError("sumw_pre must be positive")
    return float(ssw) / float(sumw_pre)
