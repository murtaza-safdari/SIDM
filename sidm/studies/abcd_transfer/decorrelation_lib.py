"""The LJ-pair mass as a third variable of the 2mu2e ABCD estimate, and three ways of removing its
effect: the numbers of mass_decorrelation.ipynb.

Notation.  In a plane and a mass bin w, a_w is the background's probability to pass the isolation
cut, measured among the events that fail |dphi| (a_w = C_w / (C_w + D_w)), and b_w its probability
to pass the |dphi| cut, measured among the events that fail the isolation (b_w = B_w / (B_w + D_w)).
r_w = a_w / (1 - a_w) = C_w / D_w is the isolation's pass/fail odds and beta_w = B_w / D_w the
|dphi| odds.  If the two cuts are independent within each bin, A_w = B_w r_w and the one-bin
estimate (sum B)(sum C)/(sum D) differs from the true sum of A_w by the mixing factor
sum_w B_w C_w / D_w / ((sum B)(sum C)/(sum D)) = <beta r>_D / (<beta>_D <r>_D), where <.>_D is the
average over the bins weighted by D_w: one plus the D-weighted covariance of the two odds over the
mass, normalised.  It is one when either odds is constant in the mass (:func:`synthetic_check`).

The three routes:

1. stratify: the estimate summed over mass bins, sum_w B_w C_w / D_w (:func:`stratified_table`),
   with kappa and the closure systematic per bin (:func:`per_bin_systematic`, the rule of
   nominal.ipynb section 8 applied in each bin);
2. a smooth transfer ratio: a(M) fitted with a logistic in ln M (two parameters, a binomial
   likelihood, which is the Poisson likelihood of C and D conditioned on their sum), so that
   r(M) = exp(alpha + s ln(M / PIVOT)), applied to B's mass spectrum (:func:`smooth_route`);
3. a mass-flattened isolation cut: per mass bin the stored iso0 edge whose a(M) is closest to the
   anchor bin's (:func:`choose_edges`), with a buffer below 0.25 so that the sidebands of the
   search plane stay outside the hidden box (:func:`flat_cfg`).

Blinding: the data cube comes from ``transfer_lib.summed_cubes`` with the hidden cells NaN.  Every
function here reads A of DD only through ``plane_search_lib.regions``, where it is NaN, and never
uses it: the DD predictions use B, C and D only, the toys draw A only where it is finite, and a
hidden count is never summed with an open one into a printed number.
"""
import dataclasses

import numpy as np
import pandas as pd
from scipy import stats

import abcd6d_io as io
import estimate_lib as el
import nominal_lib as nl
import plane_search_lib as ps
import transfer_lib as tl

NOMINAL = nl.NOMINAL
PLANES = ("PP", "PD", "DD")
OPEN_PLANES = ("PP", "PD")
FIVE = tuple(ps.MASS_BINS)                                    # the five bins of nominal.ipynb section 7d
TWO = (("150-300", (150.0, 300.0)), (">=300", (300.0, None)))
ONE = ((">=150", (150.0, None)),)
BINNINGS = {"5 bins": FIVE, "2 windows": TWO, "1 bin": ONE}
OPEN_END = 1000.0          # the open-ended bin's centre is taken as the geometric centre of [low edge, OPEN_END)
PIVOT = 200.0              # the logistic's intercept is the log-odds at this mass
EDGE_CANDIDATES = (0.1, 0.15, 0.2, 0.25, 0.3, 0.4, 0.5, 0.75, 1.0)   # stored iso0 edges a bin may take
ANCHOR_EDGE = 0.25
N_TOYS = nl.BOOTSTRAP_TOYS
SEED = nl.BOOTSTRAP_SEED
MAX_PRECISION = nl.MAX_PRECISION


# ----------------------------------------------------------------------------- bins

def bin_label(lo, hi):
    """"150-200", ">=600" style label of a mass range."""
    return f">={lo:g}" if hi is None else f"{lo:g}-{hi:g}"


def centre(lo, hi, open_end=OPEN_END):
    """Geometric centre of a mass bin; the open-ended bin is taken as [lo, open_end)."""
    return float(np.sqrt(lo * (open_end if hi is None else hi)))


def bin_regions(cube, cfg, plane, bins):
    """``[{A, B, C, D: (sumw, sumw2)}]``, one per bin of ``bins`` (``[(label, (lo, hi))]``), of
    ``plane`` at the working point of ``cfg``; A is NaN where it touches the hidden box."""
    return [ps.regions(cube, dataclasses.replace(cfg, mass=rng), plane) for _, rng in bins]


def counts(regs):
    """``(nbins, 4)`` array of the A, B, C, D sums of :func:`bin_regions` (A NaN where hidden)."""
    return np.array([[r[x][0] for x in "ABCD"] for r in regs], float)


def as_vr(region):
    """A plane name ("PP", "PD", "DP", "DD") as that plane's validation region above 150 GeV with
    the egamma LJ isolated; a ``nominal_lib.VR`` as it is."""
    if isinstance(region, str):
        return nl.VR(region, "this analysis", region[0], region[1])
    return region


def region_counts(cube, cfg, region, bins):
    """``(nbins, 4)`` A, B, C, D of a plane or validation region in each bin of ``bins`` at the working
    point of ``cfg`` (A NaN where it touches the hidden box)."""
    if isinstance(region, str):
        return counts(bin_regions(cube, cfg, region, bins))
    return counts([nl.vr_regions(cube, dataclasses.replace(region, mass=rng), cfg) for _, rng in bins])


def region_name(region):
    return region if isinstance(region, str) else region.name


def merge_empty(bins, bcd):
    """Merge every bin whose D is zero into its lower neighbour (the first bin into the one above),
    so that each bin's B C / D exists: returns the merged ``bins`` and ``(nbins, 3)`` B, C, D sums.
    The merged bin is labelled by its new range."""
    bins = [(lab, tuple(rng)) for lab, rng in bins]
    bcd = [list(map(float, row)) for row in np.asarray(bcd, float)]
    while len(bins) > 1:
        empty = [i for i, row in enumerate(bcd) if row[2] <= 0]
        if not empty:
            break
        i = empty[-1]
        j = i - 1 if i > 0 else 1
        lo, hi = min(i, j), max(i, j)
        rng = (bins[lo][1][0], bins[hi][1][1])
        merged = [x + y for x, y in zip(bcd[lo], bcd[hi])]
        bins[lo:hi + 1] = [(bin_label(*rng), rng)]
        bcd[lo:hi + 1] = [merged]
    return bins, np.array(bcd, float)


def merge_groups(empty):
    """The bins :func:`merge_empty` joins for a pattern of empty D (one boolean per bin), as lists of
    bin indices.  The merging depends only on which bins have D = 0, since a merged bin's D is empty
    only when all its parts are."""
    bins = [(str(i), (float(i), float(i + 1))) for i in range(len(empty))]
    bcd = [(0.0, 0.0, 0.0 if e else 1.0) for e in empty]
    merged, _ = merge_empty(bins, bcd)
    return [list(range(int(lo), int(hi))) for _, (lo, hi) in merged]


def _ratio_or_inf(b, c, d):
    """B C / D, +inf where D = 0."""
    with np.errstate(divide="ignore", invalid="ignore"):
        return np.where(d > 0, b * c / np.where(d > 0, d, 1), np.inf)


def merge_rule(bins):
    """Whether the binned procedure over ``bins`` merges empty bins: the five-bin procedure does (its
    data path merges DD's empty >= 600 GeV bin); the two windows and the one bin have no merge rule."""
    return tuple(bins) == FIVE


def merged_toys(c, n_toys=N_TOYS, seed=SEED, merge=True):
    """Poisson toys of per-bin counts ``c`` (one row per bin, B, C, D in the last three columns, A in
    front when present) with the data's procedure applied in each toy.  With ``merge`` the bins whose
    drawn D is 0 are merged as :func:`merge_empty` merges them and the estimate is summed over the
    merged bins, and only a toy whose merged D is still 0 has no finite estimate; without it a toy with
    D = 0 in any bin has none (the convention of ``nominal_lib.bootstrap_abcd``).  A toy without a
    finite estimate has its sum set to +inf.  Returns ``{"toys": (n_toys, bins, columns), "split": the
    summed estimate per toy, "one_bin": (sum B)(sum C)/(sum D) per toy (+inf when sum D = 0), "ratio":
    split / one_bin, "kept": fraction of toys with a finite split, "fallback": fraction of toys merged
    differently from the data}``."""
    obs = np.asarray(c, float)
    if np.isnan(obs).any():
        raise ValueError("merged_toys takes open counts only")
    rng = np.random.default_rng(seed)
    toys = rng.poisson(obs, size=(n_toys,) + obs.shape).astype(float)
    b, cc, d = toys[..., -3], toys[..., -2], toys[..., -1]
    split = np.empty(n_toys)
    patterns, inverse = np.unique(d <= 0, axis=0, return_inverse=True)
    inverse = np.asarray(inverse).ravel()
    data_groups = merge_groups(obs[:, -1] <= 0) if merge else [[i] for i in range(len(obs))]
    fallback = 0
    for k, pattern in enumerate(patterns):
        sel = inverse == k
        groups = merge_groups(pattern) if merge else [[i] for i in range(len(pattern))]
        fallback += int(sel.sum()) if groups != data_groups else 0
        total = np.zeros(int(sel.sum()))
        for g in groups:
            total = total + _ratio_or_inf(b[sel][:, g].sum(axis=1), cc[sel][:, g].sum(axis=1), d[sel][:, g].sum(axis=1))
        split[sel] = total
    one_bin = _ratio_or_inf(b.sum(axis=1), cc.sum(axis=1), d.sum(axis=1))
    with np.errstate(divide="ignore", invalid="ignore"):
        ratio = np.where(np.isinf(split), np.inf, split / one_bin)
    return {"toys": toys, "split": split, "one_bin": one_bin, "ratio": ratio, "kept": float(np.isfinite(split).mean()),
            "fallback": fallback / n_toys}


# ----------------------------------------------------------------------------- section 1: the problem

def synthetic_regions(n, a, b):
    """Expected A, B, C, D per bin of a population of ``n`` events per bin whose isolation passes
    with probability ``a`` and |dphi| with probability ``b``, independently within each bin."""
    n, a, b = (np.asarray(v, float) for v in (n, a, b))
    return {"A": n * a * b, "B": n * (1 - a) * b, "C": n * a * (1 - b), "D": n * (1 - a) * (1 - b)}


def synthetic_check(n, a, b):
    """The identity of section 1 on a synthetic population: ``{"A": the true sum of A, "one_bin":
    (sum B)(sum C)/(sum D), "split": sum over bins of B C / D, "mixing": split / one_bin, "covariance":
    1 + cov_D(beta, r) / (<beta>_D <r>_D), "closure": A / one_bin}``.  ``split`` equals ``A`` by
    construction, so the one-bin closure equals the mixing factor, which equals the covariance form."""
    reg = synthetic_regions(n, a, b)
    B, C, D = reg["B"], reg["C"], reg["D"]
    one_bin = B.sum() * C.sum() / D.sum()
    split = float(np.sum(B * C / D))
    beta, r = B / D, C / D
    w = D / D.sum()
    mb, mr = np.sum(w * beta), np.sum(w * r)
    cov = np.sum(w * (beta - mb) * (r - mr))
    return {"A": float(reg["A"].sum()), "one_bin": float(one_bin), "split": split, "mixing": split / one_bin,
            "covariance": float(1 + cov / (mb * mr)), "closure": float(reg["A"].sum() / one_bin)}


# Three synthetic populations of section 1: events per bin n, isolation pass probability a and |dphi|
# pass probability b over five mass bins.
SYNTHETIC = {
    "a flat, b rising": ((400, 300, 120, 60, 20), (0.35, 0.35, 0.35, 0.35, 0.35), (0.72, 0.78, 0.85, 0.85, 0.90)),
    "a rising, b flat": ((400, 300, 120, 60, 20), (0.30, 0.35, 0.45, 0.50, 0.60), (0.78, 0.78, 0.78, 0.78, 0.78)),
    "both rising": ((400, 300, 120, 60, 20), (0.30, 0.35, 0.45, 0.50, 0.60), (0.72, 0.78, 0.85, 0.85, 0.90)),
    "a falling, b rising": ((400, 300, 120, 60, 20), (0.60, 0.50, 0.45, 0.35, 0.30), (0.72, 0.78, 0.85, 0.85, 0.90)),
}


def synthetic_table(populations=None):
    """:func:`synthetic_check` of each population of :data:`SYNTHETIC`, as a frame."""
    populations = SYNTHETIC if populations is None else populations
    rows = []
    for name, (n, a, b) in populations.items():
        r = synthetic_check(n, a, b)
        rows.append({"population": name, "true A": r["A"], "one-bin B C/D": r["one_bin"],
                     "sum over bins of B C/D": r["split"], "mixing factor": r["mixing"],
                     "1 + cov_D(beta, r)/(<beta><r>)": r["covariance"], "A / one-bin": r["closure"]})
    return pd.DataFrame(rows)


def pass_table(cube, cfg, planes=PLANES, bins=FIVE):
    """Per plane (or validation region, :func:`region_counts`) and mass bin: the counts, a = C / (C + D)
    and b = B / (B + D) with their 68% Clopper-Pearson intervals (NaN where the denominator is empty).
    A is reported as stored (NaN in the hidden box) and enters neither a nor b."""
    rows = []
    for plane in planes:
        for (label, (lo, hi)), (A, B, C, D) in zip(bins, region_counts(cube, cfg, plane, bins)):
            row = {"plane": region_name(plane), "bin": label, "M": centre(lo, hi), "A": A, "B": B, "C": C, "D": D}
            for name, k, n in (("a", C, C + D), ("b", B, B + D)):
                if n > 0:
                    lo_, hi_ = tl.clopper_pearson(k, n)
                    row.update({name: k / n, f"{name} minus": k / n - lo_, f"{name} plus": hi_ - k / n})
                else:
                    row.update({name: np.nan, f"{name} minus": np.nan, f"{name} plus": np.nan})
            rows.append(row)
    return pd.DataFrame(rows)


def spectra_table(cube, cfg, planes=PLANES, bins=FIVE):
    """Per plane, the share of B, C and D in each mass bin (each region's spectrum normalised to one
    over the bins) with the counts."""
    rows = []
    for plane in planes:
        c = counts(bin_regions(cube, cfg, plane, bins))
        tot = c[:, 1:].sum(axis=0)
        for (label, _), row in zip(bins, c):
            rows.append({"plane": plane, "bin": label, **{x: row[i + 1] for i, x in enumerate("BCD")},
                         **{f"{x} share": row[i + 1] / tot[i] for i, x in enumerate("BCD")}})
    return pd.DataFrame(rows)


def mixing_factor(cube, cfg, plane, bins):
    """The mixing factor of ``plane`` over ``bins``: ``{"bins": the bins after :func:`merge_empty`,
    "windows": their (B, C, D), "split": sum of B C / D, "one_bin": (sum B)(sum C)/(sum D), "ratio":
    split / one_bin, "raw": the (B, C, D) before merging, "merged": whether a bin was merged}``.  Reads
    B, C, D only."""
    bcd = counts(bin_regions(cube, cfg, plane, bins))[:, 1:]
    merged_bins, m = merge_empty(bins, bcd)
    split = float(np.sum(m[:, 0] * m[:, 1] / m[:, 2]))
    tot = m.sum(axis=0)
    one_bin = float(tot[0] * tot[1] / tot[2])
    return {"bins": merged_bins, "windows": [tuple(row) for row in m], "raw": [tuple(row) for row in bcd],
            "split": split, "one_bin": one_bin, "ratio": split / one_bin, "merged": len(merged_bins) < len(bins)}


def mixing_bootstrap(bcd, n_toys=N_TOYS, seed=SEED, merge=True):
    """:func:`merged_toys` of the per-bin (B, C, D) before merging (``merge``: whether the binning has
    the merge rule, :func:`merge_rule`), summarised: the 16/50/84th percentiles over all toys of the
    mixing factor and of the summed estimate, the share of toys below one, and the share with a
    finite factor."""
    t = merged_toys(bcd, n_toys=n_toys, seed=seed, merge=merge)
    ratio = t["ratio"][~np.isnan(t["ratio"])]
    return {"ratio": nl.interval68(ratio), "split": nl.interval68(t["split"]), "one_bin": nl.interval68(t["one_bin"]),
            "below_one": float(np.mean(ratio < 1)), "kept": float(np.isfinite(ratio).mean())}


def mixing_table(cube, cfg, planes=PLANES, binnings=None, n_toys=N_TOYS, seed=SEED):
    """Per plane and binning: the bins used, the summed and one-bin estimates, the mixing factor with
    its bootstrap interval, the share of toys below one and the share with a finite factor."""
    binnings = BINNINGS if binnings is None else binnings
    rows = []
    for plane in planes:
        for name, bins in binnings.items():
            m = mixing_factor(cube, cfg, plane, bins)
            boot = mixing_bootstrap(m["raw"], n_toys, seed, merge_rule(bins)) if len(m["windows"]) > 1 else None
            rows.append({"plane": plane, "binning": name, "bins": ", ".join(lab for lab, _ in m["bins"]),
                         "split": m["split"], "one_bin": m["one_bin"], "ratio": m["ratio"],
                         "lo": boot["ratio"][0] if boot else np.nan, "med": boot["ratio"][1] if boot else np.nan,
                         "hi": boot["ratio"][2] if boot else np.nan,
                         "below_one": boot["below_one"] if boot else np.nan, "kept": boot["kept"] if boot else 1.0,
                         "merged": m["merged"], "windows": m["windows"]})
    return pd.DataFrame(rows)


def mixing_sensitivity(windows):
    """The mixing factor with one more event in D of each bin in turn: ``[(bin index, ratio)]``.  It
    shows how much the factor rests on the smallest D counts."""
    m = np.asarray(windows, float)
    out = []
    for i in range(len(m)):
        x = m.copy()
        x[i, 2] += 1
        tot = x.sum(axis=0)
        out.append((i, float(np.sum(x[:, 0] * x[:, 1] / x[:, 2]) / (tot[0] * tot[1] / tot[2]))))
    return out


def shape_test(test, ref, n_toys=N_TOYS, seed=SEED):
    """Is the a(M) of one plane (``test``, an array of (C, D) per bin) the a(M) of another (``ref``)
    up to a constant factor?  The reference shape p_w = C/(C+D) is scaled by k so that the expected
    C of the test plane sums to its observed C; chi2 = sum (C - n k p)^2 / (n q (1 - q) + (n k)^2
    var p), q = k p, over the bins where the test plane has events, ndf = those bins - 1.  The
    p-value is given from the chi2 distribution and from binomial toys of both planes at the fitted
    shape (the small bins make the asymptotic one approximate).  Returns ``{"chi2", "ndf", "p", "p_toys",
    "k", "pulls", "used"}``."""
    test, ref = np.asarray(test, float), np.asarray(ref, float)
    n_t, n_r = test.sum(axis=1), ref.sum(axis=1)
    used = (n_t > 0) & (n_r > 0)
    c_t, n_t, c_r, n_r = test[used, 0], n_t[used], ref[used, 0], n_r[used]

    def chi2_of(c_t, c_r):
        p = c_r / n_r
        k = c_t.sum(axis=-1, keepdims=True) / np.sum(n_t * p, axis=-1, keepdims=True)
        q = np.clip(k * p, 1e-9, 1 - 1e-9)
        var = n_t * q * (1 - q) + (n_t * k) ** 2 * p * (1 - p) / n_r
        pulls = (c_t - n_t * q) / np.sqrt(var)
        return np.sum(pulls**2, axis=-1), k[..., 0], pulls

    chi2, k, pulls = chi2_of(c_t, c_r)
    ndf = int(used.sum()) - 1
    rng = np.random.default_rng(seed)
    p_fit = np.clip(k * c_r / n_r, 0, 1)
    toys_t = rng.binomial(n_t.astype(int), p_fit, size=(n_toys, len(n_t))).astype(float)
    toys_r = rng.binomial(n_r.astype(int), c_r / n_r, size=(n_toys, len(n_r))).astype(float)
    ok = (toys_r > 0).all(axis=1) & (toys_t.sum(axis=1) > 0)
    chi2_toys, _, _ = chi2_of(toys_t[ok], toys_r[ok])
    return {"chi2": float(chi2), "ndf": ndf, "p": float(stats.chi2.sf(chi2, ndf)) if ndf > 0 else np.nan,
            "p_toys": float(np.mean(chi2_toys >= chi2)), "k": float(k), "pulls": pulls, "used": used}


def log_odds_ratio(cd):
    """ln of the isolation odds C/D in the second of two windows over the first (``cd``: (C, D) per
    window), with its first-order error sqrt(sum of 1/count); NaN when a count is zero."""
    cd = np.asarray(cd, float)
    if (cd <= 0).any():
        return (np.nan, np.nan)
    return float(np.log(cd[1, 0] / cd[1, 1]) - np.log(cd[0, 0] / cd[0, 1])), float(np.sqrt(np.sum(1 / cd)))


def separation(test, ref):
    """The difference of the log odds ratios (:func:`log_odds_ratio`) of two regions over two windows,
    its first-order error and their ratio, the separation in standard deviations; with every count
    scaled by a factor f (more data at the same rates) the separation grows as sqrt(f)."""
    v_t, e_t = log_odds_ratio(test)
    v_r, e_r = log_odds_ratio(ref)
    diff, err = v_t - v_r, float(np.hypot(e_t, e_r))
    return {"diff": float(diff), "err": err, "z": float(diff / err) if err > 0 else np.nan}


# ----------------------------------------------------------------------------- route 1: stratify

def bootstrap_bins(bcd, n_toys=N_TOYS, seed=SEED, merge=True):
    """Poisson toys of the per-bin (B, C, D) before merging: ``{"groups": the bins the data merge into,
    "per_bin": [16/50/84th percentiles over all toys of B C / D of each merged data bin, drawn alone
    as ``nominal_lib.bootstrap_abcd`` draws one window (+inf where D is drawn as 0)], "sum":
    percentiles of the summed estimate (:func:`merged_toys` with ``merge``), "kept": share of toys
    with a finite sum, "fallback": share of toys merged differently from the data}``."""
    obs = np.asarray(bcd, float)
    t = merged_toys(obs, n_toys, seed, merge)
    groups = merge_groups(obs[:, 2] <= 0)
    per = [nl.interval68(nl.bootstrap_abcd([tuple(obs[g].sum(axis=0))], n_toys, seed)["one_bin"]) for g in groups]
    return {"groups": groups, "per_bin": per, "sum": nl.interval68(t["split"]), "kept": t["kept"], "fallback": t["fallback"]}


def binned_vrs(bins, vrs=None):
    """The validation regions of nominal.ipynb section 8 that lie above 150 GeV, restricted to each
    bin: ``{bin label: [VR]}``.  The regions below 150 GeV (DD < 150, V2-V4) have no events in a bin
    above it and are left out."""
    vrs = list(nl.our_vrs()) + list(nl.DONGYUB_VRS) if vrs is None else list(vrs)
    above = [v for v in vrs if tuple(v.mass) == nl.HIGH]
    return {label: [dataclasses.replace(v, mass=rng) for v in above] for label, rng in bins}


def per_bin_systematic(cube, cfg, bins, vrs=None, max_precision=MAX_PRECISION):
    """The rule of nominal.ipynb section 8 in each bin: among the regions of :func:`binned_vrs` whose
    kappa in the bin is measured to ``max_precision`` or better (and that the rule admits: no split
    of a plane), the largest max(|kappa - 1|, uncertainty facing one).  ``{bin label: {"value",
    "region", "qualifying": [names], "table": the nominal_lib.systematic_table of the bin}}``;
    value NaN and region None when no region qualifies."""
    out = {}
    for label, regions in binned_vrs(bins, vrs).items():
        table = nl.systematic_table(cube, regions, {"ours": cfg}, max_precision)
        value, region = nl.closure_systematic(table)
        out[label] = {"value": value, "region": region, "qualifying": list(table.loc[table["qualifies"], "region"]),
                      "table": table}
    return out


def stratified_table(cube, cfg, bins, search="DD", ref_planes=OPEN_PLANES, vrs=None, n_toys=N_TOYS, seed=SEED):
    """Route 1 per bin: B, C, D of the search plane (bins with D = 0 merged, :func:`merge_empty`), its
    B C / D with the first-order error (linear propagation of the Poisson errors of B, C and D) and
    the bootstrap interval, kappa of each reference plane in the same bin, and the closure systematic
    of the bin (:func:`per_bin_systematic`) with the region that sets it; a last row with the sum and
    its bootstrap interval (:func:`bootstrap_bins`, with the binning's merge rule, :func:`merge_rule`).
    The systematic on the sum takes the bins as fully correlated (sum of value x systematic) and is
    undefined when a bin has none."""
    bcd = counts(bin_regions(cube, cfg, search, bins))[:, 1:]
    merged, m = merge_empty(bins, bcd)
    boot = bootstrap_bins(bcd, n_toys, seed, merge_rule(bins))
    syst = per_bin_systematic(cube, cfg, merged, vrs)
    rows = []
    for i, (label, rng) in enumerate(merged):
        B, C, D = m[i]
        est = B * C / D
        err = est * np.sqrt(1 / B + 1 / C + 1 / D) if min(B, C) > 0 else np.nan
        row = {"bin": label, "B": B, "C": C, "D": D, "estimate": est, "error": err,
               "lo": boot["per_bin"][i][0], "hi": boot["per_bin"][i][2]}
        for p in ref_planes:
            k = ps.kappa(ps.regions(cube, dataclasses.replace(cfg, mass=rng), p))
            row.update({f"kappa {p}": k[0], f"kappa {p} minus": k[1], f"kappa {p} plus": k[2],
                        f"precision {p}": el.precision(k)})
        row.update({"syst": syst[label]["value"], "syst region": syst[label]["region"],
                    "qualifying": ", ".join(syst[label]["qualifying"]) or "none"})
        rows.append(row)
    df = pd.DataFrame(rows)
    est, syst, err = (df[c].to_numpy(float) for c in ("estimate", "syst", "error"))   # numpy sums: NaN propagates
    total = float(np.sum(est))
    syst_abs = float(np.sum(est * syst))
    rows.append({"bin": "sum", "B": m[:, 0].sum(), "C": m[:, 1].sum(), "D": m[:, 2].sum(), "estimate": total,
                 "error": float(np.sqrt(np.sum(err**2))), "lo": boot["sum"][0], "hi": boot["sum"][2],
                 "syst": syst_abs / total if np.isfinite(syst_abs) else np.nan, "syst region": None,
                 "qualifying": ""})
    out = pd.DataFrame(rows)
    out.attrs["kept"] = boot["kept"]
    out.attrs["fallback"] = boot["fallback"]
    return out


# ----------------------------------------------------------------------------- route 2: a smooth ratio

def logistic_fit(c, n, x, slope=None, max_iter=100, tol=1e-10):
    """Maximum-likelihood fit of p(x) = 1 / (1 + exp(-(alpha + s x))) to ``c`` successes of ``n`` trials
    per bin (a binomial likelihood: the Poisson likelihood of C and D conditioned on C + D), by
    iteratively reweighted least squares.  ``c`` and ``n`` may carry leading dimensions (toys) over the
    bins of ``x``; ``slope`` fixes s (a scalar or an array over the leading dimensions) and fits alpha
    alone.  Returns ``{"alpha", "slope", "cov" (of alpha, s; alpha alone when the slope is fixed),
    "converged", "deviance", "ndf"}``."""
    c, n = np.asarray(c, float), np.asarray(n, float)
    x = np.broadcast_to(np.asarray(x, float), c.shape)
    lead = c.shape[:-1]
    p0 = np.clip(c.sum(axis=-1) / np.maximum(n.sum(axis=-1), 1e-12), 1e-6, 1 - 1e-6)
    alpha = np.log(p0 / (1 - p0))
    fixed = slope is not None
    s = np.broadcast_to(np.asarray(slope, float), lead).copy() if fixed else np.zeros(lead)
    converged = np.zeros(lead, bool)
    for _ in range(max_iter):
        eta = alpha[..., None] + s[..., None] * x
        p = 1 / (1 + np.exp(-eta))
        w = n * p * (1 - p)
        g0 = np.sum(c - n * p, axis=-1)
        h00 = np.sum(w, axis=-1)
        if fixed:
            step_a = np.where(h00 > 0, g0 / np.where(h00 > 0, h00, 1), 0.0)
            step_s = np.zeros(lead)
        else:
            g1 = np.sum((c - n * p) * x, axis=-1)
            h01 = np.sum(w * x, axis=-1)
            h11 = np.sum(w * x * x, axis=-1)
            det = h00 * h11 - h01**2
            ok = det > 1e-12
            safe = np.where(ok, det, 1.0)
            step_a = np.where(ok, (h11 * g0 - h01 * g1) / safe, 0.0)
            step_s = np.where(ok, (h00 * g1 - h01 * g0) / safe, 0.0)
        step_a, step_s = np.clip(step_a, -5, 5), np.clip(step_s, -5, 5)
        alpha = alpha + step_a
        s = s + step_s
        converged = (np.abs(step_a) < tol) & (np.abs(step_s) < tol)
        if converged.all():
            break
    eta = alpha[..., None] + s[..., None] * x
    p = np.clip(1 / (1 + np.exp(-eta)), 1e-300, 1 - 1e-16)
    w = n * p * (1 - p)
    h00, h01, h11 = np.sum(w, axis=-1), np.sum(w * x, axis=-1), np.sum(w * x * x, axis=-1)
    with np.errstate(divide="ignore", invalid="ignore"):         # a degenerate toy (no C or no D) has no finite error
        if fixed:
            cov = 1 / h00
        else:
            det = h00 * h11 - h01**2
            cov = np.stack([np.stack([h11, -h01], -1), np.stack([-h01, h00], -1)], -2) / det[..., None, None]
    with np.errstate(divide="ignore", invalid="ignore"):
        t1 = np.where(c > 0, c * np.log(c / (n * p)), 0.0)
        t2 = np.where(n - c > 0, (n - c) * np.log((n - c) / (n * (1 - p))), 0.0)
    deviance = 2 * np.sum(t1 + t2, axis=-1)
    used = np.sum(n > 0, axis=-1)
    return {"alpha": alpha, "slope": s, "cov": cov, "converged": converged, "deviance": deviance,
            "ndf": used - (1 if fixed else 2)}


def logistic_a(alpha, slope, mass, pivot=PIVOT):
    """a(M) of the logistic in ln M."""
    return 1 / (1 + np.exp(-(alpha + slope * np.log(np.asarray(mass, float) / pivot))))


def logistic_r(alpha, slope, mass, pivot=PIVOT):
    """r(M) = a / (1 - a) = exp(alpha) (M / pivot)^slope."""
    return np.exp(alpha + slope * np.log(np.asarray(mass, float) / pivot))


def overdispersion(fit):
    """sqrt(deviance / ndf) of a :func:`logistic_fit` when it exceeds one, else one: the factor by which
    the fit's errors are scaled when the data scatter about the fitted curve more than binomial
    counts would."""
    dev, ndf = float(fit["deviance"]), int(fit["ndf"])
    return float(np.sqrt(dev / ndf)) if ndf > 0 and dev > ndf else 1.0


def smooth_route(cube, cfg, plane, bins=FIVE, slope_plane=None, n_toys=N_TOYS, seed=SEED):
    """Route 2 in one plane or validation region (``plane``: a plane name or a ``nominal_lib.VR``):
    a(M) fitted to the per-bin (C, C + D) with the logistic in ln M at the bin centres
    (:func:`centre`), and the prediction sum_w B_w r(M_w).  With ``slope_plane`` (a plane name) the
    slope is the one fitted in that plane and only the intercept is fitted here.  Poisson toys of
    every count (both when the slope is borrowed; A only where it is open) are refitted one by one.
    When a fit's deviance exceeds its ndf, its errors and the spread of its toys about the fitted
    values are scaled by sqrt(deviance / ndf) (:func:`overdispersion`; a borrowed slope carries the
    scale of its own plane's fit).  Returns the fit (alpha, slope and their errors, deviance/ndf, the
    scale), the prediction with its bootstrap interval, and where A is open the observed A, kappa =
    A / prediction and its bootstrap interval; the prediction and kappa are NaN when the region has no
    C or no D, where the fit has no finite solution."""
    name = region_name(plane)
    c4 = region_counts(cube, cfg, plane, bins)
    x = np.log(np.array([centre(*rng) for _, rng in bins]) / PIVOT)
    A, B, C, D = c4.T
    fixed, ref_scale = None, 1.0
    if slope_plane == name:
        slope_plane = None                       # the plane's own slope: the ordinary two-parameter fit
    if slope_plane is not None:
        ref = region_counts(cube, cfg, slope_plane, bins)
        ref_fit = logistic_fit(ref[:, 2], ref[:, 2] + ref[:, 3], x)
        fixed, ref_scale = float(ref_fit["slope"]), overdispersion(ref_fit)
    fit = logistic_fit(C, C + D, x, slope=fixed)
    alpha, slope = float(fit["alpha"]), float(fit["slope"])
    scale = overdispersion(fit)
    pred = float(np.sum(B * np.exp(alpha + slope * x)))
    rng = np.random.default_rng(seed)
    tB, tC, tD = (rng.poisson(v, size=(n_toys, len(v))).astype(float) for v in (B, C, D))
    if slope_plane is not None:
        tCr, tDr = (rng.poisson(v, size=(n_toys, len(v))).astype(float) for v in (ref[:, 2], ref[:, 3]))
        s_toys = fixed + ref_scale * (logistic_fit(tCr, tCr + tDr, x)["slope"] - fixed)
    else:
        s_toys = None
    tf = logistic_fit(tC, tC + tD, x, slope=s_toys)
    a_toys = alpha + scale * (tf["alpha"] - alpha)
    s_toys = slope + scale * (tf["slope"] - slope) if s_toys is None else s_toys
    with np.errstate(over="ignore"):
        pred_toys = np.sum(tB * np.exp(a_toys[:, None] + s_toys[:, None] * x), axis=1)
    ok = tf["converged"] & np.isfinite(pred_toys)
    out = {"plane": name, "x": x, "counts": c4, "alpha": alpha, "slope": slope, "fixed_slope": fixed,
           "scale": scale, "ref_scale": ref_scale,
           "alpha_err": scale * float(np.sqrt(fit["cov"] if fixed is not None else fit["cov"][0, 0])),
           "slope_err": (scale * float(np.sqrt(fit["cov"][1, 1])) if fixed is None else float(np.std(s_toys[ok]))),
           "deviance": float(fit["deviance"]), "ndf": int(fit["ndf"]), "pred": pred,
           "pred_interval": nl.interval68(pred_toys[ok]), "converged": float(ok.mean()),
           "slope_toys": s_toys[ok], "alpha_toys": a_toys[ok]}
    out["p"] = float(stats.chi2.sf(out["deviance"], out["ndf"])) if out["ndf"] > 0 else np.nan
    if C.sum() <= 0 or D.sum() <= 0:              # no C or no D: the logistic runs off to a = 0 or 1
        out.update({"pred": np.nan, "pred_interval": (np.nan,) * 3, "A": float(A.sum()), "kappa": (np.nan,) * 3})
    elif np.all(np.isfinite(A)):
        tA = rng.poisson(A, size=(n_toys, len(A))).astype(float)
        with np.errstate(divide="ignore", invalid="ignore"):
            k_toys = tA[ok].sum(axis=1) / pred_toys[ok]
        k = float(A.sum() / pred) if pred > 0 else np.nan
        lo, _, hi = nl.interval68(k_toys)
        out.update({"A": float(A.sum()), "kappa": (k, k - lo, hi - k) if np.isfinite(k) else (np.nan,) * 3})
    else:
        out.update({"A": np.nan, "kappa": (np.nan, np.nan, np.nan)})
    return out


def slope_consistency(fits):
    """chi2 of the fitted slopes of several planes (``{plane: smooth_route result}``) against their
    weighted mean, with ndf and p, and the pairwise separations in standard deviations."""
    s = np.array([f["slope"] for f in fits.values()])
    e = np.array([f["slope_err"] for f in fits.values()])
    w = 1 / e**2
    mean = float(np.sum(w * s) / np.sum(w))
    chi2 = float(np.sum(((s - mean) / e) ** 2))
    names = list(fits)
    pairs = {(names[i], names[j]): float((s[i] - s[j]) / np.hypot(e[i], e[j]))
             for i in range(len(names)) for j in range(i + 1, len(names))}
    return {"mean": mean, "chi2": chi2, "ndf": len(s) - 1, "p": float(stats.chi2.sf(chi2, len(s) - 1)), "pairs": pairs}


# ----------------------------------------------------------------------------- route 3: a flattened cut

def flat_cfg(cfg, edge, lo, hi):
    """The working point of one mass bin with the isolation cut at ``edge``: passes below it, fails at
    or above max(edge, 0.25).  Below 0.25 the band [edge, 0.25) is a buffer that belongs to no region:
    in the search plane it lies inside the hidden box, so a fail region reaching into it would be
    hidden."""
    return dataclasses.replace(cfg, wp0=edge, fail0=max(edge, io.BLIND_ISO0), mass=(lo, hi))


def flat_regions(cube, cfg, plane, bins, edges):
    """Per bin, A/B/C/D of ``plane`` with the bin's own isolation edge (:func:`flat_cfg`)."""
    return [ps.regions(cube, flat_cfg(cfg, e, *rng), plane) for (_, rng), e in zip(bins, edges)]


def odds_table(cube, cfg, plane, bins=FIVE, candidates=EDGE_CANDIDATES):
    """``(C, D)`` per bin and candidate edge: an array of shape (bins, edges, 2)."""
    out = np.zeros((len(bins), len(candidates), 2))
    for i, (_, rng) in enumerate(bins):
        for j, e in enumerate(candidates):
            r = ps.regions(cube, flat_cfg(cfg, e, *rng), plane)
            out[i, j] = r["C"][0], r["D"][0]
    return out


def choose_edges(cd, candidates=EDGE_CANDIDATES, anchor=0, anchor_edge=ANCHOR_EDGE):
    """Per bin the candidate edge whose a = C / (C + D) is closest to the anchor bin's a at
    ``anchor_edge`` (the target), from :func:`odds_table`'s ``cd``.  Ties go to the edge nearest the
    anchor edge in the list; a bin with no events at any edge keeps the anchor edge.  Returns
    ``{"edges", "target", "a": a per bin and candidate}``."""
    cd = np.asarray(cd, float)
    n = cd[..., 0] + cd[..., 1]
    with np.errstate(divide="ignore", invalid="ignore"):
        a = np.where(n > 0, cd[..., 0] / np.where(n > 0, n, 1), np.nan)
    j0 = list(candidates).index(anchor_edge)
    target = a[anchor, j0]
    edges = []
    for i in range(len(cd)):
        if i == anchor or not np.isfinite(a[i]).any():
            edges.append(anchor_edge)
            continue
        dist = np.abs(a[i] - target)
        dist = np.where(np.isfinite(dist), dist, np.inf)
        best = np.flatnonzero(np.isclose(dist, dist.min(), rtol=0, atol=1e-12))
        edges.append(candidates[min(best, key=lambda j: abs(j - j0))])
    return {"edges": tuple(edges), "target": float(target), "a": a}


def flatness(cd_bins):
    """a per bin from ``(C, D)`` per bin and the chi2 against a constant (the pooled a), binomial
    variance at the pooled value, over the bins with events: ``{"a", "pooled", "chi2", "ndf", "p"}``."""
    cd = np.asarray(cd_bins, float)
    n = cd.sum(axis=1)
    used = n > 0
    pooled = cd[used, 0].sum() / n[used].sum()
    with np.errstate(divide="ignore", invalid="ignore"):
        a = np.where(used, cd[:, 0] / np.where(used, n, 1), np.nan)
    chi2 = float(np.sum((cd[used, 0] - n[used] * pooled) ** 2 / (n[used] * pooled * (1 - pooled))))
    ndf = int(used.sum()) - 1
    cp = [tl.clopper_pearson(c, m) if m > 0 else (np.nan, np.nan) for c, m in zip(cd[:, 0], n)]
    return {"a": a, "minus": a - np.array([lo for lo, _ in cp]), "plus": np.array([hi for _, hi in cp]) - a,
            "c": cd[:, 0], "n": n, "pooled": float(pooled), "chi2": chi2, "ndf": ndf,
            "p": float(stats.chi2.sf(chi2, ndf)) if ndf > 0 else np.nan}


def summed(regs):
    """A/B/C/D summed over bins: ``{region: (sumw, sumw2)}`` (NaN when any bin's region is hidden)."""
    return {x: (float(np.sum([r[x][0] for r in regs])), float(np.sum([r[x][1] for r in regs]))) for x in "ABCD"}


def flat_route(cube, cfg, plane, bins, edges, n_toys=N_TOYS, seed=SEED):
    """Route 3 in one plane: the one-bin estimate B'C'/D' with the per-bin edges, its first-order error
    and bootstrap interval, the mixing factor over the bins with its bootstrap interval, and where A
    is open kappa (Clopper-Pearson ratios) and the observed A."""
    regs = flat_regions(cube, cfg, plane, bins, edges)
    tot = summed(regs)
    pred = ps.prediction(tot)
    boot_one = nl.interval68(nl.bootstrap_abcd([tuple(tot[x][0] for x in "BCD")], n_toys, seed)["one_bin"])
    bcd = counts(regs)[:, 1:]
    merged, m = merge_empty(bins, bcd)
    split = float(np.sum(m[:, 0] * m[:, 1] / m[:, 2]))
    mix = mixing_bootstrap(bcd, n_toys, seed, merge_rule(bins)) if len(m) > 1 else None
    return {"plane": plane, "regions": tot, "per_bin": counts(regs), "mixing_windows": [tuple(r) for r in m],
            "pred": pred, "pred_interval": boot_one,
            "mixing": split / pred[0], "mixing_interval": mix["ratio"] if mix else (np.nan,) * 3,
            "mixing_bins": [lab for lab, _ in merged], "kappa": ps.kappa(tot), "A": tot["A"][0],
            "flatness": flatness(counts(regs)[:, 2:])}


def flat_kappa(cube, cfg, region, bins, edges):
    """kappa of a plane or validation region with the per-bin isolation edges of route 3, the regions
    summed over the bins (Clopper-Pearson-based interval, ``plane_search_lib.kappa``)."""
    vr = as_vr(region)
    regs = [nl.vr_regions(cube, dataclasses.replace(vr, mass=rng), flat_cfg(cfg, e, *rng)) for (_, rng), e in zip(bins, edges)]
    return ps.kappa(summed(regs))


def signal_cost(base, names, cfg, bins, edge_sets, topology="2mu2e", prefix="abcd8d"):
    """Per signal point, the search-region yield (A of DD summed over ``bins``) with each set of per-bin
    edges of ``edge_sets`` (``{label: edges}``) over the yield with the nominal cut in every bin, at the
    stored cross section.  Signal cubes are never hidden; one sample is loaded at a time."""
    rows = []
    for name in names:
        cube = tl.sample_cubes(base, "signal", [name], topologies=(topology,), prefix=prefix)[name][topology]
        nominal = summed(flat_regions(cube, cfg, "DD", bins, [cfg.wp0] * len(bins)))["A"][0]
        row = {"point": name, "mediator": nl.mediator_mass(name), "yield": nominal}
        for label, edges in edge_sets.items():
            y = summed(flat_regions(cube, cfg, "DD", bins, edges))["A"][0]
            row[label] = y / nominal if nominal > 0 else np.nan
        rows.append(row)
        del cube
    return pd.DataFrame(rows)


def cost_summary(table, labels):
    """Per mediator mass: the number of points and the min-max of each yield ratio."""
    rows = []
    for m, part in table.groupby("mediator"):
        row = {"mediator mass [GeV]": f"{m:g}", "points": len(part)}
        for lab in labels:
            row[lab] = f"{part[lab].min():.2f}–{part[lab].max():.2f}"
        rows.append(row)
    return pd.DataFrame(rows)


# ----------------------------------------------------------------------------- comparison

def rule(readings, max_precision=MAX_PRECISION):
    """The closure-systematic rule over ``[(name, (kappa, minus, plus))]``: among the readings measured
    to ``max_precision`` or better, the largest max(|kappa - 1|, uncertainty facing one); (NaN, None)
    when none qualifies."""
    best = (np.nan, None)
    for name, k in readings:
        prec = el.precision(k)
        if not np.isfinite(k[0]) or not np.isfinite(prec) or prec > max_precision:
            continue
        v = max(abs(k[0] - 1), nl.side_facing_one(k))
        if not np.isfinite(best[0]) or v > best[0]:
            best = (float(v), name)
    return best


def route_closure(cube, cfg, region, bins, n_toys=N_TOYS, seed=SEED):
    """kappa of the stratified estimate in an open plane or validation region: A summed over ``bins``
    over sum_w B_w C_w / D_w (bins with D = 0 merged, :func:`merge_empty`), with the interval from
    Poisson toys of all four counts per bin with the binning's merge rule (:func:`merged_toys`,
    :func:`merge_rule`)."""
    c4 = region_counts(cube, cfg, region, bins)
    if not np.all(np.isfinite(c4)):
        return (np.nan, np.nan, np.nan)
    _, m = merge_empty(bins, c4[:, 1:])
    pred = float(np.sum(_ratio_or_inf(m[:, 0], m[:, 1], m[:, 2])))
    k = float(c4[:, 0].sum() / pred)
    t = merged_toys(c4, n_toys, seed, merge_rule(bins))
    with np.errstate(divide="ignore", invalid="ignore"):
        k_toys = t["toys"][..., 0].sum(axis=1) / t["split"]
    lo, _, hi = nl.interval68(k_toys)
    return (k, k - lo, hi - k)


def rule_regions(cube, cfg, vrs=None):
    """The validation regions the rule of nominal.ipynb section 8 may use for every route above 150
    GeV: those of :func:`binned_vrs` at M >= 150 GeV that touch no hidden cell and differ from the
    search region only in named respects (a split of a plane does not qualify).  Whether a region's
    kappa is measured to 20% is judged per route, by :func:`rule`."""
    regions = binned_vrs(ONE, vrs)[ONE[0][0]]
    table = nl.systematic_table(cube, regions, {"ours": cfg})
    excluded = table["reason"].str.contains("hidden|differs in")
    names = set(table.loc[~excluded, "region"])
    return [v for v in regions if v.name in names]


def fmt_p(p):
    """"p = 0.12", "p = 0.004" or "p < 0.001"."""
    if not np.isfinite(p):
        return "p n/a"
    if p < 0.001:
        return "p < 0.001"
    return f"p = {p:.3f}" if p < 0.01 else f"p = {p:.2f}"


def fmt_interval(x, digits=0):
    """"lo–hi" of a (16th, 50th, 84th) percentile triple (``nominal_lib.fmt_range``: an infinite upper
    end prints as no upper bound)."""
    return nl.fmt_range(x, digits)


def _syst_text(syst):
    return "no region qualifies" if not np.isfinite(syst[0]) else f"{100 * syst[0]:.0f}% ({syst[1]})"


def comparison(cube, cfg, flat_edges, cost=None, n_toys=N_TOYS, seed=SEED):
    """One record per route for the comparison table of mass_decorrelation.ipynb: the DD estimate with
    its bootstrap 16/50/84th percentiles, the mixing factor the estimate still carries, the closure of
    the route in PP and PD, the systematic by the rule over the same validation regions for every
    route (:func:`rule_regions`; per bin for route 1, :func:`per_bin_systematic`) with the readings
    behind it, the free choices, fitted parameters and systematics the route brings, the smallest
    count it rests on, the signal yield ratio and what is left to validate.  ``flat_edges`` is
    ``{calibration plane: per-bin edges}`` of route 3; ``cost`` the :func:`signal_cost` table with a
    column per calibration label "<plane>-calibrated"."""
    out = []
    regions = rule_regions(cube, cfg)
    mix5 = mixing_factor(cube, cfg, "DD", FIVE)
    mix2 = mixing_factor(cube, cfg, "DD", TWO)
    b5, b2 = mixing_bootstrap(mix5["raw"], n_toys, seed, True), mixing_bootstrap(mix2["raw"], n_toys, seed, False)

    def closures(fn):
        return {p: fn(p) for p in OPEN_PLANES}

    # the one-bin baseline
    one = counts(bin_regions(cube, cfg, "DD", ONE))[0]
    est = one[1] * one[2] / one[3]
    k = closures(lambda p: route_closure(cube, cfg, p, ONE, n_toys, seed))
    readings = [(v.name, ps.kappa(nl.vr_regions(cube, v, cfg))) for v in regions]
    syst = rule(readings)
    out.append({"route": "one bin (nominal.ipynb)", "estimate": est,
                "interval": nl.interval68(nl.bootstrap_abcd([tuple(one[1:])], n_toys, seed)["one_bin"]),
                "mixing": f"{mix2['ratio']:.2f} between the windows ({fmt_interval(b2['ratio'], 2)}); "
                          f"{mix5['ratio']:.2f} over the bins ({fmt_interval(b5['ratio'], 2)})",
                "kappa": k, "syst": syst, "syst text": _syst_text(syst) + "; the mixing not covered",
                "readings": {"one bin": readings},
                "choices": 0, "parameters": "0", "n syst": "1 + the mixing caveat",
                "smallest": f"D = {one[3]:.0f}", "cost": "1", "validate": "the mixing factor in DD"})

    # route 1
    for name, bins in (("2 windows", TWO), ("5 bins", FIVE)):
        t = stratified_table(cube, cfg, bins, n_toys=n_toys, seed=seed)
        body, total = t[t["bin"] != "sum"], t[t["bin"] == "sum"].iloc[0]
        k = closures(lambda p: route_closure(cube, cfg, p, bins, n_toys, seed))
        per = [("none" if not np.isfinite(v) else f"{100 * v:.0f}%") for v in body["syst"]]
        n_undefined = int(np.sum(~np.isfinite(body["syst"].to_numpy(float))))
        within = (f"{mix5['split'] / total['estimate']:.2f} inside the windows (the five-bin sum over this one)"
                  if bins is TWO else "inside the bins: not measurable (C and D of DD are empty above 600 GeV)")
        rec = {"route": f"1, stratified, {name}" + (" (≥ 600 merged)" if len(body) < len(bins) else ""),
               "estimate": float(total["estimate"]), "interval": (total["lo"], np.nan, total["hi"]),
               "kept": t.attrs["kept"], "mixing": within, "kappa": k,
               "syst": (float(total["syst"]), None),
               "syst text": " / ".join(per) + ("" if n_undefined else f" per bin; {100 * total['syst']:.0f}% on the sum"),
               "choices": len(body) - 1, "parameters": "0",
               "n syst": f"{len(body)}" + (f", {n_undefined} without a qualifying region" if n_undefined else ""),
               "smallest": f"D = {body['D'].min():.0f}", "cost": "1",
               "validate": "kappa per bin with a displaced LJ" + (" above 300 GeV" if n_undefined == 0 else
                                                                  "; no region measures it above 300 GeV")}
        if bins is TWO:
            names = {v.name for v in regions}
            merged, _ = merge_empty(bins, counts(bin_regions(cube, cfg, "DD", bins))[:, 1:])
            rec["readings"] = {}
            for label, r in per_bin_systematic(cube, cfg, merged).items():
                tab = r["table"][r["table"]["region"].isin(names)]
                rec["readings"][f"1: {label.replace('>=', '≥ ')}"] = [
                    (row["region"], (row["kappa"], row["minus"], row["plus"])) for _, row in tab.iterrows()]
        out.append(rec)

    # route 2
    fits = {p: smooth_route(cube, cfg, p, n_toys=n_toys, seed=seed) for p in PLANES}
    shared = smooth_route(cube, cfg, "DD", slope_plane="PD", n_toys=n_toys, seed=seed)
    own = {v.name: (fits[v.name] if v.name in fits else smooth_route(cube, cfg, v, n_toys=n_toys, seed=seed)) for v in regions}
    readings = [(n, f["kappa"]) for n, f in own.items()]
    syst = rule(readings)
    out.append({"route": "2, logistic a(M), DD's own fit", "estimate": fits["DD"]["pred"],
                "interval": fits["DD"]["pred_interval"], "mixing": "absorbed by the fit",
                "kappa": {p: fits[p]["kappa"] for p in OPEN_PLANES}, "syst": syst,
                "syst text": _syst_text(syst) + " + the fit form", "readings": {"2, own slope": readings},
                "choices": 1, "parameters": "2 per plane", "n syst": "2 (closure, fit form)",
                "smallest": "D = 1 at 300–400 GeV sets the slope; B = 4 at ≥ 600 GeV is extrapolated",
                "cost": "1", "validate": f"the form (PD's fit: deviance {fits['PD']['deviance']:.1f}/{fits['PD']['ndf']})"})
    transfer = abs(fits["DD"]["pred"] - shared["pred"]) / shared["pred"]
    borrowed = {v.name: (own[v.name] if v.name == "PD" else smooth_route(cube, cfg, v, slope_plane="PD", n_toys=n_toys, seed=seed))
                for v in regions}
    readings = [(n, f["kappa"]) for n, f in borrowed.items()]
    syst_sh = rule(readings)
    out.append({"route": "2, logistic a(M), PD's slope", "estimate": shared["pred"], "interval": shared["pred_interval"],
                "mixing": "absorbed by the fit", "kappa": {p: borrowed[p]["kappa"] for p in OPEN_PLANES}, "syst": syst_sh,
                "syst text": f"{_syst_text(syst_sh)} + {100 * transfer:.0f}% for the slope (DD's own gives "
                             f"{fits['DD']['pred']:.0f})", "readings": {"2, PD's slope": readings},
                "transfer": transfer, "choices": 1, "parameters": "1 in DD + PD's slope",
                "n syst": "2 (closure, slope transfer)", "smallest": f"C + D = {(one[2] + one[3]):.0f} in DD for the intercept",
                "cost": "1", "validate": "that DD's slope is PD's"})

    # route 3
    for cal, edges in flat_edges.items():
        dd = flat_route(cube, cfg, "DD", FIVE, edges, n_toys, seed)
        readings = [(v.name, flat_kappa(cube, cfg, v, FIVE, edges)) for v in regions]
        syst = rule(readings)
        label = f"{cal}-calibrated"
        cost_text = "n/a"
        if cost is not None and label in cost:
            cost_text = f"{cost[label].min():.2f}–{cost[label].max():.2f}"
        out.append({"route": f"3, flattened cut, {label}", "estimate": dd["pred"][0], "interval": dd["pred_interval"],
                    "mixing": f"{dd['mixing']:.2f} ({fmt_interval(dd['mixing_interval'], 2)})",
                    "kappa": {p: flat_kappa(cube, cfg, p, FIVE, edges) for p in OPEN_PLANES}, "syst": syst,
                    "syst text": _syst_text(syst) + " + the calibration transfer", "readings": {f"3, {label}": readings},
                    "choices": len(edges), "parameters": "0",
                    "n syst": "2 (closure, calibration)",
                    "smallest": f"D = {min(w[2] for w in dd['mixing_windows']):.0f} in a bin; D = {dd['regions']['D'][0]:.0f} summed",
                    "cost": cost_text, "validate": "a flat a(M) in DD; a new search region", "edges": edges})
    return out


def comparison_view(records):
    """The comparison records as the table of the notebook."""
    rows = []
    for r in records:
        rows.append({"route": r["route"], "DD estimate (68% interval)": f"{r['estimate']:.1f} ({fmt_interval(r['interval'])})",
                     "mixing still carried": r["mixing"],
                     "closure PP": tl.fmt(r["kappa"]["PP"], 2), "closure PD": tl.fmt(r["kappa"]["PD"], 2),
                     "systematic by the rule": r["syst text"], "free choices": r["choices"],
                     "fitted parameters": r["parameters"], "systematics carried": r["n syst"],
                     "rests on": r["smallest"], "signal yield ratio": r["cost"], "still to validate": r["validate"]})
    return pd.DataFrame(rows)


def readings_view(records, max_precision=MAX_PRECISION):
    """kappa of every validation region of the rule under each route (one column per route, or per
    window for route 1), "*" marking a reading not measured to ``max_precision`` ("not measured" when
    its interval is wider than the value or undefined, n/a when kappa is undefined), and the region
    the rule picks in brackets."""
    columns, picks = {}, {}
    for r in records:
        for label, readings in r.get("readings", {}).items():
            columns[label] = dict(readings)
            picks[label] = rule(readings, max_precision)[1]
    names = list(dict.fromkeys(n for col in columns.values() for n in col))
    rows = []
    for n in names:
        row = {"region": n}
        for label, col in columns.items():
            k = col.get(n, (np.nan,) * 3)
            prec = el.precision(k)
            if not np.isfinite(k[0]):
                row[label] = "n/a"
                continue
            if not np.isfinite(prec) or prec > 1:
                row[label] = "not measured *"
                continue
            cell = tl.fmt(k, 2) + ("" if prec <= max_precision else " *")
            row[label] = f"[{cell}]" if picks[label] == n else cell
        rows.append(row)
    return pd.DataFrame(rows)


# ----------------------------------------------------------------------------- views

def pass_view(table):
    rows = []
    for _, r in table.iterrows():
        rows.append({"plane": r["plane"], "bin": r["bin"], "A": nl.fmt_count(r["A"]),
                     "B/C/D": f"{r['B']:.0f}/{r['C']:.0f}/{r['D']:.0f}",
                     "a = C/(C+D)": "n/a" if not np.isfinite(r["a"]) else tl.fmt((r["a"], r["a minus"], r["a plus"]), 2),
                     "b = B/(B+D)": "n/a" if not np.isfinite(r["b"]) else tl.fmt((r["b"], r["b minus"], r["b plus"]), 2)})
    return pd.DataFrame(rows)


def mixing_view(table):
    rows = []
    for _, r in table.iterrows():
        boot = "" if not np.isfinite(r["lo"]) else f"{r['lo']:.2f}/{r['med']:.2f}/{r['hi']:.2f}"
        rows.append({"plane": r["plane"], "binning": r["binning"], "bins used": r["bins"],
                     "sum of B C/D": f"{r['split']:.1f}", "one-bin B C/D": f"{r['one_bin']:.1f}",
                     "mixing factor": f"{r['ratio']:.2f}", "bootstrap 16/50/84%": boot,
                     "toys below 1": "" if not np.isfinite(r["below_one"]) else f"{100 * r['below_one']:.0f}%",
                     "toys with a finite factor": f"{100 * r['kept']:.0f}%"})
    return pd.DataFrame(rows)


def stratified_view(table, ref_planes=OPEN_PLANES):
    rows = []
    for _, r in table.iterrows():
        row = {"bin": r["bin"], "B/C/D of DD": f"{r['B']:.0f}/{r['C']:.0f}/{r['D']:.0f}",
               "B C/D": f"{r['estimate']:.1f} ± {r['error']:.1f}", "bootstrap 68%": fmt_interval((r["lo"], np.nan, r["hi"]), 1)}
        for p in ref_planes:
            k = (r.get(f"kappa {p}", np.nan), r.get(f"kappa {p} minus", np.nan), r.get(f"kappa {p} plus", np.nan))
            row[f"kappa {p}"] = "" if not np.isfinite(k[0]) else tl.fmt(k, 2)
        region = r["syst region"] if isinstance(r["syst region"], str) else None
        row["systematic (rule)"] = ("none qualifies" if not np.isfinite(r["syst"]) else
                                    f"{100 * r['syst']:.0f}%" + (f" ({region})" if region else ""))
        row["regions measured to 20%"] = r["qualifying"]
        rows.append(row)
    return pd.DataFrame(rows)


def fit_view(fits):
    rows = []
    for name, f in fits.items():
        rows.append({"plane": name, "intercept alpha (log-odds at 200 GeV)": f"{f['alpha']:.2f} ± {f['alpha_err']:.2f}",
                     "slope s": (f"{f['slope']:.2f} ± {f['slope_err']:.2f}" + (" (PD's)" if f["fixed_slope"] is not None else "")),
                     "deviance / ndf": f"{f['deviance']:.1f} / {f['ndf']}",
                     "p": "" if not np.isfinite(f["p"]) else fmt_p(f["p"]).removeprefix("p = ").removeprefix("p "),
                     "errors scaled by": f"{f['scale']:.2f}" + (f" (slope {f['ref_scale']:.2f})" if f["fixed_slope"] is not None else ""),
                     "sum B r(M)": f"{f['pred']:.1f} ({fmt_interval(f['pred_interval'])})",
                     "observed A": nl.fmt_count(f["A"]),
                     "kappa = A / sum B r(M)": "hidden" if not np.isfinite(f["kappa"][0]) else tl.fmt(f["kappa"], 2)})
    return pd.DataFrame(rows)


def odds_view(cd, choice, bins=FIVE, candidates=EDGE_CANDIDATES):
    """a per bin and candidate edge, with C/D, the chosen edge marked by brackets."""
    rows = []
    for i, (label, _) in enumerate(bins):
        row = {"bin": label}
        for j, e in enumerate(candidates):
            c, d = cd[i, j]
            a = choice["a"][i, j]
            cell = "–" if not np.isfinite(a) else f"{a:.2f} ({c:.0f}/{d:.0f})"
            row[f"{e:g}"] = f"[{cell}]" if e == choice["edges"][i] else cell
        rows.append(row)
    return pd.DataFrame(rows)


def flat_view(results):
    """Route 3 per calibration and plane (``{(calibration, plane): flat_route result}``): the edges'
    flatness, the one-bin estimate with its interval, the mixing factor and kappa."""
    rows = []
    for (cal, plane), f in results.items():
        fl = f["flatness"]
        rows.append({"calibration": cal, "plane": plane,
                     "a per bin": " / ".join("–" if not np.isfinite(v) else f"{v:.2f}" for v in fl["a"]),
                     "chi2 vs constant": f"{fl['chi2']:.1f} / {fl['ndf']} ({fmt_p(fl['p'])})",
                     "B'/C'/D'": "/".join(f"{f['regions'][x][0]:.0f}" for x in "BCD"),
                     "B' C'/D'": f"{f['pred'][0]:.1f} ({fmt_interval(f['pred_interval'])})",
                     "mixing factor": f"{f['mixing']:.2f} ({fmt_interval(f['mixing_interval'], 2)})",
                     "A'": nl.fmt_count(f["A"]),
                     "kappa": "hidden" if not np.isfinite(f["kappa"][0]) else tl.fmt(f["kappa"], 2)})
    return pd.DataFrame(rows)
