"""The 2mu2e background as a direct ABCD in the search plane: validation regions, the closure
systematic and the numbers of nominal.ipynb.

The estimate is the search plane's own B C / D (DD at M >= 150 GeV, the nominal configuration of
estimate.ipynb: muon-LJ isolation passes below 0.25 and fails at or above it, |dphi| passes at or
above 2.4 and fails below it, the egamma LJ isolated below 0.5).  It assumes kappa = A D / (B C) = 1
in the search plane, which is tested where A is open, in validation regions (:class:`VR`) that differ
from the search region in named respects: the displacement of one LJ (or "any", the sum of prompt
and displaced), the LJ-pair mass window and the isolation band of the egamma LJ.  The last two are
what :class:`plane_search_lib.Config` cannot express (its ``iso1`` cut is ``(None, wp1)``), so
:func:`vr_regions` replaces the ``iso1`` and ``ljlj_mass`` ranges of ``plane_search_lib.region_ranges``.

Blinding: the cubes come from ``transfer_lib.summed_cubes``, whose hidden cells are NaN; a region
that touches one is NaN here too, is reported as hidden, and is never summed into anything.

:func:`systematic_table` and :func:`closure_systematic` implement the rule of the notebook: among
the validation regions at the nominal working point whose differences from the search region are
all named respects (:data:`NAMED_RESPECTS`) and whose kappa is measured to ``max_precision`` (20%)
or better, the largest of |kappa - 1| and the uncertainty of kappa on the side facing one.
"""
import dataclasses

import numpy as np
import pandas as pd

import abcd6d_io as io
import estimate_lib as el
import plane_search_lib as ps
import transfer_lib as tl

NOMINAL = ps.Config(fail0=0.25, dphi_fail=2.4)             # the configuration of estimate.ipynb
LOW = dataclasses.replace(NOMINAL, mass=(None, 150.0))
HIS_WP = ps.Config(wp0=0.3, fail0=0.3, dphi_pass=2.6, dphi_fail=2.6)   # Dongyub Lee's working point
WPS = {"his": HIS_WP, "ours": NOMINAL}
MAX_PRECISION = 0.20
HIGH = (150.0, None)
BELOW = (None, 150.0)

NAMED_RESPECTS = ("muon-LJ displacement", "egamma-LJ displacement", "mass window", "partner-isolation band")
STATES = ("P", "D", "any")


@dataclasses.dataclass(frozen=True)
class VR:
    """A region of the muon-LJ isolation x |dphi| plane: the displacement of the muon LJ (``mu``)
    and of the egamma LJ (``egm``), each ``"P"``, ``"D"`` or ``"any"``; the egamma-LJ isolation
    band ``iso1`` and the LJ-pair mass window ``mass`` as (low, high) with ``None`` open-ended;
    ``split`` an optional plane suffix of ``plane_search_lib`` (``"g"`` photon-only, ``"e"``
    electron egamma LJ, ``"dsa0"``, ...); ``quoted`` the (kappa, error) printed on the slides and
    ``quoted_a`` the A count there, for regions taken from another analysis code."""
    name: str
    source: str
    mu: str
    egm: str
    iso1: tuple = (None, 0.5)
    mass: tuple = HIGH
    split: str = None
    quoted: tuple = None
    quoted_a: float = None

    def __post_init__(self):
        if self.mu not in STATES or self.egm not in STATES:
            raise ValueError(f"{self.name}: displacement states must be in {STATES}")

    def definition(self):
        """One-line definition: displacement (muon LJ / egamma LJ), egamma isolation, mass."""
        return f"{self.mu} / {self.egm}{':' + self.split if self.split else ''}, egamma iso {_range(self.iso1)}, M {_range(self.mass)} GeV"


def _range(rng):
    lo, hi = rng
    if lo is None and hi is None:
        return "any"
    if lo is None:
        return f"< {hi:g}"
    if hi is None:
        return f"≥ {lo:g}"
    return f"{lo:g}–{hi:g}"


SR = VR("search region", "this analysis", "D", "D")

SLIDES_1013 = "D. Lee, 2026-10-13"
SLIDES_1016 = "D. Lee, 2026-10-16"
# Dongyub Lee's seven 2mu2e data validation regions in the muon-LJ isolation x |dphi| plane, with the
# kappa and A count printed on his slides (2026-10-13 "Data VR", 2026-10-16 LOW and HIGH MJJ decks).
DONGYUB_VRS = (
    VR("V1", SLIDES_1013, "P", "P", (None, 0.5), HIGH, quoted=(1.03, 0.0863), quoted_a=819),
    VR("V2", SLIDES_1016, "P", "any", (1.0, None), BELOW, quoted=(1.01, 0.0887), quoted_a=565),
    VR("V3", SLIDES_1016, "any", "P", (0.75, None), BELOW, quoted=(1.02, 0.0837), quoted_a=622),
    VR("V4", SLIDES_1016, "any", "any", (0.75, None), BELOW, quoted=(1.01, 0.07), quoted_a=860),
    VR("V5", SLIDES_1016, "P", "any", (0.2, 1.0), HIGH, quoted=(1.07, 0.0678), quoted_a=1680),
    VR("V6", SLIDES_1016, "P", "any", (0.15, 1.0), HIGH, quoted=(1.05, 0.0623), quoted_a=1870),
    VR("V7", SLIDES_1016, "P", "P", (0.15, 1.0), HIGH, quoted=(1.05, 0.0848), quoted_a=1020),
)


def our_vrs():
    """The open planes of this analysis as validation regions: PP, PD (and its photon-only and
    electron parts), DP at M >= 150 GeV and the search plane below 150 GeV."""
    return (VR("PP", "this analysis", "P", "P"), VR("PD", "this analysis", "P", "D"),
            VR("PD:g", "this analysis", "P", "D", split="g"), VR("PD:e", "this analysis", "P", "D", split="e"),
            VR("DP", "this analysis", "D", "P"), VR("DD < 150", "this analysis", "D", "D", mass=BELOW))


# ----------------------------------------------------------------------------- regions

def _state_range(topology, obj, state):
    return (None, None) if state == "any" else io.displacement(topology, obj, {"P": "prompt", "D": "displaced"}[state])


def vr_selection(topology, mu, egm, split=None):
    """``{"disp0": range, "disp1": range}`` (plus the split's axes) of a displacement pair; ``"any"``
    is the whole axis, the sum of prompt and displaced.  A split needs both states fixed."""
    if split is not None:
        if "any" in (mu, egm):
            raise ValueError("a plane split needs both displacement states fixed")
        return ps.plane_selection(topology, f"{mu}{egm}:{split}")
    return {"disp0": _state_range(topology, 0, mu), "disp1": _state_range(topology, 1, egm)}


def vr_ranges(vr, cfg):
    """``{region: ranges}`` of ``vr`` at the working point of ``cfg`` (a ``dphi x iso0``
    configuration): the configuration's isolation and |dphi| boundaries, with the egamma-LJ
    isolation band and the mass window of the region."""
    if cfg.kind != "dphi x iso0":
        raise ValueError("validation regions live in the dphi x iso0 plane")
    out = {}
    for r, rng in ps.region_ranges(dataclasses.replace(cfg, mass=vr.mass)).items():
        out[r] = {**rng, "iso1": vr.iso1}
    return out


def vr_regions(cube, vr, cfg):
    """A/B/C/D (sumw, sumw2) of ``vr`` at the working point of ``cfg``; NaN where a hidden cell is touched."""
    disp = vr_selection(cube.topology, vr.mu, vr.egm, vr.split)
    return {r: ps.project(cube, **rng, **disp) for r, rng in vr_ranges(vr, cfg).items()}


def side_facing_one(k):
    """The uncertainty of a (value, minus, plus) kappa on the side facing one."""
    v, m, p = k
    return m if v >= 1 else p


def kappa_row(cube, vr, cfg):
    """Numbers of one region at one working point: the counts, B C / D with its error, kappa with
    its 68% interval, its precision (the larger side over the value), the uncertainty facing one,
    and ``hidden`` when any of A, B, C, D touches the hidden box (kappa is then NaN)."""
    reg = vr_regions(cube, vr, cfg)
    k = ps.kappa(reg)
    pred = ps.prediction(reg)
    hidden = any(np.isnan(reg[r][0]) for r in "ABCD")
    row = {"region": vr.name, "source": vr.source, "definition": vr.definition(),
           **{r: reg[r][0] for r in "ABCD"}, "B C/D": pred[0], "B C/D err": pred[1],
           "kappa": k[0], "minus": k[1], "plus": k[2], "hidden": hidden,
           "hidden regions": " and ".join(r for r in "ABCD" if np.isnan(reg[r][0]))}
    row["precision"] = el.precision(k)
    row["sigma"] = side_facing_one(k) if np.isfinite(k[0]) else np.nan
    row["|kappa-1|"] = abs(k[0] - 1) if np.isfinite(k[0]) else np.nan
    return row


def kappa_table(cube, vrs, cfg):
    """:func:`kappa_row` for each region at one working point, as a numeric frame."""
    return pd.DataFrame([kappa_row(cube, vr, cfg) for vr in vrs])


def vr_table(cube, vrs=DONGYUB_VRS, wps=None):
    """Long numeric frame: every region at every working point (``{name: cfg}``, default
    :data:`WPS`), with the value quoted on the slides beside it."""
    wps = WPS if wps is None else wps
    rows = []
    for vr in vrs:
        for wp, cfg in wps.items():
            row = kappa_row(cube, vr, cfg)
            row.update({"wp": wp, "quoted kappa": vr.quoted[0] if vr.quoted else np.nan,
                        "quoted err": vr.quoted[1] if vr.quoted else np.nan,
                        "quoted A": vr.quoted_a if vr.quoted_a is not None else np.nan})
            rows.append(row)
    return pd.DataFrame(rows)


def fmt_count(v):
    return "hidden" if np.isnan(v) else f"{v:.0f}"


def fmt_kappa(row, digits=2):
    if row["hidden"]:
        return f"hidden ({row['hidden regions']} in the box)"
    return tl.fmt((row["kappa"], row["minus"], row["plus"]), digits)


def vr_comparison_view(table):
    """:func:`vr_table` side by side per region: definition, the slides' kappa, ours at his and at
    our working point, A on the slides against ours at his working point."""
    rows = []
    for name, part in table.groupby("region", sort=False):
        by = part.set_index("wp")
        his, ours = by.loc["his"], by.loc["ours"]
        q = (his["quoted kappa"], his["quoted err"])
        rows.append({"region": name, "definition (muon LJ / egamma LJ, egamma iso, M)": his["definition"],
                     "kappa, slides": "" if np.isnan(q[0]) else f"{q[0]:.2f} ± {q[1]:.2f}",
                     "kappa, these histograms at his WP": fmt_kappa(his),
                     "kappa, these histograms at our WP": fmt_kappa(ours),
                     "A slides / ours (his WP)": ("" if np.isnan(his["quoted A"]) else f"{his['quoted A']:.0f} / ")
                     + fmt_count(his["A"]),
                     "A ours (our WP)": fmt_count(ours["A"])})
    return pd.DataFrame(rows)


def kappa_view(table, digits=2):
    """A numeric kappa frame formatted: counts, B C/D, kappa, precision."""
    rows = []
    for _, r in table.iterrows():
        rows.append({"region": r["region"], "definition": r["definition"], **{x: fmt_count(r[x]) for x in "ABCD"},
                     "B C/D": "n/a" if np.isnan(r["B C/D"]) else f"{r['B C/D']:.1f} ± {r['B C/D err']:.1f}",
                     "kappa": fmt_kappa(r, digits),
                     "precision": "n/a" if not np.isfinite(r["precision"]) else f"{100 * r['precision']:.0f}%"})
    return pd.DataFrame(rows)


def route_estimates(cube, cfg, classes=("dsa0", "dsa1+"), ref_plane="PD", search="DD"):
    """Numeric form of ``estimate_lib.routes_table`` for one window: ``{"direct": B C / D of the
    search plane, "B"/"C"/"D": that sideband of the search plane times A/sideband of ``ref_plane``,
    per class of the muon LJ and summed}``, each (value, minus, plus)."""
    pred = ps.prediction(ps.regions(cube, cfg, search))
    out = {"direct": (pred[0], pred[1], pred[1])}
    for sideband, _ in el.ROUTES:
        total = (0.0, 0.0, 0.0)
        for cls in classes:
            total = el._add(total, el._route(cube, cfg, search, ref_plane, sideband, cls))
        out[sideband] = total
    return out


def _label(plane, split):
    return plane if split is None else f"{plane}:{split}"


def kappa_corrected(cube, cfg, splits=("dsa0", "dsa1+"), ref_plane="PD", search="DD"):
    """The direct estimate of ``search`` corrected by the kappa of ``ref_plane``, split by split:
    ``{"direct": inclusive B C / D of the search plane, "parts": {split: {"direct": B C / D of the
    split in the search plane, "kappa": (value, minus, plus) of the split in the reference plane,
    "share": the split's part of the summed per-split direct estimates}}, "corrected": the sum over
    splits of kappa x direct, "ratio": corrected over the inclusive direct estimate}``.  A split of
    ``None`` is the inclusive plane, so ``splits=(None,)`` gives kappa of the reference plane times
    the direct estimate.  Only B, C and D of the search plane are read."""
    direct = ps.prediction(ps.regions(cube, cfg, search))[0]
    parts = {}
    for s in splits:
        parts[s] = {"direct": ps.prediction(ps.regions(cube, cfg, _label(search, s)))[0],
                    "kappa": ps.kappa(ps.regions(cube, cfg, _label(ref_plane, s)))}
    total = sum(p["direct"] for p in parts.values())
    for p in parts.values():
        p["share"] = p["direct"] / total if total > 0 else np.nan
    corrected = float(sum(p["kappa"][0] * p["direct"] for p in parts.values()))
    return {"direct": direct, "parts": parts, "corrected": corrected,
            "ratio": corrected / direct if direct > 0 else np.nan}


def route_ratio_factors(cube, cfg, classes=("dsa0", "dsa1+"), ref_plane="PD", search="DD"):
    """The B route over the direct estimate as a product of two factors: ``{"ratio": B route /
    (B C / D), "kappa_mix": B route / (B x (C/D) of the reference plane), the reference plane's
    kappa per class weighted by the search plane's B, "sideband": (C/D) of the reference plane over
    (C/D) of the search plane}``, with ratio = kappa_mix x sideband."""
    s, r = ps.regions(cube, cfg, search), ps.regions(cube, cfg, ref_plane)
    route = route_estimates(cube, cfg, classes, ref_plane, search)["B"][0]
    direct = ps.prediction(s)[0]
    cd_s, cd_r = s["C"][0] / s["D"][0], r["C"][0] / r["D"][0]
    return {"ratio": route / direct, "kappa_mix": route / (s["B"][0] * cd_r), "sideband": cd_r / cd_s}


def b_route_parts(cube, cfg, classes=("dsa0", "dsa1+"), ref_plane="PD", search="DD"):
    """Per class of the muon LJ, the inputs of the B route: ``[(class, B sumw, B sumw2, A/B of the
    reference plane)]``, with B from the search plane."""
    out = []
    for cls in classes:
        b = ps.regions(cube, cfg, _label(search, cls))["B"]
        r = ps.regions(cube, cfg, _label(ref_plane, cls))
        out.append((cls, b[0], b[1], r["A"][0] / r["B"][0]))
    return out


def shared_b_separation(direct, route, reg, parts):
    """Separation in standard deviations of the direct estimate ``direct`` (value, err) of one
    search plane with regions ``reg`` and its B route ``route`` (value, minus, plus) built from
    ``parts`` (:func:`b_route_parts`): (treated as independent, with the shared B in first order).
    The second splits the variance of route - direct into the parts that do not involve B,
    sigma_direct'^2 = direct^2 (var C / C^2 + var D / D^2) and sigma_route'^2 = sigma_route^2 -
    sum_c r_c^2 var B_c, and the part that does: B_c enters the direct estimate with derivative C/D
    and the route with r_c = (A/B) of the reference plane, so it adds sum_c var B_c (r_c - C/D)^2.
    Each error is taken on the side facing the other estimate."""
    d, d_err = direct
    v, m, p = route
    r_err = m if v >= d else p
    (c, c2), (dd, dd2) = reg["C"], reg["D"]
    cd = c / dd
    independent = abs(v - d) / np.hypot(r_err, d_err)
    d_err_b = d * np.sqrt(c2 / c**2 + dd2 / dd**2)
    r_err_b2 = max(r_err**2 - sum(r_c**2 * b2 for _, _, b2, r_c in parts), 0.0)
    var_b = sum(b2 * (r_c - cd) ** 2 for _, _, b2, r_c in parts)
    return float(independent), float(abs(v - d) / np.sqrt(d_err_b**2 + r_err_b2 + var_b))


def same_regions(table, ref):
    """Names of the rows of a kappa or systematic table (other than ``ref``) whose A, B, C and D
    equal those of the row ``ref``: the same events under another name."""
    by = table.drop_duplicates("region").set_index("region")
    want = by.loc[ref, list("ABCD")].to_numpy(float)
    return [n for n in by.index if n != ref and np.array_equal(by.loc[n, list("ABCD")].to_numpy(float), want)]


# ----------------------------------------------------------------------------- the systematic

def distance_to_sr(vr, cfg, sr=SR, nominal=NOMINAL):
    """The respects in which a region at the working point of ``cfg`` differs from the search
    region, as a tuple of labels (empty for the search region itself)."""
    out = []
    if vr.mu != sr.mu:
        out.append("muon-LJ displacement")
    if vr.egm != sr.egm:
        out.append("egamma-LJ displacement")
    if tuple(vr.mass) != tuple(sr.mass):
        out.append("mass window")
    if tuple(vr.iso1) != tuple(sr.iso1):
        out.append("partner-isolation band")
    if vr.split is not None:
        out.append(f"split {vr.split}")
    if (cfg.wp0, cfg.fail0, cfg.dphi_pass, cfg.dphi_fail) != (nominal.wp0, nominal.fail0, nominal.dphi_pass, nominal.dphi_fail):
        out.append("working point")
    return tuple(out)


def systematic_table(cube, vrs, wps=None, max_precision=MAX_PRECISION):
    """One row per region and working point: kappa, |kappa - 1|, the uncertainty facing one
    (``sigma``), the precision, the respects in which the region differs from the search region,
    whether it ``qualifies`` for the rule (every difference a named respect, the nominal working
    point, kappa measured to ``max_precision``) with the ``reason`` when not, and ``candidate`` =
    max(|kappa - 1|, sigma).  Sorted by the number of differences, then by name order."""
    wps = {"ours": NOMINAL} if wps is None else wps
    rows = []
    for vr in vrs:
        for wp, cfg in wps.items():
            row = kappa_row(cube, vr, cfg)
            dist = distance_to_sr(vr, cfg)
            reasons = []
            if row["hidden"]:
                reasons.append("hidden")
            elif not np.isfinite(row["kappa"]):
                reasons.append("kappa undefined")
            other = [d for d in dist if d not in NAMED_RESPECTS]
            if other:
                reasons.append("differs in " + ", ".join(other))
            if np.isfinite(row["kappa"]) and row["precision"] > max_precision:
                reasons.append(f"precision {100 * row['precision']:.0f}% > {100 * max_precision:.0f}%")
            row.update({"wp": wp, "differences": ", ".join(dist) if dist else "none", "n differences": len(dist),
                        "qualifies": not reasons, "reason": "; ".join(reasons),
                        "candidate": max(row["|kappa-1|"], row["sigma"]) if np.isfinite(row["kappa"]) else np.nan,
                        "quadrature": float(np.hypot(row["|kappa-1|"], row["sigma"])) if np.isfinite(row["kappa"]) else np.nan})
            rows.append(row)
    df = pd.DataFrame(rows)
    df["order"] = range(len(df))
    return df.sort_values(["n differences", "order"], kind="stable").drop(columns="order").reset_index(drop=True)


def closure_systematic(table, column="candidate", qualifying=True):
    """(value, region) of the largest ``column`` over the qualifying rows of a
    :func:`systematic_table` (``qualifying=False``: over the rows that fail only on precision, the
    imprecise regions of the alternative); (NaN, None) when no row is eligible."""
    if qualifying:
        rows = table[table["qualifies"]]
    else:
        rows = table[~table["qualifies"] & table["reason"].str.startswith("precision")]
    rows = rows[np.isfinite(rows[column])]
    if rows.empty:
        return np.nan, None
    best = rows.loc[rows[column].idxmax()]
    return float(best[column]), best["region"]


def systematic_view(table):
    rows = []
    for _, r in table.iterrows():
        rows.append({"region": r["region"], "WP": r["wp"], "definition": r["definition"], "kappa": fmt_kappa(r),
                     "|kappa - 1|": "" if not np.isfinite(r["|kappa-1|"]) else f"{r['|kappa-1|']:.2f}",
                     "sigma (facing 1)": "" if not np.isfinite(r["sigma"]) else f"{r['sigma']:.2f}",
                     "precision": "" if not np.isfinite(r["precision"]) else f"{100 * r['precision']:.0f}%",
                     "max(|kappa - 1|, sigma)": "" if not np.isfinite(r["candidate"]) else f"{r['candidate']:.2f}",
                     "differs from the search region in": r["differences"],
                     "qualifies": "yes" if r["qualifies"] else "no: " + r["reason"]})
    return pd.DataFrame(rows)


# ----------------------------------------------------------------------------- the plane as a 2D histogram

def plane_hist2d(cube, cfg, plane):
    """``(iso0_edges, dphi_edges, sumw, sumw2)``: the muon-LJ isolation x |dphi| distribution of a
    plane (label or selection dict) with the configuration's egamma-isolation, mass and extra cuts,
    summed over every other axis, flow bins folded into the outer bins.  A cell is NaN when any
    stored cell summed into it is hidden."""
    if cfg.kind != "dphi x iso0":
        raise ValueError("plane_hist2d draws the dphi x iso0 plane")
    disp = plane if isinstance(plane, dict) else ps.plane_selection(cube.topology, plane)
    ranges = {"iso1": (None, cfg.wp1), "ljlj_mass": cfg.mass, **dict(cfg.extra), **disp}
    index = io.selection_index(cube, **ranges)
    keep = (cube.spec.index("iso0"), cube.spec.index("abs_dphi"))
    other = tuple(i for i in range(len(cube.spec.variables)) if i not in keep)
    sumw = np.sum(cube.sumw[index], axis=other)
    sumw2 = np.sum(cube.sumw2[index], axis=other)
    for axis, variable in enumerate(("iso0", "abs_dphi")):
        i = cube.spec.index(variable)
        if cube.spec.underflow[i]:
            sumw, sumw2 = (_fold(a, axis, first=True) for a in (sumw, sumw2))
        if cube.spec.overflow[i]:
            sumw, sumw2 = (_fold(a, axis, first=False) for a in (sumw, sumw2))
    return cube.spec.edges_of("iso0"), cube.spec.edges_of("abs_dphi"), sumw, sumw2


def _fold(a, axis, first):
    a = np.moveaxis(a, axis, 0)
    a = np.concatenate([[a[0] + a[1]], a[2:]]) if first else np.concatenate([a[:-2], [a[-2] + a[-1]]])
    return np.moveaxis(a, 0, axis)


def hist2d_regions(h2d, cfg):
    """A/B/C/D sums of a :func:`plane_hist2d` at the configuration's boundaries (NaN when a hidden
    cell is inside)."""
    e0, ed, sumw, _ = h2d
    i0 = io._edge_index(e0, cfg.wp0)
    f0 = io._edge_index(e0, cfg.fail0)
    ip = io._edge_index(ed, cfg.dphi_pass)
    fd = io._edge_index(ed, cfg.dphi_fail)
    return {"A": float(np.sum(sumw[:i0, ip:])), "B": float(np.sum(sumw[f0:, ip:])),
            "C": float(np.sum(sumw[:i0, :fd])), "D": float(np.sum(sumw[f0:, :fd]))}


# ----------------------------------------------------------------------------- composition

def class_share_table(cube, cfg, planes, splits, regions="ABCD"):
    """Share of each split (plane suffixes that partition the plane, e.g. ``("dsa0", "dsa1+")`` or
    ``("e", "g")``) in each region of each plane: one row per plane and region with the counts, the
    total and the shares ("hidden" where the region is)."""
    rows = []
    for plane in planes:
        total = ps.regions(cube, cfg, plane)
        parts = {s: ps.regions(cube, cfg, f"{plane}:{s}") for s in splits}
        for r in regions:
            t = total[r][0]
            row = {"plane": plane, "region": r, "total": t}
            for s in splits:
                n = parts[s][r][0]
                row[s] = n
                row[f"{s} share"] = n / t if np.isfinite(t) and t > 0 else np.nan
            row["hidden"] = bool(np.isnan(t))
            rows.append(row)
    return pd.DataFrame(rows)


def class_share_view(table, splits):
    rows = []
    for _, r in table.iterrows():
        row = {"plane": r["plane"], "region": r["region"], "events": fmt_count(r["total"])}
        for s in splits:
            row[s] = "hidden" if r["hidden"] else f"{r[s]:.0f} ({100 * r[f'{s} share']:.0f}%)" if np.isfinite(r[f"{s} share"]) else f"{r[s]:.0f}"
        rows.append(row)
    return pd.DataFrame(rows)


# ----------------------------------------------------------------------------- mass dependence

def kappa_drop(rows):
    """Inverse-variance combination of the drops kappa(low window) - kappa(high window) of several
    planes: ``rows`` is a list of ((k_low, minus, plus), (k_high, minus, plus)); the error of each
    drop takes the low window's lower and the high window's upper side.  Returns (mean drop, error,
    significance)."""
    drops, errs = [], []
    for lo, hi in rows:
        drops.append(lo[0] - hi[0])
        errs.append(np.hypot(lo[1], hi[2]))
    w = 1 / np.square(errs)
    mean = float(np.sum(w * np.array(drops)) / np.sum(w))
    err = float(1 / np.sqrt(np.sum(w)))
    return mean, err, mean / err


def projected_precision(pred, pred_err):
    """Relative precision of kappa if A held ``pred`` events: Poisson on A with the prediction's own
    relative error in quadrature."""
    return float(np.hypot(1 / np.sqrt(pred), pred_err / pred)) if pred > 0 else np.nan


# ----------------------------------------------------------------------------- signal

SHARE_BINS = (("150-200", (150.0, 200.0)), ("200-300", (200.0, 300.0)), ("300-400", (300.0, 400.0)),
              ("400-600", (400.0, 600.0)), (">=600", (600.0, None)))


def _folded_slice(edges, lo, hi):
    start = 0 if lo is None else io._edge_index(edges, lo)
    stop = len(edges) - 1 if hi is None else io._edge_index(edges, hi)
    return slice(start, stop)


def signal_share_table(profiles, bins=SHARE_BINS):
    """Per signal point, from flow-folded mass profiles of one region (``(edges, {point: (sumw,
    sumw2)})``, ``binning_lib.signal_profiles``): the search-region yield (M >= the first bin's low
    edge), the share of it in each bin of ``bins`` (which partition that range, so the shares sum
    to one), the share in 150-300 GeV, and ``below``, the yield in events below the first bin (A of
    the plane at M < 150 GeV, the open validation region)."""
    edges, prof = profiles
    low = bins[0][1][0]
    rows = []
    for name, (sumw, _) in prof.items():
        total = float(np.sum(sumw[_folded_slice(edges, low, None)]))
        row = {"point": name, "yield": total}
        for label, (lo, hi) in bins:
            y = float(np.sum(sumw[_folded_slice(edges, lo, hi)]))
            row[label] = y / total if total > 0 else np.nan
        row["150-300"] = float(np.sum(sumw[_folded_slice(edges, 150.0, 300.0)])) / total if total > 0 else np.nan
        row["below"] = float(np.sum(sumw[_folded_slice(edges, None, low)]))
        rows.append(row)
    return pd.DataFrame(rows)


def mediator_mass(name):
    """Mediator mass in GeV of a signal point name such as ``2Mu2E_200GeV_0p25GeV_1p0mm``."""
    return float(name.split("_")[1].replace("GeV", ""))


def share_summary(shares, columns=("150-200", "200-300", "150-300", ">=600")):
    """Per mediator mass: the number of points, min-max of each share column, of the search-region
    yield and of the yield below 150 GeV, both in events."""
    df = shares.assign(mass=shares["point"].map(mediator_mass))
    rows = []
    for m, part in df.groupby("mass"):
        row = {"mediator mass [GeV]": f"{m:g}", "points": len(part)}
        for c in columns:
            row[c] = f"{100 * part[c].min():.0f}–{100 * part[c].max():.0f}%"
        row["yield, M ≥ 150 [events]"] = f"{part['yield'].min():.2f}–{part['yield'].max():.2f}"
        row["yield, M < 150 [events]"] = f"{part['below'].min():.2f}–{part['below'].max():.2f}"
        rows.append(row)
    return pd.DataFrame(rows)


def below_share(shares):
    """Per signal point, the part of its yield in A of the plane (every mass) that lies below 150 GeV."""
    tot = shares["yield"] + shares["below"]
    return (shares["below"] / tot).where(tot > 0)


# ----------------------------------------------------------------------------- formatting

def fmt(x, digits=2):
    """The one number formatter of nominal.ipynb: a (value, minus, plus) triple as
    ``transfer_lib.fmt`` prints it, a (value, error) pair as "v ± e", a scalar with ``digits``."""
    if isinstance(x, (tuple, list)) and len(x) == 3:
        return tl.fmt(tuple(x), digits)
    if isinstance(x, (tuple, list)) and len(x) == 2:
        return f"{x[0]:.{digits}f} ± {x[1]:.{digits}f}"
    return f"{x:.{digits}f}"


def pct(x, digits=0):
    return f"{100 * x:.{digits}f}%"
