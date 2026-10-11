"""Tables for estimate.ipynb on top of plane_search_lib: the 6D/8D consistency check, spectra of
a stored variable per region, scans of an event cut, the comparison of validation planes of one
cube (the proxy regions) with the search plane of another, and the final estimate table.

The proxy cubes (``abcd8d_proxy``) use the plane labels of ``plane_search_lib`` unchanged: in
2mu2e ``disp0`` is the lone muon's pixel-hit count and ``ndsa0`` whether it is a DSA muon, in 4mu
``disp1``/``ndsa1`` describe the lone muon; ``nbjets`` enters through ``Config.with_cut``.
"""
import dataclasses

import numpy as np
import pandas as pd
from scipy import optimize, stats

import abcd6d_io as io
import plane_search_lib as ps
import transfer_lib as tl

BTAG = {"all": None, "no b-jet": ("nbjets", None, 0.5), ">=1 b-jet": ("nbjets", 0.5, None)}


def with_btag(cfg, name):
    cut = BTAG[name]
    return cfg if cut is None else cfg.with_cut(*cut)


def consistency_table(cube_a, cube_b, cfg, planes=("PP", "DP", "PD", "DD"), names=("6D", "8D")):
    """Region counts of the same selection in two cubes of one topology, with their difference
    (hidden cells stay hidden in both)."""
    rows = []
    for plane in planes:
        ra, rb = ps.regions(cube_a, cfg, plane), ps.regions(cube_b, cfg, plane)
        row = {"plane": plane}
        for r in "ABCD":
            row[f"{r} {names[0]}"] = tl.fmt_count(ra[r])
            row[f"{r} {names[1]}"] = tl.fmt_count(rb[r])
            diff = rb[r][0] - ra[r][0]
            row[f"{r} diff"] = "blind" if np.isnan(diff) else f"{diff:+.0f}"
        rows.append(row)
    return pd.DataFrame(rows)


def spectrum(cube, variable, cfg, plane, region):
    """``(edges, sumw, sumw2)`` of ``variable`` in one ABCD region of a plane; flow bins are
    folded into the first and last bin so the arrays match the edges."""
    ranges = {**ps.region_ranges(cfg)[region], **ps.plane_selection(cube.topology, plane)}
    ranges.pop(variable, None)
    sumw, sumw2 = io.profile(cube, variable, **ranges)
    i = cube.spec.index(variable)
    edges = cube.spec.edges[i]
    if cube.spec.underflow[i]:
        sumw, sumw2 = np.r_[sumw[0] + sumw[1], sumw[2:]], np.r_[sumw2[0] + sumw2[1], sumw2[2:]]
    if cube.spec.overflow[i]:
        sumw, sumw2 = np.r_[sumw[:-2], sumw[-2] + sumw[-1]], np.r_[sumw2[:-2], sumw2[-2] + sumw2[-1]]
    return edges, sumw, sumw2


def cut_scan_table(cube, cfg, plane, cuts):
    """Counts, A/B, B/D, C/D and kappa of a plane for each event cut in ``cuts``
    (``{label: None | (variable, low, high)}``)."""
    rows = []
    for label, cut in cuts.items():
        c = cfg if cut is None else cfg.with_cut(*cut)
        reg = ps.regions(cube, c, plane)
        rows.append({"plane": plane, "cut": label, **{r: tl.fmt_count(reg[r]) for r in "ABCD"},
                     "A/B": tl.fmt(tl.ratio(reg["A"], reg["B"]), 2), "B/D": tl.fmt(tl.ratio(reg["B"], reg["D"]), 2),
                     "C/D": tl.fmt(tl.ratio(reg["C"], reg["D"]), 2), "kappa": tl.fmt(ps.kappa(reg), 2)})
    return pd.DataFrame(rows)


def cross_compatibility_table(val_cube, val_planes, search_cube, cfg, search="DD", btag=("all",)):
    """Kappa and the open ratios of validation planes of ``val_cube`` (a proxy cube) with their
    log-space pulls against the search plane of ``search_cube``; one row per plane and b-tag
    requirement."""
    dd = ps.regions(search_cube, cfg, search)
    dd_f = {n: tl.ratio(dd[a], dd[b]) for n, (a, b) in (("B/D", ("B", "D")), ("C/D", ("C", "D")), ("A/B", ("A", "B")))}
    rows = []
    for plane in val_planes:
        for b in btag:
            reg = ps.regions(val_cube, with_btag(cfg, b), plane)
            f = {n: tl.ratio(reg[a], reg[b_]) for n, (a, b_) in (("B/D", ("B", "D")), ("C/D", ("C", "D")), ("A/B", ("A", "B")))}
            row = {"plane": plane, "b-tag": b, **{r: tl.fmt_count(reg[r]) for r in "ABCD"}, "kappa": tl.fmt(ps.kappa(reg), 2)}
            for n in ("B/D", "C/D"):
                row[n] = tl.fmt(f[n], 2)
                row[f"{n} pull vs {search}"] = f"{ps.pull(f[n], dd_f[n]):+.1f}" if np.isfinite(f[n][0]) else "n/a"
            row["A/B"] = tl.fmt(f["A/B"], 2)
            rows.append(row)
    rows.append({"plane": f"{search} (search cube)", "b-tag": "-", **{r: tl.fmt_count(dd[r]) for r in "ABCD"},
                 **{n: tl.fmt(dd_f[n], 2) for n in ("B/D", "C/D", "A/B")}})
    return pd.DataFrame(rows)


def class_grid_table(cube, cfg, planes, splits, btag=("all",)):
    """Counts, ratios and kappa for every plane x split x b-tag combination; splits are plane
    suffixes (``"mu3+"``, ``"dsa1+"``, ``"g"``, ...) or ``None`` for the whole plane; a split that
    lies outside its plane is skipped."""
    rows = []
    for plane in planes:
        for split in splits:
            label = plane if split is None else f"{plane}:{split}"
            for b in btag:
                try:
                    reg = ps.regions(cube, with_btag(cfg, b), label)
                except ValueError as err:
                    if "outside its plane" not in str(err):
                        raise
                    continue
                rows.append({"plane": label, "b-tag": b, **{r: tl.fmt_count(reg[r]) for r in "ABCD"},
                             "A/B": tl.fmt(tl.ratio(reg["A"], reg["B"]), 2), "B/D": tl.fmt(tl.ratio(reg["B"], reg["D"]), 2),
                             "C/D": tl.fmt(tl.ratio(reg["C"], reg["D"]), 2), "kappa": tl.fmt(ps.kappa(reg), 2),
                             "N_eff A": f"{tl.n_eff(reg['A']):.0f}" if np.isfinite(reg["A"][0]) else "blind"})
    return pd.DataFrame(rows)


def signal_a_table(base, names, topology, cfg_windows, planes, prefix="abcd8d", btag=("all",), data=None):
    """Region-A yield of each signal point (at the stored cross section) per mass window, one
    column per plane and b-tag requirement, with the data count of the same region in brackets
    when it is open.  Samples are loaded one at a time: an 8D cube is 0.1-0.4 GB."""
    rows = []
    for name in names:
        cube = tl.sample_cubes(base, "signal", [name], topologies=(topology,), prefix=prefix)[name][topology]
        for label, cfg in cfg_windows:
            row = {"point": name, "window": label}
            for plane in planes:
                for b in btag:
                    c = with_btag(cfg, b)
                    a = ps.regions(cube, c, plane)["A"]
                    col = f"{plane} A" if b == "all" else f"{plane} A, {b}"
                    row[col] = f"{a[0]:.2f}" if data is None else f"{a[0]:.2f} ({tl.fmt_count(ps.regions(data, c, plane)['A'])})"
            rows.append(row)
        del cube
    return pd.DataFrame(rows)


ABCD_AXES = ("iso0", "iso1", "abs_dphi", "ljlj_mass")


def reduce_plane(cube, plane, extra=()):
    """``cube`` restricted to displacement plane ``plane`` (a label such as ``"PD"`` or
    ``"DD:mu1-2"``) and to the event cuts ``extra`` (as :attr:`plane_search_lib.Config.extra`),
    summed over every axis but the four ABCD axes; hidden cells stay NaN.  Scanning hundreds of
    configurations on the result takes seconds instead of minutes on the full cube, and
    :func:`plane_search_lib.regions` reads it with an empty plane selection (``{}``) and a
    configuration without ``extra``."""
    ranges = {**ps.plane_selection(cube.topology, plane), **dict(extra)}
    index = io.selection_index(cube, **ranges)
    keep = [cube.spec.index(v) for v in ABCD_AXES]
    other = tuple(i for i in range(len(cube.spec.variables)) if i not in keep)
    order = [k for k in sorted(keep)]                    # the kept axes keep their relative order
    perm = [order.index(k) for k in keep]
    sumw = np.transpose(np.sum(cube.sumw[index], axis=other), perm)
    sumw2 = np.transpose(np.sum(cube.sumw2[index], axis=other), perm)
    spec = io.AxisSpec(ABCD_AXES, tuple(cube.spec.edges[k] for k in keep),
                       tuple(cube.spec.underflow[k] for k in keep), tuple(cube.spec.overflow[k] for k in keep))
    return io.Cube(cube.topology, sumw, sumw2, blinded=cube.blinded, spec=spec, prefix=cube.prefix)


def regions_reduced(reduced, cfg):
    """A/B/C/D of a configuration on a reduced cube (the configuration's extra cuts must already be
    part of the reduction)."""
    return ps.regions(reduced, dataclasses.replace(cfg, extra=()), {})


def precision(k):
    """Relative statistical precision of a (value, minus, plus) kappa; infinite when undefined."""
    return max(k[1], k[2]) / k[0] if np.isfinite(k[0]) and k[0] > 0 and np.isfinite(k[2]) else np.inf


def closure_scan(reduced, cfgs, select, confirm=(), target=0.15):
    """One row per configuration: kappa in every validation region of ``reduced`` (``{label:
    reduced cube}``), the worst |kappa - 1| over the ``select`` regions and over the ``confirm``
    regions, and whether the regions of each set close within ``target``.  A region whose kappa
    is undefined or less precise than the target cannot test it: it is left out of the worst
    deviation and counted as "untestable".  "all ... within target" requires every region of
    the set to be testable and within the target; "testable ... within target" asks it only of
    the testable ones (a set with none does not pass either way).  Selecting a configuration
    on one set of regions and confirming it on another keeps the choice from being tuned to the
    fluctuations of the regions that justify it.  Sorted by the worst deviation over the
    ``select`` regions."""
    rows = []
    for cfg in cfgs:
        row = {"config": cfg.label(), "kind": cfg.kind}
        worst, untestable = {}, {}
        for name, labels in (("select", select), ("confirm", confirm)):
            devs, skipped = [], 0
            for label in labels:
                k = ps.kappa(regions_reduced(reduced[label], cfg))
                row[f"kappa {label}"] = tl.fmt(k, 2)
                if precision(k) > target:
                    skipped += 1
                else:
                    devs.append(abs(k[0] - 1))
            worst[name], untestable[name] = (max(devs) if devs else np.inf), skipped
        for name in ("select", "confirm"):
            row[f"worst |kappa-1| {name}"] = worst[name]
            row[f"untestable {name}"] = untestable[name]
        row["all select within target"] = bool(worst["select"] <= target and untestable["select"] == 0)
        row["testable select within target"] = bool(worst["select"] <= target)
        row["all confirm within target"] = bool(worst["confirm"] <= target and untestable["confirm"] == 0) if confirm else None
        row["testable confirm within target"] = bool(worst["confirm"] <= target) if confirm else None
        row["DD in-plane estimate exists"] = cfg.dd_open()
        rows.append(row)
    return pd.DataFrame(rows).sort_values("worst |kappa-1| select", kind="stable").reset_index(drop=True)


def closure_table(cases, target=0.15):
    """Closure of validation regions against the target: one row per ``(label, cube, cfg, plane)``
    with the observed A, the in-plane prediction B C/D, kappa, its deviation from one in per cent
    with the 68% interval, the statistical precision of kappa and the verdict: ``closes`` when
    |kappa - 1| <= target, ``closes within errors`` when the interval reaches one but the central
    value is further than the target, ``imprecise`` when the interval itself is wider than the
    target on either side, ``fails`` otherwise."""
    rows = []
    for label, cube, cfg, plane in cases:
        reg = ps.regions(cube, cfg, plane)
        pred = ps.prediction(reg)
        k = ps.kappa(reg)
        row = {"validation region": label, "plane": plane, "A observed": tl.fmt_count(reg["A"]),
               "B C/D": "n/a" if np.isnan(pred[0]) else f"{pred[0]:.1f} ± {pred[1]:.1f}", "kappa": tl.fmt(k, 2)}
        if not np.isfinite(k[0]):
            row.update({"kappa - 1 [%]": "n/a", "precision [%]": "n/a", "verdict": "A hidden or a region empty"})
        else:
            dev, minus, plus = k[0] - 1, k[1], k[2]
            prec = precision(k)
            row["kappa - 1 [%]"] = f"{100 * dev:+.0f} (-{100 * minus:.0f}/+{100 * plus:.0f})"
            row["precision [%]"] = f"{100 * prec:.0f}"
            if prec > target:
                verdict = "imprecise"
            elif abs(dev) <= target:
                verdict = "closes"
            elif (dev > 0 and dev - minus <= 0) or (dev < 0 and dev + plus >= 0):
                verdict = "closes within errors"
            else:
                verdict = "fails"
            row["verdict"] = f"{verdict} (target {100 * target:.0f}%)"
        rows.append(row)
    return pd.DataFrame(rows)


def scaled(pred, factor):
    """``pred`` (value, error) times ``factor`` (value, minus, plus), as (value, minus, plus)."""
    if not (np.isfinite(pred[0]) and np.isfinite(factor[0])) or pred[0] <= 0 or factor[0] <= 0:
        return np.nan, np.nan, np.nan
    v = pred[0] * factor[0]
    rel = pred[1] / pred[0]
    return v, v * np.hypot(rel, factor[1] / factor[0]), v * np.hypot(rel, factor[2] / factor[0])


def final_table(cube, cfg_windows, kappas, plane="DD"):
    """The in-plane estimate of A (``B C/D``) per mass window, multiplied by each closure factor
    in ``kappas`` (``{name: (value, minus, plus)}``); a factor of one is the uncorrected ABCD."""
    rows = []
    for label, cfg in cfg_windows:
        reg = ps.regions(cube, cfg, plane)
        pred = ps.prediction(reg)
        row = {"window": label, f"{plane} B/C/D": f"{reg['B'][0]:.0f}/{reg['C'][0]:.0f}/{reg['D'][0]:.0f}",
               "B C/D": "n/a" if np.isnan(pred[0]) else f"{pred[0]:.1f} ± {pred[1]:.1f}"}
        for name, k in kappas.items():
            row[f"x kappa {name}"] = tl.fmt(scaled(pred, k), 1)
        rows.append(row)
    return pd.DataFrame(rows)


def _add(a, b):
    """Sum of two (value, minus, plus) estimates, errors added in quadrature; NaN propagates."""
    return a[0] + b[0], np.hypot(a[1], b[1]), np.hypot(a[2], b[2])


ROUTES = (("B", "A/B"), ("C", "A/C"), ("D", "A/D"))


def _route(cube, cfg, search, ref, sideband, cls=None):
    """``sideband`` of the search plane times A/``sideband`` of the reference plane, for one
    class of the muon LJ (or inclusive)."""
    tag = "" if cls is None else f":{cls}"
    count = ps.regions(cube, cfg, f"{search}{tag}")[sideband]
    r = ps.regions(cube, cfg, f"{ref}{tag}")
    return tl._pred(count, tl.ratio(r["A"], r[sideband]))


def routes_table(cube, cfg_windows, classes=("dsa0", "dsa1+"), ref_plane="PD", search="DD"):
    """Four estimates of A of the search plane per window, next to the observed A ("blind"
    where hidden): the plane's own B C / D, which needs the two axes to be independent and
    nothing else, and the three borrowed forms B × (A/B), C × (A/C) and D × (A/D), each with the
    ratio taken from ``ref_plane`` per class of the muon LJ and summed over ``classes``; the
    inclusive borrowed forms stand beside them.  The per-class and inclusive forms differ when
    the classes have different ratios and the search plane's sidebands have a different class
    mix than the reference plane."""
    rows = []
    for label, cfg in cfg_windows:
        reg = ps.regions(cube, cfg, search)
        row = {"window": label, f"{search} B/C/D": "/".join(tl.fmt_count(reg[r]) for r in "BCD")}
        pred = ps.prediction(reg)
        row["direct B C/D"] = tl.fmt((pred[0], pred[1], pred[1]) if np.isfinite(pred[0]) else (np.nan,) * 3, 1)
        for sideband, name in ROUTES:
            total = (0.0, 0.0, 0.0)
            for cls in classes:
                total = _add(total, _route(cube, cfg, search, ref_plane, sideband, cls))
            row[f"{sideband} x {name} per class"] = tl.fmt(total, 1)
            row[f"{sideband} x {name} inclusive"] = tl.fmt(_route(cube, cfg, search, ref_plane, sideband), 1)
        row[f"{search} A observed"] = tl.fmt_count(reg["A"])
        rows.append(row)
    return pd.DataFrame(rows)


def class_factors_table(cube, cfg_windows, classes=("dsa0", "dsa1+"), ref_plane="PD", search="DD"):
    """Per window and class of the muon LJ: the search plane's B, C and D, the reference plane's
    A/B, A/C and A/D (the borrowed factors), and C/D in both planes — the sideband-only test
    that the isolation pass/fail ratio of the leading muon LJ is the same in the two planes,
    which is what B × (A/B) assumes."""
    rows = []
    for label, cfg in cfg_windows:
        for cls in (*classes, None):
            tag = "" if cls is None else f":{cls}"
            reg = ps.regions(cube, cfg, f"{search}{tag}")
            ref = ps.regions(cube, cfg, f"{ref_plane}{tag}")
            row = {"window": label, "class": "all" if cls is None else cls,
                   f"{search} B/C/D": "/".join(tl.fmt_count(reg[r]) for r in "BCD"),
                   f"{ref_plane} A/B/C/D": "/".join(tl.fmt_count(ref[r]) for r in "ABCD")}
            for sideband, name in ROUTES:
                row[f"{name} {ref_plane}"] = tl.fmt(tl.ratio(ref["A"], ref[sideband]), 2)
            row[f"C/D {ref_plane}"] = tl.fmt(tl.ratio(ref["C"], ref["D"]), 2)
            row[f"C/D {search}"] = tl.fmt(tl.ratio(reg["C"], reg["D"]), 2)
            row[f"C/D {search} over {ref_plane}"] = tl.fmt(tl.ratio_of_ratios(tl.ratio(reg["C"], reg["D"]), tl.ratio(ref["C"], ref["D"])), 2)
            rows.append(row)
    return pd.DataFrame(rows)


def parse_fmt(text):
    """Inverse of ``transfer_lib.fmt``: (value, minus, plus) from "v ± e", "v -m +p", "v -m +inf"
    or "n/a" (NaN)."""
    parts = str(text).split()
    if not parts or parts[0] == "n/a":
        return np.nan, np.nan, np.nan
    v = float(parts[0])
    if len(parts) == 3 and parts[1] == "±":
        e = float(parts[2])
        return v, e, e
    return v, -float(parts[1]), float(parts[2])


def kappa_by_family(scan, column, target=0.15):
    """One row per plane family of a ``closure_scan`` table (the two axes of the configuration
    label, e.g. "dphi x iso0"): how many configurations, the span and median of the kappa in
    ``column``, the span of its precision, and how many 68% intervals lie above one, below one
    or cover it (undefined kappas counted separately).  Says whether a deviation belongs to the
    plane or to one boundary choice."""
    fam = scan["config"].str.extract(r"^(\w+) \S+ x (\w+)").agg(" x ".join, axis=1)
    parsed = pd.DataFrame([parse_fmt(t) for t in scan[column]], columns=["kappa", "minus", "plus"], index=scan.index)
    rows = []
    for name, group in parsed.groupby(fam, sort=False):
        k = group[np.isfinite(group["kappa"])]
        prec = k[["minus", "plus"]].max(axis=1) / k["kappa"]
        rows.append({"family": name, "configurations": len(group), "kappa defined": len(k),
                     "kappa min-max": f"{k['kappa'].min():.2f}-{k['kappa'].max():.2f}" if len(k) else "n/a",
                     "median": f"{k['kappa'].median():.2f}" if len(k) else "n/a",
                     "precision min-max [%]": f"{100 * prec.min():.0f}-{100 * prec.max():.0f}" if len(k) else "n/a",
                     "interval above 1": int((k["kappa"] - k["minus"] > 1).sum()),
                     "interval below 1": int((k["kappa"] + k["plus"] < 1).sum()),
                     "interval covers 1": int(((k["kappa"] - k["minus"] <= 1) & (k["kappa"] + k["plus"] >= 1)).sum()),
                     f"within {100 * target:.0f}% at that precision": int((((k["kappa"] - 1).abs() <= target) & (prec <= target)).sum())})
    return pd.DataFrame(rows)


def scan_summary(scan, labels, target=0.15):
    """One row per validation region of a ``closure_scan`` table: in how many configurations its
    kappa is defined, the best precision reached, how many are measured to ``target`` precision,
    the span and median of kappa among those, and how many of them lie within ``target`` of one.
    Read from the table's two-decimal kappa strings, so a value at the boundary can fall on the
    other side of it than in the scan's own exact flags."""
    rows = []
    for label in labels:
        parsed = pd.DataFrame([parse_fmt(t) for t in scan[f"kappa {label}"]], columns=["kappa", "minus", "plus"])
        k = parsed[np.isfinite(parsed["kappa"])]
        prec = k[["minus", "plus"]].max(axis=1) / k["kappa"]
        fine = k[prec <= target]
        rows.append({"validation region": label, "configurations": len(parsed), "kappa defined": len(k),
                     "best precision [%]": f"{100 * prec.min():.0f}" if len(k) else "n/a",
                     f"measured to {100 * target:.0f}%": len(fine),
                     "kappa min-max (those)": f"{fine['kappa'].min():.2f}-{fine['kappa'].max():.2f}" if len(fine) else "n/a",
                     "median": f"{fine['kappa'].median():.2f}" if len(fine) else "n/a",
                     f"within {100 * target:.0f}%": int(((fine["kappa"] - 1).abs() <= target).sum())})
    return pd.DataFrame(rows)


def count_grid(cube, cfg_windows, plane, splits):
    """Counts per ABCD region of ``plane`` in each window, split by the plane suffixes in
    ``splits`` (``"dsa2+:subdsa2+"`` is a 4mu event whose two muon LJs are both DSA-only) next to
    the plane's total; hidden cells print as blind.  Shows what kind of object populates each
    region."""
    rows = []
    for label, cfg in cfg_windows:
        total = ps.regions(cube, cfg, plane)
        parts = {s: ps.regions(cube, cfg, f"{plane}:{s}") for s in splits}
        for r in "ABCD":
            row = {"window": label, "region": r, "total": tl.fmt_count(total[r])}
            row.update({s: tl.fmt_count(parts[s][r]) for s in splits})
            rows.append(row)
    return pd.DataFrame(rows)


def sideband_transfer_table(cube, window, ref_window, search="DD", refs=("DP", "PD", "PP")):
    """B of the search plane in ``window`` (``(label, cfg)``) times A/B of each reference plane
    measured in ``ref_window``, where the reference plane's A is open, next to A/B of the same
    plane in ``window`` where that is open: a count-limited estimate of the search plane's A for
    a window whose in-plane estimate has no D."""
    (label, cfg), (ref_label, ref_cfg) = window, ref_window
    b = ps.regions(cube, cfg, search)["B"]
    rows = []
    for plane in refs:
        lo, hi = ps.regions(cube, ref_cfg, plane), ps.regions(cube, cfg, plane)
        ab_lo = tl.ratio(lo["A"], lo["B"])
        rows.append({"reference plane": plane, f"A, B ({ref_label})": f"{tl.fmt_count(lo['A'])}, {tl.fmt_count(lo['B'])}",
                     f"A/B ({ref_label})": tl.fmt(ab_lo, 2),
                     f"A/B ({label})": "blind" if np.isnan(hi["A"][0]) else tl.fmt(tl.ratio(hi["A"], hi["B"]), 2),
                     f"{search} B ({label})": tl.fmt_count(b),
                     f"estimate of {search} A ({label})": tl.fmt(tl._pred(b, ab_lo), 2)})
    return pd.DataFrame(rows)


def mass_transfer_table(cube, window, ref_window, search="DD", refs=(("PP", "A"), ("DP", "B"), ("PD", "B"))):
    """A of the search plane in the open window ``ref_window`` times the ratio of the counts of
    each open reference region between ``window`` and ``ref_window``: the second count-limited
    estimate of the search plane's A in ``window``."""
    (label, cfg), (ref_label, ref_cfg) = window, ref_window
    a_lo = ps.regions(cube, ref_cfg, search)["A"]
    rows = []
    for plane, region in refs:
        n_hi, n_lo = ps.regions(cube, cfg, plane)[region], ps.regions(cube, ref_cfg, plane)[region]
        r = tl.ratio(n_hi, n_lo)
        rows.append({"reference region": f"{plane} {region}", f"N ({label})": tl.fmt_count(n_hi),
                     f"N ({ref_label})": tl.fmt_count(n_lo), "ratio": tl.fmt(r, 2),
                     f"{search} A ({ref_label})": tl.fmt_count(a_lo),
                     f"estimate of {search} A ({label})": tl.fmt(tl._pred(a_lo, r), 2)})
    return pd.DataFrame(rows)


def mc_region_table(cube, cfg_windows, planes):
    """Luminosity-scaled simulation per region of each plane and window: the yield with its
    weight error and its effective count (simulation is never blinded)."""
    rows = []
    for label, cfg in cfg_windows:
        for plane in planes:
            reg = ps.regions(cube, cfg, plane)
            row = {"window": label, "plane": plane}
            for r in "ABCD":
                w, w2 = reg[r]
                row[r] = f"{w:.2f} ± {np.sqrt(w2):.2f} (N_eff {tl.n_eff(reg[r]):.1f})"
            rows.append(row)
    return pd.DataFrame(rows)


def signal_cells_table(base, names, topology, cells, prefix="abcd8d"):
    """Yield of each signal point in the ``cells`` (``(label, cfg, plane, region)``) as a fraction
    of its yield in the first cell, with the yield at the stored cross section in brackets; one
    row per cell, one column per point.  Samples are loaded one at a time."""
    columns = {}
    for name in names:
        cube = tl.sample_cubes(base, "signal", [name], topologies=(topology,), prefix=prefix)[name][topology]
        yields = [ps.regions(cube, cfg, plane)[region][0] for _, cfg, plane, region in cells]
        ref = yields[0]
        columns[name] = [f"{y / ref:.3f} [{y:.2f}]" if ref > 0 else f"n/a [{y:.2f}]" for y in yields]
        del cube
    table = pd.DataFrame(columns, index=[label for label, *_ in cells])
    table.index.name = "region"
    return table.reset_index()


def counting_limit(n_obs, b, cl=0.95):
    """Upper limit at confidence ``cl`` on the signal count of a single-bin counting experiment
    with ``n_obs`` events observed and a known background ``b``, no systematic uncertainty: the
    CLs limit, P(n <= n_obs | s + b) / P(n <= n_obs | b) = 1 - cl, which for this case coincides
    with the flat-prior Bayesian limit."""
    cl_b = stats.poisson.cdf(n_obs, b)
    return optimize.brentq(lambda s: stats.poisson.cdf(n_obs, s + b) / cl_b - (1 - cl), 0.0, 50.0 + 10.0 * n_obs)


def counting_limit_table(observations, backgrounds, cl=0.95):
    """:func:`counting_limit` for each observed count (rows) and expected background (columns):
    what an uncertainty of the background estimate costs when the counts are this small."""
    rows = []
    for n in observations:
        row = {"observed": n}
        row.update({f"b = {b:g}": f"{counting_limit(n, b, cl):.2f}" for b in backgrounds})
        rows.append(row)
    return pd.DataFrame(rows)
