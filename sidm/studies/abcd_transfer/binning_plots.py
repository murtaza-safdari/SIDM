"""Figures of the per-bin estimate: the four planes and the per-bin route as cartoons, the signal and
search-plane spectra against the LJ-pair mass, A/B against the mass with its fits, the background
and signal per bin of each binning, and the toy expected limits relative to one bin."""
import re

import matplotlib.pyplot as plt
import numpy as np
from matplotlib import patches, ticker

import binning_lib as bl
import plane_search_lib as ps
import plane_search_plots as pp
import transfer_plots as tp

TICK = pp.TICK
AXIS = pp.AXIS
TEXT = 20
M_MAX = 2500.0                              # upper edge of the last stored mass bin; open-ended bins are drawn to it
MASS_TICKS = (150, 200, 300, 400, 600, 1000, 2500)
CLASS_COLORS = {"dsa0": "tab:cyan", "dsa1+": "tab:orange", None: "black"}
REGION_COLORS = {"B": "tab:purple", "C": "tab:brown", "D": "tab:olive"}
FORM_STYLES = {"linear": "--", "exponential": "-"}
MARKERS = ("o", "s", "D", "^", "v", "P", "X")
FIRST = {"2mu2e": "muon LJ", "4mu": "leading muon LJ"}
OTHER = {"2mu2e": "egamma LJ", "4mu": "subleading muon LJ"}
DEFAULT_POINTS = ("2Mu2E_500GeV_0p25GeV_0p4mm", "2Mu2E_800GeV_0p25GeV_0p25mm")


def point_label(name):
    """``2Mu2E_500GeV_0p25GeV_0p4mm`` -> ``500 GeV, 0.25 GeV, 0.4 mm`` (mediator mass, dark-photon mass,
    dark-photon decay length)."""
    m = re.match(r"^[^_]+_(\w+)GeV_(\w+)GeV_(\w+)mm$", name)
    if not m:
        return name
    return ", ".join(f"{v.replace('p', '.')} {u}" for v, u in zip(m.groups(), ("GeV", "GeV", "mm")))


def candidate_edges():
    """The inner bin edges of every candidate binning."""
    return sorted({lo for bins in bl.BINNINGS.values() for _, (lo, _) in bins} - {150.0})


def _mass_axis(ax, label=True):
    ax.set_xscale("log")
    ax.set_xlim(150.0, M_MAX)
    ax.xaxis.set_major_locator(ticker.FixedLocator(MASS_TICKS))
    ax.xaxis.set_major_formatter(ticker.FixedFormatter([f"{t:g}" for t in MASS_TICKS]))
    ax.xaxis.set_minor_locator(ticker.NullLocator())
    ax.tick_params(labelsize=TICK)
    if label:
        ax.set_xlabel("M(LJ, LJ) [GeV]", fontsize=AXIS)


# ----------------------------------------------------------------------------- cartoons


def planes_cartoon(cube, cfg, topology="2mu2e"):
    """The four displacement planes as a 2 x 2 grid (rows: muon LJ prompt, displaced; columns: the
    other LJ prompt, displaced), each a square split into A (isolated muon LJ, back-to-back; top
    left), B (top right), C (bottom left) and D (bottom right), with the data counts of ``cfg`` in
    each cell; a hidden cell is hatched and reads "hidden".  The solid arrow is the estimate (A/B of
    PD per DSA class applied to B of DD), the dashed one the same step from PP to DP; where the A
    an arrow starts or ends in is hidden, the label says so."""
    fig, ax = plt.subplots(figsize=(18, 14))
    ax.set_axis_off()
    ax.set_aspect("equal")
    ax.set_xlim(-1.1, 4.7)
    ax.set_ylim(-0.5, 3.15)
    size, half = 1.0, 0.5
    origin = {"PP": (0.0, 1.55), "PD": (1.9, 1.55), "DP": (0.0, 0.0), "DD": (1.9, 0.0)}
    cells = {"A": (0.0, half), "B": (half, half), "C": (0.0, 0.0), "D": (half, 0.0)}
    hidden_a = {}
    for plane, (x0, y0) in origin.items():
        reg = ps.regions(cube, cfg, plane)
        color = tp.COLORS[plane]
        hidden_a[plane] = bool(np.isnan(reg["A"][0]))
        for r, (dx, dy) in cells.items():
            n = reg[r][0]
            hidden = np.isnan(n)
            ax.add_patch(patches.Rectangle((x0 + dx, y0 + dy), half, half, facecolor="0.88" if hidden else "white",
                                           edgecolor=color, lw=1.5, hatch="//" if hidden else None))
            ax.text(x0 + dx + half / 2, y0 + dy + half / 2, f"{r}\n{'hidden' if hidden else f'{n:.0f}'}",
                    ha="center", va="center", fontsize=TEXT + 2)
        ax.add_patch(patches.Rectangle((x0, y0), size, size, fill=False, edgecolor=color, lw=4))
        ax.text(x0 + size / 2, y0 + size + 0.05, plane, ha="center", va="bottom", fontsize=TEXT + 4, color=color,
                fontweight="bold")
    for x0, state in ((0.0, "prompt"), (1.9, "displaced")):
        ax.text(x0 + size / 2, 2.85, f"{OTHER[topology]} {state}", ha="center", va="bottom", fontsize=TEXT)
    for y0, state in ((1.55, "prompt"), (0.0, "displaced")):
        ax.text(-0.85, y0 + size / 2, f"{FIRST[topology]} {state}", ha="center", va="center", rotation=90,
                fontsize=TEXT)
        ax.annotate("", xy=(-0.12, y0 + 0.95), xytext=(-0.12, y0 + 0.05), arrowprops=dict(arrowstyle="-|>", lw=2))
        ax.text(-0.3, y0 + size / 2, f"|dphi| (boundary {cfg.dphi_pass:g})", ha="center", va="center", rotation=90,
                fontsize=TEXT - 3)
    for x0 in (0.0, 1.9):
        ax.annotate("", xy=(x0 + 0.95, -0.12), xytext=(x0 + 0.05, -0.12), arrowprops=dict(arrowstyle="-|>", lw=2))
        ax.text(x0 + size / 2, -0.3, f"muon-LJ isolation (boundary {cfg.wp0:g})", ha="center", va="center",
                fontsize=TEXT - 3)
    ax.add_patch(patches.FancyArrowPatch((2.93, 2.3), (2.93, 0.75), connectionstyle="arc3,rad=-0.45",
                                         arrowstyle="-|>", mutation_scale=35, lw=3.5, color=tp.COLORS["DD"],
                                         ls="--" if hidden_a["PD"] else "-"))
    borrow = "no A/B to borrow:\nA of PD hidden" if hidden_a["PD"] else "borrow A/B\nper DSA class"
    ax.text(3.65, 1.52, borrow, ha="left", va="center", fontsize=TEXT, color=tp.COLORS["DD"])
    ax.add_patch(patches.FancyArrowPatch((1.03, 2.3), (1.03, 0.75), connectionstyle="arc3,rad=-0.12",
                                         arrowstyle="-|>", mutation_scale=30, lw=2.5, ls="--", color=tp.COLORS["DP"]))
    step = "the same step\n(A of DP hidden)" if hidden_a["DP"] else "the same step,\ntested where\nA is open"
    ax.text(1.2, 1.275, step, ha="left", va="center", fontsize=TEXT - 2, color=tp.COLORS["DP"])
    window = f"M >= {cfg.mass[0]:g} GeV" if cfg.mass[1] is None else f"M {cfg.mass[0]:g}-{cfg.mass[1]:g} GeV"
    ax.text(-1.05, 3.12, f"{topology}: the four displacement planes, data, {window}", ha="left", va="top",
            fontsize=TEXT + 2)
    tp.save(fig, f"binning_planes_cartoon_{topology}")
    return fig


def route_cartoon():
    """The estimate of one mass bin as a flow diagram (schematic, no numbers)."""
    fig, ax = plt.subplots(figsize=(18, 8))
    ax.set_axis_off()
    ax.set_xlim(0, 18)
    ax.set_ylim(0, 8)
    ax.annotate("", xy=(17.0, 1.2), xytext=(0.6, 1.2), arrowprops=dict(arrowstyle="-|>", lw=2))
    ax.text(17.1, 1.2, "M(LJ, LJ)", ha="left", va="center", fontsize=TEXT - 2)
    chosen = (6.0, 11.0)
    for label, x0, x1 in (("150-300 GeV", 1.0, 6.0), ("300-600 GeV", 6.0, 11.0), (">= 600 GeV", 11.0, 16.5)):
        on = (x0, x1) == chosen
        ax.add_patch(patches.Rectangle((x0, 0.8), x1 - x0, 0.8, facecolor=tp.COLORS["DD"] if on else "white",
                                       alpha=0.3 if on else 1.0, edgecolor="black", lw=1.5))
        ax.text((x0 + x1) / 2, 0.35, label, ha="center", va="center", fontsize=TEXT - 2)
    dd, pd_ = tp.COLORS["DD"], tp.COLORS["PD"]
    items = [("B of DD,\ndsa0", dd), ("x", None), ("(A/B) of PD,\ndsa0", pd_), ("+", None),
             ("B of DD,\ndsa1+", dd), ("x", None), ("(A/B) of PD,\ndsa1+", pd_), ("=", None),
             ("background\nin the bin", "black")]
    x, y, width, gap = 0.9, 4.4, 2.65, 0.65
    for text, color in items:
        if color is None:
            ax.text(x + gap / 2, y, text, ha="center", va="center", fontsize=TEXT + 10)
            x += gap
            continue
        ax.add_patch(patches.FancyBboxPatch((x, y - 0.65), width, 1.3, boxstyle="round,pad=0.05", facecolor="white",
                                            edgecolor=color, lw=2.5))
        ax.text(x + width / 2, y, text, ha="center", va="center", fontsize=TEXT - 1)
        x += width
    for xb, xf in ((chosen[0], 0.9), (chosen[1], x)):
        ax.plot([xb, xf], [1.6, y - 0.75], color="gray", ls="--", lw=1.5)
    ax.text(9.0, 6.55, "(A/B) comes from the bin itself when PD measures it to <= 20%;\n"
                       "otherwise from the smooth fit A/B(M), at the mean mass of DD's B events in the bin,\n"
                       "with |measured - fitted| as a shape systematic",
            ha="center", va="center", fontsize=TEXT - 2,
            bbox=dict(boxstyle="round,pad=0.4", facecolor="white", edgecolor=pd_, lw=1.5))
    ax.text(0.2, 7.85, "The B route in one mass bin", ha="left", va="top", fontsize=TEXT + 2)
    ax.text(17.8, 7.85, "schematic", ha="right", va="top", fontsize=TEXT, style="italic", color="gray")
    tp.save(fig, "binning_route_cartoon")
    return fig


# ----------------------------------------------------------------------------- spectra and trends


def mass_shapes_figure(signal_shapes, dd_counts, edges, topology="2mu2e"):
    """Top: the LJ-pair mass distribution of each signal point in A of DD above 150 GeV, unit area,
    on the cube's mass bins (``signal_shapes``: ``{point: sumw}`` or ``{point: (sumw, sumw2)}`` on the
    flow-folded bins of ``edges``), with the candidate bin edges; bottom: B, C and D of DD in data
    per mass bin (``dd_counts``: ``{region: sumw}`` on the same bins), on the same axis."""
    edges = np.asarray(edges, float)
    i0 = int(np.nonzero(np.isclose(edges, 150.0))[0][0])
    e = edges[i0:]
    fig, (ax, bx) = plt.subplots(2, 1, figsize=(18, 14), sharex=True, gridspec_kw={"height_ratios": [1.3, 1.0], "hspace": 0.06})
    colors = plt.cm.tab10(np.arange(max(len(signal_shapes), 1)) % 10)
    for color, (name, values) in zip(colors, signal_shapes.items()):
        sumw = np.asarray(values[0] if isinstance(values, tuple) else values, float)[i0:]
        if sumw.sum() > 0:
            ax.stairs(sumw / sumw.sum(), e, color=color, lw=2.5, label=point_label(name))
    for x in candidate_edges():
        ax.axvline(x, color="gray", ls=":", lw=2)
    ax.plot([], [], color="gray", ls=":", lw=2, label="candidate bin edges")
    ax.set_yscale("log")
    ax.set_ylim(1e-3, 300)
    ax.set_ylabel("fraction of the yield above 150 GeV", fontsize=AXIS)
    ax.tick_params(labelsize=TICK)
    ax.legend(title=f"{topology} signal in A of DD (the search region)", loc="upper right", ncol=2, **tp.LEGEND)
    tp.label(ax)
    centres = np.sqrt(e[:-1] * e[1:])
    top = 1.0
    for r, color in REGION_COLORS.items():
        n = np.asarray(dd_counts[r], float)[i0:]
        bx.stairs(n, e, color=color, lw=2.5, label=f"{r}: {n.sum():.0f} events")
        bx.errorbar(centres, n, yerr=np.sqrt(n), fmt="none", ecolor=color, capsize=3)
        top = max(top, n.max())
    bx.set_yscale("log")
    bx.set_ylim(0.5, top * 60)
    bx.set_ylabel("events per bin", fontsize=AXIS)
    bx.legend(title=f"{topology}, DD plane in data (A hidden)", loc="upper right", ncol=3, **tp.LEGEND)
    _mass_axis(bx)
    tp.save(fig, f"binning_mass_shapes_{topology}")
    return fig


def _points(ax, df, value, shift, color, label):
    """The finite ``value`` points of ``df`` (columns ``x``, ``value``, ``minus``, ``plus``) with
    their asymmetric errors, the masses multiplied by ``shift`` to keep overlapping series apart;
    a bin with a finite ``lower`` column (no B event) is drawn as an upward arrow at that lower
    limit."""
    x, v, m, p = (df[c].to_numpy(float) for c in ("x", value, "minus", "plus"))
    ok = np.isfinite(v)
    ax.errorbar(x[ok] * shift, v[ok], yerr=[m[ok], p[ok]], fmt="o", color=color, ms=9, capsize=3, label=label)
    if "lower" in df:
        lower = df["lower"].to_numpy(float)
        low = np.isfinite(lower)
        if low.any():
            ax.errorbar(x[low] * shift, lower[low], yerr=[np.zeros(low.sum()), 0.25 * np.ones(low.sum())], lolims=True,
                        fmt="none", color=color, lw=2, capsize=0)
            ax.plot(x[low] * shift, lower[low], marker="_", ms=14, mew=2, ls="none", color=color)


def ab_mass_figure(points_by_key, fits_by_key, planes=("PD", "PP"), classes=("dsa0", "dsa1+", None),
                   band_form="exponential", topology="2mu2e"):
    """Top: A/B against the LJ-pair mass in each plane of ``planes`` per DSA class
    (``points_by_key[(plane, class)]``, :func:`binning_lib.ab_points`) with the fitted curves
    (``fits_by_key[(plane, class, form)]``) and their deviance/ndof, the 68% band of ``band_form``;
    bottom: A/B of the second plane over the first per bin and class, with its weighted mean."""
    fig = plt.figure(figsize=(18, 15))
    grid = fig.add_gridspec(2, 2, height_ratios=[1.45, 1.0], hspace=0.28, wspace=0.18)
    xs = np.geomspace(150.0, M_MAX, 200)
    other = next(f for f in bl.FORMS if f != band_form)
    for col, plane in enumerate(planes):
        ax = fig.add_subplot(grid[0, col])
        for k, cls in enumerate(classes):
            pts, color, name = points_by_key[(plane, cls)], CLASS_COLORS[cls], "all" if cls is None else cls
            deviance = []
            for form in bl.FORMS:
                fit = fits_by_key.get((plane, cls, form))
                if fit is None or not np.isfinite(fit["params"][0]):
                    continue
                v, err = fit["f"](xs)
                ax.plot(xs, v, ls=FORM_STYLES[form], color=color, lw=2)
                deviance.append(f"{form} {fit['deviance']:.1f}/{fit['ndof']}")
                if form == band_form:
                    ax.fill_between(xs, v - err, v + err, color=color, alpha=0.12, lw=0)
            _points(ax, pts, "A/B", 1 + 0.02 * (k - 1), color=color,
                    label=f"{name}: {', '.join(deviance)}" if deviance else name)
        _mass_axis(ax)
        ax.set_ylim(0, 3.5)
        ax.set_ylabel(f"A/B of {plane}", fontsize=AXIS)
        ax.legend(title=f"{topology}, {plane}: A/B measured per bin;\n"
                        f"arrow: no B event (68% lower limit);\n"
                        f"solid: {band_form} fit + 68% band, dashed: {other};\n"
                        f"per class: the deviance / ndof of each form",
                  loc="upper left", fontsize=16, **{**tp.LEGEND, "title_fontsize": 17})
        tp.label(ax, fontsize=18)
    rax = fig.add_subplot(grid[1, :])
    num, den = planes[1], planes[0]
    for k, cls in enumerate(classes):
        r = bl.plane_ratio_points(points_by_key[(num, cls)], points_by_key[(den, cls)])
        mean, err, chi2, ndof = bl.weighted_mean(r["ratio"], r["minus"], r["plus"])
        color, name = CLASS_COLORS[cls], "all" if cls is None else cls
        label = f"{name}: mean {mean:.2f} ± {err:.2f}, chi2/ndof {chi2:.1f}/{ndof}" if np.isfinite(mean) else name
        _points(rax, r, "ratio", 1 + 0.02 * (k - 1), color=color, label=label)
        if np.isfinite(mean):
            rax.axhline(mean, color=color, ls="--", lw=1.5)
    _mass_axis(rax)
    rax.set_ylim(0, 2.0)
    rax.set_ylabel(f"A/B {num} / A/B {den}", fontsize=AXIS)
    rax.legend(title=f"{topology}: {num} over {den} per mass bin; dashed: weighted mean", loc="upper right", **tp.LEGEND)
    tp.label(rax, fontsize=18)
    tp.save(fig, f"binning_ab_mass_{topology}")
    return fig


# ----------------------------------------------------------------------------- binnings and limits


def binning_figure(routes_by_binning, yields_by_binning, points=DEFAULT_POINTS, syst_rel=0.33, topology="2mu2e"):
    """One panel per binning: the selected background per bin (:func:`binning_lib.binned_routes`)
    with its statistical error and a hatched ``syst_rel`` band, and the yield of the signal
    ``points`` per bin at the stored cross section (:func:`binning_lib.signal_yields`)."""
    names = list(routes_by_binning)
    nrow = (len(names) + 1) // 2
    fig, axes = plt.subplots(nrow, 2, figsize=(18, 7.5 * nrow), squeeze=False)
    for ax, name in zip(axes.flat, names):
        df = routes_by_binning[name]
        bins = df[(df["class"] == bl.SUM) & (df["bin"] != bl.TOTAL)]
        lo = bins["lo"].to_numpy(float)
        hi = np.where(np.isnan(bins["hi"].to_numpy(float)), M_MAX, bins["hi"].to_numpy(float))
        sel = bins["selected"].to_numpy(float)
        ax.bar(lo, sel, width=hi - lo, align="edge", color=tp.COLORS["DD"], alpha=0.3, edgecolor=tp.COLORS["DD"], lw=2,
               label="background (selected A/B)")
        ax.bar(lo, sel * (1 + syst_rel) - sel / (1 + syst_rel), bottom=sel / (1 + syst_rel), width=hi - lo, align="edge",
               fill=False, hatch="//", edgecolor="0.35", lw=0, label=f"{100 * syst_rel:.0f}% systematic (log-normal)")
        xc = np.sqrt(lo * hi)
        carried = bins["shape carried"].to_numpy(float)
        ax.errorbar(xc, sel, yerr=[np.hypot(bins["selected minus"].to_numpy(float), carried),
                                   np.hypot(bins["selected plus"].to_numpy(float), carried)],
                    fmt="none", ecolor="darkgreen", capsize=6, lw=2.5, label="statistical error (+) shape systematic, in quadrature")
        top = np.nanmax(sel) if np.isfinite(sel).any() else 1.0
        y = yields_by_binning[name]
        for k, point in enumerate(points):
            ys = y[y["point"] == point].set_index("bin").loc[list(bins["bin"]), "yield"].to_numpy(float)
            ax.plot(xc * (1 + 0.04 * (k - (len(points) - 1) / 2)), ys, marker=MARKERS[k], ls="none", ms=13,
                    color=f"C{k + 3}", label=f"signal {point_label(point)}")
            top = max(top, np.nanmax(ys))
        _mass_axis(ax)
        ax.set_yscale("log")
        ax.set_ylim(0.05, top * 150)
        ax.set_ylabel("events per bin", fontsize=AXIS)
        ax.legend(title=f"{topology}, {name}", loc="upper right", fontsize=16, **{**tp.LEGEND, "title_fontsize": 17})
        tp.label(ax, fontsize=18)
    for ax in axes.flat[len(names):]:
        ax.set_visible(False)
    fig.tight_layout()
    tp.save(fig, f"binning_backgrounds_{topology}")
    return fig


def limit_figure(limits, factor="selected"):
    """The toy expected limit relative to the one-bin estimate (:func:`binning_lib.limit_table`) per
    signal point, one marker per binning, for one factor choice; the decision points are shaded."""
    part = limits[limits["factor"] == factor]
    points = list(dict.fromkeys(part["point"]))
    binnings = list(dict.fromkeys(part["binning"]))
    fig, ax = plt.subplots(figsize=(18, 8))
    x = np.arange(len(points))
    offsets = np.linspace(-0.27, 0.27, len(binnings))
    for k, (name, off) in enumerate(zip(binnings, offsets)):
        vals = part[part["binning"] == name].set_index("point").loc[points, "relative to one bin"].to_numpy(float)
        ax.plot(x + off, vals, marker=MARKERS[k], ls="none", ms=14, color=f"C{k}", label=name)
    decision = part.drop_duplicates("point").set_index("point").loc[points, "decision point"].to_numpy(bool)
    for i in np.nonzero(decision)[0]:
        ax.axvspan(i - 0.5, i + 0.5, color="tab:orange", alpha=0.1, lw=0)
    ax.axhline(1, color="gray", lw=1.5)
    ax.set_xticks(x)
    ax.set_xticklabels([point_label(p).replace(", ", "\n", 1) for p in points], fontsize=TICK - 2)
    ax.tick_params(axis="y", labelsize=TICK)
    ax.set_xlim(-0.5, len(points) - 0.5)
    values = part["relative to one bin"].to_numpy(float)
    hi = max(1.15, np.nanmax(values) * 1.05) if np.isfinite(values).any() else 1.15
    ax.set_ylim(0, 1.45 * hi)
    ax.set_ylabel("expected limit / one bin", fontsize=AXIS)
    ax.legend(title=f"toy expected limit, {factor} A/B; shaded: the 500 and 800 GeV points", loc="upper right",
              ncol=len(binnings), **tp.LEGEND)
    tp.label(ax)
    tp.save(fig, f"binning_limits_{factor}")
    return fig
