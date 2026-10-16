"""Figures of nominal.ipynb: the ABCD method as a cartoon, the muon-LJ isolation x |dphi| plane of
the data with its four regions, kappa per validation region, against the displacement and against
the mass, the photon-only share of the egamma LJs per region, and the signal share per mass bin."""
import re

import matplotlib.pyplot as plt
import mplhep as hep
import numpy as np
from matplotlib import colors, patches, ticker

import abcd6d_io as io
import binning_plots as bp
import nominal_lib as nl
import transfer_lib as tl
import transfer_plots as tp

TICK = 18
AXIS = 22
TEXT = 20
SLIDES, HIS, OURS = "slides (his code)", "these histograms, his WP", "these histograms, our WP"
SERIES = {SLIDES: ("tab:gray", "s"), HIS: ("tab:orange", "D"), OURS: ("black", "o")}
HIDDEN = dict(facecolor="0.85", edgecolor="0.4", hatch="//")


def _sim_label(ax, fontsize=22):
    hep.cms.label("Work in progress", data=False, year=2018, ax=ax, fontsize=fontsize, rlabel="13 TeV")


# ----------------------------------------------------------------------------- the method

def abcd_cartoon():
    """One plane split into A, B, C, D by the two boundaries, with the estimate and kappa (schematic)."""
    fig, ax = plt.subplots(figsize=(14, 8))
    ax.set_axis_off()
    ax.set_xlim(-1.2, 7.6)
    ax.set_ylim(-1.0, 4.4)
    half = 1.6
    cells = {"A": (0.0, half, "passes both"), "B": (half, half, "fails isolation"),
             "C": (0.0, 0.0, "fails |Δφ|"), "D": (half, 0.0, "fails both")}
    for r, (x, y, text) in cells.items():
        ax.add_patch(patches.Rectangle((x, y), half, half, facecolor="#dff0d8" if r == "A" else "white",
                                       edgecolor="black", lw=2))
        ax.text(x + half / 2, y + half * 0.62, r, ha="center", va="center", fontsize=TEXT + 14, fontweight="bold")
        ax.text(x + half / 2, y + half * 0.25, text, ha="center", va="center", fontsize=TEXT - 2)
    ax.annotate("", xy=(2 * half + 0.15, -0.35), xytext=(0, -0.35), arrowprops=dict(arrowstyle="-|>", lw=2))
    ax.text(half, -0.7, "x: muon-LJ isolation (boundary 0.25)", ha="center", va="center", fontsize=TEXT)
    ax.annotate("", xy=(-0.35, 2 * half + 0.15), xytext=(-0.35, 0), arrowprops=dict(arrowstyle="-|>", lw=2))
    ax.text(-0.75, half, "y: |Δφ| (boundary 2.4)", ha="center", va="center", rotation=90, fontsize=TEXT)
    ax.text(4.0, 2.7, "x, y independent:\nA / C = B / D", ha="left", va="center", fontsize=TEXT + 2)
    ax.text(4.0, 1.55, "estimate:  A = B · C / D", ha="left", va="center", fontsize=TEXT + 4,
            bbox=dict(boxstyle="round,pad=0.4", facecolor="white", edgecolor="tab:green", lw=2))
    ax.text(4.0, 0.45, "test where A is open:\nκ = A · D / (B · C) = 1 ?", ha="left", va="center", fontsize=TEXT + 2)
    ax.text(-1.15, 4.35, "ABCD in one plane", ha="left", va="top", fontsize=TEXT + 2)
    ax.text(7.55, 4.35, "schematic", ha="right", va="top", fontsize=TEXT, style="italic", color="gray")
    tp.save(fig, "nominal_abcd_cartoon")
    return fig


# ----------------------------------------------------------------------------- the plane

def plane_figure(h2d, cfg, title, name):
    """The muon-LJ isolation x |dphi| distribution of one plane (bins drawn with equal width, the
    stored edges as tick labels), the region boundaries, and each region's label with its count;
    hidden cells hatched grey."""
    e0, ed, sumw, _ = h2d
    regions = nl.hist2d_regions(h2d, cfg)
    fig, ax = plt.subplots(figsize=(14, 10))
    nx, ny = sumw.shape
    hidden = np.isnan(sumw)
    shown = np.where(hidden | (sumw <= 0), np.nan, sumw)
    vmax = np.nanmax(shown) if np.isfinite(shown).any() else 1.0
    mesh = ax.pcolormesh(np.arange(nx + 1), np.arange(ny + 1), shown.T, cmap="viridis",
                         norm=colors.LogNorm(vmin=0.8, vmax=max(vmax, 1.0)))
    for i, j in zip(*np.nonzero(hidden)):
        ax.add_patch(patches.Rectangle((i, j), 1, 1, lw=0.5, **HIDDEN))
    cbar = fig.colorbar(mesh, ax=ax, pad=0.02)
    cbar.set_label("events per cell (white: none)", fontsize=AXIS)
    cbar.ax.yaxis.set_major_locator(ticker.FixedLocator([t for t in (1, 2, 5, 10, 20, 50, 100, 200, 500, 1000)
                                                          if t <= max(vmax, 1.0)]))
    cbar.ax.yaxis.set_major_formatter(ticker.FormatStrFormatter("%g"))
    cbar.ax.yaxis.set_minor_formatter(ticker.NullFormatter())
    cbar.ax.tick_params(labelsize=TICK)
    i0, f0 = io._edge_index(e0, cfg.wp0), io._edge_index(e0, cfg.fail0)
    ip, fd = io._edge_index(ed, cfg.dphi_pass), io._edge_index(ed, cfg.dphi_fail)
    ax.axvline(i0, color="red", lw=3)
    ax.axhline(ip, color="red", lw=3)
    centres = {"A": (i0 / 2, (ip + ny) / 2), "B": ((f0 + nx) / 2, (ip + ny) / 2),
               "C": (i0 / 2, fd / 2), "D": ((f0 + nx) / 2, fd / 2)}
    for r, (x, y) in centres.items():
        n = regions[r]
        ax.text(x, y, f"{r}\n{'hidden' if np.isnan(n) else f'{n:.0f}'}", ha="center", va="center", fontsize=TEXT + 2,
                bbox=dict(boxstyle="round,pad=0.3", facecolor="white", edgecolor="red", alpha=0.92))
    xlabels = [f"{v:g}" for v in e0[:-1]] + ["∞"]
    ax.set_xticks(np.arange(nx + 1))
    ax.set_xticklabels(xlabels, fontsize=TICK - 2, rotation=45)
    ax.set_yticks(np.arange(ny + 1))
    ax.set_yticklabels([f"{v:.3g}" if v < 3.1 else "π" for v in ed], fontsize=TICK)
    ax.set_xlabel("muon-LJ isolation (bins drawn with equal width)", fontsize=AXIS - 2)
    ax.set_ylabel("|Δφ(LJ, LJ)|", fontsize=AXIS)
    ax.set_title(title, fontsize=TEXT, loc="left", pad=40)
    tp.label(ax)
    tp.save(fig, name)
    return fig


# ----------------------------------------------------------------------------- kappa

def _point(ax, x, v, m, p, color, marker, label, ms=10):
    if not np.isfinite(v):
        return
    if np.isinf(p):
        ax.errorbar([x], [v], yerr=[[m], [0]], fmt=marker, color=color, ms=ms, capsize=4, label=label)
        ax.annotate("", xy=(x, v + 0.5), xytext=(x, v), arrowprops=dict(arrowstyle="->", color=color, lw=1.5))
        return
    ax.errorbar([x], [v], yerr=[[m], [p]], fmt=marker, color=color, ms=ms, capsize=4, label=label)


def kappa_points_figure(points, labels, title, name, band=None, band_label=None, ylim=None, width=16):
    """kappa per region: ``points`` a list of dicts (``x`` the index into ``labels``, ``series``,
    ``value``, ``minus``, ``plus``, ``hidden``), the line at one, an optional band 1 ± ``band``;
    a hidden point is written as "hidden" at one."""
    fig, ax = plt.subplots(figsize=(width, 9))
    series = list(dict.fromkeys(p["series"] for p in points))
    offsets = dict(zip(series, np.linspace(-0.22, 0.22, len(series)) if len(series) > 1 else [0.0]))
    if band is not None:
        ax.axhspan(1 - band, 1 + band, color="tab:green", alpha=0.18, lw=0, label=band_label)
    ax.axhline(1, color="gray", lw=1.5, ls="--")
    seen = set()
    for p in points:
        color, marker = SERIES.get(p["series"], ("black", "o"))
        x = p["x"] + offsets[p["series"]]
        if p.get("hidden"):
            ax.text(x, 1.0, "hidden", ha="center", va="center", fontsize=TEXT - 4, color="0.35", rotation=90,
                    bbox=dict(boxstyle="round,pad=0.2", **HIDDEN))
            continue
        _point(ax, x, p["value"], p["minus"], p["plus"], color, marker, None if p["series"] in seen else p["series"])
        if np.isfinite(p["value"]):
            seen.add(p["series"])
    ax.set_xticks(np.arange(len(labels)))
    ax.set_xticklabels(labels, fontsize=TICK, rotation=30 if max(len(s) for s in labels) > 6 else 0,
                       ha="right" if max(len(s) for s in labels) > 6 else "center")
    ax.set_xlim(-0.6, len(labels) - 0.4)
    ax.tick_params(axis="y", labelsize=TICK)
    ax.set_ylabel("κ = A·D / (B·C)", fontsize=AXIS)
    if ylim is not None:
        ax.set_ylim(*ylim)
    ax.legend(title=title, loc="upper left", ncol=max(1, len(series) + (band is not None)), fontsize=TEXT - 2,
              **tp.LEGEND)
    tp.label(ax)
    tp.save(fig, name)
    return fig


def table_points(df, series=OURS):
    """Points and labels of a ``nominal_lib.kappa_table`` / ``systematic_table`` frame, in row order."""
    points = [{"x": i, "series": series, "value": r["kappa"], "minus": r["minus"], "plus": r["plus"],
               "hidden": bool(r["hidden"])} for i, (_, r) in enumerate(df.iterrows())]
    return points, list(df["region"])


def vr_points(table):
    """Points and labels of a ``nominal_lib.vr_table``: per region the slides' value, ours at his
    working point and ours at ours."""
    labels = list(dict.fromkeys(table["region"]))
    points = []
    for i, name in enumerate(labels):
        part = table[table["region"] == name].set_index("wp")
        q, e = part.loc["his", "quoted kappa"], part.loc["his", "quoted err"]
        if np.isfinite(q):
            points.append({"x": i, "series": SLIDES, "value": q, "minus": e, "plus": e, "hidden": False})
        for wp, series in (("his", HIS), ("ours", OURS)):
            r = part.loc[wp]
            points.append({"x": i, "series": series, "value": r["kappa"], "minus": r["minus"], "plus": r["plus"],
                           "hidden": bool(r["hidden"])})
    return points, labels


def ladder_figure(rows, name="nominal_kappa_ladder"):
    """kappa against the number of displaced LJs: ``rows`` (label, n displaced, (value, minus, plus))."""
    fig, ax = plt.subplots(figsize=(11, 8))
    ax.axhline(1, color="gray", lw=1.5, ls="--")
    by_n = {}
    for label, n, k in rows:
        by_n.setdefault(n, []).append((label, k))
    for n, items in by_n.items():
        xs = np.linspace(n - 0.12, n + 0.12, len(items)) if len(items) > 1 else [n]
        for x, (label, k) in zip(xs, items):
            _point(ax, x, *k, "black", "o", None)
            ax.text(x + 0.05, k[0], label, ha="left", va="bottom", fontsize=TEXT - 4)
    ax.set_xticks(sorted(by_n))
    ax.set_xticklabels([str(n) for n in sorted(by_n)], fontsize=TICK)
    ax.set_xlim(-0.5, max(by_n) + 0.7)
    ax.set_ylim(0, 2.4)
    ax.tick_params(axis="y", labelsize=TICK)
    ax.set_xlabel("number of displaced LJs in the plane", fontsize=AXIS)
    ax.set_ylabel("κ = A·D / (B·C)", fontsize=AXIS)
    tp.label(ax, fontsize=20)
    tp.save(fig, name)
    return fig


def kappa_mass_figure(rows, windows, name="nominal_kappa_mass"):
    """kappa per mass window and plane: ``rows`` {plane: [(value, minus, plus) per window]}."""
    fig, ax = plt.subplots(figsize=(11, 8))
    ax.axhline(1, color="gray", lw=1.5, ls="--")
    offsets = dict(zip(rows, np.linspace(-0.1, 0.1, len(rows))))
    for plane, ks in rows.items():
        for i, k in enumerate(ks):
            _point(ax, i + offsets[plane], *k, tp.COLORS[plane], "o", plane if i == 0 else None)
    ax.set_xticks(range(len(windows)))
    ax.set_xticklabels(windows, fontsize=TICK)
    ax.set_xlim(-0.5, len(windows) - 0.5)
    ax.set_ylim(0.4, 1.9)
    ax.tick_params(axis="y", labelsize=TICK)
    ax.set_xlabel("M(LJ, LJ) window [GeV]", fontsize=AXIS)
    ax.set_ylabel("κ = A·D / (B·C)", fontsize=AXIS)
    ax.legend(loc="upper right", fontsize=TEXT, **tp.LEGEND)
    tp.label(ax, fontsize=20)
    tp.save(fig, name)
    return fig


# ----------------------------------------------------------------------------- composition

def photon_share_figure(table, name="nominal_photon_share"):
    """Photon-only share of the egamma LJs per region of PD and DD (``nominal_lib.class_share_table``
    with the splits ("e", "g")), with binomial errors; a hidden region is a hatched bar."""
    fig, ax = plt.subplots(figsize=(12, 8))
    planes = list(dict.fromkeys(table["plane"]))
    width = 0.8 / len(planes)
    for k, plane in enumerate(planes):
        part = table[table["plane"] == plane].reset_index(drop=True)
        labelled = False
        for i, r in part.iterrows():
            x = i + (k - (len(planes) - 1) / 2) * width
            if r["hidden"]:
                ax.bar(x, 1.0, width * 0.9, **HIDDEN)
                ax.text(x, 0.5, "hidden", ha="center", va="center", rotation=90, fontsize=TEXT - 2)
                continue
            share = r["g share"]
            err = np.sqrt(share * (1 - share) / r["total"]) if r["total"] > 0 else 0.0
            ax.bar(x, share, width * 0.9, color=tp.COLORS[plane], alpha=0.75, label=None if labelled else plane)
            labelled = True
            ax.errorbar([x], [share], yerr=[err], fmt="none", ecolor="black", capsize=4)
            ax.text(x, share + err + 0.02, f"{100 * share:.0f}%\n({r['total']:.0f})", ha="center", va="bottom",
                    fontsize=TEXT - 6)
    ax.set_xticks(range(4))
    ax.set_xticklabels(list("ABCD"), fontsize=TICK + 2)
    ax.set_ylim(0, 1.25)
    ax.tick_params(axis="y", labelsize=TICK)
    ax.set_xlabel("region (in brackets: events)", fontsize=AXIS)
    ax.set_ylabel("photon-only share of egamma LJs", fontsize=AXIS)
    ax.legend(loc="upper right", ncol=2, fontsize=TEXT, **tp.LEGEND)
    tp.label(ax, fontsize=20)
    tp.save(fig, name)
    return fig


# ----------------------------------------------------------------------------- signal

def _point_key(name):
    """Sort key of a signal point name: mediator mass, Zd mass, ctau as numbers."""
    return tuple(float(v.replace("p", ".")) for v in re.findall(r"(\d+(?:p\d+)?)(?:GeV|mm)", name))


def signal_share_figure(shares, name="nominal_signal_share"):
    """Share of each signal point's search-region yield per LJ-pair mass bin, one horizontal stacked
    bar per point (``nominal_lib.signal_share_table`` rows, ordered by mediator mass, drawn top to bottom)."""
    order = sorted(range(len(shares)), key=lambda i: _point_key(shares["point"].iloc[i]))
    shares = shares.iloc[order]
    bins = [label for label, _ in nl.SHARE_BINS]
    cmap = plt.get_cmap("viridis", len(bins))
    n = len(shares)
    fig, ax = plt.subplots(figsize=(16, 2.0 + 0.5 * n))
    y = np.arange(n)[::-1]
    left = np.zeros(n)
    for i, b in enumerate(bins):
        v = shares[b].to_numpy(float)
        ax.barh(y, v, left=left, color=cmap(i), edgecolor="white", label=f"{b} GeV".replace(">=", "≥ "))
        left += v
    ax.set_yticks(y)
    ax.set_yticklabels([f"{bp.point_label(p)}  [{w:.2f}]" for p, w in zip(shares["point"], shares["yield"])],
                       fontsize=TICK - 2)
    ax.tick_params(axis="y", which="both", left=False, right=False)
    ax.set_xlim(0, 1)
    ax.set_ylim(-0.6, n + 1.6)
    ax.tick_params(axis="x", labelsize=TICK)
    ax.set_xlabel("share of the search-region signal yield (A of DD, M ≥ 150 GeV)", fontsize=AXIS - 2)
    ax.legend(title="M(LJ, LJ) bin; in brackets: yield at the stored cross section", loc="upper center",
              ncol=len(bins), fontsize=TEXT - 4, bbox_to_anchor=(0.5, 1.0), **tp.LEGEND)
    _sim_label(ax, fontsize=20)
    tp.save(fig, name)
    return fig
