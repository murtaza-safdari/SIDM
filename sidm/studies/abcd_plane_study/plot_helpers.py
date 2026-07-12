"""Shared plotting for the ABCD plane-choice notebooks (mplhep CMS style).

Every figure convention lives here so the four notebooks stay readable and
consistent: axis labels/ranges, process colors, the 2D plane view with the
A/B/C/D regions drawn, stacked 1D comparisons, the shape-in-slices independence
view, closure ladders, p-value matrices, factorization pull maps, and per-point
bar charts. All functions draw a complete labeled figure and call plt.show().
"""
import re

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import mplhep as hep
from mplhep.utils import append_axes

hep.style.use("CMS")
plt.rcParams["figure.figsize"] = (9, 7)
plt.rcParams["savefig.bbox"] = "tight"

LUMI_EVEN = r"29.9 fb$^{-1}$ (13 TeV, 2018), even half"
LUMI_FULL = r"59.8 fb$^{-1}$ (13 TeV, 2018)"


def cms_label(ax=None, rlabel=LUMI_EVEN):
    # fontsize is deliberately reduced: at the default CMS-label size the left
    # text and the rlabel overlap in the header, worst on axes narrowed by a
    # colorbar ("2018 sim." would be redundant next to the Simulation tag)
    hep.cms.label("Work in progress", data=False, rlabel=rlabel, ax=ax, fontsize=13)


AXIS_LABELS = {
    "muiso": "mu-LJ isolation", "egmiso": "egm-LJ isolation",
    "muiso0": "leading mu-LJ isolation", "muiso1": "subleading mu-LJ isolation",
    "dphi": r"$|\Delta\phi(\mathrm{LJ}_0,\mathrm{LJ}_1)|$ [rad]",
    "mjj": r"$m_{JJ}$ [GeV]",
    "mupix": "mu-LJ max PF-mu pixel hits", "egmlost": "egm-LJ min lost hits",
    "mupix0": "leading mu-LJ max pixel hits", "mupix1": "subleading mu-LJ max pixel hits",
}
AXIS_RANGES = {
    "muiso": (-0.02, 0.7), "egmiso": (-0.02, 0.7),
    "muiso0": (-0.02, 0.7), "muiso1": (-0.02, 0.7),
    "dphi": (0, np.pi), "mjj": (0, 1000),
}
PROC_COLORS = {"QCD": "#5790fc", "DY": "#f89c20", "TTJets": "#e42536",
               "Diboson": "#964a8b"}
MASS_COLORS = {100: "#7a21dd", 500: "#e42536", 1000: "#2ca02c"}

# human-readable plane names for figure text (raw registry identifiers in a
# rendered figure are a conventions/plotting.md finding)
PLANE_NICE = {
    "P1_iso_iso": "iso(mu)$\\times$iso(egm) [P1]",
    "P2_muiso_dphi": "muiso$\\times|\\Delta\\phi|$ [P2]",
    "P3_egmiso_dphi": "egmiso$\\times|\\Delta\\phi|$ [P3]",
    "P4_muiso_mjj": "muiso$\\times m_{JJ}$ [P4]",
    "P5_muiso_mupix": "muiso$\\times$pixel hits [P5]",
    "P6_mupix_dphi": "pixel hits$\\times|\\Delta\\phi|$ [P6]",
    "P7_egmlost_dphi": "lost hits$\\times|\\Delta\\phi|$ [P7]",
    "P8_dphi_mjj": "$|\\Delta\\phi|\\times m_{JJ}$ [P8]",
    "P9_egmiso_mjj": "egmiso$\\times m_{JJ}$ [P9]",
    "Q1_iso_iso": "iso$_0\\times$iso$_1$ [Q1]",
    "Q2_iso0_dphi": "muiso$\\times|\\Delta\\phi|$ [Q2]",
    "Q3_iso0_pix0": "muiso$\\times$pixel hits [Q3]",
    "Q4_pix_pix": "pixel$\\times$pixel hits [Q4]",
    "Q5_dphi_mjj": "$|\\Delta\\phi|\\times m_{JJ}$ [Q5]",
    "Q6_iso0_mjj": "muiso$\\times m_{JJ}$ [Q6]",
    "D1_ntight_dphi": "N$_{tight}\\times|\\Delta\\phi|$ [D1]",
    "D2_ntight_mjj": "N$_{tight}\\times m_{JJ}$ [D2]",
    "D3_jetmatch_dphi": "N$_{jet\\,match}\\times|\\Delta\\phi|$ [D3]",
    "D4_photononly_dphi": "photon-only$\\times|\\Delta\\phi|$ [D4]",
    "D5_ntight_dphi": "N$_{tight}\\times|\\Delta\\phi|$ [D5]",
}


def nice(key):
    """Humanize a 'channel/PLANE' or 'PLANE' registry key for figure text."""
    return "/".join(PLANE_NICE.get(p, p) for p in str(key).split("/"))


def parse_sig(name):
    """'2Mu2E_500GeV_1p2GeV_19p0mm' -> (500.0, 1.2, 19.0)."""
    m = re.match(r".+?_([\d.p]+)GeV_([\d.p]+)GeV_([\d.p]+)mm", name)
    return tuple(float(x.replace("p", ".")) for x in m.groups())


def sig_label(name):
    mB, mDp, ctau = parse_sig(name)
    return (rf"$m_{{B}}$={mB:g}, $m_{{Z_d}}$={mDp:g} GeV, "
            rf"$c\tau$={ctau:g} mm")


def _region_positions(xspec, yspec):
    """Axes-fraction positions of the A/B/C/D labels given pass directions.

    A = pass-x & pass-y, B = fail-x & pass-y, C = pass-x & fail-y, D = fail-both;
    'lt' passes below the cut, 'ge' passes above it.
    """
    px = 0.18 if xspec[0] == "lt" else 0.85
    py = 0.15 if yspec[0] == "lt" else 0.88
    fx, fy = 1.03 - px, 1.03 - py
    return {"A": (px, py), "B": (fx, py), "C": (px, fy), "D": (fx, fy)}


def plot_plane(vals, xe, ye, note, xspec=None, yspec=None, xname="", yname="",
               regions=True, rlabel=LUMI_EVEN):
    """2D plane (log color) with the ABCD boundaries and region letters drawn."""
    fig, ax = plt.subplots()
    masked = np.ma.masked_where(vals.T <= 0, vals.T)
    if masked.count() == 0:
        plt.close(fig)
        print(f"[{note}] empty after cuts - not plotted")
        return
    m = ax.pcolormesh(xe, ye, masked, norm=mcolors.LogNorm(vmin=0.5 * float(masked.min())),
                      cmap="viridis")
    cax = append_axes(ax, extend=True)   # conventions/plotting.md colorbar pattern
    fig.colorbar(m, cax=cax, label="events / bin")
    if xspec is not None:
        ax.axvline(xspec[1], color="red", ls="--", lw=1.8)
    if yspec is not None:
        ax.axhline(yspec[1], color="red", ls="--", lw=1.8)
    if regions and xspec is not None and yspec is not None:
        for lab, (fx, fy) in _region_positions(xspec, yspec).items():
            ax.text(fx, fy, lab, transform=ax.transAxes, fontsize=22, color="red",
                    ha="center", va="center", fontweight="bold")
    ax.set_xlabel(AXIS_LABELS.get(xname, xname))
    ax.set_ylabel(AXIS_LABELS.get(yname, yname))
    if xname in AXIS_RANGES:
        ax.set_xlim(*AXIS_RANGES[xname])
    if yname in AXIS_RANGES:
        ax.set_ylim(*AXIS_RANGES[yname])
    ax.text(0.5, 0.975, note, transform=ax.transAxes, ha="center", va="top",
            fontsize=12, bbox=dict(facecolor="white", alpha=0.75, edgecolor="none"))
    cms_label(ax, rlabel=rlabel)
    plt.show()


def stack1d(proc_entries, edges, xname="", signal_entries=(), log=True,
            rlabel=LUMI_EVEN, note="", ylabel="events / bin", xlim=None,
            sentinel=True):
    """Stacked per-process 1D distribution with optional signal overlays.

    proc_entries: [(process_name, values_array)] bottom-up; the isolation sentinel
    bin (edges[0] < 0) is shaded and labeled 'no matched jet'.
    """
    fig, ax = plt.subplots()
    bottom = np.zeros(len(edges) - 1)
    for pname, v in proc_entries:
        ax.stairs(bottom + v, edges, baseline=bottom, fill=True,
                  color=PROC_COLORS.get(pname), label=pname)
        bottom = bottom + v
    for sname, v, scale in signal_entries:
        lab = sig_label(sname) if "_" in sname else sname
        if scale != 1:
            lab += f"  ($\\times${scale:g})"
        mB = parse_sig(sname)[0] if "_" in sname else None
        ax.stairs(v * scale, edges, color=MASS_COLORS.get(mB, "k"), lw=2.2, label=lab)
    if log:
        ax.set_yscale("log")
        lo = bottom[bottom > 0]
        if lo.size:
            ax.set_ylim(max(lo.min() * 0.2, 1e-5), bottom.max() * 3e2)
    if sentinel and edges[0] < 0:
        import matplotlib.transforms as mtransforms
        tr = mtransforms.blended_transform_factory(ax.transData, ax.transAxes)
        ax.axvspan(edges[0], edges[1], color="gray", alpha=0.25, zorder=0)
        ax.text(0.5 * (edges[0] + edges[1]), 0.97, "no\njet", transform=tr,
                fontsize=9, ha="center", va="top", color="dimgray")
    ax.set_xlabel(AXIS_LABELS.get(xname, xname))
    ax.set_ylabel(ylabel)
    if xname in AXIS_RANGES:
        ax.set_xlim(*AXIS_RANGES[xname])
    if xlim is not None:
        ax.set_xlim(*xlim)
    ax.legend(title=note, fontsize=11, title_fontsize=12, framealpha=0.85)
    cms_label(ax, rlabel=rlabel)
    plt.show()


def shape_slices(vals, xe, slices_vals, slice_labels, xname="", note="",
                 rlabel=LUMI_EVEN, var=None):
    """Independence view: normalized x-shape in slices of the other variable,
    with a ratio panel to the inclusive shape. Flat ratios = the ABCD assumption.
    var: per-bin variances of the inclusive shape -> MC-stat band on the ratio."""
    fig, (ax, axr) = plt.subplots(2, 1, sharex=True, figsize=(9, 8.5),
                                  height_ratios=[3, 1], gridspec_kw={"hspace": 0.06})
    tot_incl = max(vals.sum(), 1e-300)
    incl = vals / tot_incl
    ax.stairs(incl, xe, color="k", lw=2.5, label="inclusive")
    axr.axhline(1, color="k", lw=1)
    if var is not None:
        with np.errstate(divide="ignore", invalid="ignore"):
            rel = np.where(vals > 0, np.sqrt(var) / np.maximum(vals, 1e-300), 0.0)
        axr.stairs(1 + rel, xe, baseline=1 - rel, fill=True, color="gray",
                   alpha=0.3, zorder=0, label="MC stat (inclusive)")
    for sv, lab, color in zip(slices_vals, slice_labels,
                              [f"C{i}" for i in range(len(slices_vals))]):
        tot = sv.sum()
        if tot <= 0:
            continue
        n = sv / tot
        ax.stairs(n, xe, color=color, lw=1.8, label=lab)
        with np.errstate(divide="ignore", invalid="ignore"):
            axr.stairs(np.where(incl > 0, n / incl, np.nan), xe, color=color, lw=1.8)
    ax.set_yscale("log")
    ax.set_ylabel("shape (a.u.)")
    ax.legend(title=note, fontsize=11, title_fontsize=12, framealpha=0.85)
    axr.set_ylim(0, 4)
    axr.set_ylabel("ratio to incl.")
    axr.set_xlabel(AXIS_LABELS.get(xname, xname))
    if xname in AXIS_RANGES:
        axr.set_xlim(*AXIS_RANGES[xname])
    cms_label(ax, rlabel=rlabel)
    plt.show()


def ladder_plot(pts_by_presc, note, band=0.25, rlabel=LUMI_EVEN):
    """Closure R along the staged loose->tight ladder, one series per prescription.
    Shaded band = the declared max(2sigma, 0.25) acceptance around 1 (drawn at 0.25;
    the per-point 2sigma criterion is what the gate applies)."""
    fig, ax = plt.subplots(figsize=(10, 6))
    n = None
    anchor_labeled = False
    for k, (presc, pts) in enumerate(pts_by_presc.items()):
        xs = np.arange(len(pts)) + 0.12 * k
        ok = np.array([p["neff"] >= 10 for p in pts])
        Rs = np.array([p["R"] for p in pts])
        errs = np.array([p["err"] for p in pts])
        if (~ok).any():
            ax.errorbar(xs[~ok], Rs[~ok], errs[~ok], marker="o", ls="",
                        color="lightgray", capsize=2,
                        label="min n_eff(B,C,D) < 10" if k == 0 else None)
        if ok.any():
            ax.errorbar(xs[ok], Rs[ok], errs[ok], marker="o", ls="", color=f"C{k}",
                        capsize=2, label=f"prescription ({presc}), healthy")
        healthy = np.where(ok)[0]
        if healthy.size:
            ax.scatter([xs[healthy[-1]]], [Rs[healthy[-1]]], marker="*", s=340,
                       facecolor="none", edgecolor=f"C{k}", lw=2, zorder=5,
                       label=None if anchor_labeled else "anchor")
            anchor_labeled = True
        n = pts if n is None else n
    ax.axhspan(1 - band, 1 + band, color="green", alpha=0.10, zorder=0,
               label=r"$\pm$25% (min. gate width; gate = max(2$\sigma$, 0.25))")
    ax.axhline(1, color="gray", lw=1)
    ax.set_xticks(np.arange(len(n)), [p["label"] for p in n], rotation=60,
                  ha="right", fontsize=9)
    ax.set_ylabel("R = A / (BC/D)")
    ax.set_ylim(-0.5, 3.5)
    ax.legend(title=note, loc="upper left", fontsize=11, title_fontsize=12,
              framealpha=0.85)
    cms_label(ax, rlabel=rlabel)
    plt.tight_layout()
    plt.show()


def pvalue_heatmap(pmat, row_labels, col_labels, note, rlabel=LUMI_EVEN):
    """Gate-1 matrix: green above p = 0.05, red below, gray not evaluated."""
    arr = np.array(pmat, float)
    fig, ax = plt.subplots(figsize=(1.6 + 1.15 * len(col_labels),
                                    1.8 + 0.55 * len(row_labels)))
    color = np.where(np.isnan(arr), 0.5, np.where(arr > 0.05, 1.0, 0.0))
    ax.pcolormesh(color, cmap=mcolors.ListedColormap(["#f2a2a2", "#bdbdbd", "#b6dbb6"]),
                  vmin=0, vmax=1, edgecolors="white", linewidth=1.5)
    for i in range(arr.shape[0]):
        for j in range(arr.shape[1]):
            txt = "-" if np.isnan(arr[i, j]) else f"{arr[i, j]:.3f}"
            ax.text(j + 0.5, i + 0.5, txt, ha="center", va="center", fontsize=11)
    ax.set_xticks(np.arange(len(col_labels)) + 0.5, col_labels, fontsize=11)
    ax.set_yticks(np.arange(len(row_labels)) + 0.5, row_labels, fontsize=11)
    ax.invert_yaxis()
    ax.set_title(note + "   (green: p > 0.05)", fontsize=13, pad=10)
    plt.tight_layout()
    plt.show()


def pull_map(fit, note, rlabel=LUMI_EVEN):
    """(observed - a_i b_j) / sqrt(var) on the adaptively rebinned grid."""
    v, w = fit["rebinned"]
    exp = fit["expected"]
    with np.errstate(divide="ignore", invalid="ignore"):
        pulls = np.where(w > 0, (v - exp) / np.sqrt(w), np.nan)
    fig, ax = plt.subplots(figsize=(7.5, 6.5))
    m = ax.pcolormesh(pulls.T, cmap="RdBu_r", vmin=-3, vmax=3, edgecolors="white",
                      linewidth=1)
    cax = append_axes(ax, extend=True)
    fig.colorbar(m, cax=cax, label=r"(obs $-$ fit) / $\sigma$")
    for i in range(pulls.shape[0]):
        for j in range(pulls.shape[1]):
            if np.isfinite(pulls[i, j]):
                ax.text(i + 0.5, j + 0.5, f"{pulls[i, j]:.1f}", ha="center",
                        va="center", fontsize=10)
    ax.set_xlabel(r"x super-bin (loose $\to$ tight)")
    ax.set_ylabel("y super-bin")
    ax.set_title(f"{note}   $\\chi^2$/ndf = {fit['chi2']:.1f}/{fit['ndf']}, "
                 f"p = {fit['pvalue']:.3f}", fontsize=13, pad=10)
    plt.tight_layout()
    plt.show()


def bars_by_point(names, series, ylabel, note, log=False, hline=None,
                  rlabel=LUMI_EVEN):
    """Grouped bars per signal point. series: {label: values (len == names)}."""
    fig, ax = plt.subplots(figsize=(11, 6))
    x = np.arange(len(names))
    nb = len(series)
    width = 0.8 / max(nb, 1)
    for k, (lab, v) in enumerate(series.items()):
        vv = np.array(v, float)
        ax.bar(x + (k - (nb - 1) / 2) * width, np.where(np.isfinite(vv), vv, 0),
               width, label=lab)
    if hline is not None:
        ax.axhline(hline, color="gray", ls="--", lw=1)
    if log:
        ax.set_yscale("log")
    ax.set_xticks(x, [sig_label(n) if "_" in n else n for n in names], rotation=70,
                  ha="right", fontsize=8)
    ax.set_ylabel(ylabel)
    ax.legend(title=note, fontsize=11, title_fontsize=12, framealpha=0.85)
    cms_label(ax, rlabel=rlabel)
    plt.tight_layout()
    plt.show()


def matrix_heatmap(arr, row_labels, col_labels, note, cmap="RdBu_r", vmin=-0.5,
                   vmax=0.5, fmt="{:.3f}"):
    """Annotated matrix (e.g. weighted correlations per plane x process)."""
    arr = np.array(arr, float)
    fig, ax = plt.subplots(figsize=(1.8 + 1.2 * len(col_labels),
                                    1.8 + 0.55 * len(row_labels)))
    cmap_obj = plt.get_cmap(cmap).copy()
    cmap_obj.set_bad("#bdbdbd")
    m = ax.pcolormesh(np.ma.masked_invalid(arr), cmap=cmap_obj, vmin=vmin,
                      vmax=vmax, edgecolors="white", linewidth=1.5)
    cax = append_axes(ax, extend=True)
    fig.colorbar(m, cax=cax)
    for i in range(arr.shape[0]):
        for j in range(arr.shape[1]):
            txt = "-" if not np.isfinite(arr[i, j]) else fmt.format(arr[i, j])
            ax.text(j + 0.5, i + 0.5, txt, ha="center", va="center", fontsize=11)
    ax.set_xticks(np.arange(len(col_labels)) + 0.5, col_labels, fontsize=11)
    ax.set_yticks(np.arange(len(row_labels)) + 0.5, row_labels, fontsize=11)
    ax.invert_yaxis()
    ax.set_title(note, fontsize=13, pad=10)
    plt.tight_layout()
    plt.show()


def eff_vs_ctau(points, ylabel, note, rlabel=LUMI_EVEN, ylog=True):
    """points: {signal_name: value}. One line per bound-state mass vs ctau."""
    fig, ax = plt.subplots()
    series = {}
    for n, v in points.items():
        mB, mDp, ctau = parse_sig(n)
        series.setdefault((mB, mDp), []).append((ctau, v))
    for (mB, mDp), pts in sorted(series.items()):
        pts.sort()
        xs, ys = zip(*pts)
        ax.plot(xs, ys, marker="o", lw=2, color=MASS_COLORS.get(mB),
                ls={0.25: ":", 1.2: "-", 5.0: "--"}.get(mDp, "-"),
                label=rf"$m_B$={mB:g}, $m_{{Z_d}}$={mDp:g} GeV")
    ax.set_xscale("log")
    if ylog:
        ax.set_yscale("log")
    ax.set_xlabel(r"dark-photon $c\tau$ [mm]")
    ax.set_ylabel(ylabel)
    ax.legend(title=note, fontsize=11, title_fontsize=12, framealpha=0.85)
    cms_label(ax, rlabel=rlabel)
    plt.show()
