"""Figures of the plane search: kappa against the |dphi| boundaries, the open ratios of each
plane against the search plane, the factors against the LJ-pair mass, and the factors against
the muon LJ's pixel-hit class."""
import dataclasses

import matplotlib.pyplot as plt
import numpy as np

import plane_search_lib as ps
import transfer_lib as tl
import transfer_plots as tp

COLORS = {"PP": "black", "DP": "tab:red", "PD": "tab:blue", "DD": "tab:green", "PD:g": "tab:purple",
          "PD:e": "tab:orange", "DP:mu1-2": "tab:brown"}
TICK = 18
AXIS = 22


def _errorbar(ax, x, r, color, label, marker="o", ms=8):
    """One (value, minus, plus) point; an infinite upward error is drawn as an arrow."""
    v, m, p = r
    if not np.isfinite(v):
        return
    if np.isinf(p):
        ax.errorbar([x], [v], yerr=[[m], [0]], fmt=marker, color=color, ms=ms, capsize=3, label=label)
        ax.annotate("", xy=(x, v * 3), xytext=(x, v), arrowprops=dict(arrowstyle="->", color=color, lw=1.5))
        return
    ax.errorbar([x], [v], yerr=[[m], [p]], fmt=marker, color=color, ms=ms, capsize=3, label=label)


def _axes(ax, ylabel, xticks, xlabels, log=True, rotation=0):
    ax.set_xticks(xticks)
    ax.set_xticklabels(xlabels, rotation=rotation, ha="right" if rotation else "center", fontsize=TICK)
    ax.tick_params(axis="y", labelsize=TICK)
    ax.set_ylabel(ylabel, fontsize=AXIS)
    if log:
        ax.set_yscale("log")


def boundary_figure(cube, planes=("PP", "PD", "PD:g"), passes=(2.0, 2.2, 2.4, 2.6, 2.8), cfg=ps.Config()):
    """kappa of each validation plane against the |dphi| pass boundary, with the fail boundary
    fixed at 2.0 (left) and moving with the pass boundary (right, no buffer)."""
    t = cube.topology
    fig, axes = plt.subplots(1, 2, figsize=(18, 8))
    offsets = np.linspace(-0.03, 0.03, len(planes))
    for ax, fixed, title in zip(axes, (True, False), ("fail: |dphi| < 2.0", "fail: |dphi| below the pass boundary")):
        for plane, off in zip(planes, offsets):
            first = True
            for dp in passes:
                c = dataclasses.replace(cfg, dphi_pass=dp, dphi_fail=cfg.dphi_fail if fixed else dp)
                _errorbar(ax, dp + off, ps.kappa(ps.regions(cube, c, plane)), COLORS[plane], plane if first else None)
                first = False
        ax.axhline(1, color="gray", lw=1)
        _axes(ax, "kappa = A D / (B C)", list(passes), [f"{p:g}" for p in passes], log=False)
        ax.set_xlabel("|dphi| pass boundary", fontsize=AXIS)
        ax.set_ylim(0, 2.0)
        ax.legend(title=f"{t}, {cfg.kind} plane, M >= {cfg.mass[0]:g} GeV\n{title}", loc="upper left", ncol=len(planes),
                  columnspacing=0.8, handletextpad=0.3, **tp.LEGEND)
        tp.label(ax)
    fig.tight_layout()
    tp.save(fig, f"plane_search_boundaries_{t}")
    return fig


def compatibility_figure(cube, planes=("PP", "DP", "PD", "PD:g", "PD:e", "DP:mu1-2"), cfg=ps.Config(), search="DD"):
    """The open ratios B/D and C/D of each validation plane next to the search plane's."""
    t = cube.topology
    fig, axes = plt.subplots(1, 2, figsize=(18, 8))
    dd = ps.regions(cube, cfg, search)
    x = np.arange(len(planes) + 1)
    for ax, (num, den) in zip(axes, (("B", "D"), ("C", "D"))):
        ref = tl.ratio(dd[num], dd[den])
        if np.isfinite(ref[0]):
            ax.axhspan(ref[0] - ref[1], ref[0] + ref[2], color=COLORS[search], alpha=0.18, lw=0)
            ax.axhline(ref[0], color=COLORS[search], lw=1.5, ls="--")
        for i, plane in enumerate(planes):
            reg = ps.regions(cube, cfg, plane)
            _errorbar(ax, i, tl.ratio(reg[num], reg[den]), COLORS.get(plane, "gray"), None)
        _errorbar(ax, len(planes), ref, COLORS[search], None, marker="s", ms=10)
        _axes(ax, f"{num}/{den}", x, list(planes) + [f"{search} (search plane)"], log=False, rotation=30)
        ax.set_xlim(-0.6, len(planes) + 0.6)
        ax.set_ylim(0, ax.get_ylim()[1] * 1.6)
        what = ("|dphi| pass/fail ratio among isolation failures" if (num, den) == ("B", "D")
                else "isolation pass/fail ratio among |dphi| failures")
        ax.legend(handles=[plt.Line2D([], [], color=COLORS[search], ls="--", lw=1.5, label=f"{search}, with its 68% band")],
                  title=f"{t}, {cfg.kind} plane, M >= {cfg.mass[0]:g} GeV\n{what}", loc="upper left", **tp.LEGEND)
        tp.label(ax)
    fig.tight_layout()
    tp.save(fig, f"plane_search_compatibility_{t}")
    return fig


MASS_POINTS = [("<150", (None, 150.0))] + ps.MASS_BINS


def mass_figure(cube, planes=("PP", "PD", "DD"), cfg=ps.Config(), bins=MASS_POINTS):
    """kappa, A/B, B/D and C/D of the planes against the LJ-pair mass bin.  In the search plane
    only the open ratios exist above 150 GeV; its kappa and A/B are drawn where A is open."""
    t = cube.topology
    fig, axes = plt.subplots(2, 2, figsize=(18, 14))
    x = np.arange(len(bins))
    offsets = np.linspace(-0.2, 0.2, len(planes))
    for ax, (name, num, den) in zip(axes.flat, (("kappa", None, None), ("A/B", "A", "B"), ("B/D", "B", "D"), ("C/D", "C", "D"))):
        for plane, off in zip(planes, offsets):
            first = True
            for i, (label, mass) in enumerate(bins):
                reg = ps.regions(cube, dataclasses.replace(cfg, mass=mass), plane)
                r = ps.kappa(reg) if num is None else tl.ratio(reg[num], reg[den])
                _errorbar(ax, i + off, r, COLORS[plane], plane if first else None)
                first = first and not np.isfinite(r[0])
        ax.axvline(0.5, color="gray", lw=1, ls="--")
        _axes(ax, "kappa = A D / (B C)" if num is None else f"{num}/{den}", x, [b[0] for b in bins], rotation=30)
        ax.set_xlabel("M(LJ, LJ) [GeV]", fontsize=AXIS)
        ax.set_xlim(-0.5, len(bins) - 0.5)
        # headroom for the legend above the highest error bar
        if num is None:
            ax.axhline(1, color="gray", lw=1)
            ax.set_ylim(0.1, 40)
        else:
            lo, hi = ax.get_ylim()
            ax.set_ylim(lo / 2, hi * 15)
        ax.legend(title=f"{t}, {cfg.kind} plane; dashed line: 150 GeV\nDD: kappa and A/B hidden above it",
                  loc="upper left", ncol=len(planes),
                  columnspacing=0.8, handletextpad=0.3, **tp.LEGEND)
        tp.label(ax, fontsize=18)  # four panels: the full-size label overlaps the luminosity text
    fig.tight_layout()
    tp.save(fig, f"plane_search_mass_{t}")
    return fig


PIXEL_CLASSES = (("0", "mu0"), ("1", "mu1"), ("2", "mu2"), ("3", "mu3"), ("4", "mu4"), (">=5", "mu5+"))


def pixel_class_figure(cube, cfg=ps.Config()):
    """A/B and kappa against the muon LJ's pixel-hit class for three populations: egamma LJ
    prompt at M >= 150 GeV (DP for 0-2 hits, PP for >= 3), egamma LJ displaced at M < 150 GeV
    (DD, PD) and egamma LJ displaced at M >= 150 GeV (PD only: the DD classes are the search
    region).  The dashed line separates the displaced (<= 2 hits) from the prompt classes."""
    t = cube.topology
    low = dataclasses.replace(cfg, mass=(None, 150.0))
    series = [("egamma LJ prompt, M >= 150 GeV", cfg, "DP", "PP", "tab:red", "o"),
              ("egamma LJ displaced, M < 150 GeV", low, "DD", "PD", "tab:green", "s"),
              ("egamma LJ displaced, M >= 150 GeV", cfg, "DD", "PD", "tab:blue", "D")]
    fig, axes = plt.subplots(1, 2, figsize=(18, 8))
    x = np.arange(len(PIXEL_CLASSES))
    offsets = np.linspace(-0.2, 0.2, len(series))
    for ax, which in zip(axes, ("A/B", "kappa")):
        for (name, c, disp, prompt, color, marker), off in zip(series, offsets):
            first = True
            for i, (label, split) in enumerate(PIXEL_CLASSES):
                plane = f"{disp}:{split}" if i < 3 else f"{prompt}:{split}"
                reg = ps.regions(cube, c, plane)
                r = tl.ratio(reg["A"], reg["B"]) if which == "A/B" else ps.kappa(reg)
                _errorbar(ax, i + off, r, color, name if first else None, marker=marker)
                first = first and not np.isfinite(r[0])
        ax.axvline(2.5, color="gray", lw=1, ls="--")
        ax.axvspan(-0.5, 2.5, color="tab:blue", alpha=0.06, lw=0)
        _axes(ax, "A/B (isolation pass/fail among |dphi| passes)" if which == "A/B" else "kappa = A D / (B C)",
              x, [c[0] for c in PIXEL_CLASSES])
        ax.set_xlabel("pixel hits of the muon LJ (max over its muons)", fontsize=AXIS)
        ax.set_xlim(-0.5, len(x) - 0.5)
        ax.set_ylim(0.05, 300)
        if which == "kappa":
            ax.axhline(1, color="gray", lw=1)
        ax.legend(title=f"{t}, {cfg.kind} plane; shaded: displaced muon LJ,\nwhere the blue series is the hidden search region",
                  loc="upper left", **tp.LEGEND)
        tp.label(ax)
    fig.tight_layout()
    tp.save(fig, f"plane_search_pixel_class_{t}")
    return fig
