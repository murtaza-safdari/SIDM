"""Figures of estimate.ipynb: spectra of a stored variable per ABCD region (data against the
simulated process groups) and the ABCD ratios against an object class, with the search plane's
open ratios as reference bands."""
import matplotlib.pyplot as plt
import numpy as np
from scipy import stats

import estimate_lib as el
import plane_search_lib as ps
import plane_search_plots as pp
import transfer_lib as tl
import transfer_plots as tp

GROUP_COLORS = {"DY": "tab:orange", "ttbar": "tab:green", "QCD": "tab:gray", "diboson": "tab:purple", "other": "tab:brown"}
GROUP_ORDER = ("QCD", "diboson", "ttbar", "DY")


def garwood(k, cl=0.6827):
    """Central Poisson interval of an observed count as (minus, plus)."""
    k = np.asarray(k, dtype=float)
    a = (1 - cl) / 2
    lo = np.where(k > 0, stats.chi2.ppf(a, 2 * k) / 2, 0.0)
    hi = stats.chi2.ppf(1 - a, 2 * k + 2) / 2
    return k - lo, hi - k


def wrapped_label(cfg):
    """The configuration label broken before the second isolation cut, for legend headers."""
    return cfg.label().replace(", iso1", ",\niso1")


def _centers(edges):
    e = np.asarray(edges, dtype=float)
    return 0.5 * (e[1:] + e[:-1])


def spectrum_figure(data, groups, variable, cfg, panels, xlabel, name, ncols=3, log=True, xlim=None):
    """Data counts (Garwood intervals) over the stacked simulated groups, one panel per
    ``(title, plane, region)`` in ``panels``.  A region that touches the hidden box is left empty
    and says so.  ``groups`` maps a process group to its cube of the data's topology, or to
    ``{topology: cube}`` as :func:`transfer_lib.group_cubes` returns it."""
    n = len(panels)
    ncols = min(ncols, n)
    nrows = int(np.ceil(n / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(6.5 * ncols, 6.5 * nrows), squeeze=False)
    for ax in axes.flat[n:]:
        ax.set_visible(False)
    for ax, (title, plane, region) in zip(axes.flat, panels):
        edges, sumw, sumw2 = el.spectrum(data, variable, cfg, plane, region)
        x = _centers(edges)
        width = np.diff(edges)
        base = np.zeros(len(x))
        for g in GROUP_ORDER:
            if g not in groups:
                continue
            cube = groups[g][data.topology] if isinstance(groups[g], dict) else groups[g]
            _, mc, _ = el.spectrum(cube, variable, cfg, plane, region)
            ax.stairs(base + mc, edges, baseline=base, fill=True, color=GROUP_COLORS[g], alpha=0.75, label=g)
            base = base + mc
        if np.isnan(sumw).any():
            ax.text(0.5, 0.5, "hidden", transform=ax.transAxes, ha="center", va="center", fontsize=pp.AXIS, color="tab:red")
        else:
            minus, plus = garwood(sumw)
            shown = sumw > 0                                   # empty bins are not drawn
            ax.errorbar(x[shown], sumw[shown], yerr=[minus[shown], plus[shown]], xerr=width[shown] / 2, fmt="o",
                        color="black", ms=6, capsize=0, label=f"data ({sumw.sum():.0f})")
        ax.set_xlabel(xlabel, fontsize=pp.AXIS)
        ax.set_ylabel("events / bin", fontsize=pp.AXIS)
        ax.tick_params(labelsize=pp.TICK)
        if log:
            ax.set_yscale("log")
            ax.set_ylim(0.05, max(ax.get_ylim()[1], 1) * 30)
        else:
            ax.set_ylim(0, ax.get_ylim()[1] * 1.5)
        if xlim is not None:
            ax.set_xlim(*xlim)
        head = f"{data.topology}, {wrapped_label(cfg)}\n" if ax is axes.flat[0] else ""
        ax.legend(title=f"{head}{title}: region {region}", loc="upper right", fontsize=16, title_fontsize=16,
                  **{k: v for k, v in tp.LEGEND.items() if k != "title_fontsize"})
    for ax in axes.flat[:n]:
        tp.label(ax, fontsize=16)
    fig.tight_layout()
    tp.save(fig, name)
    return fig


QUANTITIES = {"kappa": None, "A/B": ("A", "B"), "B/D": ("B", "D"), "C/D": ("C", "D")}


def class_ratio_figure(cube, cfg, series, classes, quantities=("kappa", "B/D", "C/D"), reference=None, name="",
                       xlabel="", ylim=(0.1, 300)):
    """ABCD ratios against an object class.  ``series`` is a list of ``(label, plane, color,
    marker, btag)``; for each class ``(tick label, split)`` the plane ``"<plane>:<split>"`` is
    evaluated (the plane itself for a split of ``None``) with the b-tag requirement of
    :data:`estimate_lib.BTAG`.  ``reference`` maps a quantity to a ``(value, minus, plus)``
    drawn as a band (the search plane's open ratios)."""
    fig, axes = plt.subplots(1, len(quantities), figsize=(7.5 * len(quantities), 8), squeeze=False)
    x = np.arange(len(classes))
    offsets = np.linspace(-0.25, 0.25, len(series)) if len(series) > 1 else [0.0]
    for ax, q in zip(axes.flat, quantities):
        if reference is not None and q in reference and np.isfinite(reference[q][0]):
            v, m, p = reference[q]
            ax.axhspan(v - m, v + p, color=pp.COLORS["DD"], alpha=0.18, lw=0)
            ax.axhline(v, color=pp.COLORS["DD"], lw=1.5, ls="--", label="search plane, 68% band")
        for (label, plane, color, marker, btag), off in zip(series, offsets):
            first = True
            c = el.with_btag(cfg, btag)
            for i, (_, split) in enumerate(classes):
                full = plane if split is None else f"{plane}:{split}"
                try:
                    reg = ps.regions(cube, c, full)
                except ValueError:
                    continue
                r = ps.kappa(reg) if QUANTITIES[q] is None else tl.ratio(reg[QUANTITIES[q][0]], reg[QUANTITIES[q][1]])
                pp._errorbar(ax, i + off, r, color, label if first else None, marker=marker)
                first = first and not np.isfinite(r[0])
        pp._axes(ax, "kappa = A D / (B C)" if q == "kappa" else q, x, [c[0] for c in classes], rotation=30)
        ax.set_xlabel(xlabel, fontsize=pp.AXIS)
        ax.set_xlim(-0.5, len(classes) - 0.5)
        ax.set_ylim(*ylim)
        if q == "kappa":
            ax.axhline(1, color="gray", lw=1)
        head = f"{cube.topology}, {wrapped_label(cfg)}\n" if ax is axes.flat[0] else ""
        ax.legend(title=f"{head}{q} of the proxy planes", loc="upper left", fontsize=15, title_fontsize=15, ncol=2,
                  columnspacing=0.8, handletextpad=0.3, **{k: v for k, v in tp.LEGEND.items() if k != "title_fontsize"})
        tp.label(ax, fontsize=16)
    fig.tight_layout()
    if name:
        tp.save(fig, name)
    return fig
