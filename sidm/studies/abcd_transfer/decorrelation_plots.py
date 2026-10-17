"""Figures of mass_decorrelation.ipynb: the isolation and |dphi| pass probabilities against the
LJ-pair mass per plane, the mass spectra of the sidebands, the logistic fits of a(M), and the
mass-flattened isolation cut.  Region A of the search plane is never drawn: every figure shows B, C
and D, or A of the open planes only."""
import matplotlib.pyplot as plt
import mplhep as hep
import numpy as np
from matplotlib import ticker

import decorrelation_lib as dl
import transfer_lib as tl
import transfer_plots as tp

TICK = 18
AXIS = 22
TEXT = 20
OFFSET = {"PP": 0.97, "PD": 1.0, "DD": 1.03}           # multiplicative x offsets on the log mass axis
MASS_TICKS = (150, 200, 300, 400, 600, 1000)
REGION_COLORS = {"B": "tab:purple", "C": "tab:orange", "D": "tab:cyan"}


def _mass_axis(ax, label=True):
    ax.set_xscale("log")
    ax.set_xlim(140, 1100)
    ax.xaxis.set_major_locator(ticker.FixedLocator(MASS_TICKS))
    ax.xaxis.set_major_formatter(ticker.FormatStrFormatter("%g"))
    ax.xaxis.set_minor_locator(ticker.NullLocator())
    ax.tick_params(labelsize=TICK)
    if label:
        ax.set_xlabel("M(LJ, LJ) [GeV] (≥ 600 drawn at 775)", fontsize=AXIS)


def _label_panels(axes, fontsize=20):
    """The CMS label over the first panel and the luminosity over the last, so that neither crowds a
    narrow panel."""
    hep.cms.label("Work in progress", data=True, ax=axes[0], rlabel="", fontsize=fontsize)
    axes[-1].text(1.0, 1.01, f"{tl.LUMI_FB} fb$^{{-1}}$, 2018 (13 TeV)", transform=axes[-1].transAxes,
                  ha="right", va="bottom", fontsize=fontsize - 2)


def _points(ax, df, column, plane, marker="o", label=None, hollow=False):
    part = df[(df["plane"] == plane) & np.isfinite(df[column])]
    face = dict(mfc="white", mew=2) if hollow else {}
    ax.errorbar(part["M"] * OFFSET[plane], part[column], yerr=[part[f"{column} minus"], part[f"{column} plus"]],
                fmt=marker, color=tp.COLORS[plane], ms=10, capsize=4, label=label or plane, **face)


def pass_figure(table, name="decorrelation_pass_probabilities"):
    """a(M) = C / (C + D) and b(M) = B / (B + D) per plane with their 68% Clopper-Pearson intervals
    (``decorrelation_lib.pass_table``); a bin with C + D = 0 has no a point."""
    fig, axes = plt.subplots(1, 2, figsize=(18, 8))
    for ax, column, title in ((axes[0], "a", "a(M) = C / (C + D): isolation passes"),
                              (axes[1], "b", "b(M) = B / (B + D): |Δφ| passes")):
        for plane in dl.PLANES:
            _points(ax, table, column, plane)
        _mass_axis(ax)
        ax.set_ylim(0, 1.25)
        ax.set_ylabel(title, fontsize=AXIS)
        ax.legend(loc="upper left", ncol=3, fontsize=TEXT, **tp.LEGEND)
    tp.label(axes[0], fontsize=20)
    tp.save(fig, name)
    return fig


def spectra_figure(table, name="decorrelation_spectra"):
    """Per plane, the mass spectra of B, C and D, each normalised to one over the five bins
    (``decorrelation_lib.spectra_table``), with the counts per region in the legend."""
    planes = list(dict.fromkeys(table["plane"]))
    fig, axes = plt.subplots(1, len(planes), figsize=(18, 7.5), sharey=True)
    for ax, plane in zip(axes, planes):
        part = table[table["plane"] == plane].reset_index(drop=True)
        x = np.arange(len(part))
        for k, region in enumerate("BCD"):
            share = part[f"{region} share"].to_numpy(float)
            n = part[region].to_numpy(float)
            err = np.sqrt(n) / n.sum()
            ax.errorbar(x + (k - 1) * 0.18, share, yerr=err, fmt="o-", color=REGION_COLORS[region], ms=9, capsize=3,
                        label=f"{region} ({n.sum():.0f})")
        ax.set_xticks(x)
        ax.set_xticklabels([b.replace(">=", "≥") for b in part["bin"]], rotation=35, fontsize=TICK)
        ax.set_ylim(0, 0.8)
        ax.tick_params(axis="y", labelsize=TICK)
        ax.set_xlabel("M(LJ, LJ) bin [GeV]", fontsize=AXIS - 2)
        ax.legend(title=plane, loc="upper right", fontsize=TEXT - 2, **tp.LEGEND)
    axes[0].set_ylabel("share of the region's events", fontsize=AXIS)
    _label_panels(axes)
    tp.save(fig, name)
    return fig


def fit_figure(table, fits, shared=None, name="decorrelation_logistic_fits"):
    """Per plane, a(M) with its 68% intervals and the fitted logistic in ln M with the band of the
    16th-84th percentiles of the refitted toys; in DD also the fit with the slope taken from PD
    (``shared``), dashed."""
    grid = np.geomspace(150, 1000, 80)
    fig, axes = plt.subplots(1, len(fits), figsize=(18, 7.5), sharey=True)
    for ax, (plane, fit) in zip(axes, fits.items()):
        _points(ax, table, "a", plane, label=f"{plane} data")
        curves = dl.logistic_a(fit["alpha_toys"][:, None], fit["slope_toys"][:, None], grid[None, :])
        lo, hi = np.percentile(curves, [15.87, 84.13], axis=0)
        ax.fill_between(grid, lo, hi, color=tp.COLORS[plane], alpha=0.18, lw=0)
        ax.plot(grid, dl.logistic_a(fit["alpha"], fit["slope"], grid), color=tp.COLORS[plane], lw=2.5,
                label=f"fit, slope {fit['slope']:.2f} ± {fit['slope_err']:.2f}")
        if plane == "DD" and shared is not None:
            ax.plot(grid, dl.logistic_a(shared["alpha"], shared["slope"], grid), color="tab:blue", lw=2.5, ls="--",
                    label=f"PD's slope ({shared['slope']:.2f})")
        _mass_axis(ax, label=False)
        ax.set_xlabel("M(LJ, LJ) [GeV]", fontsize=AXIS)
        ax.set_ylim(0, 1.3)
        ax.legend(loc="upper left", fontsize=TEXT - 2, **tp.LEGEND)
    axes[0].set_ylabel("a(M) = C / (C + D)", fontsize=AXIS)
    _label_panels(axes)
    tp.save(fig, name)
    return fig


def flatten_figure(choices, before, after, name="decorrelation_flattened_cut"):
    """Left: the isolation edge per mass bin for each calibration (``choices`` {label: edges}) against
    the fixed 0.25.  Right: a(M) per plane with the fixed cut (``before`` {plane: flatness dict}) and
    with each calibration's edges (``after`` {(label, plane): flatness dict}); open markers after."""
    centres = np.array([dl.centre(*rng) for _, rng in dl.FIVE])
    edges = [rng[0] for _, rng in dl.FIVE] + [1000.0]
    fig, axes = plt.subplots(1, 2, figsize=(18, 8))
    ax = axes[0]
    ax.stairs([0.25] * len(centres), edges, color="gray", lw=2.5, ls="--", baseline=None, label="fixed cut, 0.25")
    styles = {lab: (c, ls) for lab, c, ls in zip(choices, ("tab:blue", "tab:green"), ("-", "-."))}
    for lab, e in choices.items():
        color, ls = styles[lab]
        ax.stairs(e, edges, color=color, lw=3, ls=ls, baseline=None, label=lab)
    _mass_axis(ax, label=False)
    ax.set_xlabel("M(LJ, LJ) [GeV] (last bin drawn to 1000)", fontsize=AXIS)
    ax.set_ylim(0, 0.6)
    ax.set_ylabel("muon-LJ isolation edge (passes below)", fontsize=AXIS)
    ax.legend(loc="upper left", fontsize=TEXT, **tp.LEGEND)
    ax = axes[1]
    markers = {lab: m for lab, m in zip(choices, ("s", "D"))}
    for plane, flat in before.items():
        ax.errorbar(centres * OFFSET[plane], flat["a"], yerr=[flat["minus"], flat["plus"]], fmt="o",
                    color=tp.COLORS[plane], ms=11, capsize=4, label=f"{plane}, fixed 0.25")
    for k, ((lab, plane), flat) in enumerate(after.items()):
        ax.errorbar(centres * OFFSET[plane] * (1.05 + 0.03 * (k // len(before))), flat["a"],
                    yerr=[flat["minus"], flat["plus"]], fmt=markers[lab], color=tp.COLORS[plane], ms=10, mfc="white",
                    mew=2, capsize=3, label=f"{plane}, {lab}")
    _mass_axis(ax)
    ax.set_ylim(0, 1.5)
    ax.set_ylabel("a(M) = C / (C + D)", fontsize=AXIS)
    ax.legend(loc="upper left", ncol=2, fontsize=TEXT - 2, **tp.LEGEND)
    tp.label(axes[0], fontsize=20)
    tp.save(fig, name)
    return fig
