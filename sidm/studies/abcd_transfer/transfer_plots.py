"""Figures of the transfer study: fail-slice isolation shapes, factor scans, dphi shapes."""
import os

import matplotlib.pyplot as plt
import mplhep as hep
import numpy as np

import abcd6d_io as io
import transfer_lib as tl

FIG_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "figures")
COLORS = {"PP": "black", "DP": "tab:red", "PD": "tab:blue", "DD": "tab:green"}


LEGEND = dict(frameon=True, facecolor="white", edgecolor="none", framealpha=0.92, title_fontsize=19)


def set_style():
    plt.style.use(hep.style.CMS)
    # Figures are 18 in wide and print at text width (7.5 in), a factor 0.42: every size below
    # is chosen so that the printed text stays above 7 pt.
    plt.rcParams.update({"figure.dpi": 100, "savefig.bbox": "tight", "legend.fontsize": 20})


def label(ax, fontsize=22):
    hep.cms.label("Work in progress", data=True, lumi=tl.LUMI_FB, year=2018, ax=ax, fontsize=fontsize)


def save(fig, name):
    os.makedirs(FIG_DIR, exist_ok=True)
    fig.savefig(os.path.join(FIG_DIR, name + ".pdf"))


def bin_labels(edges):
    e = [f"{x:g}" for x in edges]
    return [f"<{e[0]}"] + [f"{a}-{b}" for a, b in zip(e[:-1], e[1:])] + [f">{e[-1]}"]


def events(n):
    n = np.nansum(n)
    return f"{n:.0f} event" + ("" if n == 1 else "s")


def step_with_errors(ax, p, err, color, text):
    x = np.arange(len(p))
    ax.step(x, p, where="mid", color=color, lw=2, label=text)
    ax.errorbar(x, p, yerr=err, fmt="none", ecolor=color, elinewidth=1.2, capsize=2)


def _ratio_points(rax, p_num, e_num, n_num, p_den, e_den, color):
    """Ratio of two shapes; bins with no event in the numerator sample are not drawn."""
    with np.errstate(divide="ignore", invalid="ignore"):
        r = p_num / p_den
        re = r * np.hypot(e_num / p_num, e_den / p_den)
    ok = np.isfinite(r) & (n_num > 0)
    x = np.arange(len(p_num))
    rax.errorbar(x[ok], r[ok], yerr=re[ok], fmt="o", color=color, ms=5)


def _shape_panel(ax, rax, shapes, a, b, edges, title):
    pa, ea, na = shapes[a]
    pb, eb, nb = shapes[b]
    step_with_errors(ax, pa, ea, COLORS[a], f"{a}: {events(na)}")
    step_with_errors(ax, pb, eb, COLORS[b], f"{b}: {events(nb)}")
    _ratio_points(rax, pb, eb, nb, pa, ea, COLORS[b])
    rax.axhline(1, color="gray", lw=1)
    rax.set_ylim(0, 3)
    rax.set_ylabel(f"{b} / {a}", fontsize=20)
    x = np.arange(len(pa))
    for a_ in (ax, rax):
        a_.set_xticks(x)
        a_.set_xlim(-0.5, len(x) - 0.5)
    rax.set_xticklabels(bin_labels(edges), rotation=60, ha="right", fontsize=18)
    rax.tick_params(axis="y", labelsize=18)
    ax.tick_params(labelbottom=False, labelsize=18)
    ax.set_ylabel("fraction of events", fontsize=22)
    ax.set_yscale("log")
    ax.set_ylim(2e-3, 40)
    ax.legend(title=title, loc="upper left", **LEGEND)
    label(ax)


def fail_slice_shapes_figure(cube, mass="M>=150"):
    """Isolation shape of each object inside the other object's fail slice, prompt vs displaced."""
    t = cube.topology
    o0, o1 = tl.OBJECTS[t]
    fig = plt.figure(figsize=(18, 18))
    grid = fig.add_gridspec(5, 2, height_ratios=[3, 1.2, 1.1, 3, 1.2], hspace=0.08, wspace=0.22)
    panels = [("r1", ("PP", "DP"), 0, 0), ("r1", ("PD", "DD"), 0, 1),
              ("r0", ("PP", "PD"), 3, 0), ("r0", ("DP", "DD"), 3, 1)]
    for factor, (a, b), row, col in panels:
        var, shapes = tl.fail_slice_shapes(cube, factor, mass)
        measured, other = (o1, o0) if factor == "r1" else (o0, o1)
        ax = fig.add_subplot(grid[row, col])
        rax = fig.add_subplot(grid[row + 1, col])
        state = tl.PLANES[a][1 if factor == "r1" else 0]
        title = f"{t}, {mass}, {other} iso >= {tl.FAIL:g}\n{measured} {state}\n{other} prompt vs displaced"
        _shape_panel(ax, rax, shapes, a, b, io.EDGES["iso"], title)
        rax.set_xlabel(f"{measured} isolation", fontsize=22)
    save(fig, f"fail_slice_shapes_{t}_{mass.replace('>=', 'ge').replace('<', 'lt')}")
    return fig


def factor_scan_figure(cube, mass="M>=150"):
    """The four pass/fail ratios against the displacement bin of the object each one holds fixed.

    Left column: object-1 ratios against object-0 displacement (r1 among object-0 failures,
    A/C among its passes); right column: object-0 ratios against object-1 displacement.
    """
    t = cube.topology
    fig, axes = plt.subplots(2, 2, figsize=(18, 15))
    panels = ((axes[0, 0], "r1"), (axes[0, 1], "r0"), (axes[1, 0], "ac"), (axes[1, 1], "ab"))
    for ax, factor in panels:
        num, den, measured, held, state = tl.FACTORS[factor]
        kind = tl.disp_kind(t, held)
        bins = [b for b, _ in tl.DISP_BINS[kind]]
        prompt_bins = tl.DISP_PROMPT_BINS[kind]
        x = np.arange(len(bins))
        ticks = list(bins)
        for other_state, color, dx, tag in (("prompt", "black", -0.1, "P"), ("displaced", "tab:red", 0.1, "D")):
            scan = tl.factor_scan(cube, factor, other_state, mass)
            ok = np.isfinite(scan["value"]) & (scan["value"] > 0)
            yerr = np.vstack([scan["minus"], np.where(np.isinf(scan["plus"]), 0, scan["plus"])])
            ax.errorbar(x[ok] + dx, scan["value"][ok], yerr=yerr[:, ok], fmt="o", color=color, ms=8,
                        capsize=3, label=f"{tl.OBJECTS[t][measured]} {other_state} ({tag})")
            counts = [("blind" if np.isnan(n) or np.isnan(d) else f"{n:.0f}/{d:.0f}") for n, d in zip(scan[num], scan[den])]
            ticks = [f"{tk}\n{tag} {c}" for tk, c in zip(ticks, counts)]
        for i, b in enumerate(bins):
            if b not in prompt_bins:
                ax.axvspan(i - 0.5, i + 0.5, color="tab:green", alpha=0.08, lw=0)
        ax.set_xticks(x)
        ax.set_xticklabels(ticks, fontsize=18)
        ax.tick_params(axis="y", labelsize=18)
        ax.set_xlim(-0.5, len(bins) - 0.5)
        unit = "pixel hits (max over muons)" if kind == "mu" else "lost hits (min over e/gamma)"
        ax.set_xlabel(f"{tl.OBJECTS[t][held]} {unit}", fontsize=22)
        ax.set_ylabel(f"{factor} = {num}/{den}: {tl.OBJECTS[t][measured]} pass/fail ratio", fontsize=20)
        ax.set_yscale("log")
        ax.set_ylim(0.03, 60)
        ax.legend(title=f"{t}, {mass}, {tl.OBJECTS[t][held]} {state}\ngreen = displaced bins; counts: {num}/{den}",
                  loc="upper left", **LEGEND)
        label(ax)
    fig.tight_layout()
    save(fig, f"factor_scan_{t}_{mass.replace('>=', 'ge').replace('<', 'lt')}")
    return fig


def dphi_shapes_figure(cube, mass="M>=150"):
    """|dphi| shape per plane, left among object-0 isolation failures, right among its passes.

    Object 1 is isolated (< 0.5) in both.  The pass slice of the DD plane is hidden at
    |dphi| >= 2 and is not drawn.
    """
    t = cube.topology
    fig = plt.figure(figsize=(18, 10))
    grid = fig.add_gridspec(2, 2, height_ratios=[3, 1.2], hspace=0.08, wspace=0.2)
    edges = io.EDGES["abs_dphi"]
    labels = ["<0"] + [f"{a:g}-{b:g}" for a, b in zip(edges[:-1], edges[1:])] + [">pi"]
    labels[-2] = f"{edges[-2]:g}-pi"
    for col, which in enumerate(("fail", "pass")):
        shapes = tl.dphi_shapes(cube, mass, which)
        ax = fig.add_subplot(grid[0, col])
        rax = fig.add_subplot(grid[1, col])
        pa, ea, _ = shapes["PP"]
        x = np.arange(len(pa))
        for plane in tl.PLANES:
            p, e, n = shapes[plane]
            if np.isnan(n).any():
                continue
            step_with_errors(ax, p, e, COLORS[plane], f"{plane}: {events(n)}")
            if plane != "PP":
                _ratio_points(rax, p, e, n, pa, ea, COLORS[plane])
        rax.axhline(1, color="gray", lw=1)
        rax.set_ylim(0, 3)
        rax.set_ylabel("plane / PP", fontsize=20)
        for a_ in (ax, rax):
            a_.set_xticks(x)
            a_.set_xlim(-0.5, len(x) - 0.5)
        rax.set_xticklabels(labels, rotation=60, ha="right", fontsize=18)
        rax.tick_params(axis="y", labelsize=18)
        rax.set_xlabel("|dphi(LJ, LJ)|", fontsize=22)
        ax.tick_params(labelbottom=False, labelsize=18)
        ax.set_ylabel("fraction of events", fontsize=22)
        ax.set_yscale("log")
        ax.set_ylim(2e-3, 40)
        iso0 = f"iso >= {tl.FAIL:g}" if which == "fail" else f"iso < {tl.WP:g}"
        ax.legend(title=f"{t}, {mass}\n{tl.OBJECTS[t][0]} {iso0}\n{tl.OBJECTS[t][1]} iso < {tl.OTHER_ISO:g}",
                  loc="upper left", **LEGEND)
        label(ax)
    save(fig, f"dphi_shapes_{t}_{mass.replace('>=', 'ge').replace('<', 'lt')}")
    return fig


MASS_LABELS = [b for b, _ in tl.MASS_BINS]


def mass_window_dphi_figure(cube, plane="PP"):
    """|dphi| shape of one plane in the two mass windows, among object-0 failures (left) and passes (right)."""
    t = cube.topology
    fig = plt.figure(figsize=(18, 10))
    grid = fig.add_gridspec(2, 2, height_ratios=[3, 1.2], hspace=0.08, wspace=0.2)
    edges = io.EDGES["abs_dphi"]
    labels = ["<0"] + [f"{a:g}-{b:g}" for a, b in zip(edges[:-1], edges[1:])] + [">pi"]
    labels[-2] = f"{edges[-2]:g}-pi"
    colors = {"M>=150": "black", "M<150": "tab:orange"}
    for col, which in enumerate(("fail", "pass")):
        ax = fig.add_subplot(grid[0, col])
        rax = fig.add_subplot(grid[1, col])
        shapes = {m: tl.dphi_shapes(cube, m, which)[plane] for m in colors}
        pa, ea, _ = shapes["M>=150"]
        x = np.arange(len(pa))
        for m, (p, e, n) in shapes.items():
            if np.isnan(n).any():
                continue
            step_with_errors(ax, p, e, colors[m], f"{m}: {events(n)}")
        p, e, n = shapes["M<150"]
        if not (np.isnan(n).any() or np.isnan(pa).any()):
            _ratio_points(rax, p, e, n, pa, ea, colors["M<150"])
        rax.axhline(1, color="gray", lw=1)
        rax.set_yscale("log")
        rax.set_ylim(0.05, 50)
        rax.set_ylabel("M<150 / M>=150", fontsize=20)
        for a_ in (ax, rax):
            a_.set_xticks(x)
            a_.set_xlim(-0.5, len(x) - 0.5)
        rax.set_xticklabels(labels, rotation=60, ha="right", fontsize=18)
        rax.tick_params(axis="y", labelsize=18)
        rax.set_xlabel("|dphi(LJ, LJ)|", fontsize=22)
        ax.tick_params(labelbottom=False, labelsize=18)
        ax.set_ylabel("fraction of events", fontsize=22)
        ax.set_yscale("log")
        ax.set_ylim(2e-3, 40)
        iso0 = f"iso >= {tl.FAIL:g}" if which == "fail" else f"iso < {tl.WP:g}"
        ax.legend(title=f"{t}, {plane} plane\n{tl.OBJECTS[t][0]} {iso0}\n{tl.OBJECTS[t][1]} iso < {tl.OTHER_ISO:g}",
                  loc="upper left", **LEGEND)
        label(ax)
    save(fig, f"mass_window_dphi_{t}_{plane}")
    return fig


def mass_factor_figure(cube, kind="dphi x iso0"):
    """The two factors among failures (B/D, C/D), the pass-side ratio A/C and kappa of each plane
    against the LJ-pair mass bin."""
    t = cube.topology
    names = tl.DPHI_FACTORS if kind == "dphi x iso0" else tl.FACTORS
    fig, axes = plt.subplots(2, 2, figsize=(18, 14))
    x = np.arange(len(MASS_LABELS))
    offsets = {"PP": -0.24, "DP": -0.08, "PD": 0.08, "DD": 0.24}
    for ax, factor in zip(axes.flat, ("r1", "ac", "r0", "kappa")):
        for plane in tl.PLANES:
            scan = tl.mass_factor_scan(cube, factor, plane, kind)
            ok = np.isfinite(scan["value"]) & (scan["value"] > 0)
            yerr = np.vstack([scan["minus"], np.where(np.isinf(scan["plus"]), 0, scan["plus"])])
            ax.errorbar(x[ok] + offsets[plane], scan["value"][ok], yerr=yerr[:, ok], fmt="o", color=COLORS[plane],
                        ms=8, capsize=3, label=plane)
        ax.axvline(1.5, color="gray", lw=1, ls="--")
        ax.set_xticks(x)
        ax.set_xticklabels(MASS_LABELS, rotation=45, ha="right", fontsize=18)
        ax.tick_params(axis="y", labelsize=18)
        ax.set_xlim(-0.5, len(x) - 0.5)
        ax.set_xlabel("M(LJ, LJ) [GeV]", fontsize=22)
        if factor == "kappa":
            ax.set_ylabel("kappa = A D / (B C)", fontsize=22)
            ax.axhline(1, color="gray", lw=1)
        else:
            short = names[factor][3] if kind == "dphi x iso0" else f"{factor} = {names[factor][0]}/{names[factor][1]}"
            ax.set_ylabel(short, fontsize=22)
        ax.set_yscale("log")
        ax.set_ylim(0.05, 3000)
        ax.legend(title=f"{t}, {kind} plane\ndashed line: the 150 GeV requirement", loc="upper left", ncol=4,
                  columnspacing=0.8, handletextpad=0.3, **LEGEND)
        label(ax)
    fig.tight_layout()
    save(fig, f"mass_factors_{t}_{kind.replace(' ', '')}")
    return fig


def mass_shape_figure(cube, kind="dphi x iso0", planes=("PP", "DD")):
    """Normalised LJ-pair mass distribution of the regions of a plane; hidden regions are not drawn."""
    t = cube.topology
    fig, axes = plt.subplots(1, len(planes), figsize=(9 * len(planes), 8))   # 18 in wide for two planes
    edges = io.EDGES["ljlj_mass"]
    labels = ["<0"] + [f"{a:g}-{b:g}" for a, b in zip(edges[:-1], edges[1:])] + [f">{edges[-1]:g}"]
    region_colors = {"A": "black", "B": "tab:red", "C": "tab:blue", "D": "tab:green"}
    for ax, plane in zip(np.atleast_1d(axes), planes):
        shapes = tl.mass_shapes(cube, plane, kind)
        x = np.arange(len(edges) + 1)
        for r, (p, e, n) in shapes.items():
            if np.isnan(n).any():
                continue
            step_with_errors(ax, p, e, region_colors[r], f"{r}: {events(n)}")
        ax.axvline(3.5, color="gray", lw=1, ls="--")
        ax.set_xticks(x)
        ax.set_xticklabels(labels, rotation=60, ha="right", fontsize=18)
        ax.tick_params(axis="y", labelsize=18)
        ax.set_xlim(-0.5, len(x) - 0.5)
        ax.set_xlabel("M(LJ, LJ) [GeV]", fontsize=22)
        ax.set_ylabel("fraction of events", fontsize=22)
        ax.set_yscale("log")
        ax.set_ylim(2e-3, 300)      # the legend sits above the spectra
        ax.legend(title=f"{t}, {plane} plane, {kind}\ndashed line: the 150 GeV requirement", loc="upper right", **LEGEND)
        label(ax)
    fig.tight_layout()
    save(fig, f"mass_shapes_{t}_{kind.replace(' ', '')}")
    return fig
