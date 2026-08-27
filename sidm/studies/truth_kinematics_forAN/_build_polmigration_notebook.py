"""Builder for polarization_migration_forAN.ipynb: writes the notebook JSON.

Part 1 ports the polarization-fit summary from
signal_kinematics/lepton_kinematics_summary_grid.ipynb onto the canonical
truth_kinematics_forAN output (channels genOnly / genOnly_born). Part 2 ports the
reconstruction-migration maps from lepton_reco/leptonReco.ipynb onto the
trigger-free channels of the same output. Execute after building:
    jupyter nbconvert --to notebook --execute --inplace polarization_migration_forAN.ipynb
"""
import json
import os

HERE = os.path.dirname(os.path.abspath(__file__))

def md(src):
    return {"cell_type": "markdown", "metadata": {}, "source": src.splitlines(keepends=True)}

def code(src):
    return {"cell_type": "code", "execution_count": None, "metadata": {}, "outputs": [],
            "source": src.splitlines(keepends=True)}

cells = [
md("""# Dark-photon polarization and lepton reconstruction migration

**Inputs and selections.** Generator-level truth from the canonical
`truth_kinematics_forAN` output (180 v10 signal samples, every file, unweighted).
Polarization uses the strictly-generator channels, **`genOnly`** (status-1 leptons) and
**`genOnly_born`** (status-23), with **no reconstruction, trigger, or vertex
requirements**; they reproduce the `gen_leptons_final`/`gen_leptons_born` channels of the
original study minus the primary-vertex filter. The reconstruction-migration maps use
**`baseNoLj_noTrigger`** (analysis LJ-source object definitions, PV filter, **no HLT**,
so the migration story is trigger-independent) and its `NoLjsource` variant (same, before
the object quality cuts).

The polarization methodology and its full phase-space discussion live in
`signal_kinematics/lepton_kinematics_summary_grid.ipynb`; this notebook reruns the fits
on the canonical full-statistics output and saves the analysis-note figures. The
migration maps fill the note's `fig:reco_lepton_kinematics` reference.
"""),
code("""import sys, os, importlib
import numpy as np
import matplotlib.pyplot as plt
import mplhep as hep
from scipy import optimize as opt

here = os.getcwd()
repo = here.split("/sidm")[0]
for p in (repo, os.path.join(repo, "sidm", "studies", "lifetime_study"),
          os.path.join(repo, "sidm", "studies", "truth_kinematics_forAN")):
    if p not in sys.path:
        sys.path.insert(1, p)
from sidm.tools import utilities
import lifetime_analysis as la
import an_style
importlib.reload(la)

utilities.set_plot_style(dpi=110)
CH_FINAL, CH_BORN = "genOnly", "genOnly_born"
"""),
code("""from _lifetime_refit import load_truthkin
output = load_truthkin()
print(f"loaded {len(output)} samples")

SAMPLE_GRID = {
    100:  {0.25: "2Mu2E_100GeV_0p25GeV_2p0mm",
           1.2:  "2Mu2E_100GeV_1p2GeV_9p6mm",
           5.0:  "2Mu2E_100GeV_5p0GeV_40p0mm"},
    200:  {0.25: "2Mu2E_200GeV_0p25GeV_1p0mm",
           1.2:  "2Mu2E_200GeV_1p2GeV_4p8mm",
           5.0:  "2Mu2E_200GeV_5p0GeV_20p0mm"},
    500:  {0.25: "2Mu2E_500GeV_0p25GeV_0p4mm",
           1.2:  "2Mu2E_500GeV_1p2GeV_1p9mm",
           5.0:  "2Mu2E_500GeV_5p0GeV_8p0mm"},
    800:  {0.25: "2Mu2E_800GeV_0p25GeV_0p25mm",
           1.2:  "2Mu2E_800GeV_1p2GeV_1p2mm",
           5.0:  "2Mu2E_800GeV_5p0GeV_5p0mm"},
    1000: {0.25: "2Mu2E_1000GeV_0p25GeV_0p2mm",
           1.2:  "2Mu2E_1000GeV_1p2GeV_0p96mm",
           5.0:  "2Mu2E_1000GeV_5p0GeV_4p0mm"},
}
BS_MASSES = [100, 200, 500, 800, 1000]
ZD_MASSES = [0.25, 1.2, 5.0]
BS_COLORS = {m: c for m, c in zip(BS_MASSES, plt.cm.viridis(np.linspace(0.0, 0.9, len(BS_MASSES))))}

def sample_label(mxx, mzd):
    return rf"$m_{{XX}}={mxx}$, $m_{{Z_d}}={mzd}$ GeV"
"""),
md("""## Polarization fits, α summary

For each sample the lepton `|cosθ*|` distribution in the dark-photon rest frame is fit
with `1 + α·cos²θ*` over a range chosen per sample, for the reasons in the next
section: at `m_Zd ≥ 1.2 GeV`, `[0, 0.6]` at `m_XX = 100 GeV` and `[0, 0.8]` at
`m_XX ≥ 200 GeV`; at `m_Zd = 0.25 GeV`, `[0, 0.8]` only at `m_XX = 200 GeV` and
`[0, 0.6]` everywhere else on that row. A spin-1 dark photon decaying to relativistic
leptons gives `α → 1` (transverse polarization); velocity suppression drives `α → 0`
as `m_ℓ/m_Zd` grows, visible for muons at `m_Zd = 0.25 GeV`, where `m_μ` eats half
the two-body momentum.
"""),
code("""# m_Zd >= 1.2 GeV: [0, 0.6] at m_XX = 100 GeV, where the gen filter's
# acceptance edge sits too close to 0.8; [0, 0.8] at m_XX >= 200 GeV, where it
# does not. m_Zd = 0.25 GeV does not follow that same split -- see the prose
# below -- and only m_XX = 200 GeV is safe to widen there.
def fit_range_for(mxx, mzd):
    if mzd <= 0.25:
        return (0.0, 0.8) if mxx == 200 else (0.0, 0.6)
    return (0.0, 0.6) if mxx <= 100 else (0.0, 0.8)

def fit_alpha(h, fit_range):
    raw = h.values().flatten().astype(float)
    edges = h.axes[-1].edges
    centers = 0.5 * (edges[:-1] + edges[1:])
    n = min(len(raw), len(centers))
    raw, centers = raw[:n], centers[:n]
    mask = (centers >= fit_range[0]) & (centers <= fit_range[1])
    x, y = centers[mask], raw[mask]
    yerr = np.sqrt(y); yerr[yerr == 0] = 1.0
    def spin1(x, A, alpha): return A * (1 + alpha * x**2)
    try:
        popt, pcov = opt.curve_fit(spin1, x, y, sigma=yerr, p0=[max(y.max(), 1), 0.5])
        return popt[1], float(np.sqrt(np.diag(pcov))[1])
    except Exception:
        return np.nan, np.nan

rows = []
for mxx in BS_MASSES:
    for mzd in ZD_MASSES:
        sample = SAMPLE_GRID[mxx][mzd]
        for status_label, ch in [("status 1", CH_FINAL), ("status 23", CH_BORN)]:
            for flavor, hname in [("mu", "genMu_AFrame_absCosTheta"),
                                  ("e",  "genE_AFrame_absCosTheta")]:
                a, da = fit_alpha(output[sample]["hists"][hname][{"channel": ch}],
                                 fit_range_for(mxx, mzd))
                rows.append((mxx, mzd, status_label, flavor, a, da))

fig, axes = plt.subplots(2, 2, figsize=(18, 12), layout="constrained")
for r, flavor in enumerate(["mu", "e"]):
    for c, status in enumerate(["status 1", "status 23"]):
        ax = axes[r, c]
        for mxx in BS_MASSES:
            pts = [(mzd, a, da) for (b, mzd, s, f, a, da) in rows
                   if b == mxx and s == status and f == flavor]
            ax.errorbar([p[0] for p in pts], [p[1] for p in pts],
                        yerr=[p[2] for p in pts], marker="o", markersize=8,
                        color=BS_COLORS[mxx], label=rf"$m_{{XX}}={mxx}$ GeV",
                        capsize=4, linewidth=2)
        ax.axhline(1.0, color="grey", linestyle=":", alpha=0.6)
        ax.axhline(0.0, color="grey", linestyle=":", alpha=0.6)
        flv = r"\\mu" if flavor == "mu" else "e"
        ax.set_xlabel(r"$m_{Z_d}$ [GeV]", fontsize=14)
        ax.set_ylabel(r"$\\alpha$", fontsize=14)
        ax.set_xscale("log")
        ax.text(0.5, 0.06, rf"${flv}$, {status} leptons", transform=ax.transAxes,
                ha="center", va="bottom", fontsize=15)
        ax.grid(True, alpha=0.2)
        ax.legend(loc="center left", fontsize=12)
        hep.cms.label(ax=ax, data=False)
an_style.save(fig, "polarization_alpha_summary")
for row in rows:
    print("  m_XX=%4d  m_Zd=%.2f  %-9s %-2s  alpha = %6.3f +- %.3f" % row)
"""),
md("""**Reading the summary.** Two unrelated things restrict the fit range, and they do
not affect the same points. The first is a generator-level acceptance cliff, present
at every `m_Zd`: several samples carry a sharp statistical collapse in `|cosθ*|`
starting around 0.7-0.85, worst at the lightest bound-state mass. This is
generator-level truth (channel `genOnly`, no reconstruction), so it cannot be a
detector effect; it is the production gen filter's per-lepton `pT > 5` GeV cut. As
`|cosθ*| → 1` the two-body decay becomes maximally asymmetric (the sub-leading
lepton's `pT` falls as `(1-β* cosθ*)/(1+β* cosθ*)` relative to the leading one,
section 6 of the anatomy notebook), and once it drops below 5 GeV the *whole event*
fails the filter and was never generated: at `m_XX = 100 GeV`, `m_Zd = 1.2 GeV`
muons, the fraction of surviving events with `pT`(sub)/`pT`(lead) `< 0.15` jumps
from 5% to 100% between `|cosθ*| = 0.75` and `0.81`, exactly where the raw event
count collapses from its peak. This is what keeps `m_XX = 100 GeV` at `[0, 0.6]`
everywhere on the grid, including `m_Zd = 0.25 GeV`: the electron fit there reads
`χ²/dof = 20.2` at `[0, 0.8]` despite `m_Zd = 0.25 GeV` muons themselves being
immune to this particular mechanism (below).

The second restriction is specific to `m_Zd = 0.25 GeV`, is a completely different
shape, and only shows up at the higher masses. The raw `|cosθ*|` counts there never
collapse -- they rise monotonically all the way to `1`, at every `m_XX` from 100 to
1000 -- but at `m_XX ≥ 500 GeV` the rise accelerates far faster near `|cosθ*| = 1`
than `1 + α cos²θ*` can describe: for `m_XX = 1000 GeV` muons the count nearly
triples between `|cosθ*| = 0.8` and `0.98`, well beyond what any physical `α ≤ 1`
predicts. Because this is excess pile-up rather than a hard edge, there is no safe
wide window that avoids it -- widening only makes it worse, monotonically, and the
fitted `α` climbs past the unphysical value of `1` if allowed to (muons at
`m_XX = 1000 GeV`: `α = 0.59` at `[0, 0.6]`, `0.91` at `[0, 0.8]`, `1.33` at
`[0, 0.9]`). `[0, 0.6]` is not a good fit there either (`χ²/dof` up to `6.0`), just
the least-bad option. The earlier `lepton_kinematics_summary_grid.ipynb` study
identified the same mechanism directly: at the highest boost the sub-leading
lepton's lab-frame `pT` runs to zero regardless of the decay angle, so recovering
`cosθ*` by boosting back into the `Z_d` rest frame is numerically ill-conditioned --
confirmed independently there by the sub/lead `pT` ratio vs `cosθ*` relation, which
should be a thin, well-defined band and instead degenerates into a diffuse blob at
exactly this corner.

`m_XX = 200 GeV`, `m_Zd = 0.25 GeV` sits in neither trap: the acceptance cliff has
moved out of the way (unlike `m_XX = 100 GeV`) and the high-boost pile-up has not
yet set in (unlike `m_XX ≥ 500 GeV`), so it is the one `m_Zd = 0.25 GeV` point where
widening genuinely helps -- `χ²/dof` stays at `2.0` for muons whether fit to
`[0, 0.6]` or `[0, 0.8]`, and the wider range roughly halves the error on `α`
(`0.16 ± 0.06` against `0.25 ± 0.03`).

With that accounted for, electrons recover transverse polarization (`α ≈ 1`) across
essentially the whole grid; muons do too, except at `m_Zd = 0.25 GeV`, where
`m_μ/m_Zd` is no longer negligible and the velocity suppression is real and large.
Read the suppression there as real and large, not as a precise number -- the fitted
value still depends on which side of the acceptance cliff and the pile-up onset a
given `m_XX` happens to sit on, not on the polarization alone.

**Why the fit range depends on both `m_XX` and `m_Zd`.** `m_Zd ≥ 1.2 GeV` samples
with `m_XX ≥ 200 GeV` stay well-behaved out past `0.8`, and fitting only to `0.6`
there throws away real, clean data for no benefit beyond a slightly larger error on
`α` -- widening costs nothing in fit quality (median `χ²/dof` across the widened
points is `1.9`, the same as at `[0, 0.6]`) and buys a tighter constraint.
`m_XX = 100 GeV` cannot be widened the same way even at `m_Zd ≥ 1.2 GeV`: its filter
edge sits close enough to `0.8` that including it still measures the collapse
(`χ²/dof` at `m_XX = 100 GeV`, `m_Zd = 1.2 GeV` muons is `11.1` at `[0, 0.8]` against
`0.9` at `[0, 0.6]`).

`m_Zd = 0.25 GeV` does not follow that same `m_XX ≤ 100` / `m_XX ≥ 200` split, because
it answers to the pile-up mechanism above rather than the acceptance cliff:
`m_XX = 100 GeV` still needs `[0, 0.6]` (electrons crash there for the acceptance
reason, same as every other `m_Zd`), `m_XX = 200 GeV` is the one point safe to widen,
and `m_XX ≥ 500 GeV` needs to stay at `[0, 0.6]` because widening only ever makes
the pile-up worse (`χ²/dof` at `m_XX = 500 GeV` muons: `6.0`, `21.9`, `49.5`, `61.0`
at `[0, 0.6]`, `[0, 0.8]`, `[0, 0.9]`, `[0, 0.95]` respectively).
"""),
code("""def fit_and_plot_polarization(h, ax, color, label_prefix, fit_range):
    edges = h.axes[-1].edges
    centers = 0.5 * (edges[:-1] + edges[1:])
    raw = h.values().flatten().astype(float)
    n = min(len(raw), len(centers))
    raw, centers = raw[:n], centers[:n]
    widths = (edges[1:] - edges[:-1])[:n]
    scale = 1.0 / max(np.sum(raw * widths), 1e-12)
    y, yerr = raw * scale, np.sqrt(raw) * scale
    yerr_safe = np.where(yerr == 0, scale, yerr)
    ax.errorbar(centers, y, yerr=yerr, fmt="o", markersize=4, color=color,
                capsize=2, label=label_prefix)
    mask = (centers >= fit_range[0]) & (centers <= fit_range[1])
    def spin1(x, A, alpha): return A * (1 + alpha * x**2)
    try:
        popt, pcov = opt.curve_fit(spin1, centers[mask], y[mask],
                                   sigma=yerr_safe[mask], p0=[max(y.max(), 1e-3), 0.5])
        perr = np.sqrt(np.diag(pcov))
        x_m = np.linspace(0, 1, 100)
        ax.plot(x_m, spin1(x_m, *popt), "-", color=color, lw=2,
                label=rf"fit ($\\alpha={popt[1]:.2f}\\pm{perr[1]:.2f}$)")
    except Exception:
        pass

fig, axes = plt.subplots(len(BS_MASSES), 3, figsize=(22, 6 * len(BS_MASSES)))
for i, mxx in enumerate(BS_MASSES):
    for j, mzd in enumerate(ZD_MASSES):
        ax = axes[i, j]
        sample = SAMPLE_GRID[mxx][mzd]
        rng = fit_range_for(mxx, mzd)
        fit_and_plot_polarization(
            output[sample]["hists"]["genMu_AFrame_absCosTheta"][{"channel": CH_BORN}],
            ax, "tab:blue", r"$\\mu$", rng)
        fit_and_plot_polarization(
            output[sample]["hists"]["genE_AFrame_absCosTheta"][{"channel": CH_BORN}],
            ax, "tab:orange", r"$e$", rng)
        ax.set_xlabel(r"$|\\cos\\theta^*|$ ($Z_d$ frame)", fontsize=13)
        ax.set_xlim(0, 1.0)
        ax.text(0.5, 0.06, sample_label(mxx, mzd), transform=ax.transAxes,
                ha="center", va="bottom", fontsize=13)
        ax.grid(True, alpha=0.15)
        ax.legend(loc="upper left", fontsize=11)
        hep.cms.label(ax=ax, data=False)
an_style.save(fig, "polarization_grid_born")
"""),
md("""## Lepton reconstruction migration vs displacement

For each dark-photon decay, the number of nearby reconstructed electrons, photons, PF
muons, and DSA muons is histogrammed against the decay's `Lxy`, and each `Lxy` column is
normalized to the number of decays there, so every panel reads as "the fraction of dark
photon decays yielding N reconstructed objects of this type, at this displacement". The
migration the note describes in prose (GED electrons giving way to photons, PF muons to
DSA muons, as `Lxy` grows) is directly visible. Objects here pass the analysis LJ-source
definitions; the trigger is deliberately not applied.
"""),
code("""def make_eff(h):
    weights = 1 / np.sum(h.values(), axis=-1)
    weights = np.nan_to_num(weights)
    for ix in range(h.shape[1]):
        h[:, ix] = np.column_stack([h.values()[:, ix] * weights,
                                    h.variances()[:, ix] * weights**2])
    return h

def plot_single(h, channel, skip_label=True, remove_xlabel=True):
    h = h[{"channel": channel}]
    h = make_eff(h)
    utilities.plot(h[:400j, 1:3], skip_label=skip_label, cbar=False, cmax=1, flow="none")
    if remove_xlabel:
        plt.xlabel(None)
    plt.draw()

def plot_all_vs_lxy(sample, channel, tag):
    fig, ax = plt.subplots(4, 1, figsize=(30, 10), sharex=True)
    for k, obj in enumerate(["electron", "photon", "muon", "dsaMuon"]):
        plt.subplot(4, 1, k + 1)
        plot_single(output[sample]["hists"][f"{obj}_nearGenA_n_genA_lxy"], channel,
                    skip_label=(k != 0), remove_xlabel=(k != 3))
    fig.subplots_adjust(hspace=0, right=0.9)
    mxx, mzd = sample.split("_")[1:3]
    fig.suptitle(rf"$m_{{XX}}$ = {mxx.replace('GeV', ' GeV')},  "
                 rf"$m_{{Z_d}}$ = {mzd.replace('p', '.').replace('GeV', ' GeV')}")
    cax = fig.add_axes([0.91, 0.11, 0.013, 0.77])
    mappable = [c for a in fig.axes for c in a.collections][0]
    cb = fig.colorbar(mappable, cax=cax)
    cb.set_label("Fraction of dark photon decays", labelpad=20)
    an_style.save(fig, f"reco_migration_{tag}")

plot_all_vs_lxy("2Mu2E_1000GeV_0p25GeV_2p0mm", "baseNoLj_noTrigger", "1000GeV_0p25GeV")
plot_all_vs_lxy("2Mu2E_100GeV_5p0GeV_400mm", "baseNoLj_noTrigger", "100GeV_5p0GeV")
plot_all_vs_lxy("2Mu2E_1000GeV_0p25GeV_2p0mm", "baseNoLjNoLjsource_noTrigger",
                "1000GeV_0p25GeV_noquality")
"""),
md("""## Summary

The canonical output reproduces the polarization structure of the original study at full
statistics, now across all five `m_XX` points from 100 to 1000 GeV rather than three:
transverse polarization (`α` consistent with 1) holds for both flavors across essentially
the entire grid, and the one real departure is muon velocity suppression at
`m_Zd = 0.25 GeV`, where `m_μ` is no longer negligible next to `m_Zd`. Getting a clean
read on that required tightening the polarization fit range from `[0, 0.8]` to
`[0, 0.6]` -- the wider range let the gen filter's acceptance edge into the fit and had
previously been misread as a broad, uniform depression of the `m_XX = 100 GeV` row for
both flavors; the filter's real effect is a sharp, localized statistical collapse past
`|cosθ*| ≈ 0.75`, not a uniform shift, and does not survive into a `[0, 0.6]` fit. The migration maps show
the reconstruction handoff (electron→photon, PF muon→DSA muon) as a function of
displacement with the analysis object definitions and no trigger, the truth-to-
reconstruction bridge the note's lepton-jet motivation rests on.
"""),
]

nb = {"cells": cells,
      "metadata": {"kernelspec": {"display_name": "sidm_venv", "language": "python",
                                  "name": "python3"},
                   "language_info": {"name": "python", "version": "3.11"}},
      "nbformat": 4, "nbformat_minor": 5}

out = os.path.join(HERE, "polarization_migration_forAN.ipynb")
with open(out, "w") as f:
    json.dump(nb, f, indent=1, ensure_ascii=False, sort_keys=True)
    f.write("\n")
print(f"wrote {out} ({len(cells)} cells)")
