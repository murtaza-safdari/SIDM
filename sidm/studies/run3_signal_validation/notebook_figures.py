#!/usr/bin/env python3
"""AN versions (vector PDF, CMS Simulation label) of the three gen-level draft figures of
run3_signal_validation.ipynb, computed with the same helpers and inputs as the notebook cells:

  run3_ctau_representative.pdf  measured vs nominal proper ctau, six 2022 points (notebook section 2)
  run3_ctau_grid.pdf            measured/nominal proper ctau over the 180-point GEN scan (section 4)
  run3_vs_2018_gen.pdf          2022 vs 2018 v10 at 4mu 200 GeV / 1.2 GeV / 4.8 mm (section 6)

run3_study_plots has no functions for these and no saved arrays exist, so the inputs are recomputed
(V.analyze over EOS, V.load_grid_scan) and cached in notebook_figures.npz; --replot redraws from it.
The 180-point scan is a standalone LHE+GEN job (500 events per point) with 2022 conditions, not a
read of the produced samples (SampleFactory sidm_run3_production/gen_validate_job.sh).
Run from the worktree root with the notebook venv.
"""
import argparse, os, sys
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _DIR)
sys.path.insert(0, os.path.join(_DIR, "..", "truth_kinematics_forAN"))
import run3_signal_validation as V
import an_style

an_style.FIG_DIR = os.path.join(_DIR, "figures")
NPZ = os.path.join(_DIR, "notebook_figures.npz")
POINTS = [("4Mu_100GeV_0p25GeV_2p0mm", "signal_4mu_run3.yaml"),     # notebook section 2
          ("4Mu_200GeV_1p2GeV_4p8mm", "signal_4mu_run3.yaml"),
          ("4Mu_1000GeV_5p0GeV_4p0mm", "signal_4mu_run3.yaml"),
          ("2Mu2E_150GeV_0p25GeV_1p3mm", "signal_2mu2e_run3.yaml"),
          ("2Mu2E_500GeV_5p0GeV_8p0mm", "signal_2mu2e_run3.yaml"),
          ("2Mu2E_1000GeV_1p2GeV_0p96mm", "signal_2mu2e_run3.yaml")]
CMP = "4Mu_200GeV_1p2GeV_4p8mm"
NDP_SCAN = 1000                      # dark photons per scan point (500 events x 2), all 180 RESULT lines
LEG = dict(frameon=True, facecolor="white", edgecolor="none", framealpha=0.92)


def compute():
    D = {}
    for s, cfg in POINTS:                                     # nfiles=4, as the notebook
        r = V.analyze(s, "2022", cfg)
        p = r["proper_mm"]
        D[f"rep_{s}"] = np.array([r["nom"], p.mean(), p.std() / np.sqrt(len(p)), len(p)])
        print(f"{s:30s} nom {r['nom']:g} meas {p.mean():.3f} +- {p.std() / np.sqrt(len(p)):.3f} (N={len(p)})")
    rows = V.load_grid_scan()
    D["grid"] = np.array([x["meas_over_nom"] for x in rows])
    D["grid_chan"] = np.array([x["chan"] for x in rows])
    r18 = V.analyze(CMP, "llpNanoAOD_v2", "signal_4mu_v10.yaml", nfiles=2)
    r22 = V.analyze(CMP, "2022", "signal_4mu_run3.yaml", nfiles=4)
    r22c = V.analyze(CMP, "2022", "signal_4mu_run3.yaml", nfiles=4, lep_abseta_max=2.4, lep_pt_min=1.0)
    for tag, r in (("r18", r18), ("r22", r22), ("r22c", r22c)):
        for k in ("dpmass", "proper_mm", "lep_abseta", "lep_pt", "dR"):
            D[f"{tag}_{k}"] = r[k]
    np.savez_compressed(NPZ, **D)
    print(f"wrote {NPZ}")
    return D


def fig_ctau_representative(D):
    fig, ax = plt.subplots(figsize=(10, 9), layout="constrained")
    lo, hi = 0.5, 12
    ax.plot([lo, hi], [lo, hi], "k--", alpha=0.6, label="measured = nominal")
    for chan, mk, col, lab in (("4Mu", "o", "C0", r"4$\mu$"), ("2Mu2E", "s", "C1", r"2$\mu$2e")):
        v = np.array([D[f"rep_{s}"] for s, _ in POINTS if s.startswith(chan + "_")])
        ax.errorbar(v[:, 0], v[:, 1], yerr=v[:, 2], fmt=mk, color=col, ms=11, capsize=3, label=lab, zorder=3)
    ax.set_xscale("log"); ax.set_yscale("log"); ax.set_xlim(lo, hi); ax.set_ylim(lo, hi)
    ax.set_xlabel(r"nominal proper $c\tau$  [mm]"); ax.set_ylabel(r"measured proper $c\tau$  [mm]")
    ax.legend(loc="upper left", fontsize=16, title="2022, six grid points", title_fontsize=16, **LEG)
    an_style.cms_sim_labels([ax], com=13.6)
    return an_style.save(fig, "run3_ctau_representative")


def fig_ctau_grid(D):
    fig, ax = plt.subplots(figsize=(10, 7.5), layout="constrained")
    g = D["grid"]
    edges = np.linspace(0.85, 1.15, 31)
    ax.hist(g, bins=edges, histtype="stepfilled", color="C0", alpha=0.35)
    ax.hist(g, bins=edges, histtype="step", color="C0", lw=2,
            label=f"{len(g)} grid points (mean {g.mean():.3f}, std {g.std():.3f})")
    x = np.linspace(edges[0], edges[-1], 300); sig = 1 / np.sqrt(NDP_SCAN)
    ax.plot(x, len(g) * np.diff(edges)[0] * np.exp(-0.5 * ((x - 1) / sig) ** 2) / (sig * np.sqrt(2 * np.pi)),
            "k--", lw=1.8, label=rf"statistical expectation ($\sigma$ = 1/$\sqrt{{{NDP_SCAN}}}$)")
    ax.set_xlabel(r"measured / nominal proper $c\tau$"); ax.set_ylabel("grid points / 0.01")
    ax.set_xlim(edges[0], edges[-1]); ax.set_ylim(0, ax.get_ylim()[1] * 1.45)
    ax.legend(loc="upper left", fontsize=15, title=r"GEN, 500 events per point, 2022 conditions",
              title_fontsize=15, **LEG)
    an_style.cms_sim_labels([ax], com=13.6)
    return an_style.save(fig, "run3_ctau_grid")


def fig_vs_2018(D):
    fig, axs = plt.subplots(3, 2, figsize=(15, 17), layout="constrained")
    fig.get_layout_engine().set(hspace=0.1)   # room for the CMS labels, placed after the layout freezes
    axs = axs.ravel()
    panels = [("dpmass", np.linspace(1.0, 1.4, 50), "gen dark-photon mass  [GeV]", False, False),
              ("proper_mm", np.logspace(np.log10(0.05), 2, 40), r"proper $c\tau$  [mm]", True, False),
              ("lep_abseta", np.linspace(0, 4, 40), r"gen lepton $|\eta|$", False, True),
              ("lep_pt", np.linspace(0, 150, 40), r"gen lepton $p_{T}$  [GeV]", False, True),
              ("dR", np.linspace(0, 0.2, 40), r"gen lepton-pair $\Delta R$", False, True)]
    for ax, (k, bins, xl, logx, cut) in zip(axs, panels):
        ax.hist(D[f"r18_{k}"], bins=bins, density=True, histtype="step", lw=2.2, color="k", label="2018 (13 TeV)")
        ax.hist(D[f"r22_{k}"], bins=bins, density=True, histtype="step", lw=2.2, color="#d62728",
                label="2022 (13.6 TeV)")
        if cut:
            ax.hist(D[f"r22c_{k}"], bins=bins, density=True, histtype="step", lw=2.2, color="#1f77b4", ls="--",
                    label="2022, 2018 acceptance")
        if logx:
            ax.set_xscale("log")
        ax.set_xlabel(xl); ax.set_ylabel("a.u.")
        ax.set_ylim(0, ax.get_ylim()[1] * 1.15)
    axs[5].axis("off")
    h, l = axs[2].get_legend_handles_labels()
    axs[5].legend(h, l, loc="center", fontsize=19,
                  title=r"4$\mu$, $m_{\chi\bar{\chi}}$ = 200 GeV" "\n"
                        r"$m_{Z_d}$ = 1.2 GeV, $c\tau$ = 4.8 mm" "\n"
                        r"2018 acceptance: $|\eta|$ < 2.4, $p_{T}$ > 1 GeV",
                  title_fontsize=19, **LEG)
    an_style.cms_sim_labels(list(axs[:5]), com="13 / 13.6 TeV")
    return an_style.save(fig, "run3_vs_2018_gen")


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--replot", action="store_true", help=f"redraw from {os.path.basename(NPZ)}, no EOS")
    args = ap.parse_args()
    an_style.set_style()
    D = dict(np.load(NPZ, allow_pickle=True)) if args.replot else compute()
    fig_ctau_representative(D); fig_ctau_grid(D); fig_vs_2018(D)


if __name__ == "__main__":
    main()
