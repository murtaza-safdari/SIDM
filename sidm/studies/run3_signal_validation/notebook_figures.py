#!/usr/bin/env python3
"""AN versions (vector PDF, CMS Simulation label) of the three gen-level draft figures of
run3_signal_validation.ipynb, computed with the same helpers and inputs as the notebook cells:

  run3_ctau_representative.pdf  measured vs nominal proper ctau, six 2022 points (notebook section 2)
  run3_ctau_grid.pdf            measured/nominal proper ctau over the 180-point GEN scan (section 4)
  run3_vs_2018_gen.pdf          2022 vs 2018 v10 at 4mu 200 GeV / 1.2 GeV / 4.8 mm (section 6), with
                                the 2018 generator-level requirement emulated per event on 2022

run3_study_plots has no functions for these and no saved arrays exist, so the inputs are recomputed
(V.analyze over EOS, V.load_grid_scan) and cached in notebook_figures.npz; --replot redraws from it.
The 180-point scan is a standalone LHE+GEN job (500 events per point) with 2022 conditions, not a
read of the produced samples (SampleFactory sidm_run3_production/gen_validate_job.sh).
Run from the worktree root with the notebook venv.
"""
import argparse, os, sys
import numpy as np
import awkward as ak
import uproot
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
LEG = dict(frameon=True, facecolor="white", edgecolor="none", framealpha=0.92)
# 2018 v10 generator-level requirement, per event: all four dark-photon leptons pass
REQ_PT, REQ_ETA, REQ_VXY, REQ_VZ = 5.0, 2.4, 740.0, 960.0      # GeV, -, cm, cm
N22, N18 = 24, 10                                               # files read for the comparison
GBR = ["GenPart_pdgId", "GenPart_genPartIdxMother", "GenPart_pt", "GenPart_eta", "GenPart_phi",
       "GenPart_mass", "GenPart_vx", "GenPart_vy", "GenPart_vz"]


def gen_events(sample, era, cfg, nfiles):
    """Per dark photon (mass, pT, proper ctau, dilepton dR) and per lepton (pT, |eta|) from the
    GenPart table, each tagged with whether its event passes the 2018 requirement (all four
    dark-photon leptons pT > 5 GeV, |eta| < 2.4, decay vertex V_xy < 740 cm, |V_z| < 960 cm)."""
    dp = {k: [] for k in ("mass", "pt", "proper_mm", "dR", "req")}
    lp = {k: [] for k in ("pt", "abseta", "req")}
    nev = npass = 0
    for u in V._fileset_files(sample, era, cfg, nfiles):
        a = uproot.open(u)["Events"].arrays(GBR)
        for ev in ak.to_list(a):
            pid, mom = ev["GenPart_pdgId"], ev["GenPart_genPartIdxMother"]
            pt, eta, phi = ev["GenPart_pt"], ev["GenPart_eta"], ev["GenPart_phi"]
            vx, vy, vz = ev["GenPart_vx"], ev["GenPart_vy"], ev["GenPart_vz"]
            daus = {}
            for i, (q, m) in enumerate(zip(pid, mom)):
                if abs(q) in (11, 13) and m >= 0 and abs(pid[m]) == 32:
                    daus.setdefault(m, []).append(i)
            leps = [i for d in daus.values() for i in d]
            ok = (len(daus) == 2 and len(leps) == 4 and all(
                pt[i] > REQ_PT and abs(eta[i]) < REQ_ETA and np.hypot(vx[i], vy[i]) < REQ_VXY
                and abs(vz[i]) < REQ_VZ for i in leps))
            nev += 1; npass += ok
            for i in leps:
                lp["pt"].append(pt[i]); lp["abseta"].append(abs(eta[i])); lp["req"].append(ok)
            for m, d in daus.items():
                mass = ev["GenPart_mass"][m]
                bg = pt[m] * np.cosh(eta[m]) / mass if mass > 0 else 0
                dl = np.sqrt((vx[d[0]] - vx[m]) ** 2 + (vy[d[0]] - vy[m]) ** 2 + (vz[d[0]] - vz[m]) ** 2)
                dr = np.nan
                if len(d) == 2:
                    dphi = (phi[d[0]] - phi[d[1]] + np.pi) % (2 * np.pi) - np.pi
                    dr = np.hypot(eta[d[0]] - eta[d[1]], dphi)
                dp["mass"].append(mass); dp["pt"].append(pt[m]); dp["req"].append(ok)
                dp["proper_mm"].append(dl / bg * 10.0 if bg > 0 else np.nan); dp["dR"].append(dr)
    out = {f"dp_{k}": np.array(v) for k, v in dp.items()}
    out.update({f"lep_{k}": np.array(v) for k, v in lp.items()})
    out["nev"], out["npass"] = nev, npass
    return out


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
    D["grid_ndecays"] = np.array([x["n_decays"] for x in rows])
    assert len({x["name"] for x in rows}) == 180, "expected 180 unique scan points"
    for tag, era, cfg, nf in (("r18", "llpNanoAOD_v2", "signal_4mu_v10.yaml", N18),
                              ("r22", "2022", "signal_4mu_run3.yaml", N22)):
        for k, v in gen_events(CMP, era, cfg, nf).items():
            D[f"{tag}_{k}"] = v
        print(f"{tag}: {D[f'{tag}_nev']} events, {D[f'{tag}_npass']} pass the 2018 requirement")
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
            label=rf"{len(g)} grid points: mean {g.mean():.3f} $\pm$ {g.std() / np.sqrt(len(g)):.3f}, std {g.std():.3f}")
    ndp = int(np.median(D["grid_ndecays"]))          # decays per point, from the RESULT lines
    x = np.linspace(edges[0], edges[-1], 300); sig = 1 / np.sqrt(ndp)
    ax.plot(x, len(g) * np.diff(edges)[0] * np.exp(-0.5 * ((x - 1) / sig) ** 2) / (sig * np.sqrt(2 * np.pi)),
            "k--", lw=1.8, label=rf"statistical expectation ($\sigma$ = 1/$\sqrt{{{ndp}}}$)")
    ax.set_xlabel(r"measured / nominal proper $c\tau$"); ax.set_ylabel("grid points / 0.01")
    ax.set_xlim(edges[0], edges[-1]); ax.set_ylim(0, ax.get_ylim()[1] * 1.45)
    ax.legend(loc="upper left", fontsize=15, title=r"GEN, 500 events per point, 2022 conditions",
              title_fontsize=15, **LEG)
    an_style.cms_sim_labels([ax], com=13.6)
    return an_style.save(fig, "run3_ctau_grid")


def _sel(D, tag, kind, k, req=False):
    x = D[f"{tag}_{kind}_{k}"]
    return x[D[f"{tag}_{kind}_req"].astype(bool)] if req else x


def fig_vs_2018(D):
    fig, axs = plt.subplots(3, 2, figsize=(15, 17), layout="constrained")
    fig.get_layout_engine().set(hspace=0.1)   # room for the CMS labels, placed after the layout freezes
    axs = axs.ravel()
    panels = [("dp", "mass", np.linspace(1.0, 1.4, 50), "gen dark-photon mass  [GeV]", False),
              ("dp", "proper_mm", np.logspace(np.log10(0.05), 2, 40), r"proper $c\tau$  [mm]", True),
              ("lep", "abseta", np.linspace(0, 4, 40), r"gen lepton $|\eta|$", False),
              ("lep", "pt", np.linspace(0, 150, 40), r"gen lepton $p_{T}$  [GeV]", False),
              ("dp", "dR", np.linspace(0, 0.2, 40), r"gen lepton-pair $\Delta R$", False)]
    for ax, (kind, k, bins, xl, logx) in zip(axs, panels):
        ax.hist(_sel(D, "r18", kind, k), bins=bins, density=True, histtype="step", lw=2.2, color="k",
                label="2018 (13 TeV)")
        ax.hist(_sel(D, "r22", kind, k), bins=bins, density=True, histtype="step", lw=2.2, color="#d62728",
                label="2022 (13.6 TeV)")
        ax.hist(_sel(D, "r22", kind, k, req=True), bins=bins, density=True, histtype="step", lw=2.2,
                color="#1f77b4", ls="--", label="2022, 2018 requirement")
        if logx:
            ax.set_xscale("log")
        ax.set_xlabel(xl); ax.set_ylabel("a.u.")
        ax.set_ylim(0, ax.get_ylim()[1] * 1.15)
    axs[5].axis("off")
    h, l = axs[2].get_legend_handles_labels()
    axs[5].legend(h, l, loc="center", fontsize=19,
                  title=r"4$\mu$, $m_{\chi\bar{\chi}}$ = 200 GeV" "\n"
                        r"$m_{Z_d}$ = 1.2 GeV, $c\tau$ = 4.8 mm" "\n"
                        "2018 requirement: all four leptons\n"
                        r"$|\eta|$ < 2.4, $p_{T}$ > 5 GeV",
                  title_fontsize=19, **LEG)
    an_style.cms_sim_labels(list(axs[:5]), com="13 / 13.6 TeV")
    return an_style.save(fig, "run3_vs_2018_gen")


def summary(D):
    """Medians/means of the compared quantities for 2018, Run 3 raw and Run 3 with the requirement."""
    rows = [("2018", "r18", False), ("2022 raw", "r22", False), ("2022 + 2018 req.", "r22", True)]
    print(f"2022 events passing the 2018 requirement: {D['r22_npass']}/{D['r22_nev']} = "
          f"{D['r22_npass'] / D['r22_nev']:.4f}; 2018 events passing it: {D['r18_npass']}/{D['r18_nev']} = "
          f"{D['r18_npass'] / D['r18_nev']:.4f}")
    for kind, k in (("lep", "pt"), ("lep", "abseta"), ("dp", "dR"), ("dp", "pt"), ("dp", "mass")):
        print(f"{kind}_{k:8s}", "  ".join(
            f"{lab}: median {np.nanmedian(_sel(D, t, kind, k, r)):.4g} mean {np.nanmean(_sel(D, t, kind, k, r)):.4g}"
            for lab, t, r in rows))
    for lab, t, r in rows:
        x = _sel(D, t, "dp", "proper_mm", r); x = x[np.isfinite(x)]
        print(f"proper ctau {lab}: mean {x.mean():.4f} +- {x.std() / np.sqrt(len(x)):.4f} mm (N={len(x)})")
    for kind, k in (("dp", "pt"), ("lep", "pt")):
        m18 = np.nanmean(_sel(D, "r18", kind, k))
        print(f"{kind} pT mean, 2022 / 2018 - 1: raw {np.nanmean(_sel(D, 'r22', kind, k)) / m18 - 1:+.4f}, "
              f"with requirement {np.nanmean(_sel(D, 'r22', kind, k, True)) / m18 - 1:+.4f}")


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--replot", action="store_true", help=f"redraw from {os.path.basename(NPZ)}, no EOS")
    args = ap.parse_args()
    an_style.set_style()
    D = dict(np.load(NPZ, allow_pickle=True)) if args.replot else compute()
    summary(D)
    fig_ctau_representative(D); fig_ctau_grid(D); fig_vs_2018(D)


if __name__ == "__main__":
    main()
