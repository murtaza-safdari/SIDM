#!/usr/bin/env python3
# High-stat gen kinematics across the reduced 2022 grid, using the committed analyze() helper.
# Representative points at fixed M_Zd=1.2 GeV, M_Bs in {200,500,800} -> shows the boost trend:
# dilepton dR (collimation) shrinks and lepton pT hardens as M_Bs (hence Z_d boost) rises.
# Saves the arrays to an npz (so the notebook cell loads + plots reproducibly, no re-read of EOS)
# plus PNG overlays per channel. Run from the SIDM worktree root with the venv.
import sys, os, numpy as np, matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import mplhep as hep
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))   # repo-relative: works in any checkout
import run3_signal_validation as V
plt.style.use(hep.style.CMS)
RO = os.environ.get("RUN3_EFF_OUT", "/uscms_data/d3/murtazas/review_out")
NF = 12

POINTS = {
    "4Mu":   [("4Mu_200GeV_1p2GeV_4p8mm", 200), ("4Mu_500GeV_1p2GeV_1p9mm", 500), ("4Mu_800GeV_1p2GeV_1p2mm", 800)],
    "2Mu2E": [("2Mu2E_200GeV_1p2GeV_4p8mm", 200), ("2Mu2E_500GeV_1p2GeV_1p9mm", 500), ("2Mu2E_800GeV_1p2GeV_1p2mm", 800)],
}
COL = {200: "#1f77b4", 500: "#2ca02c", 800: "#d62728"}
NPZ = f"{RO}/kin_hi_2022.npz"
if os.path.exists(NPZ) and "--replot" in sys.argv:
    store = dict(np.load(NPZ)); print("loaded arrays from npz (replot only)")
else:
    store = {}
    for chan, pts in POINTS.items():
        cfg = f"signal_{chan.lower()}_run3.yaml"
        for sample, mbs in pts:
            r = V.analyze(sample, "2022", cfg, nfiles=NF)
            store[f"{chan}_{mbs}_dR"] = r["dR"]; store[f"{chan}_{mbs}_pt"] = r["lep_pt"]
            store[f"{chan}_{mbs}_mass"] = r["dpmass"]
            print(f"{sample:26s} N_dp_pairs={len(r['dR']):6d}  <dR>={np.mean(r['dR']):.3f}  "
                  f"<lep pT>={np.median(r['lep_pt']):.1f} GeV  <dpmass>={np.median(r['dpmass']):.3f}")
    np.savez_compressed(NPZ, **store)
    print("wrote kin_hi_2022.npz")

for chan in ["4Mu", "2Mu2E"]:
    fig, axes = plt.subplots(1, 2, figsize=(15, 6))
    for _, mbs in POINTS[chan]:
        dR = store[f"{chan}_{mbs}_dR"]; pt = store[f"{chan}_{mbs}_pt"]
        axes[0].hist(dR, bins=np.linspace(0, 0.5, 50), histtype="step", lw=2, density=True,
                     color=COL[mbs], label=f"$M_{{B_s}}$={mbs} GeV")
        axes[1].hist(pt, bins=np.linspace(0, 250, 50), histtype="step", lw=2, density=True,
                     color=COL[mbs], label=f"$M_{{B_s}}$={mbs} GeV")
    axes[0].set_xlabel(r"dilepton $\Delta R$ (per dark photon)"); axes[0].set_ylabel("a.u."); axes[0].legend(fontsize=13)
    axes[0].set_xlim(0, 0.5)
    axes[0].axvline(0.4, ls=":", color="gray"); axes[0].text(0.36, axes[0].get_ylim()[1]*0.75, "LJ $R{=}0.4$", fontsize=11, color="gray", rotation=90)
    axes[1].set_xlabel(r"gen lepton $p_T$  [GeV]"); axes[1].set_ylabel("a.u."); axes[1].legend(fontsize=13)
    hep.cms.label("", data=False, com=13.6, ax=axes[0]); hep.cms.label("", data=False, com=13.6, ax=axes[1])
    fig.suptitle(f"Run 3 (2022) {chan} gen kinematics vs boost ($M_{{Z_d}}$=1.2 GeV, high stats)", y=0.99, fontsize=15)
    fig.tight_layout(); fig.savefig(f"{RO}/kin_hi_{chan}_2022.png", dpi=110, bbox_inches="tight")
    print(f"wrote kin_hi_{chan}")
