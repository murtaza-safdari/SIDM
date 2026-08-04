"""Plot helpers for the Run 3 signal-validation notebook: signal LJ-clustering geometry
(Run 3 vs Run 2), the DSA-cross-cleaned mu-LJ multiplicity, trigger efficiency, and high-stat
kinematics. Each function loads a committed result file (produced by run3_lj_eff.py /
trigger_eff_grid.py / run3_kin_hi.py) from this directory and returns a matplotlib Figure, so the
notebook renders without re-reading EOS. Overlay figures mix Run 2 (2018, 13 TeV) and Run 3 (2022,
13.6 TeV), so the energy is stated per era rather than via a single com=."""
import os, json, numpy as np
import matplotlib.pyplot as plt
import mplhep as hep
from scipy.stats import beta

_DIR = os.path.dirname(os.path.abspath(__file__))
RLAB = "13 / 13.6 TeV"    # Run 2 (2018) / Run 3 (2022); eras also labelled in each legend
plt.style.use(hep.style.CMS)

def _load(name):
    return json.load(open(os.path.join(_DIR, name)))

def _cp(k, n, cl=0.68):
    if n == 0:
        return 0.0, 0.0, 0.0
    a = 1 - cl; p = k / n
    lo = 0.0 if k == 0 else beta.ppf(a / 2, k, n - k + 1)
    hi = 1.0 if k == n else beta.ppf(1 - a / 2, k + 1, n - k)
    return p * 100, (p - lo) * 100, (hi - p) * 100

def _series(rs, era, field):
    ys, los, his = [], [], []
    for r in rs:
        d = r[era][field]; p, lo, hi = _cp(d["k"], d["n"]); ys.append(p); los.append(lo); his.append(hi)
    return [r["ctau"] for r in rs], ys, np.array([los, his])

_CHLAB = {"4Mu": r"4$\mu$: $\geq$2 $\mu$-LJ", "2Mu2E": r"2$\mu$2e: $\geq$1 $\mu$-LJ & $\geq$1 gen-matched e$\gamma$-LJ"}

def fig_geometry_efficiency():
    """Loose clustering-geometry efficiency in acceptance vs proper ctau, Run 3 vs Run 2."""
    rows = [r for r in _load("lj_eff_geom_2022.json") if r["run3"] and r["v10"]]
    fig, axes = plt.subplots(1, 2, figsize=(15, 6.7), sharey=True)
    for ax, chan in zip(axes, ["4Mu", "2Mu2E"]):
        rs = sorted([r for r in rows if r["chan"] == chan], key=lambda r: r["ctau"])
        x, y, e = _series(rs, "v10", "incl_acc")
        ax.errorbar(x, y, yerr=e, fmt="o", ms=12, mfc="none", mec="#d62728", mew=2.2, ecolor="#d62728", capsize=3, label="Run 2 (2018 v10), in acc.")
        x, y, e = _series(rs, "run3", "incl_acc")
        ax.errorbar(x, y, yerr=e, fmt="D", ms=8, color="#2ca02c", capsize=3, label="Run 3 (2022), in acc.")
        x, y, e = _series(rs, "run3", "incl_raw")
        ax.errorbar(x, y, yerr=e, fmt="s", ms=6, color="#7f7f7f", alpha=0.7, capsize=2, label="Run 3, full phase space")
        ax.set_xscale("log"); ax.set_xlabel(r"proper $c\tau$  [mm]"); ax.set_ylim(40, 103); ax.grid(alpha=0.25)
        ax.text(0.03, 0.08, _CHLAB[chan], transform=ax.transAxes, fontsize=13, fontweight="bold")
        if chan == "4Mu":
            ax.legend(loc="lower left", fontsize=12, framealpha=0.93, bbox_to_anchor=(0.0, 0.14))
        hep.cms.label("", data=False, rlabel=RLAB, ax=ax)
    axes[0].set_ylabel("LJ-clustering efficiency  [%]")
    fig.suptitle("Run 3 lepton-jet clustering geometry vs Run 2 (generator-driven; loose preselection, not full SR)", y=1.0, fontsize=14)
    fig.tight_layout()
    return fig

def fig_geometry_agreement():
    rows = [r for r in _load("lj_eff_geom_2022.json") if r["run3"] and r["v10"]]
    fig, ax = plt.subplots(figsize=(8.4, 7.8))
    ax.plot([65, 101], [65, 101], "--", color="gray", lw=1.4, zorder=1, label="$y=x$")
    for chan, col, mk in [("4Mu", "#1f77b4", "D"), ("2Mu2E", "#ff7f0e", "o")]:
        rs = [r for r in rows if r["chan"] == chan]
        xv = [_cp(r["v10"]["incl_acc"]["k"], r["v10"]["incl_acc"]["n"]) for r in rs]
        yv = [_cp(r["run3"]["incl_acc"]["k"], r["run3"]["incl_acc"]["n"]) for r in rs]
        ax.errorbar([v[0] for v in xv], [v[0] for v in yv],
                    xerr=np.array([[v[1] for v in xv], [v[2] for v in xv]]),
                    yerr=np.array([[v[1] for v in yv], [v[2] for v in yv]]),
                    fmt=mk, ms=8, color=col, capsize=2, label=chan, zorder=3)
    ax.set_xlabel("Run 2 (v10) efficiency in acc.  [%]", labelpad=8)
    ax.set_ylabel("Run 3 (2022) efficiency in acc.  [%]", labelpad=8)
    ax.set_xlim(65, 101); ax.set_ylim(65, 101); ax.grid(alpha=0.25)
    ax.legend(loc="upper left", fontsize=13, framealpha=0.93)
    hep.cms.label("", data=False, rlabel=RLAB, ax=ax)
    fig.tight_layout()
    return fig

def fig_mu_multiplicity():
    """mu-LJ multiplicity after the full object selection + DSA cross-cleaning, Run 3 vs Run 2.
    Uses the IN-ACCEPTANCE histograms (nmu_hist_acc) so both eras share the same denominator:
    v10 is gen-filtered at production (all events in acceptance) while Run 3 is full phase space,
    so raw histograms would show a spurious ~10pp acceptance gap at the signal bin."""
    rows = [r for r in _load("lj_eff_real_2022.json") if r["run3"] and r["v10"]]
    fig, axes = plt.subplots(1, 2, figsize=(15, 6), sharey=True)
    xm = np.arange(5); lab = ["0", "1", "2", "3", r"$\geq$4"]
    for ax, chan in zip(axes, ["4Mu", "2Mu2E"]):
        rs = [r for r in rows if r["chan"] == chan]
        h2 = np.sum([r["v10"]["nmu_hist_acc"] for r in rs], axis=0); h2 = h2 / h2.sum()
        h3 = np.sum([r["run3"]["nmu_hist_acc"] for r in rs], axis=0); h3 = h3 / h3.sum()
        w = 0.38
        ax.bar(xm - w/2, h2*100, w, color="#d62728", alpha=0.8, label="Run 2 (2018 v10)")
        ax.bar(xm + w/2, h3*100, w, color="#2ca02c", alpha=0.85, label="Run 3 (2022)")
        exp = 2 if chan == "4Mu" else 1
        ax.axvline(exp, ls=":", color="k", alpha=0.5); ax.text(exp + 0.06, 2, f"signal = {exp}", fontsize=11)
        ax.set_xticks(xm); ax.set_xticklabels(lab); ax.set_xlabel(r"$N(\mu$-LJ$)$ per event")
        ax.text(0.5, 0.92, chan, transform=ax.transAxes, fontsize=16, fontweight="bold", ha="center")
        ax.grid(alpha=0.25, axis="y")
        if chan == "4Mu":
            ax.legend(loc="upper right", fontsize=12)
        hep.cms.label("", data=False, rlabel=RLAB, ax=ax)
    axes[0].set_ylabel("in-acceptance events  [%]", labelpad=8)
    fig.suptitle(r"$\mu$-LJ multiplicity (in acceptance) after full object selection + DSA segment-match cross-cleaning: collapses to the signal count in both eras", y=1.0, fontsize=12)
    fig.tight_layout()
    return fig

def fig_trigger():
    """HLT efficiency vs ctau over the full 2022 grid. Small faint markers = 2000-event points
    (stat error ~1%); large markers with Clopper-Pearson error bars = the 100k-event points that
    the quoted mean gains are computed from. The VetoL3 group is drawn to show its genuine,
    ctau-dependent firing (it peaks at the longest lifetimes)."""
    rows = _load("trig_grid_2022.json")
    fig, axes = plt.subplots(1, 2, figsize=(15, 6.5), sharey=True)
    series = [("e2018", "#7f7f7f", "s", "2018-menu L2 paths"),
              ("eOR", "#2ca02c", "D", "full Run 3 displaced-dimuon OR"),
              ("eL3", "#1f77b4", "^", "Run 3 L3 DxyMin subset"),
              ("eVeto", "#9467bd", "v", "Run 3 L2 VetoL3 subset")]
    for ax, chan in zip(axes, ["4Mu", "2Mu2E"]):
        rs = sorted([r for r in rows if r["chan"] == chan], key=lambda r: r["ctau"])
        if not rs:
            continue
        lo = [r for r in rs if r["n"] <= 50000]; hi = [r for r in rs if r["n"] > 50000]
        for key, col, mk, lab in series:
            ax.plot([r["ctau"] for r in lo], [r[key] for r in lo], mk, color=col, ms=4,
                    alpha=0.35, label=f"{lab}" if chan == "2Mu2E" else None)
            if hi:
                ys, los, his = [], [], []
                for r in hi:
                    k = round(r[key] / 100 * r["n"])
                    p, l, h = _cp(k, r["n"]); ys.append(p); los.append(l); his.append(h)
                ax.errorbar([r["ctau"] for r in hi], ys, yerr=np.array([los, his]), fmt=mk,
                            color=col, ms=9, capsize=2, zorder=3)
        ax.set_xscale("log"); ax.set_xlabel(r"proper $c\tau$  [mm]"); ax.set_ylim(0, 105); ax.grid(alpha=0.25)
        ax.text(0.03, 0.06, chan, transform=ax.transAxes, fontsize=20, fontweight="bold")
        if chan == "2Mu2E":
            ax.plot([], [], "o", color="k", ms=9, label="large = 100k-event points (with CP errors)")
            ax.legend(loc="upper right", fontsize=10.5, framealpha=0.92)
        hep.cms.label("", data=False, com=13.6, ax=ax)
    axes[0].set_ylabel("HLT efficiency  [%]")
    fig.suptitle("Run 3 (2022) signal HLT: 2018-path subset vs full Run 3 displaced-dimuon OR (both on Run 3 MC)", y=0.99, fontsize=13)
    fig.tight_layout()
    return fig

def fig_kinematics(chan):
    d = np.load(os.path.join(_DIR, "kin_hi_2022.npz"))
    col = {200: "#1f77b4", 500: "#2ca02c", 800: "#d62728"}
    fig, axes = plt.subplots(1, 2, figsize=(15, 6))
    for mbs in (200, 500, 800):
        axes[0].hist(d[f"{chan}_{mbs}_dR"], bins=np.linspace(0, 0.5, 50), histtype="step", lw=2, density=True, color=col[mbs], label=f"$M_{{B_s}}$={mbs} GeV")
        axes[1].hist(d[f"{chan}_{mbs}_pt"], bins=np.linspace(0, 250, 50), histtype="step", lw=2, density=True, color=col[mbs], label=f"$M_{{B_s}}$={mbs} GeV")
    axes[0].set_xlabel(r"dilepton $\Delta R$ (per dark photon)"); axes[0].set_ylabel("a.u."); axes[0].legend(fontsize=13); axes[0].set_xlim(0, 0.5)
    axes[0].axvline(0.4, ls=":", color="gray"); axes[0].text(0.36, axes[0].get_ylim()[1]*0.75, "LJ $R{=}0.4$", fontsize=11, color="gray", rotation=90)
    axes[1].set_xlabel(r"gen lepton $p_T$  [GeV]"); axes[1].set_ylabel("a.u."); axes[1].legend(fontsize=13)
    hep.cms.label("", data=False, com=13.6, ax=axes[0]); hep.cms.label("", data=False, com=13.6, ax=axes[1])
    fig.suptitle(f"Run 3 (2022) {chan} gen kinematics vs boost ($M_{{Z_d}}$=1.2 GeV, high stats)", y=0.99, fontsize=15)
    fig.tight_layout()
    return fig


# ---------- per-era extension (full-grid campaign complete: all four eras) ----------

ERAS = ["2022", "2022EE", "2023", "2023BPix"]
_ECOL = {"2022": "#1f77b4", "2022EE": "#2ca02c", "2023": "#d62728", "2023BPix": "#9467bd"}

def parse_trigger_log(path):
    """Parse a trigger_eff_grid.py log into row dicts. Handles both the original 10-number rows
    and the canonical format with the hybrid + exclusive-marginal columns appended."""
    import re
    rows = []
    for line in open(path):
        m = re.match(r"^\s*(4Mu|2Mu2E)\s+(.*)", line)
        if not m:
            continue
        nums = re.findall(r"-?\d+\.?\d*", m.group(2))
        if len(nums) < 10:
            continue
        r = dict(chan=m.group(1), mbs=int(nums[0]), mdp=float(nums[1]), ctau=float(nums[2]),
                 n=int(nums[3]), e2018=float(nums[4]), eL3=float(nums[5]), eHi=float(nums[6]),
                 eVeto=float(nums[7]), eOR=float(nums[8]), gain=float(nums[9]))
        if len(nums) >= 15:   # canonical: hybrid + x2018/xL3/xVeto/xHyb
            r.update(eHyb=float(nums[10]), x2018=float(nums[11]), xL3=float(nums[12]),
                     xVeto=float(nums[13]), xHyb=float(nums[14]))
        rows.append(r)
    return rows

def fig_kinematics_eras(mbs=200):
    """Gen kinematics of one grid point overlaid across all four eras: the generated content is
    era-identical (same gridpacks + fragment; only detector conditions differ per era)."""
    fig, axes = plt.subplots(1, 2, figsize=(15, 6))
    for era in ERAS:
        d = np.load(os.path.join(_DIR, f"kin_hi_{era}.npz"))
        axes[0].hist(d[f"4Mu_{mbs}_dR"], bins=np.linspace(0, 0.3, 60), histtype="step", lw=1.8,
                     density=True, color=_ECOL[era], label=era)
        axes[1].hist(d[f"4Mu_{mbs}_pt"], bins=np.linspace(0, 150, 60), histtype="step", lw=1.8,
                     density=True, color=_ECOL[era], label=era)
    axes[0].set_xlabel(r"dilepton $\Delta R$ (per dark photon)"); axes[0].set_ylabel("a.u.")
    axes[1].set_xlabel(r"gen lepton $p_T$  [GeV]"); axes[1].set_ylabel("a.u.")
    axes[0].legend(fontsize=12, title=f"4Mu, $M_{{B_s}}$={mbs} GeV"); axes[1].legend(fontsize=12)
    hep.cms.label("", data=False, com=13.6, ax=axes[0]); hep.cms.label("", data=False, com=13.6, ax=axes[1])
    fig.suptitle("Generated kinematics are era-identical (curves coincide)", y=0.99, fontsize=14)
    fig.tight_layout()
    return fig

def fig_geometry_eras():
    """Per-era LJ-clustering geometry agreement: mean Run3-v10 in-acceptance efficiency difference
    per channel, with the max |difference| across grid points as the whisker."""
    fig, ax = plt.subplots(figsize=(10, 6))
    xs = np.arange(len(ERAS))
    for off, (chan, col, mk) in zip((-0.12, 0.12), [("4Mu", "#1f77b4", "D"), ("2Mu2E", "#ff7f0e", "o")]):
        means, maxs = [], []
        for era in ERAS:
            rows = [r for r in _load(f"lj_eff_geom_{era}.json") if r["run3"] and r["v10"]]
            d = np.array([100 * (r["run3"]["incl_acc"]["p"] - r["v10"]["incl_acc"]["p"])
                          for r in rows if r["chan"] == chan])
            means.append(d.mean()); maxs.append(np.abs(d).max())
        ax.errorbar(xs + off, means, yerr=maxs, fmt=mk, ms=9, color=col, capsize=4, label=chan)
    ax.axhline(0, color="gray", lw=1)
    ax.set_xticks(xs); ax.set_xticklabels(ERAS)
    ax.set_ylabel(r"Run 3 $-$ Run 2 eff. in acc.  [%]", labelpad=8)
    ax.set_ylim(-2, 2); ax.grid(alpha=0.25, axis="y")
    ax.legend(fontsize=13); ax.text(0.02, 0.04, "whiskers = max |difference| over grid points",
                                    transform=ax.transAxes, fontsize=11, color="gray")
    hep.cms.label("", data=False, rlabel=RLAB, ax=ax)
    fig.tight_layout()
    return fig
