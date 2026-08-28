#!/usr/bin/env python3
"""Analysis of the LJ jet-sum isolation MC evaluation run (run ON LPC, in the venv).

Produces /uscms_data/d3/murtazas/ljiso_study/metrics.json and PNGs under plots/.
"""
import json
import os
import sys

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import coffea.util

OUT = "/uscms_data/d3/murtazas/ljiso_study"
PLOTS = os.path.join(OUT, "plots")
os.makedirs(PLOTS, exist_ok=True)

# categorical hues in fixed slot order (dataviz reference palette, light mode);
# every multi-series plot also carries a distinct linestyle so identity never
# rests on color alone.
C = {"old": "#2a78d6", "sum": "#eb6834", "sum_soft": "#1baf7a",
     "soft": "#eda100", "extra": "#e87ba4"}
LS = {"old": "-", "sum": "--", "sum_soft": "-.", "soft": ":"}
LBL = {"old": "matched jet (current)", "sum": "jet-cone sum",
       "sum_soft": "jet + soft-jet cone sum", "soft": "soft-jet cone sum"}

VARS = ["old", "sum", "sum_soft"]
HIST_WIDE = {"old": "isolation_matched_jet", "sum": "isolation_sum",
             "sum_soft": "isolation_sum_soft", "soft": "isolation_soft"}
HIST_ZOOM = {k: v + "_zoom" for k, v in HIST_WIDE.items()}

CTAU_SHORT = ["0p25GeV_0p004mm", "1p2GeV_0p019mm", "5p0GeV_0p08mm"]
CTAU_LONG = ["0p25GeV_4p0mm", "1p2GeV_19p0mm", "5p0GeV_80p0mm"]
LONGEST = "5p0GeV_80p0mm"
BKG = ["QCD_Pt80To120", "QCD_Pt300To470", "QCD_Pt1000", "DYJetsToMuMu_M50", "TTJets"]
CHANNELS = ["base_ljObjCut", "4mu", "2mu2e"]
VIEWS = ["mu_lj", "egm_lj", "lj"]

warnings = []

out = coffea.util.load(os.path.join(OUT, "ljiso_eval.coffea"))
res = out["out"]
print("samples in output:", sorted(res.keys()))


def sig_names(sig_chan, tags):
    return ["%s_500GeV_%s" % (sig_chan, t) for t in tags]


def get(sample, hname, channel):
    """Return the flow-inclusive value array of hist `hname` for one sample+channel."""
    if sample not in res:
        warnings.append("sample %s missing from output" % sample)
        return None
    hists = res[sample]["hists"]
    if hname not in hists:
        warnings.append("hist %s missing for %s" % (hname, sample))
        return None
    h = hists[hname]
    if channel not in list(h.axes["channel"]):
        warnings.append("channel %s missing in %s/%s" % (channel, sample, hname))
        return None
    return h[{"channel": channel}].values(flow=True)


def stack(samples, hname, channel):
    """Sum the flow-inclusive values of `hname` over a list of samples."""
    tot = None
    for s in samples:
        v = get(s, hname, channel)
        if v is None:
            continue
        tot = v.astype(np.float64) if tot is None else tot + v
    return tot


# ------------------------------------------------------------------ coverage
def coverage(samples, channel, view):
    """P(n_cone_jets>=1), P(n_cone_jets+n_cone_softjets>=1), and the soft-only gain."""
    v = stack(samples, "%s_n_cone_jets_vs_softjets" % view, channel)
    if v is None:
        return None
    # axes: Regular(11, -0.5, 10.5) x same, with flow -> index 0 = underflow (empty),
    # 1..11 = counts 0..10, 12 = overflow (count >= 11)
    n = v.sum()
    if n == 0:
        return {"n_lj": 0.0}
    njet0 = v[1, :].sum()                      # n_cone_jets == 0
    p_jet = 1.0 - njet0 / n
    both0 = v[1, 1]                            # no jet and no soft jet
    p_any = 1.0 - both0 / n
    gain = (njet0 - both0) / n                 # n_cone_jets==0 & n_cone_softjets>=1
    p_soft = 1.0 - v[:, 1].sum() / n
    return {"n_lj": float(n), "P_njet_ge1": float(p_jet),
            "P_nsoft_ge1": float(p_soft), "P_any_ge1": float(p_any),
            "gain_soft_only": float(gain),
            "max_njet": int(np.max(np.nonzero(v.sum(axis=1))[0]) - 1),
            "max_nsoft": int(np.max(np.nonzero(v.sum(axis=0))[0]) - 1)}


# ---------------------------------------------------------------- efficiency
def eff_curve(samples, view, var, channel):
    """Cumulative efficiency of the cut `var < t`, on a merged zoom+wide threshold grid.

    Values below zero (possible only for the matched-jet isolation, whose lepton
    fraction can exceed 1) sit in the underflow bin and are counted as passing.
    """
    z = stack(samples, "%s_%s" % (view, HIST_ZOOM[var]), channel)
    w = stack(samples, "%s_%s" % (view, HIST_WIDE[var]), channel)
    if z is None or w is None:
        return None
    ntot = z.sum()
    if ntot == 0:
        return None
    if abs(ntot - w.sum()) > 1e-6:
        warnings.append("zoom/wide totals disagree for %s %s %s: %g vs %g"
                        % (view, var, channel, ntot, w.sum()))
    ez = np.linspace(0, 0.3, 61)
    ew = np.linspace(0, 3, 61)
    thr, cum = [], []
    for k, t in enumerate(ez):
        thr.append(t)
        cum.append(z[0] + z[1:1 + k].sum())
    for k, t in enumerate(ew):
        if t <= 0.3 + 1e-9:
            continue
        thr.append(t)
        cum.append(w[0] + w[1:1 + k].sum())
    thr = np.array(thr)
    eff = np.array(cum, dtype=np.float64) / ntot
    # continuity check where the two binnings meet
    k = int(round(0.3 / 0.05))
    seam = abs((w[0] + w[1:1 + k].sum()) / ntot - eff[len(ez) - 1])
    return {"thr": thr, "eff": eff, "n": float(ntot), "seam_mismatch": float(seam)}


def bkg_at_sig(sig, bkg, targets=(0.7, 0.8, 0.9)):
    """Background efficiency at fixed signal efficiency, along the shared threshold grid."""
    outd = {}
    se, be = sig["eff"], bkg["eff"]
    for t in targets:
        key = "%.4f" % t
        if se[-1] < t:
            outd[key] = None
            continue
        i = int(np.searchsorted(se, t))
        if i == 0:
            outd[key] = {"bkg_eff": float(be[0]), "threshold": float(sig["thr"][0]),
                         "sig_eff": float(se[0])}
            continue
        f = 0.0 if se[i] == se[i - 1] else (t - se[i - 1]) / (se[i] - se[i - 1])
        outd[key] = {
            "bkg_eff": float(be[i - 1] + f * (be[i] - be[i - 1])),
            "threshold": float(sig["thr"][i - 1] + f * (sig["thr"][i] - sig["thr"][i - 1])),
            "sig_eff": float(t)}
    return outd


def eff_at(curve, t):
    return float(np.interp(t, curve["thr"], curve["eff"]))


# ------------------------------------------------------------------ sentinel
def sentinel(samples, view, channel):
    """Distribution of the new variable for LJs in the old-iso first bin [0, 0.005)."""
    v = stack(samples, "%s_iso_old_vs_new" % view, channel)
    if v is None:
        return None
    row = v[1, :]                 # old-iso bin 0 = [0, 0.005)
    n = row.sum()
    if n == 0:
        return {"n_sentinel": 0.0}
    # new axis Regular(60, 0, 3) with flow: index 0 = underflow, 1..60 = bins, 61 = overflow
    def frac_above(x):
        # new-axis bin j (0-based) covers [0.05*j, 0.05*(j+1)); row[0] is the underflow
        k = int(round(x / 0.05))
        return float(row[1 + k:].sum() / n)
    return {"n_sentinel": float(n), "n_all": float(v.sum()),
            "sentinel_fraction_of_all": float(n / v.sum()),
            "frac_new_gt_0p05": frac_above(0.05),
            "frac_new_gt_0p2": frac_above(0.2),
            "frac_new_eq_0": float(row[1] / n)}


# ===================================================================== driver
metrics = {"coverage": {}, "sentinel": {}, "roc": {}, "operating_points": {},
           "n_lj": {}, "warnings": warnings}

GROUPS = {}
for ch, sc in (("4mu", "4Mu"), ("2mu2e", "2Mu2E")):
    GROUPS[ch] = {
        "signal_all": sig_names(sc, CTAU_SHORT + CTAU_LONG),
        "signal_short": sig_names(sc, CTAU_SHORT),
        "signal_long": sig_names(sc, CTAU_LONG),
        "signal_longest_%s" % LONGEST: sig_names(sc, [LONGEST]),
    }
    for b in BKG:
        GROUPS[ch][b] = [b]
    GROUPS[ch]["bkg_all"] = list(BKG)
# base_ljObjCut: everything, both signal channels merged for the signal groups
GROUPS["base_ljObjCut"] = {
    "signal_all": sig_names("4Mu", CTAU_SHORT + CTAU_LONG) + sig_names("2Mu2E", CTAU_SHORT + CTAU_LONG),
    "signal_short": sig_names("4Mu", CTAU_SHORT) + sig_names("2Mu2E", CTAU_SHORT),
    "signal_long": sig_names("4Mu", CTAU_LONG) + sig_names("2Mu2E", CTAU_LONG),
    "bkg_all": list(BKG),
}
for b in BKG:
    GROUPS["base_ljObjCut"][b] = [b]
# per-point entries for the coverage table
for ch, sc in (("4mu", "4Mu"), ("2mu2e", "2Mu2E")):
    for t in CTAU_SHORT + CTAU_LONG:
        GROUPS[ch]["point_%s" % t] = sig_names(sc, [t])

for ch in CHANNELS:
    metrics["coverage"][ch] = {}
    metrics["sentinel"][ch] = {}
    metrics["n_lj"][ch] = {}
    for g, samples in GROUPS[ch].items():
        metrics["coverage"][ch][g] = {}
        metrics["sentinel"][ch][g] = {}
        for view in VIEWS:
            c = coverage(samples, ch, view)
            if c is None or c.get("n_lj", 0) == 0:
                metrics["coverage"][ch][g][view] = {"n_lj": 0}
                warnings.append("ZERO LJs: channel=%s group=%s view=%s" % (ch, g, view))
                continue
            metrics["coverage"][ch][g][view] = c
        for view in ["mu_lj", "egm_lj"]:
            s = sentinel(samples, view, ch)
            metrics["sentinel"][ch][g][view] = s
            if s is None or s.get("n_sentinel", 0) == 0:
                warnings.append("ZERO sentinel LJs: channel=%s group=%s view=%s" % (ch, g, view))

# -------- ROC
CUR_CUT = {"mu_lj": 0.1, "egm_lj": 0.2, "lj": 0.1}
roc_pairs = [("4mu", "mu_lj"), ("2mu2e", "mu_lj"), ("2mu2e", "egm_lj"),
             ("base_ljObjCut", "mu_lj"), ("base_ljObjCut", "egm_lj")]
curves_store = {}
for ch, view in roc_pairs:
    key = "%s/%s" % (ch, view)
    metrics["roc"][key] = {}
    metrics["operating_points"][key] = {}
    sig_groups = [g for g in GROUPS[ch] if g.startswith("signal_")]
    curves = {}
    for g in sig_groups + BKG + ["bkg_all"]:
        for var in VARS:
            c = eff_curve(GROUPS[ch][g], view, var, ch)
            if c is None:
                warnings.append("no eff curve: %s %s %s" % (key, g, var))
                continue
            curves[(g, var)] = c
    curves_store[key] = curves
    for sg in sig_groups:
        for var in VARS:
            if (sg, var) not in curves:
                continue
            s = curves[(sg, var)]
            entry = {"n_sig_lj": s["n"], "seam_mismatch": s["seam_mismatch"], "vs": {}}
            for b in BKG + ["bkg_all"]:
                if (b, var) not in curves:
                    continue
                entry["vs"][b] = bkg_at_sig(s, curves[(b, var)])
                entry["vs"][b]["n_bkg_lj"] = curves[(b, var)]["n"]
            metrics["roc"][key].setdefault(sg, {})[var] = entry
    # operating points: current cut on the old variable, and the new-variable
    # threshold matched to the same signal efficiency
    if ("signal_all", "old") in curves and ("bkg_all", "old") in curves:
        s_old, b_old = curves[("signal_all", "old")], curves[("bkg_all", "old")]
        t0 = CUR_CUT[view]
        se0, be0 = eff_at(s_old, t0), eff_at(b_old, t0)
        op = {"current": {"var": "old", "threshold": t0, "sig_eff": se0, "bkg_eff": be0}}
        for var in ["sum", "sum_soft"]:
            if (("signal_all", var) not in curves) or (("bkg_all", var) not in curves):
                continue
            m = bkg_at_sig(curves[("signal_all", var)], curves[("bkg_all", var)],
                           targets=(se0,))
            op[var] = m.get("%.4f" % se0)
            if op[var] is not None:
                op[var]["bkg_eff_ratio_vs_current"] = (op[var]["bkg_eff"] / be0
                                                       if be0 > 0 else None)
        metrics["operating_points"][key] = op
    # full curves to JSON (thinned)
    metrics["roc"][key]["_curves"] = {
        "%s|%s" % (g, var): {"thr": [round(float(x), 4) for x in c["thr"]],
                             "eff": [round(float(x), 5) for x in c["eff"]]}
        for (g, var), c in curves.items()}

with open(os.path.join(OUT, "metrics.json"), "w") as f:
    json.dump(metrics, f, indent=1)
print("wrote metrics.json ; warnings:", len(warnings))

# ======================================================================= plots
def save(fig, name):
    fig.tight_layout()
    p = os.path.join(PLOTS, name)
    fig.savefig(p, dpi=140)
    plt.close(fig)
    print("plot:", p)


# 1) coverage bars
for ch in ["4mu", "2mu2e", "base_ljObjCut"]:
    groups = ["signal_short", "signal_long"] + BKG
    for view in ["mu_lj", "egm_lj"]:
        vals = [metrics["coverage"][ch].get(g, {}).get(view, {}) for g in groups]
        if not any(v.get("n_lj", 0) for v in vals):
            warnings.append("coverage plot skipped (all empty): %s %s" % (ch, view))
            continue
        x = np.arange(len(groups))
        pj = [v.get("P_njet_ge1", np.nan) for v in vals]
        pa = [v.get("P_any_ge1", np.nan) for v in vals]
        fig, ax = plt.subplots(figsize=(9, 4.2))
        ax.bar(x - 0.19, pj, 0.36, color=C["old"], label="jet in cone (= current match)")
        ax.bar(x + 0.19, pa, 0.36, color=C["sum_soft"], hatch="//",
               edgecolor="white", linewidth=0.6, label="jet or soft jet in cone")
        for i, v in enumerate(vals):
            if v.get("n_lj", 0):
                ax.text(i, min(pa[i] + 0.02, 1.02), "N=%d" % v["n_lj"],
                        ha="center", fontsize=7, color="#555")
        ax.set_xticks(x)
        ax.set_xticklabels(groups, rotation=25, ha="right", fontsize=8)
        ax.set_ylim(0, 1.15)
        ax.set_ylabel("fraction of lepton jets")
        ax.set_title("Isolation-cone coverage - %s, %s (unweighted MC)" % (ch, view))
        ax.grid(axis="y", alpha=0.25)
        ax.legend(fontsize=8, loc="lower right")
        save(fig, "coverage_%s_%s.png" % (ch, view))

# 2) shape overlays on the zoom range
for ch, view in roc_pairs:
    edges = np.linspace(0, 0.3, 61)
    ctr = 0.5 * (edges[1:] + edges[:-1])
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2), sharey=False)
    for ax, (gname, ttl) in zip(axes, [("signal_all", "signal (6 grid points merged)"),
                                       ("bkg_all", "background (all processes)")]):
        any_drawn = False
        for var in VARS:
            v = stack(GROUPS[ch][gname], "%s_%s" % (view, HIST_ZOOM[var]), ch)
            if v is None or v.sum() == 0:
                continue
            body = v[1:61].astype(float)
            n = v.sum()
            ax.step(ctr, body / n, where="mid", color=C[var], ls=LS[var],
                    lw=2, label=LBL[var])
            any_drawn = True
        if not any_drawn:
            ax.text(0.5, 0.5, "no entries", transform=ax.transAxes, ha="center")
        ax.set_xlabel("isolation variable")
        ax.set_ylabel("fraction of LJs per bin")
        ax.set_title("%s\n%s, %s" % (ttl, ch, view), fontsize=9)
        ax.grid(alpha=0.25)
        ax.legend(fontsize=8)
    save(fig, "shapes_zoom_%s_%s.png" % (ch, view))

# 3) ROC overlays
for ch, view in roc_pairs:
    key = "%s/%s" % (ch, view)
    curves = curves_store[key]
    fig, ax = plt.subplots(figsize=(5.6, 5.0))
    drawn = False
    for var in VARS:
        if ("signal_all", var) not in curves or ("bkg_all", var) not in curves:
            continue
        s, b = curves[("signal_all", var)], curves[("bkg_all", var)]
        ax.plot(s["eff"], b["eff"], color=C[var], ls=LS[var], lw=2, label=LBL[var])
        drawn = True
    if ("signal_all", "old") in curves and ("bkg_all", "old") in curves:
        t0 = CUR_CUT[view]
        ax.plot([eff_at(curves[("signal_all", "old")], t0)],
                [eff_at(curves[("bkg_all", "old")], t0)], "o", ms=9,
                mfc="none", mec="#111", mew=1.6)
        ax.annotate("current cut %.2f" % t0,
                    (eff_at(curves[("signal_all", "old")], t0),
                     eff_at(curves[("bkg_all", "old")], t0)),
                    textcoords="offset points", xytext=(8, -12), fontsize=8)
    if not drawn:
        ax.text(0.5, 0.5, "no curves", transform=ax.transAxes, ha="center")
    ax.set_xlabel("signal efficiency (LJ level)")
    ax.set_ylabel("background efficiency (LJ level)")
    ax.set_title("Isolation ROC - %s, %s\n(unweighted MC, cut = variable < t)" % (ch, view),
                 fontsize=10)
    ax.grid(alpha=0.25)
    ax.legend(fontsize=8, loc="upper left")
    save(fig, "roc_%s_%s.png" % (ch, view))

# 4) sentinel rescue
for ch, view in [("4mu", "mu_lj"), ("2mu2e", "mu_lj"), ("2mu2e", "egm_lj"),
                 ("base_ljObjCut", "mu_lj"), ("base_ljObjCut", "egm_lj")]:
    groups = [g for g in ["signal_short", "signal_long"] if g in GROUPS[ch]] + BKG
    vals = [metrics["sentinel"][ch].get(g, {}).get(view) or {} for g in groups]
    if not any(v.get("n_sentinel", 0) for v in vals):
        warnings.append("sentinel plot skipped (all empty): %s %s" % (ch, view))
        continue
    x = np.arange(len(groups))
    fig, ax = plt.subplots(figsize=(9, 4.2))
    ax.bar(x - 0.19, [v.get("frac_new_gt_0p05", np.nan) for v in vals], 0.36,
           color=C["sum"], label="new > 0.05")
    ax.bar(x + 0.19, [v.get("frac_new_gt_0p2", np.nan) for v in vals], 0.36,
           color=C["sum_soft"], hatch="//", edgecolor="white", linewidth=0.6,
           label="new > 0.2")
    for i, v in enumerate(vals):
        if v.get("n_sentinel", 0):
            ax.text(i, 0.02, "N=%d" % v["n_sentinel"], ha="center", fontsize=7, color="#555")
    ax.set_xticks(x)
    ax.set_xticklabels(groups, rotation=25, ha="right", fontsize=8)
    ax.set_ylabel("fraction of LJs with old iso in [0, 0.005)")
    ax.set_title("Rescue of the no-match sentinel by the jet+soft-jet sum\n%s, %s"
                 % (ch, view), fontsize=10)
    ax.grid(axis="y", alpha=0.25)
    ax.legend(fontsize=8)
    save(fig, "sentinel_%s_%s.png" % (ch, view))

with open(os.path.join(OUT, "metrics.json"), "w") as f:
    json.dump(metrics, f, indent=1)
print("\nWARNINGS (%d):" % len(warnings))
for w in warnings:
    print("  ", w)
print("DONE")
