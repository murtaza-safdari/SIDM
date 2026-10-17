"""Build mass_decorrelation.ipynb.

Run from this directory on the LPC after merge_abcd8d.py has filled the local directory
(ABCD8D_LOCAL, default /uscms_data/d3/murtazas/abcd8d_local):
    python _build_mass_decorrelation.py
    jupyter nbconvert --to notebook --execute --inplace mass_decorrelation.ipynb

The builder reads the same blinded cubes the notebook reads (``transfer_lib.summed_cubes``) and
computes every number that appears in the markdown with the functions the code cells call (the
bootstraps with the same seed), formatted with ``nominal_lib.fmt`` / ``nominal_lib.pct``, so that the
prose and the tables cannot drift apart.  Nothing is computed at import time.
"""
import os

import nbformat as nbf

KERNEL = {"kernelspec": {"name": "sidm_venv",
                         "display_name": "SIDM (LCG_107 Py3.11)",
                         "language": "python"},
          "language_info": {"name": "python"}}
BASE_DEFAULT = "/uscms_data/d3/murtazas/abcd8d_local"
CALIBRATIONS = ("PD", "DD")


def compute():
    """Every number of the prose, as formatted strings keyed by name."""
    import numpy as np

    import abcd6d_io as io
    import binning_lib as bl
    import plane_search_lib as ps
    import decorrelation_lib as dl
    import nominal_lib as nl
    import transfer_lib as tl

    f, pct = nl.fmt, nl.pct
    N = {}
    base = os.environ.get("ABCD8D_LOCAL", BASE_DEFAULT)
    d2 = tl.summed_cubes(base, "data", prefix="abcd8d")["2mu2e"]
    nom = dl.NOMINAL

    def lst(values, digits=2):
        return ", ".join("n/a" if not np.isfinite(v) else f"{v:.{digits}f}" for v in values)

    def iv(x, digits=0):
        return dl.fmt_interval(x, digits)

    fp = dl.fmt_p
    N["box"] = io.blind_signature(prefix="abcd8d")
    N["n_toys"] = f"{dl.N_TOYS}"

    # section 1: the synthetic identity
    syn = dl.synthetic_table().set_index("population")
    N["syn_both"] = f"{syn.loc['both rising', 'mixing factor']:.3f}"
    N["syn_fall"] = f"{syn.loc['a falling, b rising', 'mixing factor']:.3f}"

    # section 1: a(M), b(M), spectra
    pt = dl.pass_table(d2, nom)
    for p in dl.PLANES:
        part = pt[pt["plane"] == p]
        N[f"a_{p}"] = lst(part["a"])
        N[f"b_{p}"] = lst(part["b"])
        N[f"bcd_{p}"] = "; ".join(f"{r['B']:.0f}/{r['C']:.0f}/{r['D']:.0f}" for _, r in part.iterrows())
    dd = pt[pt["plane"] == "DD"].set_index("bin")
    N["dd_34"] = f"{dd.loc['300-400', 'C']:.0f}/{dd.loc['300-400', 'D']:.0f}"
    N["dd_600"] = f"{dd.loc['>=600', 'B']:.0f}/{dd.loc['>=600', 'C']:.0f}/{dd.loc['>=600', 'D']:.0f}"
    N["a_dd_34"] = f(tuple(dd.loc["300-400", ["a", "a minus", "a plus"]]), 2)
    sp = dl.spectra_table(d2, nom).set_index(["plane", "bin"])
    hi_bins = [b for b, _ in dl.FIVE if b not in ("150-200", "200-300")]
    for p in dl.PLANES:
        N[f"hi_share_{p}"] = "/".join(pct(sum(sp.loc[(p, b), f"{x} share"] for b in hi_bins)) for x in "BCD")

    # section 1: the mixing factor
    mt = dl.mixing_table(d2, nom).set_index(["plane", "binning"])
    for p in dl.PLANES:
        for key, b in (("5", "5 bins"), ("2", "2 windows")):
            r = mt.loc[(p, b)]
            N[f"mix{key}_{p}"] = f"{r['ratio']:.2f}"
            N[f"mix{key}_{p}_iv"] = f"{r['lo']:.2f}–{r['hi']:.2f}"
            N[f"mix{key}_{p}_below"] = pct(r["below_one"])
    N["mix2_DD_finite"] = pct(mt.loc[("DD", "2 windows"), "kept"])
    N["split2_DD"] = f"{mt.loc[('DD', '2 windows'), 'split']:.1f}"
    N["split5_DD"] = f"{mt.loc[('DD', '5 bins'), 'split']:.1f}"
    N["one_DD"] = f"{mt.loc[('DD', '1 bin'), 'one_bin']:.1f}"
    N["dd_bins5"] = mt.loc[("DD", "5 bins"), "bins"].replace(">=", "≥ ")
    w5 = mt.loc[("DD", "5 bins"), "windows"]
    w2 = mt.loc[("DD", "2 windows"), "windows"]
    N["p_d0_hi"] = pct(np.exp(-w2[1][2]))
    N["p_d0_34"] = pct(np.exp(-dd.loc["300-400", "D"]))
    sens5, sens2 = dict(dl.mixing_sensitivity(w5)), dict(dl.mixing_sensitivity(w2))
    i34 = [lab for lab, _ in dl.mixing_factor(d2, nom, "DD", dl.FIVE)["bins"]].index("300-400")
    N["sens5_34"] = f"{sens5[i34]:.2f}"
    N["sens2_hi"] = f"{sens2[1]:.2f}"
    N["sens5_other"] = lst([v for i, v in sens5.items() if i != i34])
    N["d_hi"] = f"{w2[1][2]:.0f}"
    N["cd_w"] = f"{w2[0][1] / w2[0][2]:.2f} and {w2[1][1] / w2[1][2]:.2f}"
    cd = {p: pt.loc[pt["plane"] == p, ["C", "D"]].to_numpy() for p in dl.PLANES}
    for ref in ("PD", "PP"):
        st = dl.shape_test(cd["DD"], cd[ref])
        N[f"shape_{ref}"] = f"χ² = {st['chi2']:.1f}/{st['ndf']}, {fp(st['p'])} ({fp(st['p_toys'])} from toys)"
        N[f"pull34_{ref}"] = f"{st['pulls'][2]:+.1f}"
        N[f"k_{ref}"] = f"{st['k']:.2f}"
    N["dd_empty"] = "≥ 600 GeV" if not np.isfinite(dd.loc[">=600", "a"]) else "none"

    # section 1d: the open band of the search plane
    band = nl.dd_band_vrs()[0]
    N["band"] = band.name.replace("DD, egamma iso ", "")
    others = [v for v in nl.dd_band_vrs()[1:] if nl.vr_contains(band, v)]
    N["band_others"] = ", ".join(v.name.replace("DD, egamma iso ", "") for v in others)
    b2 = dl.pass_table(d2, nom, [band], dl.TWO).set_index("bin")
    b5 = dl.pass_table(d2, nom, [band], dl.FIVE)
    N["band_bcd2"] = (f"{b2.loc['150-300', 'B']:.0f}/{b2.loc['150-300', 'C']:.0f}/{b2.loc['150-300', 'D']:.0f} in 150–300 GeV and "
                      f"{b2.loc['>=300', 'B']:.0f}/{b2.loc['>=300', 'C']:.0f}/{b2.loc['>=300', 'D']:.0f} above 300 GeV")
    N["band_a_lo"] = f(tuple(b2.loc["150-300", ["a", "a minus", "a plus"]]), 2)
    N["band_cd_hi"] = f"{b2.loc['>=300', 'C'] + b2.loc['>=300', 'D']:.0f}"
    N["band_a5"] = lst(b5["a"])
    cd_b5 = b5[["C", "D"]].to_numpy()
    for ref in ("DD", "PD"):
        st = dl.shape_test(cd_b5, cd[ref])
        N[f"band_shape_{ref}"] = f"χ² = {st['chi2']:.1f}/{st['ndf']}, {fp(st['p'])}"
        N["band_used"] = f"{int(st['used'].sum())}"
    two = dl.pass_table(d2, nom, ["PD", "DD"], dl.TWO)
    cd2 = {p: two.loc[two["plane"] == p, ["C", "D"]].to_numpy() for p in ("PD", "DD")}
    for p in ("PD", "DD"):
        v, e = dl.log_odds_ratio(cd2[p])
        N[f"lor_{p}"] = f"{v:.2f} ± {e:.2f}"
        N[f"or_{p}"] = f"{np.exp(v):.1f}"
    for p in ("PD", "DD"):
        row = two[(two["plane"] == p) & (two["bin"] == "150-300")].iloc[0]
        N[f"a_lo_{p}"] = f(tuple(row[["a", "a minus", "a plus"]]), 2)
        row = two[(two["plane"] == p) & (two["bin"] == ">=300")].iloc[0]
        N[f"a_hi_{p}"] = f(tuple(row[["a", "a minus", "a plus"]]), 2)
    sep = dl.separation(cd2["DD"], cd2["PD"])
    N["sep_z"] = f"{sep['z']:.1f}"
    for z in (2, 3):
        factor = (z / sep["z"]) ** 2
        N[f"f{z}"] = f"{factor:.1f}"
        N[f"dneed{z}"] = f"{factor * cd2['DD'][1, 1]:.0f}"

    # section 2: route 1
    kt = bl.kappa_table(d2, nom, list(dl.FIVE), planes=dl.OPEN_PLANES)
    for p in dl.OPEN_PLANES:
        part = kt[(kt["plane"] == p) & (kt["bin"] != "all bins")]
        N[f"k5_{p}"] = lst(part["value"])
    st5 = dl.stratified_table(d2, nom, dl.FIVE)
    st2 = dl.stratified_table(d2, nom, dl.TWO)
    for key, st in (("5", st5), ("2", st2)):
        body, tot = st[st["bin"] != "sum"], st[st["bin"] == "sum"].iloc[0]
        N[f"r1_{key}"] = f"{tot['estimate']:.1f}"
        N[f"r1_{key}_iv"] = iv((tot["lo"], np.nan, tot["hi"]))
        N[f"r1_{key}_finite"] = pct(st.attrs["kept"])
        N[f"r1_{key}_fallback"] = pct(st.attrs["fallback"])
        N[f"r1_{key}_bins"] = "; ".join(f"{r['bin'].replace('>=', '≥ ')}: {r['estimate']:.1f} ({iv((r['lo'], np.nan, r['hi']))})"
                                         for _, r in body.iterrows())
        N[f"r1_{key}_syst"] = "; ".join(
            f"{r['bin'].replace('>=', '≥ ')}: " + ("no region qualifies" if not np.isfinite(r["syst"])
                                                   else f"{pct(r['syst'])} ({r['syst region']})") for _, r in body.iterrows())
        none = [r["bin"].replace(">=", "≥ ") for _, r in body.iterrows() if not np.isfinite(r["syst"])]
        N[f"r1_{key}_none"] = " and ".join(none) if none else "none"
        N[f"r1_{key}_sum_syst"] = pct(tot["syst"]) if np.isfinite(tot["syst"]) else "undefined"
        for p in dl.OPEN_PLANES:
            N[f"rc{key}_{p}"] = f(dl.route_closure(d2, nom, p, dl.FIVE if key == "5" else dl.TWO))
        N[f"r1_{key}_qual"] = "; ".join(f"{r['bin'].replace('>=', '≥ ')}: {r['qualifying']}" for _, r in body.iterrows())
    N["r1_5_merged"] = st5.iloc[-2]["bin"].replace(">=", "≥ ")
    hi2 = st2[st2["bin"] == ">=300"].iloc[0]
    N["r1_hi"] = f"{hi2['estimate']:.0f}"
    N["r1_hi_iv"] = iv((hi2["lo"], np.nan, hi2["hi"]))
    lo2 = st2[st2["bin"] == "150-300"].iloc[0]
    N["r1_lo"] = f"{lo2['estimate']:.0f}"
    N["syst_w1"], N["syst_w2"] = pct(lo2["syst"]), pct(hi2["syst"])
    N["syst_r1"], N["syst_r2"] = lo2["syst region"], hi2["syst region"]
    N["k2_lo"] = f"PP {f((lo2['kappa PP'], lo2['kappa PP minus'], lo2['kappa PP plus']))}, PD {f((lo2['kappa PD'], lo2['kappa PD minus'], lo2['kappa PD plus']))}"
    N["k2_hi"] = f"PP {f((hi2['kappa PP'], hi2['kappa PP minus'], hi2['kappa PP plus']))}, PD {f((hi2['kappa PD'], hi2['kappa PD minus'], hi2['kappa PD plus']))}"
    N["prec_hi_PD"] = pct(hi2["precision PD"], 1)
    N["kpd_lo"] = f((lo2["kappa PD"], lo2["kappa PD minus"], lo2["kappa PD plus"]))
    N["kpd_hi"] = f((hi2["kappa PD"], hi2["kappa PD minus"], hi2["kappa PD plus"]))
    named = {v.name: v for v in nl.DONGYUB_VRS}
    N["vr_defs"] = "; ".join(f"{n}: {named[n].definition()}" for n in ("V1", "V5", "V6", "V7"))
    N["within"] = f"{float(st5.iloc[-1]['estimate']) / float(st2.iloc[-1]['estimate']):.2f}"
    for p in dl.OPEN_PLANES:
        N[f"rc1_{p}"] = f(dl.route_closure(d2, nom, p, dl.ONE))
    N["qual_hi"] = hi2["qualifying"]
    N["hi_factor"] = f"{hi2['hi'] / hi2['lo']:.1f}"

    # section 3: route 2
    fits = {p: dl.smooth_route(d2, nom, p) for p in dl.PLANES}
    shared = dl.smooth_route(d2, nom, "DD", slope_plane="PD")
    for p, r in fits.items():
        N[f"s_{p}"] = f"{r['slope']:.2f} ± {r['slope_err']:.2f}"
        N[f"al_{p}"] = f"{r['alpha']:.2f} ± {r['alpha_err']:.2f}"
        N[f"dev_{p}"] = f"{r['deviance']:.1f}/{r['ndf']} ({fp(r['p'])})"
        N[f"scale_{p}"] = f"{r['scale']:.2f}"
        N[f"sm_{p}"] = f"{r['pred']:.1f}"
        N[f"sm_{p}_iv"] = iv(r["pred_interval"])
        if p in dl.OPEN_PLANES:
            N[f"ks_{p}"] = f(r["kappa"])
            N[f"A_{p}"] = f"{r['A']:.0f}"
    N["sm_sh"] = f"{shared['pred']:.1f}"
    N["sm_sh_iv"] = iv(shared["pred_interval"])
    N["al_sh"] = f"{shared['alpha']:.2f} ± {shared['alpha_err']:.2f}"
    sc = dl.slope_consistency(fits)
    N["slope_chi2"] = f"{sc['chi2']:.1f}/{sc['ndf']} ({fp(sc['p'])})"
    scaled = [f"{p} (factor {fits[p]['scale']:.2f})" for p in dl.PLANES if fits[p]["scale"] > 1]
    if shared["scale"] > 1:
        scaled.append(f"DD's fit with PD's slope (factor {shared['scale']:.2f})")
    N["scaled"] = ", ".join(scaled[:-1]) + (" and " if len(scaled) > 1 else "") + scaled[-1] if scaled else "none of the fits"
    N["slope_dd_pd"] = f"{abs(sc['pairs'][('PD', 'DD')]):.1f}"
    N["slope_dd_pp"] = f"{abs(sc['pairs'][('PP', 'DD')]):.1f}"
    N["slope_pp_pd"] = f"{abs(sc['pairs'][('PP', 'PD')]):.1f}"
    x = fits["DD"]["x"]
    B = fits["DD"]["counts"][:, 1]
    parts = B * dl.logistic_r(fits["DD"]["alpha"], fits["DD"]["slope"], np.exp(x) * dl.PIVOT)
    N["sm_parts"] = ", ".join(f"{v:.1f}" for v in parts)
    N["sm_600"] = f"{parts[-1]:.1f}"
    N["r_600"] = f"{dl.logistic_r(fits['DD']['alpha'], fits['DD']['slope'], dl.centre(600.0, None)):.1f}"
    N["transfer"] = pct(abs(fits["DD"]["pred"] - shared["pred"]) / shared["pred"])
    N["centres"] = ", ".join(f"{dl.centre(*rng):.0f}" for _, rng in dl.FIVE)

    # section 4: route 3
    choice, flat = {}, {}
    for cal in CALIBRATIONS:
        choice[cal] = dl.choose_edges(dl.odds_table(d2, nom, cal))
        N[f"edges_{cal}"] = ", ".join(f"{e:g}" for e in choice[cal]["edges"])
        N[f"target_{cal}"] = f"{choice[cal]['target']:.3f}"
        for p in dl.PLANES:
            flat[(cal, p)] = dl.flat_route(d2, nom, p, dl.FIVE, choice[cal]["edges"])
    a_dd = choice["DD"]["a"]
    N["dd34_range"] = f"{np.nanmin(a_dd[2]):.2f}–{np.nanmax(a_dd[2]):.2f}"
    od_dd = dl.odds_table(d2, nom, "DD")
    j = {e: i for i, e in enumerate(dl.EDGE_CANDIDATES)}
    N["dd34_c_low"] = f"{od_dd[2, j[0.1], 0]:.0f}"
    N["dd34_c"] = f"{od_dd[2, j[0.25], 0]:.0f}"
    N["dd34_d_high"] = f"{od_dd[2, j[1.0], 1]:.0f}"
    N["dd600_empty"] = ("the ≥ 600 GeV bin has no C or D at any edge" if not np.isfinite(a_dd[4]).any()
                        else "the ≥ 600 GeV bin has events at some edge")
    for p in dl.PLANES:
        bf = dl.flatness(cd[p])
        N[f"flat0_{p}"] = f"{bf['chi2']:.1f}/{bf['ndf']} ({fp(bf['p'])})"
        for cal in CALIBRATIONS:
            fl = flat[(cal, p)]["flatness"]
            N[f"flat_{cal}_{p}"] = f"{fl['chi2']:.1f}/{fl['ndf']} ({fp(fl['p'])})"
            N[f"aflat_{cal}_{p}"] = lst(fl["a"])
    for cal in CALIBRATIONS:
        r = flat[(cal, "DD")]
        N[f"fe_{cal}"] = f"{r['pred'][0]:.1f}"
        N[f"fe_{cal}_iv"] = iv(r["pred_interval"])
        N[f"fmix_{cal}"] = f"{r['mixing']:.2f} ({iv(r['mixing_interval'], 2)})"
        N[f"fbcd_{cal}"] = "/".join(f"{r['regions'][x][0]:.0f}" for x in "BCD")
        for p in dl.OPEN_PLANES:
            N[f"fk_{cal}_{p}"] = f(flat[(cal, p)]["kappa"])
            N[f"fmix_{cal}_{p}"] = f"{flat[(cal, p)]['mixing']:.2f}"
    one_regs = dl.summed(dl.bin_regions(d2, nom, "DD", dl.FIVE))
    N["bcd_one"] = "/".join(f"{one_regs[x][0]:.0f}" for x in "BCD")
    for cal in CALIBRATIONS:
        edges = choice[cal]["edges"]
        up = [(lab, e) for (lab, _), e in zip(dl.FIVE, edges) if e > dl.ANCHOR_EDGE]
        N[f"up_{cal}"] = " and ".join(f"{e:g} at {lab.replace('-', '–')} GeV" for lab, e in up) or "none"
        nominal_b = dl.counts(dl.bin_regions(d2, nom, "DD", dl.FIVE))[:, 1]
        flat_b = flat[(cal, "DD")]["per_bin"][:, 1]
        N[f"moved_{cal}"] = f"{np.sum(nominal_b - flat_b):.0f}"
    names = [n for n in sorted(os.path.basename(p)[: -len(".coffea")] for p in tl.sample_paths(base, "signal")
                               if os.path.basename(p).startswith("2Mu2E_")) if nl.mediator_mass(n) >= 200]
    labels = {cal: f"{cal}-calibrated" for cal in CALIBRATIONS}
    cost = dl.signal_cost(base, names, nom, dl.FIVE, {labels[c]: choice[c]["edges"] for c in CALIBRATIONS})
    N["n_sig"] = f"{len(names)}"
    N["masses"] = ", ".join(f"{m:g}" for m in sorted(cost["mediator"].unique()))
    for cal in CALIBRATIONS:
        col = cost[labels[cal]]
        N[f"cost_{cal}"] = f"{col.min():.2f}–{col.max():.2f}"
        worst = cost.loc[col.idxmin()]
        N[f"cost_{cal}_worst"] = f"{worst['point']} ({col.min():.2f})"
        per = cost.groupby("mediator")[labels[cal]].min()
        N[f"cost_{cal}_per"] = ", ".join(f"{m:g} GeV {v:.2f}" for m, v in per.items())
    N["max_cost"] = pct(1 - min(cost[labels[c]].min() for c in CALIBRATIONS))

    # section 5: comparison
    rec = dl.comparison(d2, nom, {c: choice[c]["edges"] for c in CALIBRATIONS}, cost)
    by = {r["route"]: r for r in rec}
    N["n_rule"] = f"{len(dl.rule_regions(d2, nom))}"
    N["base_syst"] = f"{pct(by['one bin (nominal.ipynb)']['syst'][0])} ({by['one bin (nominal.ipynb)']['syst'][1]})"
    rs = by["2, logistic a(M), DD's own fit"]["syst"]
    N["syst_smooth"] = f"{pct(rs[0])} ({rs[1]})"
    rs = by["2, logistic a(M), PD's slope"]["syst"]
    N["syst_shared"] = f"{pct(rs[0])} ({rs[1]})"
    for cal in CALIBRATIONS:
        rr = by[f"3, flattened cut, {cal}-calibrated"]["syst"]
        N[f"fsyst_{cal}"] = f"{pct(rr[0])} ({rr[1]})"
    two = next(r for r in rec if r["route"].startswith("1, stratified, 2"))
    N["rec_est"] = f"{two['estimate']:.0f}"
    N["rec_iv"] = iv(two["interval"])
    N["rec_k"] = f"PP {f(two['kappa']['PP'])}, PD {f(two['kappa']['PD'])}"
    one = by["one bin (nominal.ipynb)"]
    N["base_k"] = f"PP {f(one['kappa']['PP'])}, PD {f(one['kappa']['PD'])}"
    N["pd_all"] = f"{min(r['kappa']['PD'][0] for r in rec):.2f}–{max(r['kappa']['PD'][0] for r in rec):.2f}"
    N["mix_excess"] = pct(float(N["mix2_DD"]) - 1)
    N["pd_mix"] = (f"{mt.loc[('PD', '2 windows'), 'ratio']:.3f} with two windows and "
                   f"{mt.loc[('PD', '5 bins'), 'ratio']:.3f} with five bins")
    N["one_est"] = f"{one['estimate']:.1f}"
    N["one_iv"] = iv(one["interval"])
    return N


def md(text, N=None):
    return nbf.v4.new_markdown_cell(text if N is None else text.format(**N))


def code(text):
    return nbf.v4.new_code_cell(text)


def build(N):
    cells = []
    cells.append(md(r"""# Removing the pair-mass confounding from the 2μ2e ABCD estimate

**Summary.** `nominal.ipynb` estimates the 2μ2e background in the search region as the search
plane's own A = B·C/D = {one_DD} events. That estimate assumes that the two variables defining the
regions, the muon lepton jet's isolation and the azimuthal angle |Δφ| between the two lepton jets, are
independent for the background. Their pass probabilities can change with the lepton-jet pair mass
M, and when both change, a single ABCD over all masses mixes populations: summed over
mass bins, the same estimate reads {mix2_DD} times the one-bin value with two windows (split at
300 GeV) and {mix5_DD} with the five bins of the histograms ({mix2_PP} and {mix5_PP} in PP,
{mix2_PD} and {mix5_PD} in PD). In the search plane this rests on one bin: at 300–400 GeV, C/D =
{dd_34}, and one more event in D there would bring the five-bin factor from {mix5_DD} to
{sens5_34}. The search plane's isolation pass probability against M is compatible with PD's shape
({shape_PD}) and with PP's ({shape_PP}).

This notebook tries three ways of removing the mixing and compares them by the simplicity of the
analysis they lead to, systematics included: (1) stratify, estimating the background separately in
each mass bin; (2) fit the isolation pass probability with a smooth function of M and apply it to
B's mass spectrum; (3) choose the isolation cut per mass bin so that its pass probability is flat in
M. In two windows, route 1 gives {r1_2} events (68% interval {r1_2_iv}), with a closure systematic
by the rule of `nominal.ipynb` in each window ({syst_w1} and {syst_w2}); in five bins it gives
{r1_5} and the rule has no qualifying region in {r1_5_none}. Route 2 gives {sm_DD} with the search
plane's own slope and {sm_sh} with PD's. Route 3 cannot flatten the search plane: its 300–400 GeV
bin reads a = {dd34_range} at every stored edge, and the mixing factor stays at {fmix_PD}. In every
route PD's closure stays at {pd_all}: PD's non-closure, which sets the 20% systematic of
`nominal.ipynb`, is not mass mixing (PD's mixing factor is {mix2_PD}), although PD's κ does change
with the mass ({kpd_lo} below 300 GeV, {kpd_hi} above). The open bands of the search plane have no
C or D event above 300 GeV and cannot test the rise. Section 5 proposes route 1 in two windows.

**How to read this notebook.** Every number is from data unless it is marked as simulation. The
search region (region A of the search plane above 150 GeV) is hidden: the histograms are loaded with
those cells set to NaN, a region that touches them prints as "hidden" or NaN, and no estimate uses
them. Section 1 states the problem and measures it, sections 2–4 try the three routes, section 5
compares them and makes the proposal. Terms defined in `nominal.ipynb` (lepton jet, the planes PP,
PD, DP and DD, the regions A–D, κ, the working point, the closure systematic's rule) are repeated
briefly where they are first used.""", N))

    cells.append(code(r"""import dataclasses
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import abcd6d_io as io
import transfer_lib as tl
import transfer_plots as tp
import plane_search_lib as ps
import binning_lib as bl
import nominal_lib as nl
import decorrelation_lib as dl
import decorrelation_plots as dp

pd.set_option("display.max_columns", 40)
pd.set_option("display.max_rows", 200)
pd.set_option("display.width", 250)
pd.set_option("display.max_colwidth", 120)
tp.set_style()

BASE = os.environ.get("ABCD8D_LOCAL", "/uscms_data/d3/murtazas/abcd8d_local")
print("8D inputs:", BASE)
data = tl.summed_cubes(BASE, "data", prefix="abcd8d")     # the only data path: hidden cells are NaN
print("hidden box:", io.blind_signature(prefix="abcd8d"))
d2 = data["2mu2e"]
print(f"2mu2e: {d2.sumw.shape}, {int(np.isnan(d2.sumw).sum())} hidden cells")
NOMINAL = dl.NOMINAL
print("working point:", NOMINAL.label())
print("mass bins:", [label for label, _ in dl.FIVE], "; windows:", [label for label, _ in dl.TWO])
print("bootstrap toys:", dl.N_TOYS, "seed", dl.SEED)"""))

    cells.append(md(r"""## 1. The problem: the pair mass as a third variable

**The ABCD method in one paragraph.** Two cuts split a plane into four regions: A passes both
(the search region), B fails the isolation cut and passes |Δφ|, C passes the isolation cut and fails
|Δφ|, D fails both. Here the isolation cut is "muon-LJ isolation < 0.25" and the |Δφ| cut
"|Δφ| ≥ 2.4"; the egamma lepton jet is isolated (< 0.5) and M ≥ 150 GeV. If the two cuts are
independent for the background, A = B·C/D. Where A is open, the closure ratio κ = A·D/(B·C)
tests that; κ = 1 means the method works there. The planes are named by the displacement of the muon
lepton jet and then of the egamma lepton jet (P prompt, D displaced): DD is the search plane, PP and
PD are open planes with a prompt muon lepton jet whose A can be looked at.

**Why the mass matters.** Split the events into mass bins w. In bin w let n_w be the number of
background events, a_w the probability that an event passes the isolation cut and b_w the
probability that it passes the |Δφ| cut. If the two cuts are independent within each bin,

* A_w = n_w a_w b_w, B_w = n_w (1 − a_w) b_w, C_w = n_w a_w (1 − b_w), D_w = n_w (1 − a_w)(1 − b_w),

so A_w = B_w C_w / D_w holds bin by bin, and the true A is A = Σ_w n_w a_w b_w = Σ_w B_w C_w / D_w.
The one-bin estimate instead uses the sums, (Σ B)(Σ C)/(Σ D). Write r_w = a_w/(1 − a_w) = C_w/D_w
for the isolation **odds** (pass over fail) and β_w = b_w/(1 − b_w) = B_w/D_w for the |Δφ| odds. Then
B_w = β_w D_w and C_w = r_w D_w, and

* Σ_w B_w C_w / D_w = Σ_w D_w β_w r_w, while (Σ B)(Σ C)/(Σ D) = (Σ_w D_w β_w)(Σ_w D_w r_w)/(Σ_w D_w).

If r_w = r is the same in every bin, both are r Σ_w D_w β_w; if β_w = β is the same in every bin,
both are β Σ_w D_w r_w. So **the one-bin estimate is exact when either a(M) or b(M) is constant in
M**, whatever the other does. When both vary, the ratio of the two is the **mixing factor**

* m = Σ_w B_w C_w / D_w ÷ (B·C/D) = ⟨β r⟩_D / (⟨β⟩_D ⟨r⟩_D) = 1 + cov_D(β, r) / (⟨β⟩_D ⟨r⟩_D),

where ⟨·⟩_D is the average over the bins weighted by D_w and cov_D the covariance with the same
weights. If the cuts are independent within each bin, the one-bin κ equals m: the one-bin estimate is
low by m when the two odds rise (or fall) together with M, and high when one rises while the other
falls. In data m is computed from B, C and D alone, so it can be measured in the search plane without
opening A.

The table below checks this on four synthetic populations of five mass bins, with exact expected
counts: with a flat a(M) or a flat b(M) the one-bin estimate equals the true A, with both rising it
is low (m = {syn_both}), with opposite trends high (m = {syn_fall}), and the covariance form gives
the same number as the ratio.""", N))
    cells.append(code(r"""display(dl.synthetic_table().round(3))"""))

    cells.append(md(r"""### 1a. a(M) and b(M) in the data

Per plane and mass bin, a = C/(C + D) is the isolation pass probability among the events that fail
|Δφ|, and b = B/(B + D) the |Δφ| pass probability among the events that fail the isolation; both use
only B, C and D, so in DD above 150 GeV both are open. The intervals are 68% **Clopper–Pearson**
intervals (the exact binomial interval, which stays inside [0, 1] for small counts). In the five
bins 150–200, 200–300, 300–400, 400–600 and ≥ 600 GeV:

* PP: a = {a_PP}; b = {b_PP} (B/C/D {bcd_PP});
* PD: a = {a_PD}; b = {b_PD} (B/C/D {bcd_PD});
* DD: a = {a_DD}; b = {b_DD} (B/C/D {bcd_DD}).

b rises with M in all three planes. a rises slowly in PP, has no trend in PD, and in DD jumps at
300–400 GeV to {a_dd_34}, from C/D = {dd_34}; the ≥ 600 GeV bin of DD has B/C/D = {dd_600}, so its a
is undefined. Is DD's a(M) PD's shape? Scaling PD's a(M) to DD's mean (the factor that makes the
expected C of DD sum to the observed, {k_PD}) and comparing bin by bin with binomial errors of both
planes gives {shape_PD}, over the bins where DD has events; against PP's shape {shape_PP}. The
largest deviation in both is the 300–400 GeV bin (pull {pull34_PD} against PD, {pull34_PP} against
PP). DD's a(M) is therefore compatible with the flat or slowly rising shape of the open planes; the
rise it shows rests on that one bin.

*How to read the figure:* left a(M), right b(M), per plane (PP black, PD blue, DD green), points at
the geometric centres of the bins (the ≥ 600 GeV bin at 775 GeV), shifted slightly sideways per
plane; error bars are the 68% intervals.""", N))
    cells.append(code(r"""PASS = dl.pass_table(d2, NOMINAL)
display(dl.pass_view(PASS))
fig = dp.pass_figure(PASS)"""))

    cells.append(md(r"""### 1b. The mass spectra of B, C and D

The mixing factor depends on how the regions spread over the mass: the shares of B/C/D above
300 GeV are {hi_share_PP} in PP, {hi_share_PD} in PD and {hi_share_DD} in DD. DD's C sits at higher
mass than its D, PP's and PD's do not by much.

*How to read the figure:* one panel per plane; each region's mass spectrum is normalised to one over
the five bins (B purple, C orange, D cyan, the total count in brackets); error bars are √n over the
region's total.""", N))
    cells.append(code(r"""SPECTRA = dl.spectra_table(d2, NOMINAL)
fig = dp.spectra_figure(SPECTRA)"""))

    cells.append(md(r"""### 1c. The mixing factor

The mixing factor per plane with five bins, two windows (split at 300 GeV) and one bin (one by
definition). A bin whose D is empty has no B·C/D; it is merged into the bin below (in DD the
≥ 600 GeV bin, so DD's five-bin factor is computed over {dd_bins5}). The intervals are from a
**Poisson bootstrap**: each B, C and D is redrawn {n_toys} times from a Poisson distribution whose
mean is the observed count, the factor is recomputed, and the 16th and 84th percentiles over all toys
are quoted; no toy is dropped. Each toy follows the procedure of its binning: the five-bin procedure
merges a bin with an empty D (its data path merges DD's ≥ 600 GeV bin), so its toys merge the same
way, while the two windows have no merge rule (no window of the data is empty), so a toy with no D
event in a window has no estimate and counts as above every other toy, as in `nominal.ipynb`
section 7d.

* PP: {mix5_PP} with five bins ({mix5_PP_iv}), {mix2_PP} with two windows ({mix2_PP_iv});
* PD: {mix5_PD} ({mix5_PD_iv}) and {mix2_PD} ({mix2_PD_iv});
* DD: {mix5_DD} ({mix5_DD_iv}, below one in {mix5_DD_below} of the toys) and {mix2_DD} ({mix2_DD_iv},
  below one in {mix2_DD_below}; {mix2_DD_finite} of the toys have a finite factor, the rest drew no D
  event above 300 GeV, a chance of {p_d0_hi}).

The open planes mix by 1–4%. In DD the two windows have C/D = {cd_w}: the summed estimate is
{split2_DD} events with two windows and {split5_DD} with five, against {one_DD} in one bin. **What
the mixing in DD rests on:** the 300–400 GeV bin, C/D = {dd_34}. With one more event in D of that
bin the five-bin factor would be {sens5_34} (one more D in each of the other bins in turn:
{sens5_other}); with one more event in D above 300 GeV the two-window factor would be {sens2_hi}
(D = {d_hi} there now). The factor in DD is real if DD's a(M) rises as its four filled bins say, and
an accident of one event if it has the open planes' shape, which section 1a cannot exclude.""", N))
    cells.append(code(r"""MIX = dl.mixing_table(d2, NOMINAL)
display(dl.mixing_view(MIX))
for plane in dl.PLANES:
    m = dl.mixing_factor(d2, NOMINAL, plane, dl.FIVE)
    labels = [label for label, _ in m["bins"]]
    print(f"{plane}: five-bin factor with one more D event in ...", ", ".join(f"{labels[i]}: {v:.2f}" for i, v in dl.mixing_sensitivity(m["windows"])))
CD = {p: PASS.loc[PASS["plane"] == p, ["C", "D"]].to_numpy() for p in dl.PLANES}
for ref in ("PD", "PP"):
    st = dl.shape_test(CD["DD"], CD[ref])
    print(f"DD's a(M) against {ref}'s shape: chi2 {st['chi2']:.1f}/{st['ndf']}, p = {st['p']:.2f} (toys {st['p_toys']:.2f}), "
          f"scale {st['k']:.2f}, pulls", np.round(st["pulls"], 1))"""))

    cells.append(md(r"""### 1d. An open region with both lepton jets displaced above 300 GeV

The search plane has open regions above 150 GeV: the bands where the egamma lepton jet is not
isolated (egamma-LJ isolation ≥ 0.5, ≥ 0.75, ≥ 1 and 0.5–1). Their A is open, because the hidden box
ends at egamma-LJ isolation 0.5, and they have both lepton jets displaced. Could they show whether
DD's a(M) rises? The widest band, egamma-LJ isolation {band}, holds B/C/D = {band_bcd2}; the
bands {band_others} lie inside it. Its a = C/(C + D) is {band_a_lo} in 150–300 GeV, against
{a_lo_DD} for DD and {a_lo_PD} for PD, and above 300 GeV it has C + D = {band_cd_hi}, so a is
undefined there. In the five bins its a is {band_a5}; against DD's shape it gives {band_shape_DD}
and against PD's {band_shape_PD}, over the {band_used} bins where it has events, both below 300 GeV.
**The open bands cannot test the rise:** none has a C or D event above 300 GeV.

What the search plane's own sidebands show is the window-to-window change of the isolation odds:
ln(r above 300 GeV / r in 150–300 GeV) is {lor_DD} in DD (a factor {or_DD}) and {lor_PD} in PD (a
factor {or_PD}), a difference of {sep_z} standard deviations (first-order errors, √(Σ 1/count)). The
separation grows as the square root of the data. If the central values held, it would reach two
standard deviations with {f2} times the present data, about {dneed2} D events above 300 GeV, and three
with {f3} times, about {dneed3} (now {d_hi}).""", N))
    cells.append(code(r"""BAND = nl.dd_band_vrs()[0]
for name, bins in (("2 windows", dl.TWO), ("5 bins", dl.FIVE)):
    print(f"{BAND.name}, {name}:")
    display(dl.pass_view(dl.pass_table(d2, NOMINAL, [BAND], bins)))
CD_BAND = dl.pass_table(d2, NOMINAL, [BAND], dl.FIVE)[["C", "D"]].to_numpy()
for ref in ("DD", "PD"):
    st = dl.shape_test(CD_BAND, CD[ref])
    print(f"the band's a(M) against {ref}'s shape: chi2 {st['chi2']:.1f}/{st['ndf']}, {dl.fmt_p(st['p'])}, bins used {st['used']}")
PASS2 = dl.pass_table(d2, NOMINAL, ["PD", "DD"], dl.TWO)
CD2 = {p: PASS2.loc[PASS2["plane"] == p, ["C", "D"]].to_numpy() for p in ("PD", "DD")}
for p in ("PD", "DD"):
    print(f"{p}: log odds ratio between the windows", np.round(dl.log_odds_ratio(CD2[p]), 2))
print("DD against PD:", {k: round(v, 2) for k, v in dl.separation(CD2["DD"], CD2["PD"]).items()})"""))

    cells.append(md(r"""## 2. Route 1: stratify

The direct fix: estimate the background in each mass bin from that bin's own B, C and D and sum,
Σ_w B_w C_w / D_w. If the two cuts are independent within each bin this is exact, whatever a(M) and
b(M) do; what it needs instead is a validation of the method per bin. A mass-binned search region
needs per bin the background estimate, its closure κ in the open planes, and a closure systematic.

**κ per bin in the open planes** (reproducing `nominal.ipynb` section 7d): PP {k5_PP}; PD {k5_PD}
for the five bins.

**The estimate per bin.** The interval of each bin is over all toys of its own B, C and D, with no
upper bound where D is drawn as 0 often enough; the interval of the sum follows the convention of
section 1c (two windows: a toy with an empty D has no finite sum, {r1_2_finite} of the toys have
one; five bins: the merging is redone in each toy). With two windows: {r1_2_bins}; the sum is
**{r1_2} events** (bootstrap 68% interval {r1_2_iv}, as in `nominal.ipynb` section 7d). With five bins (the ≥ 600 GeV bin merged with 400–600 GeV into {r1_5_merged} GeV, since its C and D
are empty): {r1_5_bins}; the sum is {r1_5} events ({r1_5_iv}). The 300–400 GeV bin has D = 1 and is
drawn with D = 0 in {p_d0_34} of the toys; those toys merge it with 200–300 GeV and land low, which is
why the sum's interval sits mostly below the central value. In all, {r1_5_fallback} of the toys merge
more than the data do (beyond the ≥ 600 GeV bin), so the five-bin procedure falls back to fewer bins
that often. The five-bin sum over the
two-window one, {within}, is the mixing that is left inside the windows, and it rests on the same
300–400 GeV bin.

**Closure of the summed estimate in the open planes**, A over Σ_w B_w C_w / D_w, with a bootstrap
interval: two windows PP {rc2_PP}, PD {rc2_PD}; five bins PP {rc5_PP}, PD {rc5_PD} (one bin, from
the same bootstrap: PP {rc1_PP}, PD {rc1_PD}). Stratifying does not move PD towards one.

**The systematic per bin, by the rule of `nominal.ipynb` section 8:** among the validation regions
above 150 GeV, each restricted to the bin, those whose κ is measured to 20% or better (the larger
side of its 68% interval at most 20% of κ) and that are not a sub-population of a plane; the
systematic is the largest of |κ − 1| and the side of the interval facing one. The regions are PP,
PD and its egamma-type parts (PD split by whether the egamma lepton jet is a photon-only or an
electron lepton jet; as sub-populations of a plane they do not qualify), DP, the four
egamma-isolation bands of DD (section 1d), and Dongyub Lee's data validation regions above
150 GeV, V1, V5, V6 and V7, reproduced in `nominal.ipynb` from the same histograms (displacement of
the muon LJ / the egamma LJ, egamma-LJ isolation, mass: {vr_defs}). Two windows: {r1_2_syst}; on the
sum, with the windows fully correlated, {r1_2_sum_syst}. Full correlation is a choice: the regions
that set the two systematics pull in opposite directions (κ_PD is {kpd_lo} below 300 GeV and
{kpd_hi} above), and a sum taken as uncorrelated would be smaller. Five bins:
{r1_5_syst}. The regions that qualify, per bin: two windows {r1_2_qual}; five bins {r1_5_qual}. So
in five bins the rule has no region in {r1_5_none}, and a five-bin search region would need an
assumed systematic there. In two windows the regions that qualify above 300 GeV are {qual_hi} (PD
with a precision of {prec_hi_PD}).

The table under the code lists, per bin, what a mass-binned search region needs: the background with
its first-order error (the linear propagation of the Poisson errors of B, C and D,
B·C/D·√(1/B + 1/C + 1/D)) and its bootstrap interval, κ in PP and PD, and the systematic with the
regions that qualify.""", N))
    cells.append(code(r"""KAPPA5 = bl.kappa_table(d2, NOMINAL, list(dl.FIVE), planes=dl.OPEN_PLANES)
display(KAPPA5[["plane", "bin", "A", "B", "C", "D", "kappa"]])
ST = {name: dl.stratified_table(d2, NOMINAL, bins) for name, bins in (("2 windows", dl.TWO), ("5 bins", dl.FIVE))}
for name, table in ST.items():
    print(f"route 1, {name} (toys with a finite sum: {100 * table.attrs['kept']:.0f}%)")
    display(dl.stratified_view(table))
for name, bins in (("2 windows", dl.TWO), ("5 bins", dl.FIVE), ("1 bin", dl.ONE)):
    print(f"closure of the summed estimate, {name}:",
          ", ".join(f"{p} {tl.fmt(dl.route_closure(d2, NOMINAL, p, bins), 2)}" for p in dl.OPEN_PLANES))"""))

    cells.append(md(r"""## 3. Route 2: a smooth transfer ratio

Instead of one ratio per bin, describe a(M) by a smooth function with few parameters, so that the
small bins borrow strength from the large ones. The function is a **logistic** in ln M,
a(M) = 1/(1 + exp(−(α + s·ln(M/200 GeV)))), equivalently the isolation odds
r(M) = a/(1 − a) = C/D = e^α (M/200 GeV)^s: α is the log-odds at 200 GeV and s the slope (s = 0 is a
flat a(M)). The fit is a binned maximum likelihood over the five bins: C_w out of C_w + D_w events
is a binomial count, which is the Poisson likelihood of C_w and D_w conditioned on their sum, so no
Gaussian approximation enters in the small bins. The bins sit at their geometric centres
({centres} GeV; the ≥ 600 GeV bin as 600–1000 GeV). The estimate is then A_smooth = Σ_w B_w r(M_w).
Its interval is from the Poisson bootstrap, refitting α and s on each toy. The **deviance** is twice
the log-likelihood ratio of the fit against a free a per bin; for a good fit it is about its number
of degrees of freedom (bins − 2). Where it is larger, the data scatter about the curve more than
binomial counts would, and the fit's errors and the spread of its toys are scaled by
√(deviance/ndf). The rule is applied to every fit whose deviance/ndf exceeds one: here {scaled}.

* PP: α = {al_PP}, s = {s_PP}, deviance {dev_PP}; Σ B r(M) = {sm_PP} ({sm_PP_iv}) against
  A = {A_PP}, so κ_smooth = A/A_smooth = {ks_PP};
* PD: α = {al_PD}, s = {s_PD}, deviance {dev_PD}; Σ B r(M) = {sm_PD} ({sm_PD_iv}) against
  A = {A_PD}, κ_smooth = {ks_PD};
* DD: α = {al_DD}, s = {s_DD}, deviance {dev_DD}; **A_smooth = {sm_DD} events** ({sm_DD_iv}).

PD's deviance says the logistic does not describe PD's five bins well (its a(M) zigzags, and the
≥ 600 GeV bin sits high); PP is described. In DD the per-bin terms B_w r(M_w) are {sm_parts}; the
last, {sm_600}, is B = 4 at ≥ 600 GeV times the fit's extrapolation r = {r_600} into a bin with no C
or D.

**Does the shape transfer?** The slopes: PP {s_PP}, PD {s_PD}, DD {s_DD}. Against their weighted
mean χ² = {slope_chi2}; DD differs from PD by {slope_dd_pd} standard deviations and from PP by
{slope_dd_pp} (PP from PD by {slope_pp_pd}), with the scaled errors. The slopes are consistent,
and DD's is the least precise. Taking PD's slope and fitting only DD's intercept gives α = {al_sh} and
**A_smooth = {sm_sh} events** ({sm_sh_iv}), with the toys of PD refitted alongside so that PD's
slope error is in the interval. That is close to the one-bin {one_DD}: PD's slope is nearly flat,
and with a flat a(M) the one-bin estimate is exact (section 1).

**The systematic by the rule** for this method, over the same validation regions as route 1 (each
region's own logistic fit, the closure of its smooth estimate; section 5 lists them): {syst_smooth}
(PP {ks_PP}, PD {ks_PD}); with PD's slope in every region, {syst_shared}. With PD's slope the method
carries a second systematic, the slope transfer itself: DD's own slope gives an estimate {transfer}
away.

*How to read the figure:* one panel per plane, a(M) with its 68% intervals, the fitted logistic (solid)
with the band of the 16th–84th percentiles of the refitted toys; in DD the dashed blue line is the
fit with PD's slope.""", N))
    cells.append(code(r"""FITS = {p: dl.smooth_route(d2, NOMINAL, p) for p in dl.PLANES}
SHARED = dl.smooth_route(d2, NOMINAL, "DD", slope_plane="PD")
display(dl.fit_view({**FITS, "DD with PD's slope": SHARED}))
SC = dl.slope_consistency(FITS)
print(f"slopes against their weighted mean {SC['mean']:.2f}: chi2 {SC['chi2']:.1f}/{SC['ndf']}, p = {SC['p']:.2f}; separations:",
      ", ".join(f"{a}-{b} {z:+.1f}" for (a, b), z in SC["pairs"].items()))
x = FITS["DD"]["x"]
parts = FITS["DD"]["counts"][:, 1] * dl.logistic_r(FITS["DD"]["alpha"], FITS["DD"]["slope"], np.exp(x) * dl.PIVOT)
print("DD, B_w r(M_w) per bin:", np.round(parts, 1), "sum", round(float(parts.sum()), 1))
fig = dp.fit_figure(PASS, FITS, SHARED)"""))

    cells.append(md(r"""## 4. Route 3: a mass-flattened isolation cut

If the isolation cut is chosen per mass bin so that a(M) is the same in every bin, the one-bin
estimate is exact again (section 1), with no fit and no binning of the search region. The stored
isolation axis offers the edges 0.1, 0.15, 0.2, 0.25, 0.3, 0.4, 0.5, 0.75 and 1. The rule used here:
keep 0.25 in the 150–200 GeV bin, which holds most of the background (the **anchor**; its a is the
target), and in every other bin take the edge whose a is closest to the target (a tie goes to the
edge nearer 0.25). Below 0.25 the band between the edge and 0.25 is a **buffer** that belongs to no
region: in DD it lies inside the hidden box, so a sideband reaching into it would be hidden; the fail
side stays at 0.25. The cut is calibrated two ways.

* **In PD's C and D** (large counts; target a = {target_PD}): edges {edges_PD} for the five bins.
  PD's a(M) is {aflat_PD_PD} after (χ² against a constant {flat_PD_PD}, before {flat0_PD}); applied
  to DD it gives a = {aflat_PD_DD} ({flat_PD_DD}; before {flat0_DD}).
* **In DD's own C and D** (target {target_DD}): edges {edges_DD}. DD's a(M) is {aflat_DD_DD}
  ({flat_DD_DD}). The 300–400 GeV bin of DD reads a = {dd34_range} at every edge: {dd34_c_low} of its
  {dd34_c} C events sit below 0.1 and its D = {dd34_d_high} at or above 1, so no stored edge moves
  it; {dd600_empty}. At these counts DD's a(M) is compatible with a constant already with the fixed
  cut ({flat0_DD}); **what the available edges cannot do is move the one bin that drives the mixing
  factor, so the best achievable, in the table, leaves the mixing where it was.** Applied to PD, DD's
  edges make PD's a(M) less flat ({flat_DD_PD}, against {flat0_PD} with the fixed cut): they follow
  DD's fluctuations.

With the flattened cut, the one-bin estimate in DD and the mixing factor it still carries:

* PD-calibrated: B'/C'/D' = {fbcd_PD}, **B'·C'/D' = {fe_PD} events** ({fe_PD_iv}), mixing factor
  {fmix_PD}; closure in PP {fk_PD_PP} and PD {fk_PD_PD} (mixing {fmix_PD_PP} and {fmix_PD_PD});
  systematic by the rule over the validation regions of section 2 with the same edges {fsyst_PD};
* DD-calibrated: B'/C'/D' = {fbcd_DD}, B'·C'/D' = {fe_DD} events ({fe_DD_iv}), mixing factor
  {fmix_DD}; closure in PP {fk_DD_PP} and PD {fk_DD_PD}; systematic {fsyst_DD}.

The search region itself changes with the cut (A' is the region below each bin's edge), so these
estimates are of a different region than the one-bin {one_DD} (B/C/D = {bcd_one} with the fixed
cut). The PD-calibrated edges above 0.25 ({up_PD}) move {moved_PD} events that were open in B into
the new search region A'; A' stays hidden. **Signal cost** (simulation, {n_sig}
2μ2e points with mediator mass {masses} GeV, at the stored cross sections): the search-region yield
with the flattened cut over the yield with the fixed 0.25 cut is {cost_PD} for the PD-calibrated
cut and {cost_DD} for the DD-calibrated one (lowest per mediator mass: PD-calibrated {cost_PD_per};
DD-calibrated {cost_DD_per}); the lowest is {cost_DD_worst}. The edges between 0.1 and 0.4 cost the
signal at most {max_cost}.

*How to read the figure:* left, the isolation edge per mass bin, fixed (grey dashed) and from each
calibration (where two choose the same edge their lines lie on top of each other); right, a(M) with
the fixed cut for PP, PD and DD (filled) and with each calibration's edges for PD and DD (open
squares PD-calibrated, open diamonds DD-calibrated), with 68% intervals.""", N))
    cells.append(code(r"""CHOICE, FLAT = {}, {}
for cal in ("PD", "DD"):
    cd = dl.odds_table(d2, NOMINAL, cal)
    CHOICE[cal] = dl.choose_edges(cd)
    print(f"{cal}-calibrated: target a = {CHOICE[cal]['target']:.3f}, edges {CHOICE[cal]['edges']}; a (C/D) per bin and edge, chosen in brackets:")
    display(dl.odds_view(cd, CHOICE[cal]))
    for p in dl.PLANES:
        FLAT[(cal, p)] = dl.flat_route(d2, NOMINAL, p, dl.FIVE, CHOICE[cal]["edges"])
BEFORE = {p: dl.flatness(CD[p]) for p in dl.PLANES}
for p, fl in BEFORE.items():
    print(f"fixed 0.25, {p}: a = {np.round(fl['a'], 2)}, chi2 against a constant {fl['chi2']:.1f}/{fl['ndf']} (p = {fl['p']:.2f})")
display(dl.flat_view(FLAT))
fig = dp.flatten_figure({f"{c}-calibrated": CHOICE[c]["edges"] for c in CHOICE}, BEFORE,
                        {(f"{c}-calibrated", p): FLAT[(c, p)]["flatness"] for c in CHOICE for p in ("PD", "DD")})"""))
    cells.append(code(r"""SIG_NAMES = [n for n in sorted(os.path.basename(p)[: -len(".coffea")] for p in tl.sample_paths(BASE, "signal")
                                if os.path.basename(p).startswith("2Mu2E_")) if nl.mediator_mass(n) >= 200]
COST = dl.signal_cost(BASE, SIG_NAMES, NOMINAL, dl.FIVE, {f"{c}-calibrated": CHOICE[c]["edges"] for c in CHOICE})   # simulation
print(f"{len(SIG_NAMES)} 2mu2e signal points: search-region yield with the flattened cut over the fixed 0.25 cut, per mediator mass")
display(dl.cost_summary(COST, [f"{c}-calibrated" for c in CHOICE]))"""))

    cells.append(md(r"""## 5. Comparison and proposal

One row per route, with the one-bin estimate of `nominal.ipynb` as the baseline. "Mixing still
carried" is the mixing factor the estimate does not remove (for route 1 in two windows, the
five-bin sum over the two-window one); the closure is A over the route's prediction in PP and PD
(the baseline and route 1: Poisson bootstrap of the four counts; route 2: refitted toys; route 3:
Clopper–Pearson-based intervals of the summed regions). The systematic is the rule of
`nominal.ipynb` section 8 over the same {n_rule} validation regions for every route (those of
section 2 that are not a sub-population of a plane), each region's κ computed with the route's own
method (per window for route 1); for the baseline it reproduces the {base_syst} of `nominal.ipynb`.
The second table lists those κ, "*" marking one not measured to 20% and brackets the one that sets
the systematic. In the first table, "free choices" counts what the analyst sets (bin boundaries, a functional form, an edge per bin),
"fitted parameters" what the data set; "rests on" the smallest count that drives the number; the
signal yield ratio is the search-region yield over the fixed cut's, min–max over the {n_sig} points.""", N))
    cells.append(code(r"""REC = dl.comparison(d2, NOMINAL, {c: CHOICE[c]["edges"] for c in CHOICE}, COST)
display(dl.comparison_view(REC))
display(dl.readings_view(REC))"""))

    cells.append(md(r"""**What the table says.** No route reduces PD's non-closure: PD's closure is {pd_all} in every row.
That non-closure is not mass mixing, since PD's own mixing factor is {pd_mix}; PD's κ does vary with
the mass ({kpd_lo} below 300 GeV, {kpd_hi} above), which is what route 1's per-window systematics
see. What the routes differ in is how they treat DD's rise of a(M) above 300 GeV, which rests on
D = 1 at 300–400 GeV. Route 1 in five bins and route 2 with DD's own slope take it at face value and
inherit its statistics (five bins: {r1_5}, {r1_5_iv}; smooth: {sm_DD}, {sm_DD_iv}). Route 2 with PD's
slope assumes it away and lands near the one-bin value ({sm_sh}). Route 3 leaves it in place: its
mixing factor stays {fmix_PD}, and its lower estimates ({fe_PD} and {fe_DD}) come from a changed
region (B/C/D = {bcd_one} becomes B'/C'/D' = {fbcd_PD} with the PD-calibrated edges), not from
removing the rise; the PD-calibrated edges also move {moved_PD} open B events into the hidden A'.

**Proposal to the group.** We propose route 1 at the granularity of two mass windows, 150–300 GeV and
≥ 300 GeV: estimate the background in each window from that window's own B, C and D, and give each
window its own closure systematic by the rule of `nominal.ipynb` section 8. It is the simplest
analysis that removes the mixing between the windows: no fit, no new cut, and one free choice, the
split at 300 GeV, which `binning.ipynb` makes for sensitivity anyway (a factor 2.6 in the expected
limit at 800 GeV from that split alone). It is the only binned route in which the rule yields a
systematic in every bin: {syst_w1} below 300 GeV (set by {syst_r1}) and {syst_w2} above (set by {syst_r2}). The
numbers this rests on: the two-window estimate {rec_est} events (68% interval {rec_iv}; {r1_lo} below
300 GeV and {r1_hi} above, {r1_hi_iv}), its closure in the open planes ({rec_k}, against {base_k} in
one bin), the per-window systematics just quoted, and the five-bin and smooth routes' dependence on
the one D event at 300–400 GeV. Its cost is statistical: the high window rests on D = {d_hi}, and
its interval spans a factor {hi_factor}.

Two routes tie on simplicity with it. The one-bin estimate with the mixing factor added as a
systematic also carries two systematics and no fitted parameter; it would carry the mixing as a
one-sided +{mix_excess} resting on the same {d_hi} events, where the two-window estimate carries it as
the statistical error of the high window. Route 2 with PD's slope also carries two systematics (the
closure, {syst_shared}, and the slope transfer, {transfer}), plus a fit whose form PD does not
support (deviance {dev_PD}). **What decides between them is D of the search plane above 300 GeV**
({d_hi} events now, 1 at 300–400 GeV): it sets the two-window estimate, the mixing factor and the
slope difference ({slope_dd_pd} standard deviations between DD and PD). The open bands of the search
plane (section 1d) hold no C or D event above 300 GeV and cannot help. DD's window-to-window
isolation odds differ from PD's by {sep_z} standard deviations now; at the present rates two
standard deviations need about {dneed2} D events above 300 GeV ({f2} times the data) and three about
{dneed3} ({f3} times). Route 3 is not proposed: it changes the search region, its five edges are free
choices, and it leaves DD's mixing where it was.

## Appendix: reproducibility

The data are read only through `transfer_lib.summed_cubes(BASE, "data", prefix="abcd8d")`, which sums
the merged 8D histograms and sets the hidden box ({box}) to NaN before anything is cached or
returned; A of DD above 150 GeV is NaN in every mass bin and is never read. The signal-cost table of
section 4 reads the signal histograms one point at a time (`decorrelation_lib.signal_cost`;
simulation is never hidden). Code: `decorrelation_lib.py` (the identity, a(M) and b(M), the mixing
factor and its bootstrap, the shape test, the three routes, the comparison), `decorrelation_plots.py`
(the figures), `_build_mass_decorrelation.py` (writes this notebook and computes every number in its
text with the same functions and seeds), `test_decorrelation_lib.py` (tests on synthetic
populations and histograms). On the LPC: `python _build_mass_decorrelation.py` then
`jupyter nbconvert --to notebook --execute --inplace mass_decorrelation.ipynb` in `sidm_venv`.""", N))

    nb = nbf.v4.new_notebook(cells=cells, metadata=KERNEL)
    nbf.write(nb, "mass_decorrelation.ipynb")


if __name__ == "__main__":
    numbers = compute()
    for key in sorted(numbers):
        print(f"{key}: {numbers[key]}")
    build(numbers)
