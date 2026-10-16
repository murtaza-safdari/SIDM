"""Build nominal.ipynb.

Run from this directory on the LPC after merge_abcd8d.py has filled the local directory
(ABCD8D_LOCAL, default /uscms_data/d3/murtazas/abcd8d_local):
    python _build_nominal.py
    jupyter nbconvert --to notebook --execute --inplace nominal.ipynb

The builder reads the same blinded cubes the notebook reads (``transfer_lib.summed_cubes``) and
computes every number that appears in the markdown with the functions the code cells call,
formatted with ``nominal_lib.fmt`` / ``nominal_lib.pct``, so that the prose and the tables cannot
drift apart.  Nothing is computed at import time.
"""
import os

import nbformat as nbf

KERNEL = {"kernelspec": {"name": "sidm_venv",
                         "display_name": "SIDM (LCG_107 Py3.11)",
                         "language": "python"},
          "language_info": {"name": "python"}}
BASE_DEFAULT = "/uscms_data/d3/murtazas/abcd8d_local"
SB_BINS = ((">=150", (150.0, None)), ("150-300", (150.0, 300.0)), (">=300", (300.0, None)))
WINDOWS = (("150–300", (150.0, 300.0)), ("≥ 300", (300.0, None)))
OPEN_BINS = (("150–200", (150.0, 200.0)), ("200–300", (200.0, 300.0)), ("150–300", (150.0, 300.0)))
EARLIER_SYST = 0.33          # the systematic of estimate.ipynb
RELAXED_PRECISION = 0.50     # the precision cut of the comparison in section 8
PROMPT_CTAU_MM = 0.01        # signal points with a shorter ctau are called prompt
# Dongyub Lee's 4mu numbers, quoted from his slides (2026-10-13 data VR; 2026-10-16 LOW and HIGH decks)
SLIDES_4MU = {"iso x iso": "4.66 ± 0.54", "muon iso x dphi": "1.06 ± 0.18", "range low": "0.3–1.3", "range high": "0.75 to above 2"}


def compute():
    """Every number of the prose, as formatted strings keyed by name."""
    import numpy as np
    from scipy import stats

    import abcd6d_io as io
    import binning_lib as bl
    import estimate_lib as el
    import nominal_lib as nl
    import plane_search_lib as ps
    import transfer_lib as tl

    f, pct = nl.fmt, nl.pct
    N = {}
    base = os.environ.get("ABCD8D_LOCAL", BASE_DEFAULT)
    data = tl.summed_cubes(base, "data", prefix="abcd8d")
    d2, d4 = data["2mu2e"], data["4mu"]
    nom = nl.NOMINAL
    his_wp = nl.HIS_WP

    def tri(r):
        return r["kappa"], r["minus"], r["plus"]

    def counts(r, regions="ABCD"):
        return "/".join(nl.fmt_count(r[x]) for x in regions)

    def rng(values, fmtr):
        values = list(values)
        return f"{fmtr(min(values))}–{fmtr(max(values))}"

    def names_and(items):
        items = list(items)
        if not items:
            return "none"
        return items[0] if len(items) == 1 else ", ".join(items[:-1]) + " and " + items[-1]

    N["box"] = io.blind_signature(prefix="abcd8d")
    N["hidden_cells"] = f"{int(np.isnan(d2.sumw).sum())}"
    N["n_toys"] = f"{nl.BOOTSTRAP_TOYS}"

    # section 2: the variables
    for p in ("DD", "PD", "PP"):
        reg = ps.regions(d2, nom, p)
        N[f"bd_{p}"] = f"{reg['B'][0] / reg['D'][0]:.1f}"
    dphi_edges = d2.spec.edges_of("abs_dphi")
    N["bd_flat"] = f"{(dphi_edges[-1] - nom.dphi_pass) / (nom.dphi_fail - dphi_edges[0]):.2f}"
    bkg = tl.summed_cubes(base, "bkg", prefix="abcd8d")["2mu2e"]
    neff = [tl.n_eff(ps.regions(bkg, nom, "DD")[r]) for r in "BCD"]
    N["neff_DD"] = names_and([f"{v:.1f}" for v in neff])
    del bkg

    # section 4: the estimate
    sr = nl.kappa_row(d2, nl.SR, nom)
    est = (sr["B C/D"], sr["B C/D err"])
    N.update(B=f"{sr['B']:.0f}", C=f"{sr['C']:.0f}", D=f"{sr['D']:.0f}", est=f(est, 1), est0=f(est, 0),
             est_rel=pct(est[1] / est[0]))
    hb = nl.interval68(nl.bootstrap_abcd([(sr["B"], sr["C"], sr["D"])])["one_bin"])
    N["hb"] = f"{hb[0]:.0f}–{hb[2]:.0f}"
    N["hb_err"] = f"−{est[0] - hb[0]:.0f}/+{hb[2] - est[0]:.0f}"
    N["est_err0"] = f"{est[1]:.0f}"
    N["pp_counts"] = counts(nl.kappa_row(d2, nl.our_vrs()[0], nom))

    # section 5: kappa where A is open
    vrs = {v.name: v for v in nl.our_vrs()}
    val = nl.kappa_table(d2, [vrs[n] for n in ("PP", "PD", "DP", "DD < 150")], nom).set_index("region")
    for name, key in (("PP", "PP"), ("PD", "PD"), ("DP", "DP"), ("DD < 150", "DD150")):
        r = val.loc[name]
        N[f"k_{key}"] = f(tri(r))
        N[f"n_{key}"] = counts(r)
        N[f"prec_{key}"] = pct(r["precision"])
        N[f"z_{key}"] = f(abs(r["kappa"] - 1) / r["sigma"], 1)
        N[f"closes_{key}"] = "closes" if abs(r["kappa"] - 1) <= r["sigma"] else "does not close"
    N["dev_PD"] = pct(val.loc["PD", "kappa"] - 1)
    N["prec_open"] = rng((val.loc["DP", "precision"], val.loc["DD < 150", "precision"]), pct)
    bands = nl.dd_band_vrs()
    bt = nl.kappa_table(d2, bands, nom).set_index("region")
    band_label = {v.name: f"egamma-LJ isolation {nl._range(v.iso1)}" for v in bands}
    parts, finite = [], []
    for v in bands:
        r = bt.loc[v.name]
        if np.isfinite(r["kappa"]):
            parts.append(f"{band_label[v.name]}: κ = {f(tri(r))} ({counts(r)})")
            finite.append(r)
        else:
            empty = [x for x in "ABCD" if r[x] == 0]
            parts.append(f"{band_label[v.name]}: κ undefined ({counts(r)}; {' and '.join(empty)} empty)")
    N["band_k"] = "; ".join(parts)
    N["band_names"] = names_and([nl._range(v.iso1) for v in bands])
    word = {2: "two", 3: "three", 4: "four", 5: "five", 6: "six"}
    N["n_bands"] = word.get(len(bands), f"{len(bands)}")
    N["n_bands_cap"] = N["n_bands"][:1].upper() + N["n_bands"][1:]
    N["band_prec"] = rng((r["precision"] for r in finite), pct) if finite else "undefined"
    band_qual = [r.name for r in finite if r["precision"] <= nl.MAX_PRECISION]
    N["band_verdict"] = ("none of them is measured to 20% or better, so none can carry the systematic of section 8"
                         if not band_qual else f"{names_and(band_qual)} {'is' if len(band_qual) == 1 else 'are'} "
                         "measured to 20% or better and enter the rule of section 8")
    N["band_values"] = names_and([
        f"{bt.loc[v.name, 'kappa']:.2f}" if np.isfinite(bt.loc[v.name, "kappa"]) else "undefined"
        for v in bands]) + " for egamma-LJ isolation " + names_and([nl._range(v.iso1) for v in bands])

    # section 6: Dongyub Lee's regions
    vrt = nl.vr_table(d2)
    his, ours = (vrt[vrt["wp"] == w].set_index("region") for w in ("his", "ours"))
    names = list(his.index)
    N["vr_quoted"] = "/".join(f"{his.loc[n, 'quoted kappa']:.2f}" for n in names)
    N["vr_his"] = "/".join(f"{his.loc[n, 'kappa']:.2f}" for n in names)
    N["vr_ours"] = "/".join(f"{ours.loc[n, 'kappa']:.2f}" for n in names)
    match = [n for n in names if round(his.loc[n, "kappa"], 2) == round(his.loc[n, "quoted kappa"], 2)]
    N["vr_match"] = f"{len(match)} of {len(names)}"
    nomatch = [f"{n} reads {his.loc[n, 'kappa']:.3f} here against {his.loc[n, 'quoted kappa']:.2f} on the slides "
               "and so rounds differently" for n in names if n not in match]
    N["vr_nomatch"] = ("; " + " and ".join(nomatch)) if nomatch else ""
    N["vr_dA"] = rng((abs(his.loc[n, "A"] - his.loc[n, "quoted A"]) / his.loc[n, "quoted A"] for n in names),
                     lambda x: pct(x, 1))
    N["vr_err"] = rng((his.loc[n, "minus"] for n in names), lambda x: f(x, 2))
    low = ["V2", "V3", "V4"]
    N["vr_low_ours"] = rng((ours.loc[n, "kappa"] for n in low), lambda x: f(x, 2))
    N["vr_low_his"] = rng((his.loc[n, "kappa"] for n in low), lambda x: f(x, 2))
    shift = nl.wp_shift_table(d2, nl.DONGYUB_VRS).set_index("region")
    N["vr_low_shift"] = rng((abs(shift.loc[n, "shift"]) for n in low), lambda x: f(x, 2))
    N["vr_low_shift_err"] = rng((shift.loc[n, "error of the shift"] for n in low), lambda x: f(x, 2))
    N["vr_low_z"] = rng((abs(shift.loc[n, "shift / error"]) for n in low), lambda x: f(x, 1))
    N["vr_low_meas"] = rng((ours.loc[n, "plus"] for n in low), lambda x: f(x, 2))
    high = ["V1", "V5", "V6", "V7"]
    N["vr_high_ours"] = rng((ours.loc[n, "kappa"] for n in high), lambda x: f(x, 2))
    N["vr_high_z"] = rng((abs(shift.loc[n, "shift / error"]) for n in high), lambda x: f(x, 1))
    N["wp_edges"] = (f"{nom.wp0:g} and {his_wp.wp0:g} in the isolation and {nom.dphi_pass:g} and "
                     f"{his_wp.dphi_pass:g} in |Δφ|")
    N["n_blocks"] = f"{len(nl.wp_shift(d2, nl.DONGYUB_VRS[0], his_wp, nom)['blocks'])}"
    prompt_mu = [v.name for v in nl.DONGYUB_VRS if v.mu == "P"]
    any_mu = [v.name for v in nl.DONGYUB_VRS if v.mu != "P"]
    any_mu_low = all(v.mass[1] is not None and v.mass[1] <= 150.0 for v in nl.DONGYUB_VRS if v.mu != "P")
    N["dy_prompt"] = names_and(prompt_mu)
    N["dy_any"] = (f"; {names_and(any_mu)} admit{'s' if len(any_mu) == 1 else ''} either state of the muon LJ"
                   + (" but lie below 150 GeV" if any_mu_low else "")) if any_mu else ""
    N["dy_none_high"] = ("None has a displaced muon LJ above 150 GeV" if any_mu_low else
                         "Some admit a displaced muon LJ above 150 GeV")
    N["sr_his_hidden"] = nl.kappa_row(d2, nl.SR, his_wp)["hidden regions"]
    touching = sorted(set(vrt.loc[vrt["hidden"], "region"]))
    N["vr_hidden"] = "None of the seven touches the hidden box" if not touching else (
        f"{', '.join(touching)} touch the hidden box and are shown as hidden")

    # section 7: what differs between the validation regions and the search region
    cls = nl.kappa_table(d2, [vrs["PD:g"], vrs["PD:e"]], nom).set_index("region")
    N["k_PDg"], N["k_PDe"] = f(tri(cls.loc["PD:g"])), f(tri(cls.loc["PD:e"]))
    N["k_PDg_asym"] = nl.fmt_asym(tri(cls.loc["PD:g"]))
    gs = nl.class_share_table(d2, nom, ("PD", "DD"), ("e", "g")).set_index(["plane", "region"])
    N["g_PD"] = "/".join(pct(gs.loc[("PD", r), "g share"]) for r in "ABCD")
    N["g_DD"] = "/".join(pct(gs.loc[("DD", r), "g share"]) for r in "BCD")
    side = {p: sum(gs.loc[(p, r), "g"] for r in "BCD") / sum(gs.loc[(p, r), "total"] for r in "BCD") for p in ("PD", "DD")}
    N["g_PD_bcd"], N["g_DD_bcd"] = pct(side["PD"]), pct(side["DD"])
    N["g_cmp"] = ("as photon-rich as" if abs(side["DD"] - side["PD"]) < 0.05 else
                  "more photon-rich than" if side["DD"] > side["PD"] else "less photon-rich than")
    N["g_A_PD"] = pct(gs.loc[("PD", "A"), "e share"])
    dsa_vrs = [nl.VR(f"{p}:{c}", "this analysis", p[0], p[1], split=c) for p in ("PP", "PD") for c in ("dsa0", "dsa1+")]
    dk = nl.kappa_table(d2, dsa_vrs, nom).set_index("region")
    for p in ("PP", "PD"):
        for c, key in (("dsa0", "0"), ("dsa1+", "1")):
            N[f"k_{p}_dsa{key}"] = f(tri(dk.loc[f"{p}:{c}"]))
        a, b = dk.loc[f"{p}:dsa0"], dk.loc[f"{p}:dsa1+"]
        err = np.hypot(a["plus"] if a["kappa"] < b["kappa"] else a["minus"], b["minus"] if a["kappa"] < b["kappa"] else b["plus"])
        N[f"dsa_z_{p}"] = f(abs(a["kappa"] - b["kappa"]) / err, 1)
        N[f"dsa_dir_{p}"] = "lower" if b["kappa"] < a["kappa"] else "higher"
    ds = nl.class_share_table(d2, nom, ("PD", "DD"), ("dsa0", "dsa1+")).set_index(["plane", "region"])
    N["dsa_PD"] = "/".join(pct(ds.loc[("PD", r), "dsa1+ share"]) for r in "BCD")
    N["dsa_DD"] = "/".join(pct(ds.loc[("DD", r), "dsa1+ share"]) for r in "BCD")
    ratio = [ds.loc[("DD", r), "dsa1+ share"] / ds.loc[("PD", r), "dsa1+ share"] for r in "BCD"]
    N["dsa_mix"] = f"{min(ratio):.1f}–{max(ratio):.1f}"
    same = N["dsa_dir_PP"] == N["dsa_dir_PD"]
    N["dsa_dir"] = (f"In both planes κ of the `dsa1+` class is the {N['dsa_dir_PD']} one" if same else
                    f"κ of the `dsa1+` class is {N['dsa_dir_PP']} in PP and {N['dsa_dir_PD']} in PD")
    kc_inc = nl.kappa_corrected(d2, nom, (None,))
    kc_dsa = nl.kappa_corrected(d2, nom, ("dsa0", "dsa1+"))
    kc_typ = nl.kappa_corrected(d2, nom, ("g", "e"))
    kc_typ_pd = nl.kappa_corrected(d2, nom, ("g", "e"), search="PD")
    e_counts = {x: ps.regions(d2, nom, "DD:e")[x][0] for x in "BCD"}
    N.update(kc_kPD=f(kc_inc["ratio"], 2), kc_inc=f"{kc_inc['corrected']:.0f}", kc_est=f(est[0], 1),
             kc_dsa=f"{kc_dsa['corrected']:.0f}", kc_dsa_ratio=f(kc_dsa["ratio"], 2),
             kc_dsa_terms=" + ".join(f"{p['kappa'][0]:.2f} × {p['direct']:.1f}" for p in kc_dsa["parts"].values()),
             kc_typ=f"{kc_typ['corrected']:.0f}", kc_typ_ratio=f(kc_typ["ratio"], 2),
             kc_typ_terms=" + ".join(f"{p['kappa'][0]:.2f} × {p['direct']:.1f}" for p in kc_typ["parts"].values()),
             g_direct_DD=f"{kc_typ['parts']['g']['direct']:.1f}", e_direct_DD=f"{kc_typ['parts']['e']['direct']:.1f}",
             g_pred_DD=pct(kc_typ["parts"]["g"]["share"]), g_pred_PD=pct(kc_typ_pd["parts"]["g"]["share"]),
             e_counts_DD="/".join(nl.fmt_count(e_counts[x]) for x in "BCD"),
             e_C=f"{e_counts['C']:.0f}", e_C_word="event" if e_counts["C"] == 1 else "events")
    for key, kc in (("typ", kc_typ), ("cls", kc_dsa)):
        total = sum(p["direct"] for p in kc["parts"].values())
        N[f"{key}_sum"] = f"{total:.1f}"
        N[f"{key}_k"] = f(kc["corrected"] / total, 3 if key == "cls" else 2)
        N[f"{key}_split"] = f(total / kc["direct"], 2)
    pull = abs(kc_dsa["ratio"] - kc_inc["ratio"]) >= 0.005
    N["dsa_pull"] = (f"down, from {N['kc_kPD']} to about {N['kc_dsa_ratio']} (section 8), against the photon argument of 7b, "
                     "which pulls it up" if same and N["dsa_dir_PD"] == "lower" and pull else
                     f"up, from {N['kc_kPD']} to about {N['kc_dsa_ratio']} (section 8), the same way as the photon argument of 7b"
                     if same and pull else "in no consistent direction")
    mass_vrs = [nl.VR(f"{p}, {w}", "this analysis", p[0], p[1], mass=m) for p in ("PP", "PD") for w, m in WINDOWS]
    mk = nl.kappa_table(d2, mass_vrs, nom).set_index("region")
    for p in ("PP", "PD"):
        for (w, _), key in zip(WINDOWS, ("lo", "hi")):
            N[f"k_{p}_{key}"] = f(tri(mk.loc[f"{p}, {w}"]))
        _, _, chi2, ndf = ps.kappa_vs_mass(d2, nom, p)
        N[f"chi2_{p}"] = f"{chi2:.1f}/{ndf}"
        N[f"chi2p_{p}"] = f"{stats.chi2.sf(chi2, ndf):.2f}"
    lo_cfg, hi_cfg = bl.bin_cfg(nom, 150.0, 300.0), bl.bin_cfg(nom, 300.0, None)
    for p in ("DD", "PD"):
        N[f"dshare_{p}"] = pct(ps.regions(d2, lo_cfg, p)["D"][0] / ps.regions(d2, nom, p)["D"][0])
    lo_d, hi_d = nl.route_estimates(d2, lo_cfg)["direct"], nl.route_estimates(d2, hi_cfg)["direct"]
    N.update(two_lo=f"{lo_d[0]:.1f}", two_hi=f"{hi_d[0]:.1f}", two_sum=f"{lo_d[0] + hi_d[0]:.0f}",
             d_hi=f"{ps.regions(d2, hi_cfg, 'DD')['D'][0]:.0f}")
    split_windows = tuple(m for _, m in WINDOWS)
    spl = {p: nl.split_ratio(d2, nom, p, split_windows) for p in ("PP", "PD", "DD")}
    for p in ("PP", "PD", "DD"):
        N[f"split_{p}"] = f(spl[p]["ratio"], 2)
        above = [w / t for w, t in zip(spl[p]["windows"][1], spl[p]["one"])]
        N[f"above_{p}"] = "/".join(pct(x) for x in above)
        N[f"cd_win_{p}"] = " and ".join(f"{b_c_d[1] / b_c_d[2]:.2f}" for b_c_d in spl[p]["windows"])
    N["split_open"] = rng((spl[p]["ratio"] for p in ("PP", "PD")), lambda x: f(x, 2))
    N["hi_bcd"] = "/".join(f"{x:.0f}" for x in spl["DD"]["windows"][1])
    boot = nl.bootstrap_abcd(spl["DD"]["windows"])
    boot_hi = nl.bootstrap_abcd([spl["DD"]["windows"][1]])
    i_hi, i_sum, i_ratio = nl.interval68(boot_hi["one_bin"]), nl.interval68(boot["split"]), nl.interval68(boot["ratio"])
    N["hi_int"] = f"{i_hi[0]:.0f}–{i_hi[2]:.0f}"
    N["sum_int"] = f"{i_sum[0]:.0f}–{i_sum[2]:.0f}"
    N["ratio_pct"] = "/".join(f"{x:.2f}" for x in i_ratio)
    N["ratio_below1"] = pct(float(np.mean(boot["ratio"] < 1)))
    N["boot_kept"] = pct(boot["kept"])
    drop = nl.kappa_drop([(tri(mk.loc[f"{p}, {WINDOWS[0][0]}"]), tri(mk.loc[f"{p}, {WINDOWS[1][0]}"])) for p in ("PP", "PD")])
    N["drop"], N["drop_z"] = f(drop[:2], 2), f(drop[2], 1)
    N["win_dev_max"] = pct(max(abs(k - 1) for k in mk["kappa"]))
    sb = bl.sideband_test_table(d2, nom, SB_BINS)
    for label, key in ((">=150", "all"), ("150-300", "lo"), (">=300", "hi")):
        r = sb[(sb["bin"] == label) & (sb["class"] == "all")].iloc[0]
        N[f"cd_{key}"] = f((r["ratio"], r["ratio minus"], r["ratio plus"]))
    r_lo = sb[(sb["bin"] == "150-300") & (sb["class"] == "all")].iloc[0]
    N["cd_lo_z"] = f((1.0 - r_lo["ratio"]) / r_lo["ratio plus"], 1)
    r = sb[(sb["bin"] == ">=150") & (sb["class"] == "all")].iloc[0]
    reaches_one = r["ratio"] - r["ratio minus"] <= 1.0 <= r["ratio"] + r["ratio plus"]
    N["cd_ratio0"] = f"{r['ratio']:.2f}"
    N["cd_transfer"] = (f"C/D does not transfer at the central value, {r['ratio']:.2f}, and the interval reaches one only at "
                        "its upper edge" if reaches_one else f"C/D does not transfer: {r['ratio']:.2f}, with one outside the interval")
    for c in ("dsa0", "dsa1+"):
        r = sb[(sb["bin"] == ">=150") & (sb["class"] == c)].iloc[0]
        N[f"cd_{c.replace('+', '')}"] = f((r["ratio"], r["ratio minus"], r["ratio plus"]))

    # section 8: the systematic
    sys_t = nl.systematic_table(d2, nl.our_vrs() + nl.DONGYUB_VRS)
    syst, syst_name = nl.closure_systematic(sys_t)
    quad, _ = nl.closure_systematic(sys_t, "quadrature")
    alt, alt_name = nl.closure_systematic(sys_t, qualifying=False)
    alt_row = sys_t.set_index("region").loc[alt_name]
    qual = sys_t[sys_t["qualifies"]].sort_values("candidate", ascending=False)
    N.update(syst=pct(syst), syst_name=syst_name, syst_ev=f(est[0] * syst, 0), syst_quad=pct(quad),
             syst_quad_ev=f(est[0] * quad, 0), syst_alt=pct(alt), syst_alt_name=alt_name,
             syst_alt_ev=f(est[0] * alt, 0), syst_33=pct(EARLIER_SYST), syst_33_ev=f(est[0] * EARLIER_SYST, 0),
             n_qual=f"{len(qual)}", n_rows=f"{len(sys_t)}", qual_names=", ".join(qual["region"]),
             second=f"{qual.iloc[1]['region']} ({pct(qual.iloc[1]['candidate'])})" if len(qual) > 1 else "none")
    N["syst_alt_why"] = (", a value set by its own statistical error, which is why the rule asks for 20% precision"
                         if alt_row["sigma"] >= alt_row["|kappa-1|"] else "")
    sys_50 = nl.systematic_table(d2, nl.our_vrs() + nl.DONGYUB_VRS, max_precision=RELAXED_PRECISION)
    s50, s50_name = nl.closure_systematic(sys_50)
    N.update(relaxed=pct(RELAXED_PRECISION), syst_50=pct(s50), syst_50_name=s50_name, syst_50_ev=f(est[0] * s50, 0),
             syst_50_prec=pct(sys_50.set_index("region").loc[s50_name, "precision"]))
    dev, dev_name = nl.largest_deviation(sys_t)
    dev_row = sys_t.set_index("region").loc[dev_name]
    N["dev_name"], N["dev_k"], N["dev_pct"] = dev_name, f(tri(dev_row)), pct(dev)
    N["dev_z"] = f(dev / dev_row["sigma"], 1)
    N["dev_carrier"] = (f", carried by its photon-only egamma LJs (PD:g, {f(tri(cls.loc['PD:g']))}); DD's sidebands are "
                        f"{N['g_cmp']} PD's" if dev_name == "PD" else "")
    dups = nl.same_regions(qual, "PP")
    distinct = qual[~qual["region"].isin(dups)]
    miss, touch = [], []
    for _, r in distinct.iterrows():
        d_, s_ = round(r["|kappa-1|"], 2), round(r["sigma"], 2)
        if d_ > s_:
            miss.append(r["region"])
        elif d_ == s_:
            touch.append(r["region"])
    order = list(sys_t["region"])
    miss.sort(key=order.index)
    N["miss"] = (f"{names_and(miss)} miss{'es' if len(miss) == 1 else ''} one at 68%"
                 + (f" ({names_and(touch)} touch{'es' if len(touch) == 1 else ''} it)" if touch else "")) if miss else \
        "every qualifying region contains one in its 68% interval"
    g_row = sys_t.set_index("region").loc["PD:g"]
    N["pdg_cand"] = pct(g_row["candidate"])
    N["typ_dev"] = pct(kc_typ["ratio"] - 1)
    lo_g, hi_g = sorted((g_row["candidate"], kc_typ["ratio"] - 1))
    N["syst_g"] = f"{100 * lo_g:.0f}–{100 * hi_g:.0f}%"
    n_distinct = len({tuple(r[x] for x in "ABCD") for _, r in qual.iterrows()})
    N["n_distinct"] = f"{n_distinct}"
    N["dup_clause"] = (f" ({' and '.join(dups)} at our working point {'is' if len(dups) == 1 else 'are'} PP)" if dups else "")
    v1_dup = nl.same_regions(nl.kappa_table(d2, [vrs["PP"], nl.DONGYUB_VRS[0]], nom), "PP")
    N["v1_pp"] = (f"At our working point V1 selects exactly the events of PP ({N['pp_counts']} in A/B/C/D), so the "
                  "two are one validation region, not two." if v1_dup else "V1 and PP differ in their events.")
    N["headline"] = f"{est[0]:.0f} ± {est[1]:.0f} (stat.) ± {est[0] * syst:.0f} (syst.)"
    N["total_err"] = f(float(np.hypot(est[1], est[0] * syst)), 0)
    nonq = sys_t[~sys_t["qualifies"]]["region"].tolist()
    N["nonqual_names"] = names_and(nonq)
    # every reading of what kappa of the search plane might be
    ro = nl.route_estimates(d2, nom)
    all_vrs = {v.name: v for v in nl.our_vrs() + nl.DONGYUB_VRS}
    sys_i = sys_t.set_index("region")

    def displaced(name):
        v = all_vrs[name]
        return v.mu == "D" or v.egm == "D"

    def label(name):
        same = nl.same_regions(sys_t, name)
        return name + (f" (= {', '.join(same)})" if same else "")

    rows_d = [n for n in sys_t["region"] if n not in dups and displaced(n)]
    precise = [n for n in rows_d if sys_i.loc[n, "precision"] <= nl.MAX_PRECISION]
    imprecise = [n for n in rows_d if n not in precise]
    reweighted = {"the reweighting by egamma type": kc_typ["ratio"], "the reweighting by DSA class": kc_dsa["ratio"]}
    up_name = max(reweighted, key=reweighted.get)
    up = reweighted[up_name] - 1
    derived = list(reweighted.items()) + [("the κ the B route implies", ro["B"][0] / est[0]),
                                          ("the mass-split ratio of 7d", spl["DD"]["ratio"])]
    group_a = [(n, sys_i.loc[n, "kappa"]) for n in precise] + derived
    a_text = [f"{n} {v:.2f}" for n, v in group_a]
    a_text[-1] += f" (bootstrap 16th/50th/84th percentiles {N['ratio_pct']})"
    side = ("all lie above one" if all(v > 1 for _, v in group_a) else
            "all lie below one" if all(v < 1 for _, v in group_a) else "lie on both sides of one")
    N["read_a"] = f"{side}: {names_and(a_text)}"
    imp_k = [f"{n} {f(tri(sys_i.loc[n]))}" if np.isfinite(sys_i.loc[n, "kappa"]) else f"{n} undefined" for n in imprecise]
    imp_prec = [sys_i.loc[n, "precision"] for n in imprecise if np.isfinite(sys_i.loc[n, "precision"])]
    N["read_imp"] = names_and(imp_k)
    N["read_imp_prec"] = rng(imp_prec, pct)
    rows_p = [n for n in sys_t["region"] if n not in dups and not displaced(n) and np.isfinite(sys_i.loc[n, "kappa"])]
    N["read_b"] = names_and([f"{label(n)} {sys_i.loc[n, 'kappa']:.2f}" for n in rows_p])
    N["read_b_dev"] = pct(max(abs(sys_i.loc[n, "kappa"] - 1) for n in rows_p))
    qual_vrs = sorted((v for v in nl.our_vrs() + nl.DONGYUB_VRS if v.name in set(distinct["region"])),
                      key=lambda v: order.index(v.name))
    nest = nl.nesting(qual_vrs)
    by_outer = {}
    for inner, outer in nest:
        by_outer.setdefault(outer, []).append(inner)
    nest_txt = [f"{names_and(v)} lie{'s' if len(v) == 1 else ''} inside {o}" for o, v in by_outer.items()]
    pairs = [f"{a.name} and {b.name}" for i, a in enumerate(qual_vrs) for b in qual_vrs[i + 1:]
             if nl.vr_overlap(a, b) and not nl.vr_contains(a, b) and not nl.vr_contains(b, a)]
    share_txt = [f"these pairs share events without one containing the other: {'; '.join(pairs)}"] if pairs else []
    N["overlap"] = "; ".join(([f"V1 is PP"] if "V1" in dups else []) + nest_txt + share_txt)
    N["n_disjoint"] = f"{len(nl.max_disjoint(qual_vrs))}"
    N["up_name"], N["up_k"], N["up"] = up_name, f(reweighted[up_name], 2), pct(max(up, syst))
    if up > syst:
        N["syst_phrase"] = (f"{N['syst']} by the rule of section 8 (set by {syst_name}); −{N['syst']}/+{N['up']} if the "
                            f"upper side is taken from the largest reweighted reading ({up_name}, {N['up_k']}), "
                            "which this notebook recommends; the choice is the group's")
    else:
        N["syst_phrase"] = f"{N['syst']} by the rule of section 8 (set by {syst_name}), which also covers the reweighted readings"

    # section 9: the borrowed-factor routes
    for x in "BCD":
        N[f"r_{x}"], N[f"r_{x}0"] = f(ro[x], 1), f"{ro[x][0]:.0f}"
    N["r_B_total"] = f((ro["B"][0] - est[0]) / np.hypot(est[1], est[0] * syst), 1)
    N["r_B_diff"] = f"{ro['B'][0] - est[0]:.0f}"
    N["implied_k"] = f(ro["B"][0] / est[0], 2)

    # section 10: opening the low-mass window of the search plane
    for label, (lo, hi) in OPEN_BINS:
        key = label.replace("–", "_")
        r = nl.route_estimates(d2, bl.bin_cfg(nom, lo, hi))
        N[f"o_{key}_direct"], N[f"o_{key}_B"] = f(r["direct"], 1), f(r["B"], 1)
        N[f"o_{key}_direct0"] = f"{r['direct'][0]:.0f}"
        N[f"o_{key}_prec"] = pct(nl.projected_precision(r["direct"][0], r["direct"][1]))
        N[f"o_{key}_prec_a"] = pct(1 / np.sqrt(r["direct"][0]))
        N[f"o_{key}_ratio"] = f(r["B"][0] / r["direct"][0], 1)
        reg = nl.ps.regions(d2, bl.bin_cfg(nom, lo, hi), "DD")
        sep, sep_b = nl.shared_b_separation(r["direct"][:2], r["B"], reg, nl.b_route_parts(d2, bl.bin_cfg(nom, lo, hi)))
        N[f"o_{key}_sep"], N[f"o_{key}_sep_b"] = f(sep, 1), f(sep_b, 1)
        fac = nl.route_ratio_factors(d2, bl.bin_cfg(nom, lo, hi))
        N[f"o_{key}_kmix"], N[f"o_{key}_sbr"] = f(fac["kappa_mix"], 2), f(fac["sideband"], 2)
        N[f"o_{key}_bcd"] = "/".join(nl.fmt_count(reg[x][0]) for x in "BCD")
    win_k = nl.kappa_table(d2, [nl.VR(f"PD:{c}, 150–300", "this analysis", "P", "D", mass=(150.0, 300.0), split=c)
                                for c in ("dsa0", "dsa1+")], nom).set_index("region")
    N["win_k_dsa0"], N["win_k_dsa1"] = (f(tri(win_k.loc[f"PD:{c}, 150–300"])) for c in ("dsa0", "dsa1+"))
    N["win_b_dsa0"], N["win_b_dsa1"] = (f"{ps.regions(d2, lo_cfg, f'DD:{c}')['B'][0]:.0f}" for c in ("dsa0", "dsa1+"))
    N["max_prec"] = pct(nl.MAX_PRECISION)
    names = sorted(os.path.basename(p)[: -len(".coffea")] for p in tl.sample_paths(base, "signal")
                   if os.path.basename(p).startswith("2Mu2E_"))
    shares = nl.signal_share_table(bl.signal_profiles(base, names, "2mu2e", nom, plane="DD", region="A"))
    mass = shares["point"].map(nl.mediator_mass)
    s200, heavy = shares[mass == 200], shares[mass >= 500]
    s200_below = (s200["below"] / s200["yield"]).where(s200["yield"] > 0).dropna()
    bshare = nl.below_share(shares)
    light = sorted(set(mass[mass < 200]))
    N["light_masses"] = " and ".join(f"{m:g}" for m in light)
    N["light_below"] = ", ".join(f"{rng(bshare[mass == m].dropna(), lambda x: f'{100 * x:.0f}')}% at {m:g} GeV" for m in light)
    N.update(n_points=f"{len(shares)}", n200=f"{len(s200)}", s200_lo=rng(s200["150-200"], pct),
             s200_150300=rng(s200["150-300"], pct), s200_below=rng(s200_below, lambda x: pct(x, 1)),
             s200_yield=rng(s200["yield"], lambda x: f(x, 2)), heavy_150300=pct(heavy["150-300"].max(), 1),
             heavy_masses=", ".join(f"{m:g}" for m in sorted(set(mass[mass >= 500]))),
             s200_win_max=f"{(s200['yield'] * s200['150-300']).max():.2f}")
    shown = set(shares["point"][mass == 200]) | set(bl.SIGNAL_POINTS["2mu2e"])
    N["n_shown"], N["n_est_other"] = f"{len(shown)}", f"{len(shown) - len(s200)}"
    p800 = shares.set_index("point").loc[nl.REFERENCE_POINT]
    N.update(p800_yield=f(p800["yield"], 1), p800_600=pct(p800[">=600"], 1), p800_300=pct(p800["150-300"], 1))

    # section 7f: signal in the validation regions
    sig_names = [n for n in names if nl.mediator_mass(n) >= 200]
    row_vrs = list(nl.our_vrs()) + list(nl.DONGYUB_VRS)
    sig_counts = nl.signal_region_counts(base, sig_names, row_vrs + [nl.SR], nom)
    ct = nl.signal_contamination_table(sig_counts, d2, row_vrs, nom)
    ct["cell"] = ct["ABCD region"] + " of " + ct["region"]
    N["n_sig"] = f"{len(sig_names)}"
    N["ref"] = nl.REFERENCE_POINT
    band_rows = ct[ct["region"].isin([v.name for v in nl.dd_band_vrs()])].sort_values("largest signal / observed",
                                                                                      ascending=False)
    b_top = band_rows.iloc[0]
    N["sig_bands"] = (f"the DD bands above 150 GeV hold up to {pct(b_top['largest signal / observed'])} signal "
                      f"({b_top['region']}, {b_top['point']}), the other "
                      f"{word.get(len(band_rows) - 1, len(band_rows) - 1)} bands {rng(band_rows['largest signal / observed'].iloc[1:], pct)}, which "
                      "rules them out as validation regions for that reason too")
    other = ct[~ct["region"].isin(band_rows["region"])].sort_values("largest signal / observed", ascending=False).iloc[0]
    N["sig_other"] = f"{other['cell']}, {pct(other['largest signal / observed'])} ({other['point']})"
    cell = ct.set_index("cell")
    N["ref_DP"] = pct(cell.loc["A of DP", "reference signal / observed"])
    N["ref_PD"] = pct(cell.loc["A of PD", "reference signal / observed"])
    N["ref_C"] = pct(cell.loc["C of search region", "reference signal / observed"])
    ref_yield = sig_counts[nl.REFERENCE_POINT][nl.SR.name]["A"]
    s_up = el.counting_limit(int(round(est[0])), est[0])
    N["ref_yield"] = f"{ref_yield:.1f}"
    N["n_obs_lim"], N["s_up"] = f"{int(round(est[0]))}", f"{s_up:.1f}"
    frac = s_up / ref_yield
    N["s_up_frac"] = pct(frac)
    N["ref_DP_scaled"] = pct(frac * cell.loc["A of DP", "reference signal / observed"], 1)
    N["ref_C_scaled"] = pct(frac * cell.loc["C of search region", "reference signal / observed"], 1)
    qset = set(distinct["region"])
    qrows = ct[ct["region"].isin(qset)]
    qmax = qrows.loc[qrows["largest signal / observed"].idxmax()]
    N["sig_qual"] = (f"{qmax['cell']}, {pct(qmax['largest signal / observed'])} from {qmax['point']}, whose search-region "
                     f"yield is {qmax['its search-region yield']:.1f} events")
    ctau = float(qmax["point"].split("_")[-1].replace("mm", "").replace("p", "."))
    N["sig_qual_prompt"] = (f" That point is a prompt one (cτ = {ctau:g} mm), to which the displaced search is not "
                            f"sensitive ({qmax['its search-region yield']:.1f} events in the search region), so its "
                            f"{pct(qmax['largest signal / observed'])} in {qmax['cell']} is not a contamination of the search."
                            if ctau <= PROMPT_CTAU_MM else "")
    N["sig_qual_ref"] = pct(qrows["reference signal / observed"].max(), 1)

    # section 11: 4mu
    lo4, hi4 = ps.regions(d4, nl.LOW, "DD"), ps.regions(d4, nom, "DD")
    N["dd4_lo"], N["dd4_hi"] = counts({r: lo4[r][0] for r in "ABCD"}), counts({r: hi4[r][0] for r in "BCD"}, "BCD")
    N["k4_lo"] = f(ps.kappa(lo4), 1)
    pairs = 0.0
    for cfg in (nl.LOW, nom):
        reg = ps.regions(d4, cfg, "DD:dsa2+:subdsa2+")
        pairs += sum(reg[r][0] for r in "ABCD" if np.isfinite(reg[r][0]))
    N["dsa_pairs"] = f"{pairs:.0f}"
    pp_hi, pp_lo = ps.regions(d4, nom, "PP"), ps.regions(d4, nl.LOW, "PP")
    ab_hi, ab_lo = tl.ratio(pp_hi["A"], pp_hi["B"]), tl.ratio(pp_lo["A"], pp_lo["B"])
    N["ab_pp_hi"], N["ab_pp_lo"], N["ab_pp_factor"] = f(ab_hi, 2), f(ab_lo, 2), f(ab_hi[0] / ab_lo[0], 2)
    readings4 = nl.fourmu_readings(d4, nom, nl.LOW)
    central = [k[0] for _, k in readings4 if np.isfinite(k[0])]
    med = float(np.median(central))
    N.update(four_range=rng(central, lambda x: f(x, 1)), four_med=f(med, 1),
             four_rel=rng((el.precision(k) for _, k in readings4 if np.isfinite(k[0])), lambda x: f"{100 * x:.0f}") + "%",
             four_factor=f(max(central) / min(central), 1))
    scan = nl.fourmu_b_scan(readings4)
    wide = [(b, *nl.expected_counting_limit(b)) for b in scan["b"]]
    wl = [lim for _, lim, _ in wide]
    N.update(b4_lo=f(scan["b"][0], 2), b4_hi=f(scan["b"][-1], 2), b4_factor=f(scan["factor"], 1),
             b4_err=pct(scan["error"]), b4_err_x=f"{scan['error']:.1f}", elim_lo=f(min(wl), 1), elim_hi=f(max(wl), 1),
             elim_factor=f(max(wl) / min(wl), 1))
    drop = next(((b0, l0, b1, l1, n1) for (b0, l0, _), (b1, l1, n1) in zip(wide, wide[1:]) if l1 < l0), None)
    N["elim_step"] = ""
    if drop:
        b0, l0, b1, l1, n1 = drop
        N["elim_step"] = (f" The expected limit is not monotonic in b, because the median count steps in integers: it "
                          f"falls from {l0:.1f} at b = {b0:.2f} to {l1:.1f} at b = {b1:.2f}, where the median count is "
                          f"still {n1}.")
    lims = [(b, *nl.expected_counting_limit(b)) for b in scan["central"]]
    N["b4"] = [b for b, _, _ in lims]
    N["elim"] = "; ".join(f"{lim:.1f} for b = {b:.2f} ({what}, median n = {n})" for (b, lim, n), what in
                          zip(lims, ("the lowest reading", "the median", "the highest reading")))
    lv = [lim for _, lim, _ in lims]
    N["elim_spread"] = pct(max(lv) / min(lv) - 1)
    N["elim_range"] = f"{min(lv):.1f}–{max(lv):.1f}"
    return N


def md(text, N=None):
    return nbf.v4.new_markdown_cell(text if N is None else text.format(**N))


def code(text):
    return nbf.v4.new_code_cell(text)


def build(N):
    cells = []
    cells.append(md(r"""# The 2μ2e background from the search plane's own sidebands

**Summary.** The 2μ2e search region holds an estimated **{headline}** background events
(2018 data, 59.8 fb⁻¹). The estimate is the plain ABCD method in the search plane itself: the
background in the search region, A, is predicted from the three neighbouring regions B, C and D of
the same plane as A = B·C/D = {est} events (from {B}, {C} and {D} events). That prediction is
right when the two variables that define the regions are independent for the background. We test
this in every region of the data where the corresponding A is open (not blinded): the three planes
in which at least one lepton jet is prompt; the search plane itself below 150 GeV and, above
150 GeV, in {n_bands} bands of the egamma lepton jet's isolation outside the search region; and the
seven validation regions of Dongyub Lee's programme, recomputed from these histograms. His regions
test the method on populations with a prompt muon lepton jet. The displacement that defines the
search region is tested on the egamma side by PD (only the egamma lepton jet displaced) and only
imprecisely by the search plane's own open regions: its bands above 150 GeV read {band_values},
and {band_verdict}. Among the regions measured to 20% or better, the largest deviation from one is
in {dev_name} (κ = {dev_k}), and at our working point {miss}. The systematic uncertainty is
{syst_phrase}. One caveat the rule does not cover: in the search plane the lepton-jet pair mass
correlates with both ABCD variables, and the estimate made separately below and above 300 GeV sums
to {two_sum} events, {split_DD} times the one-bin estimate; that ratio rests on {d_hi} events in D
above 300 GeV and falls below one in {ratio_below1} of the toys of a Poisson bootstrap (section
7d). The earlier notebooks borrowed a factor from PD and quoted {r_B0} events; here that number is
a cross-check.

**How to read this notebook.** Every number is from data unless it is marked as simulation. The
search region itself (region A of the search plane above 150 GeV, defined in section 3) is hidden:
the histograms are loaded with those cells set to NaN, any region that touches them prints as
"hidden", and nothing is computed from them. Everything else is open. Sections 1–3 explain the
method and the objects, section 4 gives the estimate, sections 5–7 test it, section 8 turns the
tests into a systematic uncertainty, sections 9–10 give the cross-checks and what would settle the
remaining question, section 11 covers 4μ, and section 12 summarises. The longer record is in
`estimate.ipynb` (the four estimate routes, the configuration scan, the simulation) and
`binning.ipynb` (the estimate per mass bin and the expected limits).

The first code cell prints the inputs: the hidden box (iso0 is the muon-LJ isolation, iso1 the
egamma-LJ isolation, mass the LJ-pair mass in GeV, dphi the azimuthal angle |Δφ| between the two
LJs), the shape of the histograms with their number of hidden cells, and two working points, ours
and Dongyub Lee's (section 6), each written as "dphi a/b x iso0 a/b": a is the edge at which the
variable passes (|Δφ| at or above it, the isolation below it) and b the edge at which it fails; the
two are equal here, so no buffer separates the regions.""", N))

    cells.append(code(r"""import dataclasses
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import stats

import abcd6d_io as io
import transfer_lib as tl
import transfer_plots as tp
import plane_search_lib as ps
import estimate_lib as el
import binning_lib as bl
import binning_plots as bp
import nominal_lib as nl
import nominal_plots as npl

pd.set_option("display.max_columns", 40)
pd.set_option("display.max_rows", 200)
pd.set_option("display.width", 250)
pd.set_option("display.max_colwidth", 200)
tp.set_style()

BASE = os.environ.get("ABCD8D_LOCAL", "/uscms_data/d3/murtazas/abcd8d_local")
print("8D inputs:", BASE)
data = tl.summed_cubes(BASE, "data", prefix="abcd8d")     # the only data path: hidden cells are NaN
print("hidden box:", io.blind_signature(prefix="abcd8d"))
for t, c in data.items():
    print(f"{t}: {c.sumw.shape}, {int(np.isnan(c.sumw).sum())} hidden cells")
d2, d4 = data["2mu2e"], data["4mu"]
NOMINAL, LOW, HIS_WP = nl.NOMINAL, nl.LOW, nl.HIS_WP
VRS = {v.name: v for v in nl.our_vrs()}
WINDOWS = [("M >= 150", NOMINAL),
           ("150-300", dataclasses.replace(NOMINAL, mass=(150.0, 300.0))),
           ("M >= 300", dataclasses.replace(NOMINAL, mass=(300.0, None)))]
print("our working point (WP):", NOMINAL.label())
print("Dongyub Lee's working point (WP):", HIS_WP.label().split(", M")[0])"""))

    cells.append(md(r"""## 1. ABCD in one page

The ABCD method predicts a background in a region we cannot look at from three regions we can. Take
two variables, x and y, that the signal and the background distribute differently, and cut on each:
the signal sits where both cuts pass. Those two cuts split the plane of (x, y) into four regions:

* **A** passes both cuts (the search region);
* **B** fails x and passes y;
* **C** passes x and fails y;
* **D** fails both.

If x and y are independent for the background, the fraction of events that pass x is the same
whether y passes or not: A/B = C/D, equivalently A/C = B/D, and so **A = B·C/D**. B, C and D are dominated by background,
so the right-hand side can be counted in data.

The test of the assumption is the closure ratio **κ = A·D/(B·C)**, measured in a plane where A is
open: κ = 1 within its error means the method works there. "Closes" in this notebook means exactly
that, κ consistent with one within its quoted 68% interval. Two things break the method: a
correlation between x and y in the background, or a background component that lives in one corner
of the plane only (a component in A that B, C and D do not see).

*How to read the figure:* the square is the plane, with x to the right and y upwards; the shaded
corner is A.""", None))
    cells.append(code(r"""fig = npl.abcd_cartoon()
plt.show()"""))

    cells.append(md(r"""## 2. Our objects and variables

A **lepton jet (LJ)** is a cluster of collimated leptons and photons, the signature of a light
dark photon decaying to a lepton pair. A **muon LJ** is built from muons, an **egamma LJ** from
electrons and photons. A 2μ2e event has one of each.

The two ABCD variables are properties of the pair:

* **x = the muon LJ's isolation.** Take the nearest jet within ΔR < 0.4 of the muon LJ and remove
  from its energy all of its electromagnetic and muon energy (the jet's charged and neutral
  electromagnetic fractions and its muon fraction, whichever particles carry them, not only the
  LJ's own); the remainder divided by the LJ's energy is the isolation. It is 0 when no jet is near.
  The histograms clip negative values at zero: the jet's energy fractions can sum to slightly more
  than one, which makes an isolated LJ's value slightly negative. A signal LJ is isolated; a muon LJ
  inside a QCD jet is not.
* **y = |Δφ|, the azimuthal angle between the two LJs.** The two LJs of a signal event come from
  one heavy particle at rest and are back-to-back (|Δφ| near π). Most background pairs are
  back-to-back too: among the events that fail the isolation cut, the ratio of pairs with |Δφ| ≥ 2.4
  to the rest, B/D, is {bd_DD} in DD, {bd_PD} in PD and {bd_PP} in PP (the planes of section 3),
  against {bd_flat} for a flat |Δφ| distribution; the signal is more sharply peaked at π.

The background in B, C and D is a mix (Drell–Yan, photon conversions, jets misidentified as lepton
jets) whose composition the simulation cannot give at these counts: its effective event counts in
B, C and D of DD are {neff_DD}. Independence of x and y is therefore tested in data, not argued
from the physics of the background.

The search region's other cuts (the validation regions relax them as stated in sections 5 and 6): the egamma LJ is isolated (its isolation, defined the same way,
< 0.5); the LJ-pair mass M(LJ, LJ) ≥ 150 GeV; both LJs displaced (section 3).

The **working point** (WP) is the pair of cuts that defines A: isolated means isolation < 0.25,
back-to-back means |Δφ| ≥ 2.4. A **sideband** is a region next to A that fails one or both cuts:
here B (isolation ≥ 0.25), C (|Δφ| < 2.4) and D. The sidebands start right at A's edges, with no
**buffer band** between them. The earlier version of `estimate.ipynb` left a buffer (B at
isolation ≥ 0.5, C and D at |Δφ| < 2.0); the buffers were most of the non-closure it saw in the
prompt plane, and were removed (`estimate.ipynb`, section 4).

*How to read the figure:* the isolation × |Δφ| distribution of the plane with both LJs prompt
(PP, section 3) above 150 GeV; colour is events per cell on a log scale; the isolation and the |Δφ|
bins are both non-uniform and are drawn with equal width, with the stored edges as tick labels; the
red lines are the working point, and each region's total is written in it ({pp_counts} events in
A/B/C/D).""", N))
    cells.append(code(r"""fig = npl.plane_figure(nl.plane_hist2d(d2, NOMINAL, "PP"), NOMINAL,
                       "2μ2e, PP plane (both LJs prompt), M ≥ 150 GeV, egamma-LJ isolation < 0.5", "nominal_plane_PP")
plt.show()
for p in ("DD", "PD", "PP"):
    reg = ps.regions(d2, NOMINAL, p)
    print(f"{p}: B/D = {reg['B'][0]:.0f}/{reg['D'][0]:.0f} = {reg['B'][0] / reg['D'][0]:.2f}")
BKG = tl.summed_cubes(BASE, "bkg", prefix="abcd8d")["2mu2e"]     # simulation, never hidden
print("simulated background in B/C/D of DD at the stored cross sections, yield (effective event count):",
      ", ".join(f"{r} {v[0]:.1f} ({tl.n_eff(v):.1f})" for r, v in ps.regions(BKG, NOMINAL, "DD").items() if r != "A"))
del BKG"""))

    cells.append(md(r"""## 3. The four planes and the hidden box

Each LJ is **prompt** or **displaced**, judged from hit counts in the tracker:

* a muon LJ is displaced when the largest number of **pixel hits** (hits in the innermost tracker
  layers) over its PF muons (the muons reconstructed with a tracker track) is ≤ 2, prompt when it
  is ≥ 3. A **DSA muon** (displaced standalone muon, reconstructed from the muon chambers alone)
  adds no pixel hits, so it never makes an LJ prompt: the LJ is displaced only when none of its PF
  muons has 3 or more pixel hits, and an LJ made only of DSA muons is displaced by that rule;
* an egamma LJ is displaced when the smallest number of **lostHits** (missing hits in the inner
  tracker before the first hit of an electron's track) over its electrons is ≥ 1, prompt when it is
  0. A photon-only egamma LJ has no track and counts as displaced.

That gives four planes, named by the muon LJ first: **PP**, **PD** (muon LJ prompt, egamma LJ
displaced), **DP** and **DD**. DD is the search plane. **Blinding** means that a region where the
signal is expected is not looked at until the method is fixed. The hidden box is region A of DD
above 150 GeV: muon-LJ isolation < 0.25, egamma-LJ isolation < 0.5, |Δφ| ≥ 2.4, M(LJ, LJ) ≥
150 GeV, both LJs displaced. Its B, C and D are
open, and so is everything below 150 GeV and every other plane.

*How to read the figure:* the four planes as a 2 × 2 grid (rows: muon LJ prompt, displaced;
columns: egamma LJ prompt, displaced), each split into A (top left), B, C and D with the data
counts above 150 GeV; the hatched cell is the hidden box.""", N))
    cells.append(code(r"""fig = bp.planes_cartoon(d2, NOMINAL, "2mu2e", routes=False)
plt.show()"""))

    cells.append(md(r"""## 4. The estimate

In DD above 150 GeV, B, C and D hold {B}, {C} and {D} events, and

**A = B·C/D = {est} events** (statistical error {est_rel}, from the Poisson errors of the three
counts propagated through the product to first order, as `estimate_lib` and `plane_search_lib` do
it). A Poisson bootstrap of B, C and D (each count redrawn {n_toys} times from a Poisson
distribution whose mean is the observed count) gives the asymmetric 68% interval {hb} ({hb_err});
the headline keeps the symmetric ± {est_err0} of first-order propagation.

The table gives the counts and the prediction; κ cannot be computed because A is hidden. In its
definition column, "D / D" gives the displacement of the muon LJ and of the egamma LJ (P prompt, D
displaced, "any" either), followed by the egamma-LJ isolation band and the mass window. This
number assumes κ = 1 in the search plane above 150 GeV. The rest of the notebook is about how well
that is justified and what uncertainty to assign to it.

*How to read the figure:* as in section 2, for the search plane; the hatched cells are the hidden
box and region A reads "hidden".""", N))
    cells.append(code(r"""display(nl.kappa_view(nl.kappa_table(d2, [nl.SR], NOMINAL)))
EST = nl.kappa_row(d2, nl.SR, NOMINAL)
HB = nl.interval68(nl.bootstrap_abcd([(EST["B"], EST["C"], EST["D"])])["one_bin"])
print(f"Poisson bootstrap of B, C, D ({nl.BOOTSTRAP_TOYS} toys): 16th/50th/84th percentiles {HB[0]:.1f}/{HB[1]:.1f}/{HB[2]:.1f}")
fig = npl.plane_figure(nl.plane_hist2d(d2, NOMINAL, "DD"), NOMINAL,
                       "2μ2e, DD plane (search plane), M ≥ 150 GeV, egamma-LJ isolation < 0.5", "nominal_plane_DD")
plt.show()"""))

    cells.append(md(r"""## 5. Validation 1: κ wherever A is open

A **validation region** is a region like the search region whose A can be looked at. Four come
for free: the planes PP, PD and DP above 150 GeV, and the search plane itself below 150 GeV.
{n_bands_cap} more are the search plane above 150 GeV with the egamma LJ not isolated, in the bands of
egamma-LJ isolation {band_names} (the hidden box ends at 0.5): they are the only open regions with
both LJs displaced in the search mass window, and they differ from the search region only in the
egamma LJ's isolation. The table gives A, B, C, D, the prediction B·C/D and κ with its 68%
interval. That interval combines the Clopper–Pearson intervals of the two ratios A/C and B/D (each
the interval of the fraction A/(A+C), or B/(B+D), mapped to the ratio): their relative errors are
added in quadrature, the lower side of A/C with the upper side of B/D and the other way round. The
error of B·C/D is first-order propagation of the three Poisson errors. "Precision" is the larger
side of κ's interval over κ.

* PP {closes_PP}: κ = {k_PP} ({n_PP}).
* PD is {dev_PD} above one, {z_PD} standard deviations: κ = {k_PD} ({n_PD}).
* DP, κ = {k_DP} ({n_DP}), and DD below 150 GeV, κ = {k_DD150} ({n_DD150}), are consistent with
  one but imprecise: their errors are {prec_open} of the value, so they cannot tell 1.0 from 1.3.
* The search plane above 150 GeV with the egamma LJ not isolated, {band_k}. Their errors are
  {band_prec} of the value: {band_verdict}.

*How to read the figure:* κ per region with its 68% interval; the dashed line is κ = 1; a value
above the drawn range is marked by an arrow at the top edge, with its value written beside the error bar, and an undefined κ (an empty
region) is marked as such.""", N))
    cells.append(code(r"""VAL1 = nl.kappa_table(d2, [VRS[n] for n in ("PP", "PD", "DP", "DD < 150")] + list(nl.dd_band_vrs()), NOMINAL)
display(nl.kappa_view(VAL1))
fig = npl.kappa_points_figure(*npl.table_points(VAL1), "2μ2e, our WP; A open", "nominal_kappa_open", ylim=(0, 2.6), width=16)
plt.show()"""))

    cells.append(md(r"""## 6. Validation 2: the seven validation regions of Dongyub Lee's programme

Dongyub Lee tested the same plane (muon-LJ isolation × |Δφ|) in seven data validation regions
(slides of 2026-10-13 and 2026-10-16), with his own analysis code reading his 6D histograms: the
group's earlier histograms of the data in six variables (the isolation and the displacement
variable of each LJ, |Δφ| and the LJ-pair mass), filled by his campaigns. This notebook's 8D
histograms add the mass of the muon LJ and its number of DSA muons. His histograms carry the data
of the search region unblinded: no blinding cut is applied in his campaigns. The table defines the regions: the
displacement of each LJ (P prompt, D displaced, "any" = either), the egamma-LJ isolation band and
the mass window. His working point is isolation < 0.3 and |Δφ| ≥ 2.6 (ours: < 0.25 and ≥ 2.4);
V1 keeps the egamma LJ isolated (< 0.5); V2–V4 put it in a non-isolated band (≥ 1 or ≥ 0.75),
which keeps their A away from the signal; V5–V7 use bands, 0.2–1 and 0.15–1, that straddle the
isolation boundary 0.5 of the search region, so they hold isolated and non-isolated egamma LJs
alike (V7 overlaps V1 in 0.15–0.5). {v1_pp}

His regions test the method on populations with a prompt muon LJ: {dy_prompt} require one{dy_any}.
{dy_none_high}, the population of the search region. The displacement that defines the search
region is tested by PD on the egamma side, imprecisely by DP and by DD below 150 GeV, and by the DD
bands of section 5. We recompute all seven from our histograms at his working point and at ours.

* **At his working point** the slides read {vr_quoted}, these histograms {vr_his} (errors
  {vr_err}): equal to the printed digit in {vr_match} regions{vr_nomatch}. The A counts agree
  within {vr_dA}. This is two codes on one data set: a cross-check of both codes,
  not an independent measurement. The A counts differ by selection details we cannot remove (our
  HEM15/16 veto applies to runs ≥ 319077 only, his to every run; our raw-event pre-filter; our
  isolation clipped at zero).
* **At our working point** the same regions read {vr_ours}. In the three low-mass regions V2–V4,
  κ falls from {vr_low_his} at his working point to {vr_low_ours} at ours, by {vr_low_shift}. The
  two working points share most events, so the error of the shift is not that of either κ
  ({vr_low_meas}): the edges {wp_edges} cut the plane into {n_blocks} disjoint blocks with
  independent Poisson counts, and propagating them to first order into the difference of the two κ
  gives {vr_low_shift_err}. The shift is {vr_low_z} standard deviations in V2–V4, and {vr_high_z}
  in the four regions above 150 GeV, which read {vr_high_ours} at ours. κ depends on the working
  point in the low-mass regions. No working-point systematic is assigned: the group sets the working
  point, κ and "closes" are statements at a working point, and the rule of section 8 is applied at
  ours.
* {vr_hidden}. His working point applied to the search plane puts
  {sr_his_hidden} of DD inside the box (C then reaches into 2.4 ≤ |Δφ| < 2.6 at isolation
  < 0.25): there is no DD estimate at his working point, as the second table shows.

*How to read the figure:* κ per region from the slides (grey squares), from these histograms at
his working point (orange diamonds) and at ours (black circles); the dashed line is κ = 1.""", N))
    cells.append(code(r"""VRT = nl.vr_table(d2, nl.DONGYUB_VRS)
display(nl.vr_comparison_view(VRT))
print("the search region at both working points")
display(nl.vr_comparison_view(nl.vr_table(d2, (nl.SR,))))
print("the shift of kappa from his working point to ours, with the error of the difference (shared events counted once)")
display(nl.wp_shift_table(d2, nl.DONGYUB_VRS).round(3))
fig = npl.kappa_points_figure(*npl.vr_points(VRT), "2μ2e, Dongyub Lee's validation regions", "nominal_kappa_dongyub",
                              ylim=(0.6, 1.5))
plt.show()"""))

    cells.append(md(r"""## 7. Validation 3: does κ depend on what differs between the validation regions and the search region?

The search region differs from every validation region in at least one respect. If κ depends on
one of them, the validation regions can mislead. Six checks.

### 7a. The displacement ladder

From PP (no displaced LJ) through PD and DP (one) to DD below 150 GeV and the DD bands above
150 GeV (two): κ = {k_PP}, {k_PD}, {k_DP}, {k_DD150}, and, in the bands, {band_values}. Nothing
trends with the number of displaced LJs beyond the errors; the two-displaced points carry errors of
{prec_DD150} and more.

*How to read the figure:* κ against the number of displaced LJs in the plane, each point labelled
with its region; the dashed line is κ = 1; a value above the drawn range is marked by an arrow at the top edge, with its value written beside the error bar; an undefined κ is marked as such.""", N))
    cells.append(code(r"""fig = npl.ladder_figure([(r["region"], (VRS[r["region"]].mu == "D") + (VRS[r["region"]].egm == "D"),
                          (r["kappa"], r["minus"], r["plus"])) for _, r in VAL1.iterrows()])
plt.show()"""))

    cells.append(md(r"""### 7b. The egamma LJ type

Here **PD:g** denotes the events of PD whose egamma LJ is photon-only (it contains no electron,
so no track), and **PD:e** those whose egamma LJ contains at least one electron, with or without
photons: an LJ with one electron and some photons is "e", its lost hits being the electron's.
PD's deviation sits in its photon-only egamma LJs: κ = {k_PDg} for them, {k_PDe} for the electron
ones, which make up {g_A_PD} of PD's A. Photon-only egamma LJs count as displaced (no track, section 3), so no prompt plane contains
them; they enter PD and DD only. The photon-only share per region is {g_PD} in A/B/C/D of PD and
{g_DD} in B/C/D of DD: summed over B, C and D, DD's sidebands ({g_DD_bcd} photon-only) are
{g_cmp} PD's ({g_PD_bcd}). If κ follows the photon-only share, κ of DD is more likely to sit near
PD's {k_PD} than near PP's {k_PP}. That is the case the systematic of section 8 has to cover.

*How to read the figure:* the photon-only share of the egamma LJs in each region of PD (blue) and
DD (green), with binomial errors and the number of events in brackets; A of DD is hidden.""", N))
    cells.append(code(r"""display(el.class_grid_table(d2, NOMINAL, ("PD",), (None, "g", "e")).drop(columns=["b-tag", "N_eff A"]))
SHARES_G = nl.class_share_table(d2, NOMINAL, ("PD", "DD"), ("e", "g"))
display(nl.class_share_view(SHARES_G, ("e", "g")))
fig = npl.photon_share_figure(SHARES_G)
plt.show()"""))

    cells.append(md(r"""### 7c. The muon LJ's DSA class

A muon LJ with a DSA muon (class `dsa1+`) is a different object from one without (`dsa0`). κ per
class: PP {k_PP_dsa0} (`dsa0`) and {k_PP_dsa1} (`dsa1+`), PD {k_PD_dsa0} and {k_PD_dsa1}. The
`dsa1+` share of B/C/D is {dsa_PD} in PD and {dsa_DD} in DD. The two classes' κ differ by
{dsa_z_PP} standard deviations in PP and {dsa_z_PD} in PD. {dsa_dir}. DD's sidebands hold
{dsa_mix} times PD's `dsa1+` share, so if the class difference carries over to DD it pulls κ of DD
{dsa_pull}. The direct estimate uses DD's own sidebands and needs no class weighting; the class
matters for any argument that borrows κ, or a ratio, from PD.""", N))
    cells.append(code(r"""display(el.class_grid_table(d2, NOMINAL, ("PP", "PD"), (None, "dsa0", "dsa1+")).drop(columns=["b-tag", "N_eff A"]))
display(nl.class_share_view(nl.class_share_table(d2, NOMINAL, ("PD", "DD"), ("dsa0", "dsa1+")), ("dsa0", "dsa1+")))"""))

    cells.append(md(r"""### 7d. The mass

κ in 150–300 GeV and above 300 GeV: PP {k_PP_lo} → {k_PP_hi}, PD {k_PD_lo} → {k_PD_hi}. κ falls
with the mass in both planes; the drop, combined over the two planes, is {drop} ({drop_z} standard
deviations). That is a test with one degree of freedom of a split at 300 GeV chosen after looking
at the bins. The table under the figure also gives the χ² of the five mass bins against a constant
κ equal to the integrated value, an omnibus test that spreads a monotone trend over four degrees of
freedom and so dilutes it: {chi2_PP} in PP (p = {chi2p_PP}) and {chi2_PD} in PD (p = {chi2p_PD}).
Both are shown.

For a one-bin search region the integrated κ of section 5 is the right test if the validation
plane's mass distribution resembles the search plane's, so that the mass dependence averages the
same way in both. In D it roughly does: {dshare_DD} of DD's D lies in 150–300 GeV, against
{dshare_PD} of PD's.

The same question asked of the estimate itself. The direct estimate above 300 GeV alone is
{two_hi} events from B/C/D = {hi_bcd}, with a 68% interval of {hi_int} from a Poisson bootstrap of
the three counts ({n_toys} toys, those with D = 0 dropped); in 150–300 GeV it is {two_lo}, and the
sum of the two windows, {two_sum} events, has the interval {sum_int}. The ratio of that sum to the
one-bin estimate, Σ_w B_w C_w/D_w ÷ (B·C/D), is {split_PP} in PP, {split_PD} in PD and {split_DD}
in DD; for DD its bootstrap 16th/50th/84th percentiles are {ratio_pct}, and it is below one in
{ratio_below1} of the toys. The ratio has a direct meaning: the one-bin κ equals Σ_w A_w ÷ (B·C/D),
so if the two variables are independent within each window, the one-bin estimate is low by exactly
that ratio. The mass window acts as a third variable that correlates with both ABCD variables in
DD: above 300 GeV lie {above_DD} of DD's B/C/D, so C/D is {cd_win_DD} in the two windows. It does
not in PP or PD ({above_PP} and {above_PD} of B/C/D above 300 GeV, C/D {cd_win_PP} and
{cd_win_PD}), where the ratio is {split_open}; the open planes therefore cannot test it, and in DD
it rests on {d_hi} events in D above 300 GeV. This is a caveat on the one-bin estimate that the
systematic rule of section 8 does not cover; a search region binned in mass, or more data, resolves
it. For a search region binned in mass, each bin needs its own κ, and the per-window deviations of
PP and PD, up to {win_dev_max}, would be the systematic. This notebook estimates the one-bin search
region.

*How to read the figure:* κ per mass window in PP (black) and PD (blue); the dashed line is κ = 1.
The table below it gives κ in the finer bins of the histograms.""", N))
    cells.append(code(r"""MASS = nl.kappa_table(d2, [nl.VR(f"{p}, {w}", "this analysis", p[0], p[1], mass=m)
                           for p in ("PP", "PD") for w, m in (("150–300", (150.0, 300.0)), ("≥ 300", (300.0, None)))], NOMINAL)
display(nl.kappa_view(MASS))
fig = npl.kappa_mass_figure({p: [(r["kappa"], r["minus"], r["plus"]) for _, r in MASS[MASS["region"].str.startswith(p)].iterrows()]
                             for p in ("PP", "PD")}, ["150–300", "≥ 300"])
plt.show()
display(ps.mass_table(d2, NOMINAL, planes=("PP", "PD")))
for p in ("PP", "PD"):
    _, _, chi2, ndf = ps.kappa_vs_mass(d2, NOMINAL, p)
    print(f"{p}: chi2 of the five bins against the integrated kappa {chi2:.1f}/{ndf}, p = {stats.chi2.sf(chi2, ndf):.2f}")
SPLIT = {p: nl.split_ratio(d2, NOMINAL, p, ((150.0, 300.0), (300.0, None))) for p in ("PP", "PD", "DD")}
print("the direct estimate per mass window, summed, against the one-bin estimate")
display(pd.DataFrame([{"plane": p, "B/C/D, 150-300": "/".join(f"{x:.0f}" for x in v["windows"][0]),
                       "B/C/D, >= 300": "/".join(f"{x:.0f}" for x in v["windows"][1]),
                       "B C/D summed over the windows": f"{v['split']:.1f}", "B C/D, one bin": f"{v['one_bin']:.1f}",
                       "ratio": f"{v['ratio']:.2f}"} for p, v in SPLIT.items()]))
BOOT = nl.bootstrap_abcd(SPLIT["DD"]["windows"])
BOOT_HI = nl.bootstrap_abcd([SPLIT["DD"]["windows"][1]])
print(f"DD, Poisson bootstrap ({nl.BOOTSTRAP_TOYS} toys; D > 0 in both windows in {nl.pct(BOOT['kept'])}), 16th/50th/84th percentiles:")
print("  B C/D above 300 GeV:", "/".join(f"{x:.1f}" for x in nl.interval68(BOOT_HI["one_bin"])))
print("  B C/D summed over the two windows:", "/".join(f"{x:.1f}" for x in nl.interval68(BOOT["split"])))
print("  ratio of the sum to the one-bin estimate:", "/".join(f"{x:.2f}" for x in nl.interval68(BOOT["ratio"])),
      f"; below one in {nl.pct(np.mean(BOOT['ratio'] < 1))} of the toys")"""))

    cells.append(md(r"""### 7e. The borrowed-factor premise: is PD's sideband ratio DD's?

The borrowed-factor routes of section 9 take a ratio from PD and apply it to DD. One of the
premises they rest on can be tested without opening A: C/D, the isolation pass/fail ratio among
events that fail |Δφ|, compared between the two planes. This is a comparison between planes, not a
test of independence inside DD (that would be A/B = C/D within DD, which needs A), and the direct
estimate does not depend on it: it uses DD's own C/D. The B route does: A/B of PD equals κ_PD × C/D
of PD, so B × (A/B)_PD is right for DD only if both κ and C/D transfer from PD to DD. DD over PD:
{cd_all} above 150 GeV ({cd_lo} in 150–300 GeV, {cd_hi} above 300 GeV); per DSA class {cd_dsa0}
(`dsa0`) and {cd_dsa1} (`dsa1+`). {cd_transfer}. With
DD's own C/D, the B route's {r_B0} events therefore require κ_DD = {implied_k}. In 150–300 GeV
alone the ratio reads {cd_lo}, {cd_lo_z} standard deviations low, the window where the direct and
the B-route predictions differ most (section 10).""", N))
    cells.append(code(r"""SB = bl.sideband_test_table(d2, NOMINAL, [(">=150", (150.0, None)), ("150-300", (150.0, 300.0)), (">=300", (300.0, None))])
display(SB.drop(columns=["ratio", "ratio minus", "ratio plus"]))"""))

    cells.append(md(r"""### 7f. Signal in the validation regions

A validation region that holds signal reads κ > 1 for a reason that has nothing to do with the
background: the signal adds to its A, and the test then measures the signal, not the method.
Signal in B, C or D of the search plane likewise biases the estimate. The table gives, for A of
every region of section 8 and for B, C and D of the search plane, the observed count and the
largest ratio of a signal point's yield to it over the {n_sig} 2μ2e signal points with a mediator
mass of 200 GeV or more, with the point that gives it ("at the point"), that point's yield in the
region ("its signal") and in the search region ("its search-region yield"), and the yield and
ratio of the reference point {ref}. A signal point is named by the mediator mass (the mass of the
heavy state produced in the collision, which decays to the two dark photons), the dark-photon mass
and the dark photon's proper decay length cτ, the mm value in the name. The yields are at the
stored cross sections, the cross sections stored with the samples, and every ratio in the table
scales linearly with the cross section.

At the stored cross sections {sig_bands}; elsewhere the largest ratio is in {sig_other}. For the reference point, A of DP
holds {ref_DP} of its observed count, A of PD {ref_PD} and C of the search plane {ref_C}. Among the
regions that qualify for the rule of section 8, the largest ratio is in {sig_qual}; for the
reference point it is at most {sig_qual_ref} in those regions.{sig_qual_prompt} The stored cross sections are not the ones that
matter: with the background at its estimate and {n_obs_lim} events observed, the 95% CL upper limit
on the signal count in the search region is {s_up} events (`estimate_lib.counting_limit`),
{s_up_frac} of the reference point's {ref_yield}. At the cross section where the reference point
gives that many events, each of its ratios is {s_up_frac} of the value in the table: {ref_DP_scaled}
in A of DP and {ref_C_scaled} in C of the search plane. For another point, scale its ratios by
{s_up} over its own search-region yield.""", N))
    cells.append(code(r"""SIG_NAMES = [n for n in sorted(os.path.basename(p)[: -len(".coffea")] for p in tl.sample_paths(BASE, "signal")
                                if os.path.basename(p).startswith("2Mu2E_")) if nl.mediator_mass(n) >= 200]
ROW_VRS = list(nl.our_vrs()) + list(nl.DONGYUB_VRS)
SIG_COUNTS = nl.signal_region_counts(BASE, SIG_NAMES, ROW_VRS + [nl.SR], NOMINAL)     # simulation, never hidden
CONTAM = nl.signal_contamination_table(SIG_COUNTS, d2, ROW_VRS, NOMINAL)
print(f"{len(SIG_NAMES)} 2mu2e signal points with a mediator mass >= 200 GeV, at the stored cross sections; "
      f"{nl.REFERENCE_POINT} has {SIG_COUNTS[nl.REFERENCE_POINT][nl.SR.name]['A']:.1f} events in the search region")
display(nl.contamination_view(CONTAM))
S_UP = el.counting_limit(int(round(EST["B C/D"])), EST["B C/D"])
print(f"95% CL upper limit on the signal count for n = {int(round(EST['B C/D']))} observed and b = {EST['B C/D']:.1f}: {S_UP:.1f} events")"""))

    cells.append(md(r"""## 8. The systematic uncertainty from the validation regions

The rule, written after the regions above were measured (so it is not blind to their values):

> Among the validation regions at our working point that differ from the search region only in
> the displacement of one or both lepton jets, the mass window, or the band of the egamma lepton
> jet's isolation (the partner of the muon lepton jet), and whose κ is measured to 20% or better, take for each the larger of |κ − 1| and the
> uncertainty facing one, and quote the largest of those.

"The uncertainty facing one" is the side of κ's 68% interval that points towards one; "measured to
20% or better" means that the larger side of the interval is at most 20% of κ. The table lists
every region: ours (PP, PD and its two egamma types, DP, DD below 150 GeV, the {n_bands} DD bands
above 150 GeV) and Dongyub Lee's seven at our working point; it is sorted by the number of respects
in which each region differs from the search region. Of the {n_rows} rows, {n_qual} qualify,
{n_distinct} distinct regions{dup_clause}; ordered by max(|κ − 1|, σ), largest first, they are
{qual_names}.

The photon-only part of PD (PD:g, κ = {k_PDg_asym}) is excluded by the rule's letter: it is a
sub-population of PD selected on the egamma LJ type, not a region of the plane. Whether the
inclusive κ is the right comparator depends on how much of the prediction each egamma type
carries, not on how many events it has. In DD, B·C/D per type is {g_direct_DD} events for the
photon-only LJs and {e_direct_DD} for the electron ones, so the photon-only LJs carry {g_pred_DD} of
the direct prediction in DD against {g_pred_PD} in PD. Applying PD's κ per egamma type to DD's
prediction per type, {kc_typ_terms} ≈ {kc_typ} events, gives {kc_typ_ratio} times the direct
estimate. That factor is the product of two: PD's κ per type applied to DD's per-type predictions,
{kc_typ} events against their sum of {typ_sum}, a factor {typ_k}; and that sum against the one-bin
{kc_est}, a factor {typ_split}, which rests on the {e_C} electron {e_C_word} in C of DD (the electron
LJs hold {e_counts_DD} events in B/C/D). The same reweighting by the muon LJ's DSA class (section
7c) gives {kc_dsa_ratio} = {cls_k} × {cls_split}. The quantity relevant to a systematic on A is the
reweighted estimate over the direct one, the product. The two reweightings pull in opposite
directions from the inclusive {kc_kPD}: by egamma type to {kc_typ_ratio}, by DSA class to
{kc_dsa_ratio}. The rule takes the inclusive value, {syst}; admitting PD:g as a region would give
{pdg_cand}, and taking the type reweighting at face value {typ_dev}.

**Result of the rule: {syst_name}, {syst}, ± {syst_ev} events.** The next largest qualifying value
is {second}.

Shown for comparison, not adopted:

* the quadrature form, √(|κ − 1|² + σ²): {syst_quad} (± {syst_quad_ev} events);
* admitting the imprecise regions: with no precision cut the rule is set by its noisiest row, {syst_alt_name}, {syst_alt}
  (± {syst_alt_ev} events){syst_alt_why}; with the cut relaxed to {relaxed}, {syst_50_name} ({syst_50_prec} precision) gives
  {syst_50} (± {syst_50_ev} events);
* the {syst_33} of `estimate.ipynb` (± {syst_33_ev} events).

**What the readings say beyond the rule.** They fall into two groups. The readings that bear on
the search plane's own κ, from the regions with a displaced LJ and from the search plane's
sidebands: those measured to 20% or better and the derived ones {read_a}; the imprecise ones scatter
({read_imp}; errors of {read_imp_prec} of the value). The regions in which neither LJ is required to be
displaced, all with a prompt or unconstrained muon LJ, scatter about one within ± {read_b_dev}:
{read_b}. The qualifying rows are not independent tests: {overlap}. At most
{n_disjoint} of the qualifying regions share no events with each other, so "{n_qual} of {n_rows}
rows qualify" is a listing, not a vote. Recommended: the rule's value on the lower side and the
largest reweighted reading on the upper side, −{syst}/+{up}, with the choice left to the group. In
one phrase, used in the summary as well: the systematic uncertainty is {syst_phrase}.

The 2μ2e background in the one-bin search region, with the rule's symmetric systematic:

**{headline} events.**

*How to read the figure:* κ per region at our working point, ordered from the regions closest to
the search region (fewest differences, in brackets) to the farthest; filled circles qualify for the
rule, open circles ({nonqual_names}) do not; the green band is 1 ± {syst}, the dashed line κ = 1; a
value above the drawn range is marked by an arrow at the top edge, with its value written beside the error bar; an undefined κ is marked
as such.""", N))
    cells.append(code(r"""SYS = nl.systematic_table(d2, nl.our_vrs() + nl.DONGYUB_VRS)
display(nl.systematic_view(SYS))
SYST, SYST_FROM = nl.closure_systematic(SYS)
QUAD, _ = nl.closure_systematic(SYS, "quadrature")
ALT, ALT_FROM = nl.closure_systematic(SYS, qualifying=False)
EST = nl.kappa_row(d2, nl.SR, NOMINAL)
SYS50 = nl.systematic_table(d2, nl.our_vrs() + nl.DONGYUB_VRS, max_precision=0.5)
S50, S50_FROM = nl.closure_systematic(SYS50)
print(f"rule: {nl.pct(SYST)} from {SYST_FROM}; quadrature {nl.pct(QUAD)}; no precision cut {nl.pct(ALT)} from {ALT_FROM}; "
      f"precision cut 50% {nl.pct(S50)} from {S50_FROM}")
print(f"estimate: {EST['B C/D']:.1f} ± {EST['B C/D err']:.1f} (stat.) ± {EST['B C/D'] * SYST:.1f} (syst.) events")
points, labels = npl.table_points(SYS)
SAME_AS_PP = nl.same_regions(SYS, "PP")
labels = [f"{r} ({n})" + (" = PP here" if r in SAME_AS_PP else "") for r, n in zip(labels, SYS["n differences"])]
fig = npl.kappa_points_figure(points, labels, "2μ2e, our WP; in brackets: respects differing from the search region",
                              "nominal_kappa_systematic", band=SYST, band_label=f"1 ± {nl.pct(SYST)}", ylim=(0, 2.6),
                              hollow_label="does not qualify", filled_label="qualifies", width=20)
plt.show()"""))

    cells.append(md(r"""## 9. Cross-check: the three borrowed-factor routes

`estimate.ipynb` estimated A of DD three more ways, each by borrowing a ratio from PD per DSA class
and summing: B × (A/B), C × (A/C) and D × (A/D) give {r_B}, {r_C} and {r_D} events. Their spread
is not a systematic uncertainty: they share DD's sidebands with the direct estimate, and each
assumes that a ratio of PD transfers to DD. The B route holds if A/B itself transfers from PD to
DD. A/B of PD is κ_PD × C/D of PD, and DD's C/D is {cd_all} times PD's (section 7e), so with DD's own
C/D the B route's {r_B0} events require κ_DD = {implied_k}. The C and D routes likewise assume that
A/C and A/D transfer. The direct estimate and the B route differ by {r_B_diff} events, {r_B_total} of the combined
statistical and systematic uncertainty of the direct estimate.

If the search plane's κ were PD's, A would be {kc_kPD} × {kc_est} ≈ {kc_inc} events, or
{kc_dsa_terms} ≈ {kc_dsa} with PD's κ per DSA class applied to DD's prediction per class; {r_B0}
assumes more than that.

The earlier notebooks made the B route the headline and the direct estimate a cross-check. This
one does the opposite: every qualifying validation region's κ lies within {syst} of one at our
working point, and the largest deviation, {dev_name} at {dev_z} standard deviations, is carried by
the systematic.""", N))
    cells.append(code(r"""display(el.routes_table(d2, WINDOWS))
display(el.class_factors_table(d2, WINDOWS[:1]))"""))

    cells.append(md(r"""## 10. What opening 150–300 GeV of the search plane would and would not settle

The direct test of κ in the search plane is to open A of DD in a window where the signal is small.
The background predicted there (direct, and B route):

* 150–200 GeV: {o_150_200_direct} (B route {o_150_200_B});
* 200–300 GeV: {o_200_300_direct} (B route {o_200_300_B});
* 150–300 GeV: {o_150_300_direct} (B route {o_150_300_B}).

The two predictions for 150–300 GeV differ by a factor {o_150_300_ratio}: the direct one uses DD's own
C/D in the window ({o_150_300_bcd} events in B/C/D), the B route uses PD's A/B. Their ratio is the
product of two factors. The first is PD's κ per DSA class weighted by the search plane's B,
Σ_c B_DD,c (A/B)_PD,c ÷ (B_DD (C/D)_PD), {o_150_300_kmix} here; it is close to one although PD's κ
per class in the window is not, {win_k_dsa0} (`dsa0`) and {win_k_dsa1} (`dsa1+`), because DD's B
in the window holds {win_b_dsa0} and {win_b_dsa1} events of the two classes and the two deviations
nearly cancel. The second is the sideband ratio (C/D)_PD / (C/D)_DD, {o_150_300_sbr} (the inverse of
the {cd_lo} of section 7e). The factor between the predictions is therefore almost all sideband
ratio.
A count near the first prediction would say that DD's own sidebands are right there; one near the
second, that the direct route under-predicts the window. The two predictions are {o_150_300_sep}
standard deviations apart if treated as independent, about {o_150_300_sep_b} once the shared B is
accounted for (both scale with DD's B, with different factors), so the window can tell them apart, but not sharply. κ of DD in the
window would be measured to about {o_150_300_prec} ({o_150_300_prec_a} from A's count alone, the
rest from B, C and D), which separates 1.0 from {o_150_300_ratio} but not from {kc_kPD}, and could not
qualify under the {max_prec} criterion of the rule of section 8; and section 7d shows that κ
changes with the mass, so the window's κ is not the search region's. 150–200 GeV alone
would measure it to about {o_150_200_prec}. The window is the only direct test there is; it is
not a sharp one.

The price is signal. Of the {n_points} 2μ2e signal points, the {n200} with a 200 GeV mediator put
{s200_lo} of their search-region yield in 150–200 GeV and {s200_150300} in 150–300 GeV (yields
{s200_yield} events at the stored cross sections of section 7f; below 150 GeV, in the already-open
DD validation region, they put a further {s200_below} of that yield, the column "below / yield" of
the table). The heavier points ({heavy_masses} GeV) put at most
{heavy_150300} in 150–300 GeV; the 800 GeV, 0.25 GeV, 0.25 mm point, for example, has {p800_yield}
events, {p800_300} of them below 300 GeV and {p800_600} at ≥ 600 GeV. Opening 150–300 GeV ends the
200 GeV search; opening 150–200 GeV costs about half of it. In absolute terms the loss is small:
at the stored cross sections the 200 GeV points put at most {s200_win_max} events in 150–300 GeV,
on a predicted background of {o_150_300_direct0} there (yields scale with the cross section, the
shares above do not). The decision is the group's.

The table also lists the {light_masses} GeV mediators. These masses are outside the range the
analysis targets (the group's grid of interest starts at 200 GeV) and are kept in the table for
completeness. For them much or most of the signal in A of DD sits below 150 GeV ({light_below}),
so DD below 150 GeV, used here as a validation region, would be signal-contaminated for those
hypotheses.

The first table under the background predictions gives, per mediator mass, the range of the shares
and of the yields; the second, per point, the yield in A of DD at M ≥ 150 GeV in events, its share
per mass bin, and the yield below 150 GeV in events ("below [events]") and as a share of the
search-region yield ("below / yield").

*How to read the figure:* one bar per signal point (the {n200} points with a 200 GeV mediator and
the {n_est_other} other points of `estimate.ipynb`), split into the share of its search-region
yield in each mass bin; the label gives the mediator mass, dark-photon mass and cτ, and the yield
in brackets (simulation).""", N))
    cells.append(code(r"""OPEN = [(label, bl.bin_cfg(NOMINAL, lo, hi)) for label, (lo, hi) in (("150-200", (150.0, 200.0)), ("200-300", (200.0, 300.0)), ("150-300", (150.0, 300.0)))]
display(el.routes_table(d2, OPEN)[["window", "DD B/C/D", "direct B C/D", "B x A/B per class", "DD A observed"]])
NAMES = sorted(os.path.basename(p)[: -len(".coffea")] for p in tl.sample_paths(BASE, "signal")
               if os.path.basename(p).startswith("2Mu2E_"))
SIGNAL = nl.signal_share_table(bl.signal_profiles(BASE, NAMES, "2mu2e", NOMINAL, plane="DD", region="A"))
print(f"{len(SIGNAL)} 2mu2e signal points; share of the search-region yield per mass bin, min-max per mediator mass, "
      "and the yields in A of DD above and below 150 GeV in events")
display(nl.share_summary(SIGNAL))
SHOWN = SIGNAL[SIGNAL["point"].str.startswith("2Mu2E_200GeV_") | SIGNAL["point"].isin(bl.SIGNAL_POINTS["2mu2e"])]
display(SHOWN.assign(**{"below / yield": SHOWN["below"] / SHOWN["yield"]})
        .rename(columns={"yield": "yield [events]", "below": "below [events]"}).round(3))
fig = npl.signal_share_figure(SHOWN.reset_index(drop=True))
plt.show()"""))

    cells.append(md(r"""## 11. 4μ in one section

In 4μ both LJs are muon LJs. The planes name the leading LJ (the one with the larger transverse
momentum) first, and each LJ is prompt or displaced by its pixel hits, as the muon LJ of 2μ2e. The
classes `dsaN:subdsaM` of the first table count the DSA muons in the leading (N) and the subleading
(M) muon LJ. The 4μ hidden box covers A above 150 GeV in DD and also in DP and PD, where the
signal is large too; those rows print "blind".

The search plane holds {dd4_lo} events in A/B/C/D below 150 GeV and {dd4_hi} in B/C/D above it: no
informative κ of the search plane can be measured (below 150 GeV it is {k4_lo}), and no B·C/D
exists above 150 GeV (C and D are empty). Six count-limited readings come from the open
neighbouring regions, as in `estimate.ipynb` (section 5.2), with one change. Three take B of DD
above 150 GeV times A/B of a plane: for PP, A/B above 150 GeV, which is open ({ab_pp_hi}), the same
mass window as the search region; for DP and PD, whose A is hidden there, A/B below 150 GeV. PP's
A/B is {ab_pp_factor} times higher above 150 GeV than below ({ab_pp_lo}), which suggests that the
DP and PD readings are biased low. The other three take A of DD below 150 GeV times the
above/below-150 GeV ratio of an open region of a neighbouring plane. The six read {four_range}
events, median {four_med}, each with a {four_rel} statistical error: the 4μ background is uncertain
by a factor {four_factor} between the lowest and the highest reading.

What that costs: the expected 95% CL upper limit on the signal count, the limit of a counting
experiment with a known background b (`estimate_lib.counting_limit`) at n equal to the median of a
Poisson distribution of mean b, is {elim}. Over the readings' central values the expected limit
moves by {elim_spread} ({elim_range}); allowing each reading its own statistical error, b from
{b4_lo} to {b4_hi} (a factor {b4_factor} either side of the median, which covers each reading's own
statistical error), it moves from {elim_lo} to {elim_hi}, a factor {elim_factor}.{elim_step} The tables below give the
limit at these five values of b.

Of the open DD events, {dsa_pairs} have both muon LJs made only of DSA muons. That was the
signature of the instrumental background seen in July 2026 before the cosmic and spread cuts were
added: events whose two muon LJs were made only of DSA muons, back-to-back, consistent with cosmic
rays. This statement is about the open sidebands; an artefact that is isolated and back-to-back
would sit in A, which the sidebands cannot test. What addresses it is the cosmic and spread cuts:
the cut on the opening angle between the two muons, cos α > −0.99, and the cuts on the spread of the
muons' transverse and longitudinal impact parameters within an LJ.

Dongyub Lee's 4μ regions (quoted from his slides): the isolation × isolation plane (the plane of
the two LJs' isolations, with |Δφ| as an event cut) fails ({SLIDES_4MU_iso}), the muon-isolation ×
|Δφ| plane closes ({SLIDES_4MU_dphi}) but its κ runs over {SLIDES_4MU_low} below 150 GeV and
{SLIDES_4MU_high} above as the isolation boundary moves. Nothing more can be done with these
counts.""", N))
    cells.append(code(r"""HIGH4, LOW4 = ("M >= 150", NOMINAL), ("M < 150", LOW)
display(el.count_grid(d4, [LOW4, HIGH4], "DD", [f"dsa{a}:subdsa{b}" for a in ("0", "1", "2+") for b in ("0", "1", "2+")]))
print("B of the search plane above 150 GeV times A/B below 150 GeV of the planes whose A is open there (the form of estimate.ipynb)")
display(el.sideband_transfer_table(d4, HIGH4, LOW4))
print("A of the DD plane below 150 GeV times the above/below-150 GeV ratio of the open neighbouring regions")
display(el.mass_transfer_table(d4, HIGH4, LOW4))
READ4 = nl.fourmu_readings(d4, NOMINAL, LOW)
print("the six readings used here (PP's A/B taken above 150 GeV, where it is open)")
display(pd.DataFrame([{"reading": label, "estimate of DD A (M >= 150)": tl.fmt(k, 2)} for label, k in READ4]))
B4 = nl.fourmu_b_scan(READ4)["b"]
print("expected 95% CL upper limit on the signal count at the median of the readings divided by the factor, the lowest "
      "reading, the median, the highest and the median times the factor")
display(pd.DataFrame([{"b": f"{b:.2f}", "median n": n, "expected limit": f"{lim:.2f}"}
                      for b, (lim, n) in ((b, nl.expected_counting_limit(b)) for b in B4)]))
print("95% CL upper limit on a signal count for n observed events and the same backgrounds")
display(el.counting_limit_table((0, 1, 2), [round(b, 2) for b in B4]))"""))

    cells.append(md(r"""## 12. Summary and to-do list

* The 2μ2e background in the one-bin search region is **{headline} events**, the search plane's
  own B·C/D; a Poisson bootstrap of B, C and D gives the 68% interval {hb}. The systematic
  uncertainty is {syst_phrase}.
* A caveat the rule does not cover: in DD the LJ-pair mass correlates with both ABCD variables, and
  the estimate made separately in 150–300 GeV and above 300 GeV sums to {two_sum} events, {split_DD}
  times the one-bin estimate (bootstrap 16th/50th/84th percentiles of the ratio {ratio_pct}, below
  one in {ratio_below1} of the toys). The open planes cannot test it (ratio {split_open}), and it
  rests on {d_hi} events in D above 300 GeV; a search region binned in mass, or more data, resolves it.
* κ closes in PP ({k_PP}). Dongyub Lee's seven regions agree between the two codes at his working
  point (to the printed digit in {vr_match}); they test the method on populations with a prompt
  muon LJ. At our working point {miss}. From his working point to ours κ falls by {vr_low_shift}
  in V2–V4, {vr_low_z} standard deviations of the shift.
* Among the regions measured to 20% or better, the largest deviation is in {dev_name} ({dev_k}){dev_carrier}.
  PD's C/D does not transfer to DD: DD over PD is {cd_all}.
* The open regions with both LJs displaced are imprecise: DD below 150 GeV ({k_DD150}) and the DD
  bands above 150 GeV ({band_values}); {band_verdict}. DP, with only the muon LJ displaced, reads
  {k_DP}.
* κ falls with the mass ({drop_z} standard deviations from 150–300 GeV to above 300 GeV, a split
  chosen after looking; the five-bin χ² against a constant κ is {chi2_PP} in PP and {chi2_PD} in
  PD). For a binned search region it is an effect of up to {win_dev_max} per window.
* Signal at the stored cross sections: {sig_bands}. In the regions that qualify for the rule the largest ratio is in {sig_qual}.{sig_qual_prompt} At the cross section the search can exclude for the reference point,
  its ratios are {s_up_frac} of their stored-cross-section values.
* The B route ({r_B0} events) is the cross-check; it holds if A/B transfers from PD, which with DD's
  own C/D requires κ_DD = {implied_k}. With PD's κ itself, A would be {kc_inc} ({kc_dsa} per DSA
  class).
* Opening 150–300 GeV of the search plane would measure κ there to about {o_150_300_prec}, not
  enough to qualify under the rule.
* 4μ: about {four_med} events (the median of six readings), uncertain by a factor {four_factor} between the
  readings; the expected limit moves by {elim_spread} over their central values and by a factor {elim_factor}
  ({elim_lo}–{elim_hi}) once each reading's own statistical error is allowed. Nothing more to do with these counts.

To do, for the group:

1. Fix the blinding box: Dongyub Lee's 6D histograms carry the data in the search region unblinded.
2. Dongyub to reproduce κ_PD from his 6D histograms, and to split his validation regions above
   150 GeV at 300 or 400 GeV.
3. State the criterion by which the working point is chosen.
4. Decide whether to open 150–300 GeV (or 150–200 GeV) of the search plane as a validation region.
5. Adopt the direct ABCD with the validation-region systematic as the convention.

## Appendix: reproducibility

The data are read only through `transfer_lib.summed_cubes(BASE, "data", prefix="abcd8d")`,
which sums the merged 8D histograms and sets the hidden box ({box}) to NaN before anything is
cached or returned; every projection touching a hidden cell is NaN and prints as "hidden". The
signal table of section 7f and the signal shares of section 10 read the signal histograms one point
at a time (`nominal_lib.signal_region_counts`, `binning_lib.signal_profiles`; simulation is never
hidden), and section 2 reads the summed background simulation (`transfer_lib.summed_cubes(BASE,
"bkg", prefix="abcd8d")`). Code: `nominal_lib.py` (validation
regions, the systematic rule, the working-point shift, the mass-split ratio and its bootstrap, the
plane histogram, the signal tables, the 4μ readings), `nominal_plots.py` (the figures),
`_build_nominal.py` (writes this notebook and computes every number in its text with the same
functions), `test_nominal_lib.py` (tests on synthetic histograms). On the LPC:
`python _build_nominal.py` then `jupyter nbconvert --to notebook --execute --inplace
nominal.ipynb` in `sidm_venv`.""", N))

    nb = nbf.v4.new_notebook(cells=cells, metadata=KERNEL)
    nbf.write(nb, "nominal.ipynb")


if __name__ == "__main__":
    numbers = compute()
    numbers.update({f"SLIDES_4MU_{k}": v for k, v in
                    (("iso", SLIDES_4MU["iso x iso"]), ("dphi", SLIDES_4MU["muon iso x dphi"]),
                     ("low", SLIDES_4MU["range low"]), ("high", SLIDES_4MU["range high"]))})
    for key in sorted(numbers):
        print(f"{key}: {numbers[key]}")
    build(numbers)
