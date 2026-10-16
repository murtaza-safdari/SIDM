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
# Dongyub Lee's 4mu numbers, quoted from his slides (2026-10-13 data VR; 2026-10-16 LOW and HIGH decks)
SLIDES_4MU = {"iso x iso": "4.66 ± 0.54", "muon iso x dphi": "1.06 ± 0.18", "range low": "0.3–1.3", "range high": "0.75 to above 2"}


def compute():
    """Every number of the prose, as formatted strings keyed by name."""
    import numpy as np

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

    def tri(r):
        return r["kappa"], r["minus"], r["plus"]

    def counts(r, regions="ABCD"):
        return "/".join(nl.fmt_count(r[x]) for x in regions)

    def rng(values, fmtr):
        values = list(values)
        return f"{fmtr(min(values))}–{fmtr(max(values))}"

    N["box"] = io.blind_signature(prefix="abcd8d")
    N["hidden_cells"] = f"{int(np.isnan(d2.sumw).sum())}"

    # section 4: the estimate
    sr = nl.kappa_row(d2, nl.SR, nom)
    est = (sr["B C/D"], sr["B C/D err"])
    N.update(B=f"{sr['B']:.0f}", C=f"{sr['C']:.0f}", D=f"{sr['D']:.0f}", est=f(est, 1), est0=f(est, 0),
             est_rel=pct(est[1] / est[0]))
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

    # section 6: Dongyub Lee's regions
    vrt = nl.vr_table(d2)
    his, ours = (vrt[vrt["wp"] == w].set_index("region") for w in ("his", "ours"))
    names = list(his.index)
    N["vr_quoted"] = "/".join(f"{his.loc[n, 'quoted kappa']:.2f}" for n in names)
    N["vr_his"] = "/".join(f"{his.loc[n, 'kappa']:.2f}" for n in names)
    N["vr_ours"] = "/".join(f"{ours.loc[n, 'kappa']:.2f}" for n in names)
    match = [n for n in names if round(his.loc[n, "kappa"], 2) == round(his.loc[n, "quoted kappa"], 2)]
    N["vr_match"] = f"{len(match)} of {len(names)}"
    nomatch = [f"{n}, which reads {his.loc[n, 'kappa']:.3f} here against {his.loc[n, 'quoted kappa']:.2f} on the slides"
               for n in names if n not in match]
    N["vr_nomatch"] = ("; the exception is " + " and ".join(nomatch)) if nomatch else ""
    N["vr_dA"] = rng((abs(his.loc[n, "A"] - his.loc[n, "quoted A"]) / his.loc[n, "quoted A"] for n in names),
                     lambda x: pct(x, 1))
    N["vr_err"] = rng((his.loc[n, "minus"] for n in names), lambda x: f(x, 2))
    low = ["V2", "V3", "V4"]
    N["vr_low_ours"] = rng((ours.loc[n, "kappa"] for n in low), lambda x: f(x, 2))
    N["vr_low_his"] = rng((his.loc[n, "kappa"] for n in low), lambda x: f(x, 2))
    N["vr_low_shift"] = rng((1 - ours.loc[n, "kappa"] / his.loc[n, "kappa"] for n in low), pct)
    N["vr_low_z"] = rng(((his.loc[n, "kappa"] - ours.loc[n, "kappa"]) / ours.loc[n, "plus"] for n in low), lambda x: f(x, 1))
    high = ["V1", "V5", "V6", "V7"]
    N["vr_high_ours"] = rng((ours.loc[n, "kappa"] for n in high), lambda x: f(x, 2))
    N["sr_his_hidden"] = nl.kappa_row(d2, nl.SR, nl.HIS_WP)["hidden regions"]
    touching = sorted(set(vrt.loc[vrt["hidden"], "region"]))
    N["vr_hidden"] = "None of the seven touches the hidden box" if not touching else (
        f"{', '.join(touching)} touch the hidden box and are shown as hidden")

    # section 7: what differs between the validation regions and the search region
    cls = nl.kappa_table(d2, [vrs["PD:g"], vrs["PD:e"]], nom).set_index("region")
    N["k_PDg"], N["k_PDe"] = f(tri(cls.loc["PD:g"])), f(tri(cls.loc["PD:e"]))
    gs = nl.class_share_table(d2, nom, ("PD", "DD"), ("e", "g")).set_index(["plane", "region"])
    N["g_PD"] = "/".join(pct(gs.loc[("PD", r), "g share"]) for r in "ABCD")
    N["g_DD"] = "/".join(pct(gs.loc[("DD", r), "g share"]) for r in "BCD")
    N["g_PD_side"] = rng((gs.loc[("PD", r), "g share"] for r in "BCD"), pct)
    N["g_DD_side"] = rng((gs.loc[("DD", r), "g share"] for r in "BCD"), pct)
    side = {p: sum(gs.loc[(p, r), "g"] for r in "BCD") / sum(gs.loc[(p, r), "total"] for r in "BCD") for p in ("PD", "DD")}
    N["g_PD_bcd"], N["g_DD_bcd"] = pct(side["PD"]), pct(side["DD"])
    N["g_cmp"] = ("as photon-rich as" if abs(side["DD"] - side["PD"]) < 0.05 else
                  "more photon-rich than" if side["DD"] > side["PD"] else "less photon-rich than")
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
    N.update(kc_kPD=f(kc_inc["ratio"], 2), kc_inc=f"{kc_inc['corrected']:.0f}", kc_est=f(est[0], 1),
             kc_dsa=f"{kc_dsa['corrected']:.0f}", kc_dsa_ratio=f(kc_dsa["ratio"], 2),
             kc_dsa_terms=" + ".join(f"{p['kappa'][0]:.2f} × {p['direct']:.1f}" for p in kc_dsa["parts"].values()),
             kc_typ=f"{kc_typ['corrected']:.0f}", kc_typ_ratio=f(kc_typ["ratio"], 2),
             kc_typ_terms=" + ".join(f"{p['kappa'][0]:.2f} × {p['direct']:.1f}" for p in kc_typ["parts"].values()),
             g_direct_DD=f"{kc_typ['parts']['g']['direct']:.1f}", e_direct_DD=f"{kc_typ['parts']['e']['direct']:.1f}",
             g_pred_DD=pct(kc_typ["parts"]["g"]["share"]), g_pred_PD=pct(kc_typ_pd["parts"]["g"]["share"]),
             e_counts_DD="/".join(nl.fmt_count(ps.regions(d2, nom, "DD:e")[x][0]) for x in "BCD"))
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
    lo_cfg, hi_cfg = bl.bin_cfg(nom, 150.0, 300.0), bl.bin_cfg(nom, 300.0, None)
    for p in ("DD", "PD"):
        N[f"dshare_{p}"] = pct(ps.regions(d2, lo_cfg, p)["D"][0] / ps.regions(d2, nom, p)["D"][0])
    lo_d, hi_d = nl.route_estimates(d2, lo_cfg)["direct"], nl.route_estimates(d2, hi_cfg)["direct"]
    two = (lo_d[0] + hi_d[0], float(np.hypot(lo_d[1], hi_d[1])))
    N.update(two_lo=f"{lo_d[0]:.1f}", two_hi=f"{hi_d[0]:.1f}", two_bin=f(two, 0),
             d_hi=f"{ps.regions(d2, hi_cfg, 'DD')['D'][0]:.0f}",
             two_ok="consistent with" if abs(two[0] - est[0]) <= np.hypot(two[1], est[1]) else "in tension with")
    drop = nl.kappa_drop([(tri(mk.loc[f"{p}, {WINDOWS[0][0]}"]), tri(mk.loc[f"{p}, {WINDOWS[1][0]}"])) for p in ("PP", "PD")])
    N["drop"], N["drop_z"] = f(drop[:2], 2), f(drop[2], 1)
    N["win_dev"] = rng((abs(k - 1) for k in mk["kappa"]), pct)
    N["win_dev_max"] = pct(max(abs(k - 1) for k in mk["kappa"]))
    sb = bl.sideband_test_table(d2, nom, SB_BINS)
    for label, key in ((">=150", "all"), ("150-300", "lo"), (">=300", "hi")):
        r = sb[(sb["bin"] == label) & (sb["class"] == "all")].iloc[0]
        N[f"cd_{key}"] = f((r["ratio"], r["ratio minus"], r["ratio plus"]))
    r_lo = sb[(sb["bin"] == "150-300") & (sb["class"] == "all")].iloc[0]
    N["cd_lo_z"] = f((1.0 - r_lo["ratio"]) / r_lo["ratio plus"], 1)
    r = sb[(sb["bin"] == ">=150") & (sb["class"] == "all")].iloc[0]
    reaches_one = r["ratio"] - r["ratio minus"] <= 1.0 <= r["ratio"] + r["ratio plus"]
    N["cd_verdict"] = ("is consistent with the failing-|Δφ| half of DD being like PD's" if reaches_one else
                       "finds the failing-|Δφ| half of DD unlike PD's")
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
    qual = sys_t[sys_t["qualifies"]].sort_values("candidate", ascending=False)
    N.update(syst=pct(syst), syst_name=syst_name, syst_ev=f(est[0] * syst, 0), syst_quad=pct(quad),
             syst_quad_ev=f(est[0] * quad, 0), syst_alt=pct(alt), syst_alt_name=alt_name,
             syst_alt_ev=f(est[0] * alt, 0), syst_33=pct(EARLIER_SYST), syst_33_ev=f(est[0] * EARLIER_SYST, 0),
             n_qual=f"{len(qual)}", n_rows=f"{len(sys_t)}", qual_names=", ".join(qual["region"]),
             second=f"{qual.iloc[1]['region']} ({pct(qual.iloc[1]['candidate'])})" if len(qual) > 1 else "none")
    g_row = sys_t.set_index("region").loc["PD:g"]
    N["pdg_cand"] = pct(g_row["candidate"])
    N["typ_dev"] = pct(kc_typ["ratio"] - 1)
    lo_g, hi_g = sorted((g_row["candidate"], kc_typ["ratio"] - 1))
    N["syst_g"] = f"{100 * lo_g:.0f}–{100 * hi_g:.0f}%"
    N["syst_both"] = f"{N['syst']} by the rule; {N['syst_g']} if the photon-only sub-population is taken as the carrier"
    dups = nl.same_regions(qual, "PP")
    n_distinct = len({tuple(r[x] for x in "ABCD") for _, r in qual.iterrows()})
    N["n_distinct"] = f"{n_distinct}"
    N["dup_clause"] = (f" ({' and '.join(dups)} at our working point {'is' if len(dups) == 1 else 'are'} PP)" if dups else "")
    v1_dup = nl.same_regions(nl.kappa_table(d2, [vrs["PP"], nl.DONGYUB_VRS[0]], nom), "PP")
    N["v1_pp"] = (f"At our working point V1 selects exactly the events of PP ({N['pp_counts']} in A/B/C/D), so the "
                  "two are one validation region, not two." if v1_dup else "V1 and PP differ in their events.")
    N["headline"] = f"{est[0]:.0f} ± {est[1]:.0f} (stat.) ± {est[0] * syst:.0f} (syst.)"
    N["total_err"] = f(float(np.hypot(est[1], est[0] * syst)), 0)

    # section 9: the borrowed-factor routes
    ro = nl.route_estimates(d2, nom)
    for x in "BCD":
        N[f"r_{x}"], N[f"r_{x}0"] = f(ro[x], 1), f"{ro[x][0]:.0f}"
    N["r_B_syst"] = f((ro["B"][0] - est[0]) / (est[0] * syst), 1)
    N["r_B_total"] = f((ro["B"][0] - est[0]) / np.hypot(est[1], est[0] * syst), 1)
    N["r_B_diff"] = f"{ro['B'][0] - est[0]:.0f}"
    N["implied_k"] = f(ro["B"][0] / est[0], 2)

    # section 10: opening the low-mass window of the search plane
    for label, (lo, hi) in OPEN_BINS:
        key = label.replace("–", "_")
        r = nl.route_estimates(d2, bl.bin_cfg(nom, lo, hi))
        N[f"o_{key}_direct"], N[f"o_{key}_B"] = f(r["direct"], 1), f(r["B"], 1)
        N[f"o_{key}_prec"] = pct(nl.projected_precision(r["direct"][0], r["direct"][1]))
        N[f"o_{key}_prec_a"] = pct(1 / np.sqrt(r["direct"][0]))
        N[f"o_{key}_ratio"] = f(r["B"][0] / r["direct"][0], 1)
        reg = nl.ps.regions(d2, bl.bin_cfg(nom, lo, hi), "DD")
        sep, sep_b = nl.shared_b_separation(r["direct"][:2], r["B"], reg, nl.b_route_parts(d2, bl.bin_cfg(nom, lo, hi)))
        N[f"o_{key}_sep"], N[f"o_{key}_sep_b"] = f(sep, 1), f(sep_b, 1)
        fac = nl.route_ratio_factors(d2, bl.bin_cfg(nom, lo, hi))
        N[f"o_{key}_kmix"], N[f"o_{key}_sbr"] = f(fac["kappa_mix"], 2), f(fac["sideband"], 2)
        N[f"o_{key}_bcd"] = "/".join(nl.fmt_count(reg[x][0]) for x in "BCD")
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
             heavy_masses=", ".join(f"{m:g}" for m in sorted(set(mass[mass >= 500]))))
    p800 = shares.set_index("point").loc["2Mu2E_800GeV_0p25GeV_0p25mm"]
    N.update(p800_yield=f(p800["yield"], 1), p800_600=pct(p800[">=600"], 1), p800_300=pct(p800["150-300"], 1))

    # section 11: 4mu
    lo4, hi4 = ps.regions(d4, nl.LOW, "DD"), ps.regions(d4, nom, "DD")
    N["dd4_lo"], N["dd4_hi"] = counts({r: lo4[r][0] for r in "ABCD"}), counts({r: hi4[r][0] for r in "BCD"}, "BCD")
    pairs = 0.0
    for cfg in (nl.LOW, nom):
        reg = ps.regions(d4, cfg, "DD:dsa2+:subdsa2+")
        pairs += sum(reg[r][0] for r in "ABCD" if np.isfinite(reg[r][0]))
    N["dsa_pairs"] = f"{pairs:.0f}"
    readings = []
    for plane in ("PP", "DP", "PD"):
        r = ps.regions(d4, nl.LOW, plane)
        readings.append(tl._pred(hi4["B"], tl.ratio(r["A"], r["B"])))
    for plane, region in (("PP", "A"), ("DP", "B"), ("PD", "B")):
        ratio = tl.ratio(ps.regions(d4, nom, plane)[region], ps.regions(d4, nl.LOW, plane)[region])
        readings.append(tl._pred(lo4["A"], ratio))
    central = [r[0] for r in readings if np.isfinite(r[0])]
    med = float(np.median(central))
    N.update(four_range=rng(central, lambda x: f(x, 1)), four_med=f(med, 1),
             four_rel=rng((el.precision(r) for r in readings if np.isfinite(r[0])), lambda x: f"{100 * x:.0f}") + "%")
    for n in (1, 2):
        a, b = el.counting_limit(n, med / 2.5), el.counting_limit(n, med * 2.5)
        N[f"lim{n}"] = f"{a:.1f} → {b:.1f} ({pct(1 - b / a)})"
    N["lim_b"] = f"{med / 2.5:.1f}–{med * 2.5:.1f}"
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
in which at least one lepton jet is prompt, the search plane itself below 150 GeV, and the seven
validation regions of Dongyub Lee's programme, recomputed from these histograms. In the regions
measured to 20% or better the method holds within 20%; the largest deviation is in the plane where
only the egamma lepton jet is displaced (PD, κ = {k_PD}). That deviation sets the systematic
uncertainty: {syst} by the rule of section 8, {syst_g} if the photon-only egamma lepton jets, which
carry PD's deviation, are taken as what decides κ of the search plane; that choice is put to the
group. The earlier notebooks borrowed a factor from that plane and quoted {r_B0} events; here
that number is a cross-check.

**How to read this notebook.** Every number is from data unless it is marked as simulation. The
search region itself (region A of the search plane above 150 GeV, defined in section 3) is hidden:
the histograms are loaded with those cells set to NaN, any region that touches them prints as
"hidden", and nothing is computed from them. Everything else is open. Sections 1–3 explain the
method and the objects, section 4 gives the estimate, sections 5–7 test it, section 8 turns the
tests into a systematic uncertainty, sections 9–10 give the cross-checks and what would settle the
remaining question, section 11 covers 4μ, and section 12 summarises. The longer record is in
`estimate.ipynb` (the four estimate routes, the configuration scan, the simulation) and
`binning.ipynb` (the estimate per mass bin and the expected limits).""", N))

    cells.append(code(r"""import dataclasses
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

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
print("nominal configuration:", NOMINAL.label())
print("Dongyub Lee's working point:", HIS_WP.label().split(", M")[0])"""))

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
  LJ's own); the remainder divided by the LJ's energy is the isolation. It is 0 when no jet is near. A signal LJ is isolated;
  a muon LJ inside a QCD jet is not.
* **y = |Δφ|, the azimuthal angle between the two LJs.** The two LJs of a signal event come from
  one heavy particle at rest and are back-to-back (|Δφ| near π); background pairs are spread out.

The search region's other cuts (the validation regions relax them as stated in sections 5 and 6): the egamma LJ is isolated (its isolation, defined the same way,
< 0.5); the LJ-pair mass M(LJ, LJ) ≥ 150 GeV; both LJs displaced (section 3).

The **working point** (WP) is the pair of cuts that defines A: isolated means isolation < 0.25,
back-to-back means |Δφ| ≥ 2.4. A **sideband** is a region next to A that fails one or both cuts:
here B (isolation ≥ 0.25), C (|Δφ| < 2.4) and D. The sidebands start right at A's edges, with no
**buffer band** between them. The earlier version of `estimate.ipynb` left a buffer (B at
isolation ≥ 0.5, C and D at |Δφ| < 2.0); the buffers were most of the non-closure it saw in the
prompt plane, and were removed (`estimate.ipynb`, section 4).

*How to read the figure:* the isolation × |Δφ| distribution of the plane with both LJs prompt
(PP, section 3) above 150 GeV; colour is events per cell on a log scale, bins are drawn with equal
width with the stored edges as tick labels, the red lines are the working point, and each region's
total is written in it ({pp_counts} events in A/B/C/D).""", N))
    cells.append(code(r"""fig = npl.plane_figure(nl.plane_hist2d(d2, NOMINAL, "PP"), NOMINAL,
                       "2μ2e, PP plane (both LJs prompt), M ≥ 150 GeV, egamma-LJ isolation < 0.5", "nominal_plane_PP")
plt.show()"""))

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
counts propagated through the product, as `estimate_lib` and `plane_search_lib` do it).

The table gives the counts and the prediction; κ cannot be computed because A is hidden. This
number assumes κ = 1 in the search plane above 150 GeV. The rest of the notebook is about how well
that is justified and what uncertainty to assign to it.

*How to read the figure:* as in section 2, for the search plane; the hatched cells are the hidden
box and region A reads "hidden".""", N))
    cells.append(code(r"""display(nl.kappa_view(nl.kappa_table(d2, [nl.SR], NOMINAL)))
fig = npl.plane_figure(nl.plane_hist2d(d2, NOMINAL, "DD"), NOMINAL,
                       "2μ2e, DD plane (search plane), M ≥ 150 GeV, egamma-LJ isolation < 0.5", "nominal_plane_DD")
plt.show()"""))

    cells.append(md(r"""## 5. Validation 1: κ wherever A is open

A **validation region** is a region like the search region whose A can be looked at. Four come
for free: the planes PP, PD and DP above 150 GeV, and the search plane itself below 150 GeV. The
table gives A, B, C, D, the prediction B·C/D and κ with its 68% interval (Clopper–Pearson
intervals of the two ratios A/C and B/D, combined); "precision" is the larger side of the interval
over κ.

* PP {closes_PP}: κ = {k_PP} ({n_PP}).
* PD is {dev_PD} above one, {z_PD} standard deviations: κ = {k_PD} ({n_PD}).
* DP, κ = {k_DP} ({n_DP}), and DD below 150 GeV, κ = {k_DD150} ({n_DD150}), are consistent with
  one but imprecise: their errors are {prec_open} of the value, so they cannot tell 1.0 from 1.3.

*How to read the figure:* κ per plane with its 68% interval; the dashed line is κ = 1.""", N))
    cells.append(code(r"""VAL1 = nl.kappa_table(d2, [VRS[n] for n in ("PP", "PD", "DP", "DD < 150")], NOMINAL)
display(nl.kappa_view(VAL1))
fig = npl.kappa_points_figure(*npl.table_points(VAL1), "2μ2e, our WP; A open", "nominal_kappa_open", ylim=(0, 2.6), width=12)
plt.show()"""))

    cells.append(md(r"""## 6. Validation 2: the seven validation regions of Dongyub Lee's programme

Dongyub Lee tested the same plane (muon-LJ isolation × |Δφ|) in seven data validation regions
(slides of 2026-10-13 and 2026-10-16), with his own analysis code. The table defines them: the
displacement of each LJ (P prompt, D displaced, "any" = either), the egamma-LJ isolation band and
the mass window. His working point is isolation < 0.3 and |Δφ| ≥ 2.6 (ours: < 0.25 and ≥ 2.4);
V1 keeps the egamma LJ isolated (< 0.5); V2–V4 put it in a non-isolated band (≥ 1 or ≥ 0.75),
which keeps their A away from the signal; V5–V7 use bands, 0.2–1 and 0.15–1, that straddle the
isolation boundary 0.5 of the search region, so they hold isolated and non-isolated egamma LJs
alike (V7 overlaps V1 in 0.15–0.5). {v1_pp} We recompute all seven from our histograms at his
working point and at ours.

* **At his working point** the slides read {vr_quoted}, these histograms {vr_his} (errors
  {vr_err}): equal to the printed digit in {vr_match} regions{vr_nomatch}. The A counts agree
  within {vr_dA}. This is two codes on one data set: a cross-check of both codes,
  not an independent measurement. The A counts differ by selection details we cannot remove (our
  HEM15/16 veto applies to runs ≥ 319077 only, his to every run; our raw-event pre-filter; our
  isolation clipped at zero).
* **At our working point** the same regions read {vr_ours}. The three low-mass regions V2–V4 read
  {vr_low_ours} against {vr_low_his} at his: κ moves by {vr_low_shift} with the working point
  there, {vr_low_z} standard deviations of the measurement. "Closes" is a statement at a working
  point. The four regions above 150 GeV read {vr_high_ours} at ours.
* {vr_hidden}. His working point applied to the search plane puts
  {sr_his_hidden} of DD inside the box (C then reaches into 2.4 ≤ |Δφ| < 2.6 at isolation
  < 0.25): there is no DD estimate at his working point, as the second table shows.

*How to read the figure:* κ per region from the slides (grey squares), from these histograms at
his working point (orange diamonds) and at ours (black circles); the dashed line is κ = 1.""", N))
    cells.append(code(r"""VRT = nl.vr_table(d2, nl.DONGYUB_VRS)
display(nl.vr_comparison_view(VRT))
print("the search region at both working points")
display(nl.vr_comparison_view(nl.vr_table(d2, (nl.SR,))))
fig = npl.kappa_points_figure(*npl.vr_points(VRT), "2μ2e, Dongyub Lee's validation regions", "nominal_kappa_dongyub",
                              ylim=(0.6, 1.5))
plt.show()"""))

    cells.append(md(r"""## 7. Validation 3: does κ depend on what differs between the validation regions and the search region?

The search region differs from every validation region in at least one respect. If κ depends on
one of them, the validation regions can mislead. Five checks.

### 7a. The displacement ladder

From PP (no displaced LJ) through PD and DP (one) to DD below 150 GeV (two): κ = {k_PP},
{k_PD}, {k_DP}, {k_DD150}. Nothing trends with the number of displaced LJs beyond the errors; the
two-displaced point carries a {prec_DD150} error.

*How to read the figure:* κ against the number of displaced LJs in the plane; the dashed line is
κ = 1.""", N))
    cells.append(code(r"""fig = npl.ladder_figure([(r["region"], {"PP": 0, "PD": 1, "DP": 1, "DD < 150": 2}[r["region"]],
                          (r["kappa"], r["minus"], r["plus"])) for _, r in VAL1.iterrows()])
plt.show()"""))

    cells.append(md(r"""### 7b. The egamma LJ type

PD's deviation sits in its photon-only egamma LJs: κ = {k_PDg} for them, {k_PDe} for the electron
ones. Photon-only egamma LJs count as displaced (no track, section 3), so no prompt plane contains
them; they enter PD and DD only. The photon-only share per region is {g_PD} in A/B/C/D of PD and
{g_DD} in B/C/D of DD: summed over B, C and D, DD's sidebands ({g_DD_bcd} photon-only) are
{g_cmp} PD's ({g_PD_bcd}). If κ follows the photon-only share, κ of DD is more likely to sit near
PD's {k_PD} than near PP's {k_PP}. That is the case the systematic of section 8 has to cover.

*How to read the figure:* the photon-only share of the egamma LJs in each region of PD (blue) and
DD (green), with binomial errors and the number of events in brackets; A of DD is hidden.""", N))
    cells.append(code(r"""display(el.class_grid_table(d2, NOMINAL, ("PD",), (None, "g", "e")))
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
    cells.append(code(r"""display(el.class_grid_table(d2, NOMINAL, ("PP", "PD"), (None, "dsa0", "dsa1+")))
display(nl.class_share_view(nl.class_share_table(d2, NOMINAL, ("PD", "DD"), ("dsa0", "dsa1+")), ("dsa0", "dsa1+")))"""))

    cells.append(md(r"""### 7d. The mass

κ in 150–300 GeV and above 300 GeV: PP {k_PP_lo} → {k_PP_hi}, PD {k_PD_lo} → {k_PD_hi}. κ falls
with the mass in both planes; the drop, combined over the two planes, is {drop} ({drop_z} standard
deviations). For a one-bin search region the integrated κ of section 5 is the right test if the
validation plane's mass distribution resembles the search plane's, so that the mass dependence
averages the same way in both. It roughly does: {dshare_DD} of DD's D lies in 150–300 GeV, against
{dshare_PD} of PD's. The same question asked of the estimate itself: the direct estimate in the two
windows separately, {two_lo} + {two_hi} = {two_bin} events (errors in quadrature), is {two_ok} the
one-bin {est0}, but the window above 300 GeV rests on {d_hi} events in D. For a search region binned
in mass, each bin needs its own κ, and the per-window deviations, up to {win_dev_max}, would be the
systematic. This notebook estimates the one-bin search region.

*How to read the figure:* κ per mass window in PP (black) and PD (blue); the dashed line is κ = 1.
The table below it gives κ in the finer bins of the histograms.""", N))
    cells.append(code(r"""MASS = nl.kappa_table(d2, [nl.VR(f"{p}, {w}", "this analysis", p[0], p[1], mass=m)
                           for p in ("PP", "PD") for w, m in (("150–300", (150.0, 300.0)), ("≥ 300", (300.0, None)))], NOMINAL)
display(nl.kappa_view(MASS))
fig = npl.kappa_mass_figure({p: [(r["kappa"], r["minus"], r["plus"]) for _, r in MASS[MASS["region"].str.startswith(p)].iterrows()]
                             for p in ("PP", "PD")}, ["150–300", "≥ 300"])
plt.show()
display(ps.mass_table(d2, NOMINAL, planes=("PP", "PD")))"""))

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

    cells.append(md(r"""## 8. The systematic uncertainty from the validation regions

The rule, written after the regions above were measured (so it is not blind to their values):

> Among the validation regions at our working point that differ from the search region only in
> the displacement of one or both lepton jets, the mass window, or the band of the partner's
> isolation, and whose κ is measured to 20% or better, take for each the larger of |κ − 1| and the
> uncertainty facing one, and quote the largest of those.

"The uncertainty facing one" is the side of κ's 68% interval that points towards one; "measured to
20% or better" means that the larger side of the interval is at most 20% of κ. The table lists
every region: ours (PP, PD and its two egamma types, DP, DD below 150 GeV) and Dongyub Lee's seven
at our working point, sorted by the number of respects in which each differs from the search
region. {n_qual} of the {n_rows} rows qualify, {n_distinct} distinct regions{dup_clause}:
{qual_names}.

The photon-only part of PD (PD:g, κ = {k_PDg}) is excluded by the rule's letter: it is a
sub-population of PD selected on the egamma LJ type, not a region of the plane. Whether the
inclusive κ is the right comparator depends on how much of the prediction each egamma type
carries, not on how many events it has. In DD, B·C/D per type is {g_direct_DD} events for the
photon-only LJs and {e_direct_DD} for the electron ones, so the photon-only LJs carry {g_pred_DD} of
the direct prediction in DD against {g_pred_PD} in PD. Applying PD's κ per egamma type to DD's
prediction per type, {kc_typ_terms} ≈ {kc_typ} events, gives {kc_typ_ratio} times the direct
estimate, a number that rests on DD's {e_counts_DD} electron LJs in B/C/D. The same reweighting by
the muon LJ's DSA class (section 7c) gives {kc_dsa_ratio}. The two reweightings pull in opposite
directions, by egamma type to {kc_typ_ratio} and by DSA class to {kc_dsa_ratio}, and the inclusive
{kc_kPD} sits between them. The rule takes the inclusive value, {syst}; admitting PD:g
as a region would give {pdg_cand}, and taking the type reweighting at face value {typ_dev}. That
choice, {syst} against {syst_g}, is put to the group.

**Result: {syst_name}, {syst}, ± {syst_ev} events.** The next largest qualifying value is {second}.

Shown for comparison, not adopted:

* the quadrature form, √(|κ − 1|² + σ²): {syst_quad} (± {syst_quad_ev} events);
* admitting the imprecise regions: {syst_alt_name} gives {syst_alt} (± {syst_alt_ev} events);
* the {syst_33} of `estimate.ipynb` (± {syst_33_ev} events).

The 2μ2e background in the one-bin search region:

**{headline} events.**

*How to read the figure:* κ per region at our working point, ordered from the regions closest to
the search region (fewest differences, in brackets) to the farthest; filled circles qualify for the
rule, open circles (PD:g, PD:e, DP, DD below 150 GeV) do not; the green band is 1 ± {syst}, the
dashed line κ = 1.""", N))
    cells.append(code(r"""SYS = nl.systematic_table(d2, nl.our_vrs() + nl.DONGYUB_VRS)
display(nl.systematic_view(SYS))
SYST, SYST_FROM = nl.closure_systematic(SYS)
QUAD, _ = nl.closure_systematic(SYS, "quadrature")
ALT, ALT_FROM = nl.closure_systematic(SYS, qualifying=False)
EST = nl.kappa_row(d2, nl.SR, NOMINAL)
print(f"rule: {nl.pct(SYST)} from {SYST_FROM}; quadrature {nl.pct(QUAD)}; imprecise regions admitted {nl.pct(ALT)} from {ALT_FROM}")
print(f"estimate: {EST['B C/D']:.1f} ± {EST['B C/D err']:.1f} (stat.) ± {EST['B C/D'] * SYST:.1f} (syst.) events")
points, labels = npl.table_points(SYS)
SAME_AS_PP = nl.same_regions(SYS, "PP")
labels = [f"{r} ({n})" + (" = PP here" if r in SAME_AS_PP else "") for r, n in zip(labels, SYS["n differences"])]
fig = npl.kappa_points_figure(points, labels, "2μ2e, our WP; in brackets: respects differing from the search region",
                              "nominal_kappa_systematic", band=SYST, band_label=f"1 ± {nl.pct(SYST)}", ylim=(0, 2.6),
                              hollow_label="does not qualify")
plt.show()"""))

    cells.append(md(r"""## 9. Cross-check: the three borrowed-factor routes

`estimate.ipynb` estimated A of DD three more ways, each by borrowing a ratio from PD per DSA class
and summing: B × (A/B), C × (A/C) and D × (A/D) give {r_B}, {r_C} and {r_D} events. Their spread
is not a systematic uncertainty: they share DD's sidebands with the direct estimate, and each
assumes that a ratio of PD transfers to DD. The B route holds if A/B itself transfers from PD to
DD. A/B of PD is κ_PD × C/D of PD, and DD's C/D is {cd_all} times PD's (section 7e), so with DD's own
C/D the B route's {r_B0} events require κ_DD = {implied_k}. The C and D routes likewise assume that
A/C and A/D transfer. The two estimates differ by {r_B_diff} events, {r_B_total} of the combined
statistical and systematic uncertainty of the direct estimate.

If the search plane's κ were PD's, A would be {kc_kPD} × {kc_est} ≈ {kc_inc} events, or
{kc_dsa_terms} ≈ {kc_dsa} with PD's κ per DSA class applied to DD's prediction per class; {r_B0}
assumes more than that.

The earlier notebooks made the B route the headline and the direct estimate a cross-check. This
one does the opposite: with Dongyub Lee's regions added, the validation regions favour κ = 1 within
{syst}, and the one {z_PD}σ deviation, PD, is carried by the systematic.""", N))
    cells.append(code(r"""display(el.routes_table(d2, WINDOWS))
display(el.class_factors_table(d2, WINDOWS[:1]))"""))

    cells.append(md(r"""## 10. What would settle it: opening 150–300 GeV of the search plane

The direct test of κ in the search plane is to open A of DD in a window where the signal is small.
The background predicted there (direct, and B route):

* 150–200 GeV: {o_150_200_direct} (B route {o_150_200_B});
* 200–300 GeV: {o_200_300_direct} (B route {o_200_300_B});
* 150–300 GeV: {o_150_300_direct} (B route {o_150_300_B}).

The two predictions for 150–300 GeV differ by a factor {o_150_300_ratio}: the direct one uses DD's own
C/D in the window ({o_150_300_bcd} events in B/C/D), the B route uses PD's A/B. Their ratio is the
product of two factors: PD's κ per DSA class weighted by the search plane's B, {o_150_300_kmix} here,
and the sideband ratio (C/D)_PD / (C/D)_DD, {o_150_300_sbr} (the inverse of the {cd_lo} of section
7e). The first is close to one, so the factor between the predictions is almost all sideband ratio.
A count near the first prediction would say that DD's own sidebands are right there; one near the
second, that the direct route under-predicts the window. The two predictions are {o_150_300_sep}
standard deviations apart if treated as independent, about {o_150_300_sep_b} once the shared B is
accounted for (both scale with DD's B, with different factors), so the window can tell them apart, but not sharply. κ of DD in the
window would be measured to about {o_150_300_prec} ({o_150_300_prec_a} from A's count alone, the
rest from B, C and D), which separates 1.0 from 1.9 but not from 1.2; and section 7d shows
that κ changes with the mass, so the window's κ is not the search region's. 150–200 GeV alone
would measure it to about {o_150_200_prec}. The window is the only direct test there is; it is
not a sharp one.

The price is signal. Of the {n_points} 2μ2e signal points, the {n200} with a 200 GeV mediator put
{s200_lo} of their search-region yield in 150–200 GeV and {s200_150300} in 150–300 GeV (yields
{s200_yield} events at the stored cross section; {s200_below} more below 150 GeV, in the
already-open DD validation region). The heavier points ({heavy_masses} GeV) put at most
{heavy_150300} in 150–300 GeV; the 800 GeV, 0.25 GeV, 0.25 mm point, for example, has {p800_yield}
events, {p800_300} of them below 300 GeV and {p800_600} at ≥ 600 GeV. Opening 150–300 GeV ends the
200 GeV search; opening 150–200 GeV costs about half of it. The decision is the group's.

The table also lists the {light_masses} GeV mediators. These masses are outside the range the
analysis targets (the group's grid of interest starts at 200 GeV) and are kept in the table for
completeness. For them much or most of the signal in A of DD sits below 150 GeV ({light_below}),
so DD below 150 GeV, used here as a validation region, would be signal-contaminated for those
hypotheses.

*How to read the figure:* one bar per signal point (the 200 GeV points and the seven points of
`estimate.ipynb`), split into the share of its search-region yield in each mass bin; the label
gives the mediator mass, dark-photon mass and decay length, and the yield in brackets
(simulation).""", N))
    cells.append(code(r"""OPEN = [(label, bl.bin_cfg(NOMINAL, lo, hi)) for label, (lo, hi) in (("150-200", (150.0, 200.0)), ("200-300", (200.0, 300.0)), ("150-300", (150.0, 300.0)))]
display(el.routes_table(d2, OPEN)[["window", "DD B/C/D", "direct B C/D", "B x A/B per class", "DD A observed"]])
NAMES = sorted(os.path.basename(p)[: -len(".coffea")] for p in tl.sample_paths(BASE, "signal")
               if os.path.basename(p).startswith("2Mu2E_"))
SIGNAL = nl.signal_share_table(bl.signal_profiles(BASE, NAMES, "2mu2e", NOMINAL, plane="DD", region="A"))
print(f"{len(SIGNAL)} 2mu2e signal points; share of the search-region yield per mass bin, min-max per mediator mass, "
      "and the yields in A of DD above and below 150 GeV in events")
display(nl.share_summary(SIGNAL))
SHOWN = SIGNAL[SIGNAL["point"].str.startswith("2Mu2E_200GeV_") | SIGNAL["point"].isin(bl.SIGNAL_POINTS["2mu2e"])]
display(SHOWN.round(3))
fig = npl.signal_share_figure(SHOWN.reset_index(drop=True))
plt.show()"""))

    cells.append(md(r"""## 11. 4μ in one section

In 4μ the search plane holds {dd4_lo} events in A/B/C/D below 150 GeV and {dd4_hi} in B/C/D above
it: no κ of the search plane can be measured, and no B·C/D exists above 150 GeV (C and D are
empty). `estimate.ipynb` (section 5.2) gives six count-limited readings from the open neighbouring
regions, by B of DD above 150 GeV times A/B of a plane below it, and by A of DD below 150 GeV times
the above/below ratio of the neighbouring regions: they read {four_range} events, median
{four_med}, each with a {four_rel} statistical error, so the 4μ background is uncertain by a factor
two to three. Of the open DD events, {dsa_pairs} have both muon LJs made only of DSA muons, the
signature of the instrumental background seen in July 2026 before the cosmic and spread cuts.

Dongyub Lee's 4μ regions (quoted from his slides): the isolation × isolation plane fails
({SLIDES_4MU_iso}), the muon-isolation × |Δφ| plane closes ({SLIDES_4MU_dphi}) but its κ runs over
{SLIDES_4MU_low} below 150 GeV and {SLIDES_4MU_high} above as the isolation boundary moves.

Nothing more can be done with these counts. What the factor costs: for one observed event the 95%
CL upper limit on a signal count moves {lim1} as the background runs over {lim_b} events, for two
{lim2} (a counting experiment with a known background, `estimate_lib.counting_limit`).""", N))
    cells.append(code(r"""HIGH4, LOW4 = ("M >= 150", NOMINAL), ("M < 150", LOW)
display(el.count_grid(d4, [LOW4, HIGH4], "DD", [f"dsa{a}:subdsa{b}" for a in ("0", "1", "2+") for b in ("0", "1", "2+")]))
print("B of the search plane above 150 GeV times A/B of the planes whose A is open")
display(el.sideband_transfer_table(d4, HIGH4, LOW4))
print("A of the DD plane below 150 GeV times the above/below-150 GeV ratio of the open neighbouring regions")
display(el.mass_transfer_table(d4, HIGH4, LOW4))
print("95% CL upper limit on a signal count for n observed events and an expected background b")
display(el.counting_limit_table((0, 1, 2), (0.5, 1.0, 1.7, 3.0, 4.5)))"""))

    cells.append(md(r"""## 12. Summary and to-do list

* The 2μ2e background in the one-bin search region is **{headline} events**, the search plane's
  own B·C/D.
* κ closes in PP ({k_PP}) and in Dongyub Lee's seven regions at his working point (the two codes
  agree to the printed digit in {vr_match}); DP ({k_DP}) and DD below 150 GeV ({k_DD150}) are
  consistent with one but imprecise.
* PD is {dev_PD} above one ({k_PD}), carried by photon-only egamma LJs ({k_PDg}); DD's sidebands
  are {g_cmp} PD's. PD sets the systematic: {syst_both} (the choice is put to the group).
* κ depends on the working point (V2–V4: {vr_low_his} at his, {vr_low_ours} at ours) and falls
  with the mass ({drop_z} standard deviations from 150–300 GeV to above 300 GeV). For one bin the
  integrated κ is the right test as long as the validation planes' mass distributions resemble the
  search plane's ({dshare_DD} of DD's D in 150–300 GeV, {dshare_PD} of PD's); for a binned search
  region it is an effect of up to {win_dev_max} per window.
* The B route ({r_B0} events) is the cross-check; it holds if A/B transfers from PD, which with DD's
  own C/D requires κ_DD = {implied_k}. With PD's κ itself, A would be {kc_inc} ({kc_dsa} per DSA
  class).
* 4μ: about {four_med} events, uncertain by a factor two to three; nothing more to do.

To do, for the group:

1. Fix the blinding box: Dongyub Lee's histograms carry the data in the search region unblinded.
2. Dongyub to reproduce κ_PD from his 6D histograms, and to split his validation regions above
   150 GeV at 300 or 400 GeV.
3. State the criterion by which the working point is chosen.
4. Decide whether to open 150–300 GeV (or 150–200 GeV) of the search plane as a validation region.
5. Adopt the direct ABCD with the validation-region systematic as the convention.

## Appendix: reproducibility

The data are read only through `transfer_lib.summed_cubes(BASE, "data", prefix="abcd8d")`,
which sums the merged 8D histograms and sets the hidden box ({box}) to NaN before anything is
cached or returned; every projection touching a hidden cell is NaN and prints as "hidden". The
signal shares of section 10 read the signal histograms one point at a time
(`binning_lib.signal_profiles`; simulation is never hidden). Code: `nominal_lib.py` (validation
regions, the systematic rule, the plane histogram, the shares), `nominal_plots.py` (the figures),
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
