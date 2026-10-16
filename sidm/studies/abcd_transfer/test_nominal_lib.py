"""Tests of the nominal-plane ABCD library on synthetic 8D cubes: the validation-region definitions,
the hidden box, the systematic rule, the 2D plane histogram and the signal shares."""
import itertools

import numpy as np
import pytest

import abcd6d_io as io
import estimate_lib as el
import nominal_lib as nl
import plane_search_lib as ps
from test_abcd6d_io import event8d, make_hist8d
from test_estimate_lib import routes_cube
from test_plane_search_lib import CORNERS

# Dongyub Lee's seven regions as printed on his slides: (muon LJ, egamma LJ, egamma iso band, mass window)
SLIDE_DEFINITIONS = {
    "V1": ("P", "P", (None, 0.5), (150.0, None)),
    "V2": ("P", "any", (1.0, None), (None, 150.0)),
    "V3": ("any", "P", (0.75, None), (None, 150.0)),
    "V4": ("any", "any", (0.75, None), (None, 150.0)),
    "V5": ("P", "any", (0.2, 1.0), (150.0, None)),
    "V6": ("P", "any", (0.15, 1.0), (150.0, None)),
    "V7": ("P", "P", (0.15, 1.0), (150.0, None)),
}
DISP = {"mu": {"P": 5, "D": 1}, "egm": {"P": 0, "D": 1}}
ISO1 = (0.1, 0.17, 0.3, 0.6, 0.8, 1.5)
MASS = (400.0, 100.0)
MULT = {"A": 1, "B": 3, "C": 5, "D": 7}


def fill(h, n, iso0, iso1, disp0, disp1, dphi, mass, ndsa0=0, weight=1.0):
    if n == 0:
        return
    values = dict(iso0=iso0, iso1=iso1, disp0=disp0, disp1=disp1, abs_dphi=dphi, ljlj_mass=mass, lj0_mass=1.0, ndsa0=ndsa0)
    h.fill(channel="ch", weight=np.full(n, weight), **{v: np.full(n, values[v]) for v in h.axes.name[1:]})


def groups():
    return list(itertools.product(("P", "D"), ("P", "D"), ISO1, MASS))


def tagged_cube():
    """Every (muon state, egamma state, egamma isolation, mass) group j in every region r, with
    weight MULT[r] * 2**j, so that a region sum names exactly the groups it contains; blinded."""
    h = make_hist8d("2mu2e")
    for j, (mu, egm, iso1, mass) in enumerate(groups()):
        for r, (iso0, dphi) in CORNERS.items():
            fill(h, 1, iso0, iso1, DISP["mu"][mu], DISP["egm"][egm], dphi, mass, weight=MULT[r] * 2.0 ** j)
    return io.blind(io.cube_from_hist(h, "2mu2e"))


def inside(value, rng):
    lo, hi = rng
    return (lo is None or value >= lo) and (hi is None or value < hi)


def expected(definition, region):
    mu, egm, band, window = definition
    return sum(MULT[region] * 2.0 ** j for j, (m, e, iso1, mass) in enumerate(groups())
               if mu in ("any", m) and egm in ("any", e) and inside(iso1, band) and inside(mass, window))


@pytest.mark.parametrize("name", list(SLIDE_DEFINITIONS))
def test_dongyub_regions_select_the_slide_definitions_at_both_working_points(name):
    cube = tagged_cube()
    vr = {v.name: v for v in nl.DONGYUB_VRS}[name]
    for cfg in (nl.HIS_WP, nl.NOMINAL):
        reg = nl.vr_regions(cube, vr, cfg)
        for r in "ABCD":
            assert reg[r][0] == expected(SLIDE_DEFINITIONS[name], r)
        row = nl.kappa_row(cube, vr, cfg)
        assert not row["hidden"] and np.isfinite(row["kappa"])


def test_any_displacement_is_the_sum_of_prompt_and_displaced():
    cube = tagged_cube()
    cfg = nl.NOMINAL
    v4 = {v.name: v for v in nl.DONGYUB_VRS}["V4"]
    parts = [nl.vr_regions(cube, nl.VR("x", "", mu, egm, v4.iso1, v4.mass), cfg) for mu in "PD" for egm in "PD"]
    total = nl.vr_regions(cube, v4, cfg)
    for r in "ABCD":
        assert total[r][0] == sum(p[r][0] for p in parts)


def test_a_region_touching_the_box_is_flagged_hidden():
    cube = tagged_cube()
    ours = nl.kappa_row(cube, nl.SR, nl.NOMINAL)
    assert ours["hidden"] and ours["hidden regions"] == "A" and np.isnan(ours["kappa"])
    assert ours["B C/D"] == pytest.approx(expected(("D", "D", (None, 0.5), (150.0, None)), "B")
                                          * expected(("D", "D", (None, 0.5), (150.0, None)), "C")
                                          / expected(("D", "D", (None, 0.5), (150.0, None)), "D"))
    his = nl.kappa_row(cube, nl.SR, nl.HIS_WP)
    assert his["hidden"] and his["hidden regions"] == "A and C" and np.isnan(his["B C/D"])
    below = nl.kappa_row(cube, nl.VR("DD < 150", "", "D", "D", mass=nl.BELOW), nl.NOMINAL)
    assert not below["hidden"] and np.isfinite(below["kappa"])
    any_muon = nl.kappa_row(cube, nl.VR("x", "", "any", "D"), nl.NOMINAL)
    assert any_muon["hidden"]                  # "any" includes the displaced muon LJ: the box
    view = nl.vr_comparison_view(nl.vr_table(cube, (nl.SR,)))
    assert view["kappa, these histograms at our WP"].iloc[0].startswith("hidden")
    assert view["A ours (our WP)"].iloc[0] == "hidden"


def test_a_split_needs_fixed_states_and_stays_inside_its_plane():
    with pytest.raises(ValueError):
        nl.vr_selection("2mu2e", "any", "D", "g")
    with pytest.raises(ValueError):
        nl.VR("x", "", "prompt", "D")
    sel = nl.vr_selection("2mu2e", "P", "D", "g")
    assert sel["disp1"] == (9.5, None) and sel["disp0"] == io.MU_PROMPT


def rule_cube():
    """PP closes (kappa 1), PD sits at 1.2 (its electron part at 1.0, its photon-only part at 1.4),
    DP and DD below 150 GeV are imprecise (kappa 2 and 1 on two events per region)."""
    h = make_hist8d("2mu2e")
    content = [(5, 0, 400.0, dict(A=400, B=400, C=400, D=400)),     # PP
               (5, 1, 400.0, dict(A=200, B=200, C=200, D=200)),     # PD, electron egamma LJ
               (5, 999, 400.0, dict(A=280, B=200, C=200, D=200)),   # PD, photon-only egamma LJ
               (1, 0, 400.0, dict(A=4, B=2, C=2, D=2)),             # DP
               (1, 1, 100.0, dict(A=2, B=2, C=2, D=2)),             # DD below 150 GeV
               (1, 1, 400.0, dict(A=50, B=20, C=20, D=20))]         # DD, A in the box
    for d0, d1, mass, counts in content:
        for r, n in counts.items():
            iso0, dphi = CORNERS[r]
            fill(h, n, iso0, 0.1, d0, d1, dphi, mass)
    return io.blind(io.cube_from_hist(h, "2mu2e"))


def test_systematic_rule_takes_the_largest_qualifying_deviation_or_uncertainty():
    cube = rule_cube()
    table = nl.systematic_table(cube, nl.our_vrs(), wps={"his": nl.HIS_WP, "ours": nl.NOMINAL})
    rows = table.set_index(["region", "wp"])
    assert rows.loc[("PD", "ours"), "kappa"] == pytest.approx(1.2)
    assert rows.loc[("PD", "ours"), "qualifies"] and rows.loc[("PP", "ours"), "qualifies"]
    assert not rows.loc[("PD:g", "ours"), "qualifies"] and "split g" in rows.loc[("PD:g", "ours"), "reason"]
    assert rows.loc[("PD:g", "ours"), "candidate"] > rows.loc[("PD", "ours"), "candidate"]
    for name in ("DP", "DD < 150"):
        assert rows.loc[(name, "ours"), "reason"].startswith("precision")
    assert not table[table["wp"] == "his"]["qualifies"].any()
    pd_row = rows.loc[("PD", "ours")]
    assert pd_row["sigma"] == pd_row["minus"] and pd_row["candidate"] == pytest.approx(0.2)
    pp_row = rows.loc[("PP", "ours")]
    assert pp_row["|kappa-1|"] == pytest.approx(0.0) and pp_row["candidate"] == pp_row["sigma"] > 0
    value, name = nl.closure_systematic(table)
    assert name == "PD" and value == pytest.approx(0.2)
    quad, _ = nl.closure_systematic(table, "quadrature")
    assert quad == pytest.approx(np.hypot(0.2, pd_row["sigma"]))
    alt, alt_name = nl.closure_systematic(table, qualifying=False)
    assert alt_name in ("DP", "DD < 150") and alt > 0.2
    assert table["n differences"].is_monotonic_increasing
    assert nl.distance_to_sr(nl.SR, nl.NOMINAL) == ()
    assert nl.distance_to_sr(nl.DONGYUB_VRS[1], nl.HIS_WP) == ("muon-LJ displacement", "egamma-LJ displacement",
                                                                "mass window", "partner-isolation band", "working point")


def test_side_facing_one():
    assert nl.side_facing_one((1.2, 0.1, 0.3)) == 0.1
    assert nl.side_facing_one((0.8, 0.1, 0.3)) == 0.3


def test_plane_hist2d_sums_to_the_regions_and_is_nan_only_in_the_box():
    cube = rule_cube()
    for plane in ("PP", "PD", "DD"):
        h2d = nl.plane_hist2d(cube, nl.NOMINAL, plane)
        e0, ed, sumw, sumw2 = h2d
        assert sumw.shape == (len(e0) - 1, len(ed) - 1) == sumw2.shape
        got, ref = nl.hist2d_regions(h2d, nl.NOMINAL), ps.regions(cube, nl.NOMINAL, plane)
        for r in "ABCD":
            assert got[r] == pytest.approx(ref[r][0], nan_ok=True)
        nan = np.isnan(sumw)
        if plane == "DD":
            i0, ip = io._edge_index(e0, 0.25), io._edge_index(ed, 2.4)
            box = np.zeros_like(nan)
            box[:i0, ip:] = True
            assert np.array_equal(nan, box) and np.isnan(got["A"])
        else:
            assert not nan.any()
    low = nl.plane_hist2d(cube, nl.LOW, "DD")
    assert not np.isnan(low[2]).any() and np.nansum(low[2]) == 8


def test_plane_hist2d_folds_the_overflow_into_the_last_bin():
    h = make_hist8d("2mu2e")
    fill(h, 3, 7.0, 0.1, 5, 0, 3.0, 400.0)          # iso0 beyond the last edge
    cube = io.cube_from_hist(h, "2mu2e")
    e0, ed, sumw, _ = nl.plane_hist2d(cube, nl.NOMINAL, "PP")
    assert sumw[-1, :].sum() == 3 and sumw.sum() == 3


def test_class_shares_partition_each_region():
    cube = rule_cube()
    t = nl.class_share_table(cube, nl.NOMINAL, ("PD", "DD"), ("e", "g"))
    pd_ = t[t["plane"] == "PD"].set_index("region")
    assert pd_.loc["A", "g share"] == pytest.approx(280 / 480) and pd_.loc["B", "e share"] == pytest.approx(0.5)
    dd = t[t["plane"] == "DD"].set_index("region")
    assert dd.loc["A", "hidden"] and np.isnan(dd.loc["A", "g share"])
    assert (pd_["e share"] + pd_["g share"]).tolist() == pytest.approx([1.0] * 4)


def test_signal_shares_sum_to_one_per_point():
    edges = np.array([0, 100, 150, 200, 300, 400, 600, 800, 1000, 2500], float)
    prof = {"2Mu2E_200GeV_0p25GeV_1p0mm": (np.array([1, 1, 50, 48, 2, 0, 0, 0, 0], float), None),
            "2Mu2E_800GeV_0p25GeV_0p25mm": (np.array([0, 0, 0, 0, 1, 2, 30, 40, 27], float), None)}
    shares = nl.signal_share_table((edges, prof))
    cols = [label for label, _ in nl.SHARE_BINS]
    assert shares[cols].sum(axis=1).tolist() == pytest.approx([1.0, 1.0])
    first = shares.iloc[0]
    assert first["yield"] == 100 and first["150-200"] == 0.5 and first["150-300"] == pytest.approx(0.98)
    assert first["below"] == pytest.approx(2.0)                 # events, not a share
    assert nl.below_share(shares).tolist() == pytest.approx([2 / 102, 0.0])
    assert shares.iloc[1][">=600"] == pytest.approx(97 / 100) and shares.iloc[1]["150-300"] == 0
    summary = nl.share_summary(shares)
    assert summary["mediator mass [GeV]"].tolist() == ["200", "800"]
    assert summary["yield, M < 150 [events]"].tolist() == ["2.00–2.00", "0.00–0.00"]


def test_kappa_drop_and_projected_precision():
    mean, err, z = nl.kappa_drop([((1.3, 0.1, 0.1), (0.9, 0.1, 0.1)), ((1.1, 0.1, 0.1), (0.7, 0.1, 0.1))])
    assert mean == pytest.approx(0.4) and err == pytest.approx(0.1) and z == pytest.approx(4.0)
    assert nl.projected_precision(100.0, 0.0) == pytest.approx(0.1)
    assert nl.projected_precision(100.0, 10.0) == pytest.approx(np.hypot(0.1, 0.1))


def test_fmt_is_the_one_number_formatter():
    assert nl.fmt((1.2, 0.1, 0.1)) == "1.20 ± 0.10"
    assert nl.fmt((45.7, 14.9), 0) == "46 ± 15"
    assert nl.fmt(0.204, 2) == "0.20" and nl.pct(0.204) == "20%"


def test_route_estimates_are_the_numbers_of_the_routes_table():
    cube = routes_cube()
    got = nl.route_estimates(cube, nl.NOMINAL)
    row = el.routes_table(cube, [("M >= 150", nl.NOMINAL)]).iloc[0]
    assert nl.fmt(got["direct"], 1) == row["direct B C/D"]
    for sideband, name in el.ROUTES:
        assert nl.fmt(got[sideband], 1) == row[f"{sideband} x {name} per class"]


def type_cube():
    """PD: electron egamma LJs with kappa 1 (A/B/C/D 200 each), photon-only ones with kappa 1.4
    (A 280); DD: electron B/C/D 10/10/10 (B C/D 10), photon-only 30/20/10 (B C/D 60), A in the box."""
    h = make_hist8d("2mu2e")
    content = [(5, 1, dict(A=200, B=200, C=200, D=200)), (5, 999, dict(A=280, B=200, C=200, D=200)),
               (1, 1, dict(A=7, B=10, C=10, D=10)), (1, 999, dict(A=9, B=30, C=20, D=10))]
    for d0, d1, counts in content:
        for r, n in counts.items():
            iso0, dphi = CORNERS[r]
            fill(h, n, iso0, 0.1, d0, d1, dphi, 400.0)
    return io.blind(io.cube_from_hist(h, "2mu2e"))


def test_kappa_corrected_weights_the_reference_kappa_by_the_search_planes_prediction():
    cube = type_cube()
    got = nl.kappa_corrected(cube, nl.NOMINAL, ("e", "g"))
    assert got["direct"] == pytest.approx(40 * 30 / 20)
    assert got["parts"]["e"]["direct"] == pytest.approx(10) and got["parts"]["g"]["direct"] == pytest.approx(60)
    assert got["parts"]["e"]["kappa"][0] == pytest.approx(1.0) and got["parts"]["g"]["kappa"][0] == pytest.approx(1.4)
    assert got["parts"]["g"]["share"] == pytest.approx(60 / 70)
    assert got["corrected"] == pytest.approx(1.0 * 10 + 1.4 * 60) and got["ratio"] == pytest.approx(94 / 60)
    inclusive = nl.kappa_corrected(cube, nl.NOMINAL, (None,))
    assert inclusive["ratio"] == pytest.approx(1.2) and inclusive["corrected"] == pytest.approx(72)
    in_pd = nl.kappa_corrected(cube, nl.NOMINAL, ("e", "g"), search="PD")
    assert in_pd["parts"]["g"]["share"] == pytest.approx(0.5)


def test_route_ratio_is_the_class_weighted_kappa_times_the_sideband_ratio():
    cube = routes_cube()
    got = nl.route_ratio_factors(cube, nl.NOMINAL)
    assert got["sideband"] == pytest.approx((3 / 4) / (3 / 2))           # C/D of PD over C/D of DD
    assert got["kappa_mix"] == pytest.approx(7.5 / (5 * 3 / 4))          # B route / (B x C/D of PD)
    assert got["ratio"] == pytest.approx(got["kappa_mix"] * got["sideband"]) == pytest.approx(1.0)
    ro = nl.route_estimates(cube, nl.NOMINAL)
    assert got["ratio"] == pytest.approx(ro["B"][0] / ro["direct"][0])


def test_shared_b_separation_carries_b_in_first_order():
    reg = {"B": (50.0, 50.0), "C": (10.0, 10.0), "D": (25.0, 25.0)}
    parts = [("dsa0", 30.0, 30.0, 1.0), ("dsa1+", 20.0, 20.0, 0.5)]          # route 30 x 1.0 + 20 x 0.5 = 40
    direct = (20.0, 20.0 * np.sqrt(1 / 50 + 1 / 10 + 1 / 25))
    indep, shared = nl.shared_b_separation(direct, (40.0, 7.0, 8.0), reg, parts)
    assert indep == pytest.approx(20 / np.hypot(7.0, direct[1]))
    d_rest2 = 20.0**2 * (1 / 10 + 1 / 25)                                     # C and D only
    r_rest2 = 7.0**2 - (1.0**2 * 30 + 0.5**2 * 20)                            # route error without B
    var_b = 30 * (1.0 - 0.4) ** 2 + 20 * (0.5 - 0.4) ** 2                    # C/D = 0.4
    assert shared == pytest.approx(20 / np.sqrt(d_rest2 + r_rest2 + var_b))
    below, _ = nl.shared_b_separation(direct, (10.0, 7.0, 8.0), reg, parts)
    assert below == pytest.approx(10 / np.hypot(8.0, direct[1]))           # the side facing the direct estimate


def test_b_route_parts_rebuild_the_b_route():
    cube = routes_cube()
    parts = nl.b_route_parts(cube, nl.NOMINAL)
    assert [(c, b) for c, b, _, _ in parts] == [("dsa0", 2.0), ("dsa1+", 3.0)]
    assert sum(b * r for _, b, _, r in parts) == pytest.approx(nl.route_estimates(cube, nl.NOMINAL)["B"][0])


def test_same_regions_names_the_rows_with_the_same_events():
    cube = tagged_cube()
    table = nl.systematic_table(cube, nl.our_vrs() + nl.DONGYUB_VRS)
    assert nl.same_regions(table, "PP") == ["V1"]
    assert nl.same_regions(table, "V5") == []


def test_dd_bands_are_open_and_differ_from_the_search_region_only_in_the_partner_band():
    cube = tagged_cube()
    bands = nl.dd_band_vrs()
    assert [v.iso1 for v in bands] == list(nl.DD_BANDS) and all(v in nl.our_vrs() for v in bands)
    for vr in bands:
        reg = nl.vr_regions(cube, vr, nl.NOMINAL)
        for r in "ABCD":
            assert reg[r][0] == expected(("D", "D", vr.iso1, (150.0, None)), r)
        assert not nl.kappa_row(cube, vr, nl.NOMINAL)["hidden"]
        assert nl.distance_to_sr(vr, nl.NOMINAL) == ("partner-isolation band",)
    table = nl.systematic_table(cube, nl.our_vrs()).set_index("region")
    assert table.loc[[v.name for v in bands], "n differences"].tolist() == [1] * len(bands)
    assert bands[0].name == "DD, egamma iso ≥ 0.5" and bands[-1].name == "DD, egamma iso 0.5–1"


def window_cube():
    """Two mass windows, 150-300 GeV (at 175) and >= 300 GeV (at 350).  DD: B/C/D 40/10/20 and
    20/12/4, A in the box.  PP: the same B/C/D with A = B C / D in each window (kappa 1 per window)."""
    h = make_hist8d("2mu2e")
    for d0, d1, windows in ((1, 1, ((175.0, dict(B=40, C=10, D=20)), (350.0, dict(B=20, C=12, D=4)))),
                            (5, 0, ((175.0, dict(A=20, B=40, C=10, D=20)), (350.0, dict(A=60, B=20, C=12, D=4))))):
        for mass, counts in windows:
            for r, n in counts.items():
                iso0, dphi = CORNERS[r]
                fill(h, n, iso0, 0.1, d0, d1, dphi, mass)
    return io.blind(io.cube_from_hist(h, "2mu2e"))


def test_split_ratio_is_the_one_bin_kappa_when_each_window_closes():
    cube = window_cube()
    windows = ((150.0, 300.0), (300.0, None))
    got = nl.split_ratio(cube, nl.NOMINAL, "DD", windows)
    assert got["windows"] == [(40.0, 10.0, 20.0), (20.0, 12.0, 4.0)] and got["one"] == (60.0, 22.0, 24.0)
    assert got["split"] == pytest.approx(20 + 60) and got["one_bin"] == pytest.approx(55)
    assert got["ratio"] == pytest.approx(80 / 55)
    pp = nl.split_ratio(cube, nl.NOMINAL, "PP", windows)
    assert ps.kappa(ps.regions(cube, nl.NOMINAL, "PP"))[0] == pytest.approx(pp["ratio"])


def test_bootstrap_reproduces_the_first_order_spread_for_large_counts():
    big = nl.bootstrap_abcd([(10000.0, 10000.0, 10000.0)], n_toys=20000, seed=1)
    lo, med, hi = nl.interval68(big["one_bin"])
    assert med == pytest.approx(10000, rel=0.005)
    assert (hi - lo) / 2 == pytest.approx(10000 * np.sqrt(3 / 10000), rel=0.05)
    assert np.allclose(big["ratio"], 1.0) and big["kept"] == 1.0
    two = nl.bootstrap_abcd([(4000.0, 2000.0, 1000.0), (8000.0, 4000.0, 2000.0)], n_toys=5000, seed=2)
    assert nl.interval68(two["ratio"])[1] == pytest.approx(1.0, abs=0.002)
    # B/D and C/D differ between the windows: one bin gives 3000 x 5000 / 2000, the windows 2000 + 4000
    mixed = nl.bootstrap_abcd([(2000.0, 1000.0, 1000.0), (1000.0, 4000.0, 1000.0)], n_toys=5000, seed=4)
    assert nl.interval68(mixed["one_bin"])[1] == pytest.approx(7500, rel=0.01)
    assert nl.interval68(mixed["split"])[1] == pytest.approx(6000, rel=0.01)
    assert nl.interval68(mixed["ratio"])[1] == pytest.approx(0.8, abs=0.005)
    small = nl.bootstrap_abcd([(26.0, 6.0, 3.0)], n_toys=40000, seed=3)
    assert small["kept"] == pytest.approx(1 - np.exp(-3.0), abs=0.005)
    again = nl.bootstrap_abcd([(26.0, 6.0, 3.0)], n_toys=40000, seed=3)
    assert np.array_equal(small["one_bin"], again["one_bin"])
    with pytest.raises(ValueError):
        nl.bootstrap_abcd([(np.nan, 6.0, 3.0)])


BLOCK_ISO = (0.1, 0.27, 0.6)            # < 0.25, 0.25-0.3, >= 0.3
BLOCK_DPHI = (1.0, 2.5, 3.0)            # < 2.4, 2.4-2.6, >= 2.6
BLOCKS = np.array([[30, 12, 50], [8, 3, 9], [60, 20, 80]], float)   # [iso block][dphi block]


def block_cube():
    h = make_hist8d("2mu2e")
    for i, iso0 in enumerate(BLOCK_ISO):
        for j, dphi in enumerate(BLOCK_DPHI):
            fill(h, int(BLOCKS[i, j]), iso0, 0.1, 5, 0, dphi, 400.0)
    return io.blind(io.cube_from_hist(h, "2mu2e"))


def _kappas(n):
    ours = (n[0, 1] + n[0, 2]) * (n[1, 0] + n[2, 0]) / ((n[1:, 1:].sum()) * n[0, 0])
    his = (n[0, 2] + n[1, 2]) * (n[2, 0] + n[2, 1]) / (n[2, 2] * n[:2, :2].sum())
    return his, ours


def test_wp_shift_error_is_the_first_order_propagation_over_the_shared_blocks():
    cube = block_cube()
    pp = nl.VR("PP", "", "P", "P")
    got = nl.wp_shift(cube, pp, nl.HIS_WP, nl.NOMINAL)
    his, ours = _kappas(BLOCKS)
    assert got["a"] == pytest.approx(his) and got["b"] == pytest.approx(ours)
    assert got["a"] == pytest.approx(nl.kappa_row(cube, pp, nl.HIS_WP)["kappa"])
    assert got["b"] == pytest.approx(nl.kappa_row(cube, pp, nl.NOMINAL)["kappa"])
    var = 0.0                                       # numerical derivative of the shift, block by block
    for i in range(3):
        for j in range(3):
            step = np.zeros((3, 3))
            step[i, j] = 1e-4
            d = (np.subtract(*_kappas(BLOCKS + step)[::-1]) - np.subtract(*_kappas(BLOCKS - step)[::-1])) / 2e-4
            var += BLOCKS[i, j] * d**2
    assert got["shift"] == pytest.approx(ours - his) and got["err"] == pytest.approx(np.sqrt(var), rel=1e-5)
    same = nl.wp_shift(cube, pp, nl.NOMINAL, nl.NOMINAL)
    assert same["shift"] == 0 and same["err"] == 0
    table = nl.wp_shift_table(cube, [pp])
    assert table.loc[0, "shift / error"] == pytest.approx(got["z"])


def test_signal_contamination_takes_the_largest_ratio_over_the_points():
    data = rule_cube()
    vrs = [nl.VR("PP", "", "P", "P"), nl.VR("DP", "", "D", "P")]
    counts = {"p1": {"PP": dict(A=40.0, B=0, C=0, D=0), "DP": dict(A=1.0, B=0, C=0, D=0),
                     "search region": dict(A=9.0, B=2.0, C=0.5, D=0.0)},
              nl.REFERENCE_POINT: {"PP": dict(A=4.0, B=0, C=0, D=0), "DP": dict(A=2.0, B=0, C=0, D=0),
                                   "search region": dict(A=30.0, B=1.0, C=1.0, D=1.0)}}
    t = nl.signal_contamination_table(counts, data, vrs, nl.NOMINAL).set_index(["region", "ABCD region"])
    assert list(t.index) == [("PP", "A"), ("DP", "A"), ("search region", "B"), ("search region", "C"),
                             ("search region", "D")]                     # A of the search region is not read
    assert t.loc[("PP", "A"), "observed"] == 400 and t.loc[("PP", "A"), "point"] == "p1"
    assert t.loc[("PP", "A"), "largest signal / observed"] == pytest.approx(0.1)
    assert t.loc[("DP", "A"), "point"] == nl.REFERENCE_POINT and t.loc[("DP", "A"), "largest signal / observed"] == 0.5
    assert t.loc[("search region", "B"), "reference signal / observed"] == pytest.approx(1 / 20)
    assert t.loc[("search region", "B"), "largest signal / observed"] == pytest.approx(2 / 20)
    assert t.loc[("PP", "A"), "its search-region yield"] == 9.0
    assert len(nl.contamination_view(t.reset_index())) == 5


def test_fourmu_readings_use_the_same_window_where_its_a_is_open():
    h = make_hist8d("4mu")
    for d0, d1, mass, counts in ((5, 5, 400.0, dict(A=30, B=40)), (5, 5, 100.0, dict(A=10, B=40)),
                                 (1, 5, 100.0, dict(A=20, B=40)), (5, 1, 100.0, dict(A=5, B=40)),
                                 (1, 5, 400.0, dict(B=20)), (5, 1, 400.0, dict(B=10)),
                                 (1, 1, 400.0, dict(B=4)), (1, 1, 100.0, dict(A=2))):
        for r, n in counts.items():
            iso0, dphi = CORNERS[r]
            for _ in range(n):
                event8d(h, iso0, 0.1, d0, d1, dphi=dphi, mass=mass)
    cube = io.blind(io.cube_from_hist(h, "4mu"))
    got = dict(nl.fourmu_readings(cube, nl.NOMINAL, nl.LOW))
    assert len(got) == 6
    assert got["B of DD × A/B of PP, same window"][0] == pytest.approx(4 * 30 / 40)
    assert got["B of DD × A/B of DP below the window"][0] == pytest.approx(4 * 20 / 40)
    assert got["B of DD × A/B of PD below the window"][0] == pytest.approx(4 * 5 / 40)
    assert got["A of DD below × A of PP between the windows"][0] == pytest.approx(2 * 30 / 10)
    assert got["A of DD below × B of DP between the windows"][0] == pytest.approx(2 * 20 / 40)


def test_expected_limit_is_the_limit_at_the_poisson_median():
    assert nl.expected_counting_limit(0.5) == (pytest.approx(-np.log(0.05)), 0)     # n = 0: 3.0 for any b
    lim, n = nl.expected_counting_limit(3.7)
    assert n == 4 and lim == pytest.approx(el.counting_limit(4, 3.7))
    lim, n = nl.expected_counting_limit(1.665)
    assert n == 1 and lim == pytest.approx(el.counting_limit(1, 1.665))


def test_overlap_and_nesting_of_the_qualifying_regions():
    v = {r.name: r for r in nl.our_vrs() + nl.DONGYUB_VRS}
    assert nl.vr_contains(v["V6"], v["V7"]) and nl.vr_contains(v["V6"], v["V5"]) and not nl.vr_contains(v["V7"], v["V6"])
    assert nl.vr_contains(v["V4"], v["V2"]) and nl.vr_contains(v["V4"], v["V3"])
    assert nl.vr_overlap(v["V6"], v["PP"]) and nl.vr_overlap(v["V6"], v["PD"]) and not nl.vr_overlap(v["PP"], v["PD"])
    assert not nl.vr_overlap(v["PD:g"], v["PD:e"]) and nl.vr_overlap(v["PD:g"], v["PD"])
    names = ["PP", "PD", "V1", "V2", "V3", "V4", "V5", "V6", "V7"] + [b.name for b in nl.dd_band_vrs()]
    nest = nl.nesting([v[n] for n in names])
    for pair in (("V7", "V6"), ("V5", "V6"), ("V2", "V4"), ("V3", "V4"),
                 ("DD, egamma iso ≥ 1", "DD, egamma iso ≥ 0.5"), ("DD, egamma iso 0.5–1", "DD, egamma iso ≥ 0.5")):
        assert pair in nest
    assert ("DD, egamma iso ≥ 1", "DD, egamma iso ≥ 0.75") not in nest      # only the largest container
    assert not any("V1" in pair or "PP" == pair[0] for pair in nest)        # V1 and PP are one definition
    disjoint = nl.max_disjoint([v[n] for n in names])
    assert len(disjoint) == 5 and {"PP", "PD", "DD, egamma iso ≥ 1", "DD, egamma iso 0.5–1"} <= set(disjoint)
    assert all(not nl.vr_overlap(v[a], v[b]) for a in disjoint for b in disjoint if a != b)
    assert len(nl.max_disjoint([v[n] for n in names[:9]])) == 3


def test_largest_deviation_is_over_the_qualifying_rows():
    table = nl.systematic_table(rule_cube(), nl.our_vrs())
    dev, name = nl.largest_deviation(table)
    assert name == "PD" and dev == pytest.approx(0.2)


def test_fourmu_b_scan_spans_the_smallest_error_about_the_median():
    readings = [("a", (1.2, 0.636, 0.636)), ("b", (2.0, 1.2, 1.2)), ("c", (3.0, 2.4, 2.4)),
                ("d", (np.nan, np.nan, np.nan))]
    scan = nl.fourmu_b_scan(readings)
    assert scan["central"] == (1.2, 2.0, 3.0)
    assert scan["error"] == pytest.approx(0.5)            # 0.53 rounded to 0.1
    assert scan["factor"] == pytest.approx(2.0)
    assert scan["b"] == pytest.approx([1.0, 1.2, 2.0, 3.0, 4.0])
