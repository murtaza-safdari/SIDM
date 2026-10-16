"""Tests of the nominal-plane ABCD library on synthetic 8D cubes: the validation-region definitions,
the hidden box, the systematic rule, the 2D plane histogram and the signal shares."""
import itertools

import numpy as np
import pytest

import abcd6d_io as io
import estimate_lib as el
import nominal_lib as nl
import plane_search_lib as ps
from test_abcd6d_io import make_hist8d
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
    assert first["below"] == pytest.approx(0.02)
    assert shares.iloc[1][">=600"] == pytest.approx(97 / 100) and shares.iloc[1]["150-300"] == 0
    summary = nl.share_summary(shares)
    assert summary["mediator mass [GeV]"].tolist() == ["200", "800"]


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
