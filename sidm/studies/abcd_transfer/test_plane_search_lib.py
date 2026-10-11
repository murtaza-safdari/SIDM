"""Tests of the plane-search library on synthetic cubes: region conventions, the hidden box,
split planes, the statistics and the shapes of the summary tables."""
import dataclasses

import numpy as np
import pytest

import abcd6d_io as io
import plane_search_lib as ps
import transfer_lib as tl
from test_abcd6d_io import event, event8d, make_hist, make_hist8d

# (iso0, |dphi|) placing an event in each region of the nominal dphi x iso0 configuration
CORNERS = {"A": (0.1, 3.0), "B": (0.6, 3.0), "C": (0.1, 1.0), "D": (0.6, 1.0)}


def fill_regions(h, d0, d1, counts, mass=400.0, iso1=0.1):
    for region, n in counts.items():
        iso0, dphi = CORNERS[region]
        for _ in range(n):
            event(h, iso0, iso1, d0, d1, dphi=dphi, mass=mass)


def cube_2mu2e():
    """Known counts per plane and muon pixel class, at mass 400 (hidden box) and 100 (open)."""
    h = make_hist("2mu2e")
    fill_regions(h, 5, 0, dict(A=6, B=4, C=3, D=2))                 # PP (5 pixel hits)
    fill_regions(h, 3, 0, dict(A=2, B=2, C=1, D=1))                 # PP:mu3
    fill_regions(h, 5, 1, dict(A=9, B=6, C=3, D=2))                 # PD, kappa 1, A/B 1.5
    fill_regions(h, 1, 0, dict(A=3, B=3, C=2, D=2))                 # DP:mu1-2
    fill_regions(h, 0, 0, dict(A=2, B=4, C=1, D=1))                 # DP:mu0
    fill_regions(h, 2, 999, dict(A=1, B=8, C=4, D=2))               # DD:mu1-2, photon-only egamma LJ
    fill_regions(h, 0, 1, dict(A=1, B=4, C=2, D=1))                 # DD:mu0
    fill_regions(h, 0, 1, dict(A=1, B=2, C=2, D=2), mass=100.0)     # DD:mu0 at low mass
    fill_regions(h, 2, 999, dict(A=4, B=2, C=2, D=1), mass=100.0)   # DD:mu1-2 at low mass
    fill_regions(h, 5, 1, dict(A=3, B=2, C=2, D=1), mass=100.0)     # PD at low mass
    event(h, 0.1, 0.6, 5, 0, dphi=3.0)      # fails the iso1 event cut
    event(h, 0.1, 0.1, 5, 0, dphi=2.2)      # |dphi| buffer, in no region
    event(h, 0.3, 0.1, 5, 0, dphi=3.0)      # isolation buffer, in no region
    return io.cube_from_hist(h, "2mu2e")


def counts(reg):
    return {r: reg[r][0] for r in "ABCD"}


def test_region_conventions_and_buffers():
    cube = cube_2mu2e()
    reg = ps.regions(cube, ps.Config(), "PP")
    assert counts(reg) == {"A": 8, "B": 6, "C": 4, "D": 3}         # pixel 5 plus pixel 3, buffers excluded
    assert ps.kappa(ps.regions(cube, ps.Config(), "PD"))[0] == pytest.approx(1.0)
    assert ps.prediction(ps.regions(cube, ps.Config(), "PD"))[0] == pytest.approx(9.0)
    no_buffer = ps.Config(fail0=0.25, dphi_fail=2.4)
    assert counts(ps.regions(cube, no_buffer, "PP")) == {"A": 8, "B": 7, "C": 5, "D": 3}
    assert counts(ps.regions(cube, ps.Config(wp1=1.0), "PP"))["A"] == 9


def test_search_plane_hides_exactly_the_box():
    cube = io.blind(cube_2mu2e())
    reg = ps.regions(cube, ps.Config(), "DD")
    assert np.isnan(reg["A"][0])
    assert {r: reg[r][0] for r in "BCD"} == {"B": 12, "C": 6, "D": 3}
    assert np.isnan(ps.kappa(reg)[0]) and ps.prediction(reg)[0] == pytest.approx(24.0)
    assert ps.Config().dd_open()
    no_buffer = ps.Config(fail0=0.25, dphi_fail=2.4)                    # sidebands start at the box's edges
    assert no_buffer.dd_open()
    assert {r: ps.regions(cube, no_buffer, "DD")[r][0] for r in "BCD"} == {"B": 12, "C": 6, "D": 3}
    for cfg in (ps.Config(wp0=0.2, fail0=0.2), ps.Config(dphi_pass=2.6, dphi_fail=2.6),
                ps.Config("iso0 x iso1", fail1=0.25)):                 # a sideband reaches into the box
        assert not cfg.dd_open()
        assert any(np.isnan(v[0]) for r, v in ps.regions(cube, cfg, "DD").items() if r != "A")
    straddling = ps.regions(cube, ps.Config(mass=(100.0, 200.0)), "DD")      # A reaches into the box, B/C/D do not
    assert ps.Config(mass=(100.0, 200.0)).dd_open()
    assert np.isnan(straddling["A"][0]) and all(np.isfinite(straddling[r][0]) for r in "BCD")
    low = ps.Config(mass=(None, 150.0))
    assert low.dd_open()
    assert counts(ps.regions(cube, low, "DD")) == {"A": 5, "B": 4, "C": 4, "D": 3}
    for plane in ("PP", "DP", "PD"):
        assert not np.isnan(ps.regions(cube, ps.Config(), plane)["A"][0])


def test_split_planes_stay_inside_their_base_plane():
    cube = cube_2mu2e()
    assert counts(ps.regions(cube, ps.Config(), "PP:mu3")) == {"A": 2, "B": 2, "C": 1, "D": 1}
    assert counts(ps.regions(cube, ps.Config(), "PP:mu5+")) == {"A": 6, "B": 4, "C": 3, "D": 2}
    assert counts(ps.regions(cube, ps.Config(), "DD:g")) == {"A": 1, "B": 8, "C": 4, "D": 2}
    assert counts(ps.regions(cube, ps.Config(), "DD:e")) == {"A": 1, "B": 4, "C": 2, "D": 1}
    assert counts(ps.regions(cube, ps.Config(), "DD:mu0")) == {"A": 1, "B": 4, "C": 2, "D": 1}
    assert counts(ps.regions(cube, ps.Config(), "DP:mu1-2")) == {"A": 3, "B": 3, "C": 2, "D": 2}
    for label in ("DD:mu3", "PP:mu1-2", "PP:g", "DP:e1", "PD:mu0", "DD:mu3:g"):
        with pytest.raises(ValueError):
            ps.plane_selection("2mu2e", label)
    with pytest.raises(ValueError):
        ps.plane_selection("4mu", "PP:e1")
    assert ps.plane_selection("4mu", "PP:sub3") == {"disp0": (2.5, None), "disp1": (2.5, 3.5)}
    assert ps.plane_contains("2mu2e", "PD:mu3:g", "prompt", "displaced")
    assert not ps.plane_contains("2mu2e", "PD:mu3", "prompt", "prompt")


def test_dsa_splits_and_extra_cuts_on_an_8d_cube():
    """DSA-count splits of a muon LJ and event cuts on the two added axes, on a synthetic abcd8d
    cube with the same region counts as cube_2mu2e's DD:mu0 corner at low mass."""
    h = make_hist8d("2mu2e")
    for region, n in dict(A=1, B=2, C=2, D=2).items():
        iso0, dphi = CORNERS[region]
        for ndsa0, lj0_mass in ((0, 1.0), (1, 2.0), (2, 91.0)):
            for _ in range(n):
                event8d(h, iso0, 0.1, 0, 1, dphi=dphi, mass=100.0, lj0_mass=lj0_mass, ndsa0=ndsa0)
    cube = io.cube_from_hist(h, "2mu2e")
    low = ps.Config(mass=(None, 150.0))
    assert counts(ps.regions(cube, low, "DD:mu0")) == {"A": 3, "B": 6, "C": 6, "D": 6}
    assert counts(ps.regions(cube, low, "DD:dsa0")) == {"A": 1, "B": 2, "C": 2, "D": 2}
    assert counts(ps.regions(cube, low, "DD:mu0:dsa1+")) == {"A": 2, "B": 4, "C": 4, "D": 4}
    assert counts(ps.regions(cube, low, "DD:dsa2+")) == {"A": 1, "B": 2, "C": 2, "D": 2}
    assert ps.plane_selection("2mu2e", "DD:dsa1")["ndsa0"] == (0.5, 1.5)
    assert ps.plane_selection("4mu", "DD:subdsa1+") == {"disp0": (None, 2.5), "disp1": (None, 2.5), "ndsa1": (0.5, None)}
    with pytest.raises(ValueError):
        ps.plane_selection("2mu2e", "DD:subdsa1")
    with_cut = low.with_cut("lj0_mass", None, 75.0).with_cut("ndsa0", 0.5, None)
    assert with_cut.label().endswith("M<150, lj0_mass<75, ndsa0>=0.5")
    assert ps.region_ranges(with_cut)["A"]["lj0_mass"] == (None, 75.0)
    assert counts(ps.regions(cube, with_cut, "DD")) == {"A": 1, "B": 2, "C": 2, "D": 2}   # one DSA muon, mass 2
    assert counts(ps.regions(cube, low.with_cut("lj0_mass", 75.0, None), "DD")) == {"A": 1, "B": 2, "C": 2, "D": 2}
    with pytest.raises(ValueError):
        ps.regions(cube, low.with_cut("nbjets", 0.5, None), "DD")       # not an axis of this cube


def test_cube4_reproduces_the_full_cube():
    cube = cube_2mu2e()
    for plane in tl.PLANES:
        c4 = ps.to_cube4(cube, plane)
        for cfg in (ps.Config(), ps.Config(mass=(None, 150.0)), ps.Config("iso0 x iso1")):
            assert counts(ps.regions(c4, cfg, plane)) == counts(ps.regions(cube, cfg, plane))
        full = ps.profile(cube, "ljlj_mass", iso0=(None, 0.25), **ps.plane_selection("2mu2e", plane))[0]
        assert np.array_equal(ps.profile(c4, "ljlj_mass", iso0=(None, 0.25))[0], full)


def test_statistics():
    assert ps.pull((2.0, 0.2, 0.2), (2.0, 0.3, 0.3)) == 0.0
    assert ps.pull((4.0, 0.2, 0.2), (2.0, 0.1, 0.1)) > 5
    assert ps.pull((1.0, 0.2, 0.2), (2.0, 0.1, 0.1)) == pytest.approx(np.log(0.5) / np.hypot(0.2, 0.05), rel=0.05)
    assert ps.log_sigma((1.0, 0.1, 0.1)) == pytest.approx(0.1, rel=0.01)
    assert np.isnan(ps.pull((np.nan, np.nan, np.nan), (2.0, 0.1, 0.1)))
    a, b, c, d = ((40.0, 40.0), (20.0, 20.0), (10.0, 10.0), (10.0, 10.0))
    k = ps.kappa({"A": a, "B": b, "C": c, "D": d})
    assert k[0] == pytest.approx(2.0) and 0 < k[1] < k[0] and k[2] > 0
    cube = io.blind(cube_2mu2e())
    cfg = ps.Config()
    var, sel = ps.fail_slice_selection(cfg, "iso")
    assert var == "abs_dphi" and "abs_dphi" not in sel and sel["iso0"] == (0.5, None)
    chi2, ndf, p, n_pd, n_dd = ps.shape_p(cube, var, sel, "PD", "DD")
    assert (n_pd, n_dd) == (8.0, 15.0) and ndf >= 1 and 0 <= p <= 1
    var, sel = ps.fail_slice_selection(cfg, "first")
    assert var == "iso0" and "iso0" not in sel and sel["abs_dphi"] == (None, 2.0)


def test_kappa_vs_mass_and_config_grid():
    cube = cube_2mu2e()
    bins = [("<150", (None, 150.0)), (">=150", (150.0, None))]
    df, k_all, chi2, ndf = ps.kappa_vs_mass(cube, ps.Config(mass=(None, None)), "PD", bins=bins)
    assert list(df["bin"]) == ["<150", ">=150"]
    assert df.loc[1, "kappa"] == pytest.approx(1.0) and k_all[0] == pytest.approx((12 / 5) / (8 / 3))
    assert ndf == 1
    expected = sum((np.log(k / k_all[0]) / sig) ** 2 for k, sig in zip(df["kappa"], df["log sigma"]))
    assert chi2 == pytest.approx(expected)
    df, _, chi2, ndf = ps.kappa_vs_mass(cube, ps.Config(mass=(None, None)), "DP:mu0", bins=bins)
    assert np.isnan(chi2)                                                # one bin with a finite kappa: no chi2
    grid = ps.config_grid()
    assert len(grid) == len(set(grid)) and ps.Config() in grid
    assert all(c.dphi_fail <= c.dphi_pass and c.fail0 >= c.wp0 for c in grid)
    blinded = io.blind(cube)
    every_kind = grid + ps.config_grid("iso0 x iso1") + ps.config_grid("dphi x iso1")
    assert all(c.dd_open() == all(np.isfinite(ps.regions(blinded, c, "DD")[r][0]) for r in "BCD") for c in every_kind)
    assert 0 < sum(c.dd_open() for c in every_kind) < len(every_kind)


def test_scan_and_summary_tables():
    cube = io.blind(cube_2mu2e())
    cfgs = [ps.Config(), ps.Config(dphi_pass=2.6)]
    df = ps.scan(cube, cfgs, planes=["PP", "PD"])
    assert len(df) == 4 and set(df["plane"]) == {"PP", "PD"}
    pd_row = df[(df["plane"] == "PD") & (df["dphi_pass"] == 2.4)].iloc[0]
    assert pd_row["kappa"] == pytest.approx(1.0) and pd_row["DD_pred"] == pytest.approx(24.0)
    assert pd_row["pred_from_V"] == pytest.approx(24.0) and np.isnan(pd_row["DD_A"])
    assert np.isfinite(pd_row["pull_bd"]) and np.isfinite(pd_row["pull_cd"])
    planes = ["PP", "PD", "DP:mu1-2"]
    assert len(ps.plane_table(cube, ps.Config(), planes)) == 3
    assert len(ps.compatibility_table(cube, ps.Config(), planes)) == 4
    classes = ps.class_table(cube, ps.Config())
    assert set(classes["plane"]) == {"PP", "DP", "PD", "DD"}
    assert list(classes.loc[classes["plane"] == "DD", "pixel hits"]) == ["all", "0", "1-2"]
    steps = ps.muon_step_tests(cube, ps.Config())
    assert len(steps) == 5 and steps["prompt plane"].str.startswith(("PP", "PD")).all()
    assert "ratio 1-2 hits / prompt 3 hits" in steps.columns


def test_estimate_table_arithmetic():
    cube = io.blind(cube_2mu2e())
    steps = ps.pixel0_step_ratios(cube, ps.Config())
    assert steps["egamma prompt"][0] == pytest.approx((2 / 4) / (8 / 6))               # DP:mu0 against PP
    assert steps["egamma displaced, M<150"][0] == pytest.approx((1 / 2) / (3 / 2))     # DD:mu0 against PD, low mass
    row = ps.estimate_table(cube, [(">=150", ps.Config())]).iloc[0]
    assert row["DD 1-2 hits B/C/D"] == "8/4/2"
    assert row["in-plane B C/D"].startswith("16.0")          # 8 x 4 / 2
    assert row["kappa PD"].startswith("1.00")
    assert row["x kappa PD"].startswith("16.0")
    assert row["A/B PD"].startswith("1.50")
    assert row["B x A/B PD"].startswith("12.0")              # 8 x 1.5
    assert row["DD 0 hits B"] == "4"
    assert row["0 hits: B x A/B PD x step (egamma prompt)"].startswith("2.2")                 # 4 x 1.5 x 0.375
    assert row["0 hits: B x A/B PD x step (egamma displaced, M<150)"].startswith("2.0")      # 4 x 1.5 x 1/3


def test_signal_tables_accept_full_and_exported_cubes():
    cube = cube_2mu2e()
    exported = {plane: ps.to_cube4(cube, plane) for plane in tl.PLANES}
    cfg, ref = ps.Config(), ps.Config(dphi_pass=2.0)
    for signal in (cube, exported):
        lk = ps.leakage(signal, cfg)
        assert lk["A"] == 2 and lk["fB"] == pytest.approx(6.0) and lk["fC"] == pytest.approx(3.0)
        assert ps.acceptance(signal, cfg, ref) == pytest.approx(1.0)
        table = ps.signal_table({"signal_x": {"2mu2e": signal}}, "2mu2e", cfg, ref)
        assert list(table["point"]) == ["x"] and table.loc[0, "A fraction at M>=300"] == "0.286"    # 2 of 7 A-like events
    classes = ps.signal_class_table({"signal_x": {"2mu2e": cube}}, "2mu2e", cfg)
    assert classes.loc[0, "pixel hits 0 / 1 / 2"] == "0.50 / 0.00 / 0.50"
    assert classes.loc[0, "lost hits 1 / 2-9 / photon-only"] == "0.50 / 0.00 / 0.50"
    mc = ps.mc_composition_table({"QCD": {"2mu2e": cube}, "DY": {"2mu2e": exported}}, io.blind(cube), "2mu2e", cfg)
    assert len(mc) == 16 and mc.loc[(mc["plane"] == "DD") & (mc["region"] == "A"), "data"].item() == "blind"
    assert mc.loc[(mc["plane"] == "PD") & (mc["region"] == "A"), "kappa (simulation)"].item().startswith("1.00")
    windows = [(">=150", cfg)]
    sig = ps.signal_in_validation_table({"signal_x": {"2mu2e": cube}}, "2mu2e", io.blind(cube), windows)
    assert list(sig["plane"]) == ["PD", "DP"] and sig.loc[0, "x in A"] == "9.0" and sig.loc[0, "data A"] == "9"
    assert sig.loc[0, "largest signal / data A"] == "1.00"
    half = ps.Cube4("2mu2e", 0.5 * exported["PD"].sumw, 0.25 * exported["PD"].sumw2)      # half of PD, as a "process"
    sub = ps.subtracted_table({"DY": {"2mu2e": {"PD": half}}}, io.blind(cube), "2mu2e", windows)
    assert sub.loc[0, "DY / data"] == "0.50/0.50/0.50/0.50"
    assert sub.loc[0, "DY N_eff A/B/C/D"] == "9.0/6.0/3.0/2.0"       # half the yield on the same events: N_eff unchanged
    assert sub.loc[0, "kappa, DY subtracted"].startswith("1.00") and sub.loc[0, "A/B, DY subtracted"].startswith("1.50")


def test_with_cut_refuses_abcd_axes_and_repeated_variables():
    cfg = ps.Config()
    with pytest.raises(ValueError):
        cfg.with_cut("iso1", None, 1.0)
    cut = cfg.with_cut("lj0_mass", None, 75.0)
    with pytest.raises(ValueError):
        cut.with_cut("lj0_mass", 5.0, None)
    assert cut.with_cut("ndsa0", 0.5, None).extra == (("lj0_mass", (None, 75.0)), ("ndsa0", (0.5, None)))
