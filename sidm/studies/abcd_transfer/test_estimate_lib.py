"""Tests of the estimate tables on synthetic 8D and proxy cubes."""
import dataclasses

import numpy as np

import abcd6d_io as io
import estimate_lib as el
import plane_search_lib as ps
from test_abcd6d_io import event8d, make_hist8d
from test_plane_search_lib import CORNERS, counts


def proxy_cube(topology="2mu2e"):
    """DD plane of a proxy cube: A/B/C/D = 4/2/1/1 without a b-jet (kappa 2) and 2/2/2/2 with one."""
    h = make_hist8d(topology, "abcd8d_proxy")
    for region, (n0, n1) in dict(A=(4, 2), B=(2, 2), C=(1, 2), D=(1, 2)).items():
        iso0, dphi = CORNERS[region]
        for _ in range(n0):
            event8d(h, iso0, 0.1, 0, 1, dphi=dphi, mass=400.0, nbjets=0, ndsa0=1)
        for _ in range(n1):
            event8d(h, iso0, 0.1, 2, 999, dphi=dphi, mass=400.0, nbjets=1, ndsa0=0)
    return io.cube_from_hist(h, topology)


def test_proxy_tables_split_by_btag_and_class():
    cube = proxy_cube()
    cfg = ps.Config()
    assert counts(ps.regions(cube, cfg, "DD")) == {"A": 6, "B": 4, "C": 3, "D": 3}
    assert counts(ps.regions(cube, el.with_btag(cfg, ">=1 b-jet"), "DD")) == {"A": 2, "B": 2, "C": 2, "D": 2}
    assert counts(ps.regions(cube, cfg, "DD:dsa1+")) == {"A": 4, "B": 2, "C": 1, "D": 1}
    assert counts(ps.regions(cube, cfg, "DD:mu0:e1")) == {"A": 4, "B": 2, "C": 1, "D": 1}
    grid = el.class_grid_table(cube, cfg, ["DD"], [None, "mu0", "mu1-2", "g"], btag=("all", ">=1 b-jet"))
    assert list(grid["plane"]) == ["DD", "DD", "DD:mu0", "DD:mu0", "DD:mu1-2", "DD:mu1-2", "DD:g", "DD:g"]
    assert grid.loc[(grid["plane"] == "DD") & (grid["b-tag"] == ">=1 b-jet"), "kappa"].item().startswith("1.00")
    assert grid.loc[(grid["plane"] == "DD:mu0") & (grid["b-tag"] == "all"), "kappa"].item().startswith("2.00")
    scan = el.cut_scan_table(cube, cfg, "DD", {"none": None, "no b-jet": ("nbjets", None, 0.5)})
    assert list(scan["A"]) == ["6", "4"]


def test_cross_compatibility_and_final_tables():
    h = make_hist8d("2mu2e")
    for region, n in dict(A=1, B=6, C=4, D=2).items():            # the search cube's DD plane, A hidden after blinding
        iso0, dphi = CORNERS[region]
        for _ in range(n):
            event8d(h, iso0, 0.1, 0, 1, dphi=dphi, mass=400.0)
    search = io.blind(io.cube_from_hist(h, "2mu2e"))
    cfg = ps.Config()
    table = el.cross_compatibility_table(proxy_cube(), ["DD"], search, cfg, btag=("all", ">=1 b-jet"))
    assert len(table) == 3 and table["A"].iloc[-1] == "blind" and table["B/D"].iloc[-1].startswith("3.00")
    assert table["B/D"].iloc[1].startswith("1.00") and float(table["B/D pull vs DD"].iloc[1]) < 0   # proxy 1 vs search 3
    assert table["kappa"].iloc[1].startswith("1.00") and table["kappa"].iloc[0].startswith("1.50")
    final = el.final_table(search, [("M >= 150", cfg)], {"PD": (1.26, 0.15, 0.15), "one": (1.0, 0.0, 0.0)})
    assert final["DD B/C/D"].item() == "6/4/2" and final["B C/D"].item().startswith("12.0")
    assert final["x kappa one"].item().startswith("12.0") and final["x kappa PD"].item().startswith("15.1")
    consistency = el.consistency_table(search, search, cfg, planes=("DD",))
    assert consistency["A diff"].item() == "blind" and consistency["B diff"].item() == "+0"


def test_spectrum_folds_flow_bins():
    h = make_hist8d("2mu2e")
    event8d(h, 0.1, 0.1, 5, 1, dphi=3.0, mass=400.0, lj0_mass=91.0)
    event8d(h, 0.1, 0.1, 5, 1, dphi=3.0, mass=400.0, lj0_mass=500.0)    # lj0_mass overflow
    event8d(h, 0.1, 0.1, 5, 1, dphi=3.0, mass=100.0, lj0_mass=91.0)     # outside the window
    cube = io.cube_from_hist(h, "2mu2e")
    edges, sumw, sumw2 = el.spectrum(cube, "lj0_mass", ps.Config(), "PD", "A")
    assert len(sumw) == len(edges) - 1 == 7 and sumw.sum() == 2.0 and sumw[-1] == 1.0 and sumw[4] == 1.0


def test_reduced_cube_reproduces_the_full_cube_regions():
    cube = proxy_cube()
    cfg = ps.Config()
    for plane, btag in (("DD", "all"), ("DD", ">=1 b-jet"), ("DD:dsa1+", "all")):
        c = el.with_btag(cfg, btag)
        reduced = el.reduce_plane(cube, plane, c.extra)
        assert reduced.spec.variables == el.ABCD_AXES
        assert counts(el.regions_reduced(reduced, c)) == counts(ps.regions(cube, c, plane))
    h = make_hist8d("2mu2e")
    for region, n in dict(A=1, B=6, C=4, D=2).items():
        iso0, dphi = CORNERS[region]
        for _ in range(n):
            event8d(h, iso0, 0.1, 0, 1, dphi=dphi, mass=400.0)
    blind_dd = el.reduce_plane(io.blind(io.cube_from_hist(h, "2mu2e")), "DD")
    assert np.isnan(el.regions_reduced(blind_dd, cfg)["A"][0])          # the hidden corner stays hidden


def test_closure_scan_ranks_by_the_selection_regions_and_closure_table_judges():
    cube = proxy_cube()
    cfg = ps.Config()
    reduced = {"all": el.reduce_plane(cube, "DD"), "b": el.reduce_plane(cube, "DD", el.with_btag(cfg, ">=1 b-jet").extra)}
    cfgs = [cfg, dataclasses.replace(cfg, wp0=0.3)]
    scan = el.closure_scan(reduced, cfgs, select=["b"], confirm=["all"], target=5.0)     # a few events: loose target
    assert list(scan["kind"]) == ["dphi x iso0"] * 2
    assert scan["worst |kappa-1| select"].iloc[0] == 0.0 and scan["all select within target"].iloc[0]
    assert scan["worst |kappa-1| confirm"].iloc[0] == 0.5 and scan["all confirm within target"].iloc[0]
    strict = el.closure_scan(reduced, cfgs, select=["b"], confirm=["all"])     # too few events to test 15%
    assert strict["untestable select"].iloc[0] == 1 and np.isinf(strict["worst |kappa-1| select"].iloc[0])
    assert not strict["all select within target"].any() and not strict["testable select within target"].any()
    reduced["pp"] = el.reduce_plane(cube, "PP")                                 # empty: its kappa is undefined
    mixed = el.closure_scan(reduced, cfgs, select=["b", "pp"], confirm=["all"], target=5.0)
    assert mixed["untestable select"].iloc[0] == 1 and mixed["worst |kappa-1| select"].iloc[0] == 0.0
    assert not mixed["all select within target"].any()        # one region cannot test the configuration
    assert mixed["testable select within target"].iloc[0]     # the one that can, closes
    table = el.closure_table([("b-tagged", cube, el.with_btag(cfg, ">=1 b-jet"), "DD"), ("all", cube, cfg, "DD")])
    assert table["kappa"].iloc[0].startswith("1.00") and table["verdict"].iloc[0].startswith("imprecise")
    assert table["kappa - 1 [%]"].iloc[1].startswith("+50")


def test_class_recipe_sums_the_per_class_products():
    h = make_hist8d("2mu2e")
    iso_b, dphi_b = CORNERS["B"]
    iso_a, dphi_a = CORNERS["A"]
    for ndsa, n_a, n_b in ((0, 4, 2), (1, 1, 2)):          # PD: A/B 2.0 without a DSA muon, 0.5 with one
        for _ in range(n_a):
            event8d(h, iso_a, 0.1, 3, 1, dphi=dphi_a, mass=400.0, ndsa0=ndsa)
        for _ in range(n_b):
            event8d(h, iso_b, 0.1, 3, 1, dphi=dphi_b, mass=400.0, ndsa0=ndsa)
    for ndsa, n_b in ((0, 1), (1, 3)):                     # DD: B of 1 and 3 events in the two classes
        for _ in range(n_b):
            event8d(h, iso_b, 0.1, 1, 1, dphi=dphi_b, mass=400.0, ndsa0=ndsa)
    cube = io.blind(io.cube_from_hist(h, "2mu2e"))
    table = el.class_recipe_table(cube, [("M >= 150", ps.Config())])
    assert list(table["DD B dsa0"]) == ["1"] and list(table["DD B dsa1+"]) == ["3"]
    assert table["A/B PD dsa0"].item().startswith("2.00") and table["A/B PD dsa1+"].item().startswith("0.50")
    assert table["sum over classes"].item().startswith("3.5")          # 1 x 2.0 + 3 x 0.5
    assert table["inclusive B x A/B"].item().startswith("5.0")         # 4 x (5/4)
    assert table["inclusive C x A/C"].item() == "n/a"                  # no event in C of either plane
    assert table["DD A observed"].item() == "blind"


def test_class_recipe_shows_the_observed_a_where_the_window_is_open():
    h = make_hist8d("2mu2e")
    iso_b, dphi_b = CORNERS["B"]
    iso_a, dphi_a = CORNERS["A"]
    for _ in range(4):
        event8d(h, iso_a, 0.1, 3, 1, dphi=dphi_a, mass=100.0)          # PD: A/B 2.0, one class
    for _ in range(2):
        event8d(h, iso_b, 0.1, 3, 1, dphi=dphi_b, mass=100.0)
    iso_c, dphi_c = CORNERS["C"]
    event8d(h, iso_c, 0.1, 3, 1, dphi=dphi_c, mass=100.0)              # PD: A/C 4.0
    for _ in range(3):
        event8d(h, iso_a, 0.1, 1, 1, dphi=dphi_a, mass=100.0)          # DD below 150 GeV: A 3, B 1, C 2
    event8d(h, iso_b, 0.1, 1, 1, dphi=dphi_b, mass=100.0)
    for _ in range(2):
        event8d(h, iso_c, 0.1, 1, 1, dphi=dphi_c, mass=100.0)
    cube = io.blind(io.cube_from_hist(h, "2mu2e"))
    low = dataclasses.replace(ps.Config(), mass=(None, 150.0))
    table = el.class_recipe_table(cube, [("M < 150", low)])
    assert table["inclusive B x A/B"].item().startswith("2.0")         # 1 x 2.0, against 3 observed
    assert table["inclusive C x A/C"].item().startswith("8.0")         # 2 x 4.0
    assert table["DD A observed"].item() == "3"


def test_parse_fmt_inverts_fmt():
    import transfer_lib as tl
    for k in ((1.27, 0.15, 0.15), (0.46, 0.24, 0.16), (2.0, 0.5, np.inf), (np.nan, np.nan, np.nan)):
        v, m, p = el.parse_fmt(tl.fmt(k, 2))
        assert (np.isnan(v) and np.isnan(k[0])) or (v, m, p) == tuple(round(x, 2) if np.isfinite(x) else x for x in k)


def test_kappa_by_family_groups_the_scan_by_plane_axes():
    import pandas as pd
    scan = pd.DataFrame({"config": ["dphi 2.4/2 x iso0 0.25/0.5, iso1<0.5, M>=150", "dphi 2/2 x iso0 0.3/0.5, iso1<0.25, M>=150",
                                    "iso0 0.2/0.5 x iso1 0.25/1, dphi>=2, M>=150", "dphi 2.8/2 x iso1 0.5/1, iso0<0.25, M>=150",
                                    "dphi 2/2 x iso1 0.5/0.5, iso0<0.25, M>=150"],
                         "kappa": ["1.27 ± 0.15", "1.05 ± 0.10", "n/a", "0.46 -0.24 +0.16", "0.90 ± 0.20"]})
    table = el.kappa_by_family(scan, "kappa").set_index("family")
    assert list(table.index) == ["dphi x iso0", "iso0 x iso1", "dphi x iso1"]
    assert table.loc["dphi x iso0", "interval above 1"] == 1 and table.loc["dphi x iso0", "interval covers 1"] == 1
    assert table.loc["dphi x iso0", "within 15% at that precision"] == 1          # 1.05 ± 0.10, not 1.27
    assert table.loc["iso0 x iso1", "kappa defined"] == 0 and table.loc["iso0 x iso1", "kappa min-max"] == "n/a"
    assert table.loc["dphi x iso1", "interval below 1"] == 1 and table.loc["dphi x iso1", "interval covers 1"] == 1
    assert table.loc["dphi x iso1", "kappa min-max"] == "0.46-0.90"


def test_scan_summary_counts_the_precise_configurations():
    import pandas as pd
    scan = pd.DataFrame({"config": ["a", "b", "c", "d"],
                         "kappa r": ["1.10 ± 0.10", "0.50 ± 0.05", "1.00 -0.40 +0.40", "n/a"]})
    row = el.scan_summary(scan, ["r"]).iloc[0]
    assert row["kappa defined"] == 3 and row["measured to 15%"] == 2          # the ±0.40 one is imprecise
    assert row["best precision [%]"] == "9"
    assert row["kappa min-max (those)"] == "0.50-1.10" and row["within 15%"] == 1


def small_count_cube():
    """A blinded 4mu cube whose DD plane is nearly empty: below 150 GeV A/B/C/D = 2/0/3/1 (the
    A events with one DSA muon per LJ), above it B = 1 (a DSA-only leading LJ) and a hidden A;
    the open planes carry known counts for the count-limited estimates."""
    h = make_hist8d("4mu")
    for region, n, kw in (("A", 2, dict(ndsa0=1, ndsa1=1)), ("C", 3, {}), ("D", 1, {})):
        iso0, dphi = CORNERS[region]
        for _ in range(n):
            event8d(h, iso0, 0.1, 1, 1, dphi=dphi, mass=100.0, **kw)
    event8d(h, 0.6, 0.1, 1, 1, dphi=3.0, mass=400.0, ndsa0=2, ndsa1=1)      # DD B above 150
    event8d(h, 0.1, 0.1, 1, 1, dphi=3.0, mass=400.0)                        # DD A above 150: hidden
    for (d0, d1), mass, region, n in ((( 1, 5), 100.0, "A", 3), ((1, 5), 100.0, "B", 1), ((1, 5), 400.0, "B", 2),
                                      ((5, 1), 100.0, "A", 1), ((5, 1), 100.0, "B", 2),
                                      ((5, 5), 100.0, "A", 2), ((5, 5), 100.0, "B", 4), ((5, 5), 400.0, "A", 4), ((5, 5), 400.0, "B", 2)):
        iso0, dphi = CORNERS[region]
        for _ in range(n):
            event8d(h, iso0, 0.1, d0, d1, dphi=dphi, mass=mass)
    return io.blind(io.cube_from_hist(h, "4mu"))


def test_count_grid_splits_the_regions_and_keeps_the_box():
    cube = small_count_cube()
    low = dataclasses.replace(ps.Config(), mass=(None, 150.0))
    grid = el.count_grid(cube, [("M<150", low), ("M>=150", ps.Config())], "DD", ["dsa1:subdsa1", "dsa2+:subdsa1"])
    low_a = grid[(grid["window"] == "M<150") & (grid["region"] == "A")].iloc[0]
    assert low_a["total"] == "2" and low_a["dsa1:subdsa1"] == "2" and low_a["dsa2+:subdsa1"] == "0"
    high = grid[grid["window"] == "M>=150"].set_index("region")
    assert high.loc["A", "total"] == "blind" and high.loc["A", "dsa1:subdsa1"] == "blind"
    assert high.loc["B", "total"] == "1" and high.loc["B", "dsa2+:subdsa1"] == "1"
    assert list(grid[grid["window"] == "M<150"]["total"]) == ["2", "0", "3", "1"]


def test_count_limited_estimates_multiply_the_open_counts():
    cube = small_count_cube()
    high, low = ("M>=150", ps.Config()), ("M<150", dataclasses.replace(ps.Config(), mass=(None, 150.0)))
    side = el.sideband_transfer_table(cube, high, low, refs=("DP", "PD", "PP")).set_index("reference plane")
    assert side.loc["DP", "A, B (M<150)"] == "3, 1" and side.loc["DP", "A/B (M<150)"].startswith("3.00")
    assert side.loc["DP", "A/B (M>=150)"] == "blind" and side.loc["PP", "A/B (M>=150)"].startswith("2.00")
    assert side.loc["DP", "DD B (M>=150)"] == "1" and side.loc["DP", "estimate of DD A (M>=150)"].startswith("3.00")
    assert side.loc["PD", "estimate of DD A (M>=150)"].startswith("0.50")
    mass = el.mass_transfer_table(cube, high, low, refs=(("PP", "A"), ("DP", "B"))).set_index("reference region")
    assert mass.loc["PP A", "N (M>=150)"] == "4" and mass.loc["PP A", "N (M<150)"] == "2"
    assert mass.loc["PP A", "ratio"].startswith("2.00") and mass.loc["PP A", "DD A (M<150)"] == "2"
    assert mass.loc["PP A", "estimate of DD A (M>=150)"].startswith("4.00")
    assert mass.loc["DP B", "estimate of DD A (M>=150)"].startswith("4.00")


def test_simulation_and_signal_placement_tables(monkeypatch):
    import transfer_lib as tl
    h = make_hist8d("4mu")
    for _ in range(2):
        event8d(h, 0.1, 0.1, 1, 1, dphi=3.0, mass=400.0, weight=2.0)        # DD A: w 4, w2 8 -> N_eff 2
    event8d(h, 0.6, 0.1, 1, 1, dphi=3.0, mass=100.0, weight=0.5)              # DD B below 150
    mc = io.cube_from_hist(h, "4mu")                                           # simulation: never blinded
    table = el.mc_region_table(mc, [("M>=150", ps.Config())], ("DD", "PP")).set_index("plane")
    assert table.loc["DD", "A"] == "4.00 ± 2.83 (N_eff 2.0)" and table.loc["PP", "A"].startswith("0.00")
    monkeypatch.setattr(tl, "sample_cubes", lambda base, kind, names, topologies, prefix: {names[0]: {"4mu": mc}})
    low = dataclasses.replace(ps.Config(), mass=(None, 150.0))
    cells = [("A DD M>=150", ps.Config(), "DD", "A"), ("B DD M<150", low, "DD", "B"), ("A PP M>=150", ps.Config(), "PP", "A")]
    sig = el.signal_cells_table("/nowhere", ["point"], "4mu", cells).set_index("region")
    assert sig.loc["A DD M>=150", "point"] == "1.000 [4.00]" and sig.loc["B DD M<150", "point"] == "0.125 [0.50]"
    assert sig.loc["A PP M>=150", "point"] == "0.000 [0.00]"


def test_counting_limit_reproduces_the_poisson_upper_limits():
    # flat-prior Bayesian limits for a background-free experiment (PDG), and the b-independence of n = 0
    assert np.isclose(el.counting_limit(0, 0.0), 3.00, atol=0.01)
    assert np.isclose(el.counting_limit(1, 0.0), 4.74, atol=0.01)
    assert np.isclose(el.counting_limit(2, 0.0), 6.30, atol=0.01)
    assert np.isclose(el.counting_limit(0, 2.3), 3.00, atol=0.01)
    assert el.counting_limit(1, 2.3) < el.counting_limit(1, 0.5) < el.counting_limit(1, 0.0)
    table = el.counting_limit_table((0, 1), (0.5, 2.3))
    assert list(table.columns) == ["observed", "b = 0.5", "b = 2.3"]
    assert table["b = 0.5"].iloc[0] == "3.00"
