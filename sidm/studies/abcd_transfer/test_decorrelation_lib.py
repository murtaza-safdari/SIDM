"""Tests of the mass-decorrelation library on synthetic populations and 8D cubes: the identity behind
the mixing factor, the mixing factor with and without a flat a(M), the logistic fit, the choice of
the flattening edges, and the hidden box."""
import numpy as np
import pytest

import abcd6d_io as io
import decorrelation_lib as dl
import nominal_lib as nl
import plane_search_lib as ps
from test_abcd6d_io import make_hist8d
from test_nominal_lib import DISP, fill

PASS_DPHI, FAIL_DPHI = 3.0, 1.0
MASSES = (175.0, 250.0, 350.0, 500.0, 700.0)          # one value inside each of the five bins


def make_cube(entries, blind=True):
    """An 8D 2mu2e cube from ``[(plane, mass, iso0, dphi, n)]`` with the egamma LJ isolated."""
    h = make_hist8d("2mu2e")
    for plane, mass, iso0, dphi, n in entries:
        fill(h, int(n), iso0, 0.1, DISP["mu"][plane[0]], DISP["egm"][plane[1]], dphi, mass)
    cube = io.cube_from_hist(h, "2mu2e")
    return io.blind(cube) if blind else cube


def abcd_entries(plane, per_bin, iso_pass=0.12, iso_fail=0.6):
    """Entries placing (A, B, C, D) counts of each mass bin at one pass and one fail isolation."""
    out = []
    for mass, (A, B, C, D) in zip(MASSES, per_bin):
        out += [(plane, mass, iso_pass, PASS_DPHI, A), (plane, mass, iso_fail, PASS_DPHI, B),
                (plane, mass, iso_pass, FAIL_DPHI, C), (plane, mass, iso_fail, FAIL_DPHI, D)]
    return out


# ----------------------------------------------------------------------------- the identity

@pytest.mark.parametrize("a, b", [((0.3, 0.3, 0.3, 0.3), (0.6, 0.7, 0.8, 0.9)),     # flat a(M)
                                  ((0.2, 0.3, 0.5, 0.7), (0.75, 0.75, 0.75, 0.75))])  # flat b(M)
def test_one_bin_estimate_is_exact_when_either_probability_is_flat(a, b):
    r = dl.synthetic_check((400, 300, 100, 50), a, b)
    assert r["one_bin"] == pytest.approx(r["A"])
    assert r["mixing"] == pytest.approx(1.0)
    assert r["split"] == pytest.approx(r["A"])


def test_mixing_factor_is_the_one_bin_bias_and_the_covariance_of_the_odds():
    r = dl.synthetic_check((400, 300, 100, 50), (0.2, 0.3, 0.5, 0.7), (0.6, 0.7, 0.8, 0.9))
    assert r["mixing"] > 1.1
    assert r["closure"] == pytest.approx(r["mixing"])
    assert r["covariance"] == pytest.approx(r["mixing"])
    falling = dl.synthetic_check((400, 300, 100, 50), (0.7, 0.5, 0.3, 0.2), (0.6, 0.7, 0.8, 0.9))
    assert falling["mixing"] < 1                     # opposite trends: the one-bin estimate is high


def test_mixing_factor_on_a_cube_with_and_without_a_flat_a():
    # PP: C/D = 1/2 in every bin, B/D rising, A = B C / D in each bin
    flat = [(4, 8, 10, 20), (9, 18, 6, 12), (10, 20, 2, 4), (6, 12, 1, 2), (5, 10, 1, 2)]
    cube = make_cube(abcd_entries("PP", flat))
    m = dl.mixing_factor(cube, dl.NOMINAL, "PP", dl.FIVE)
    assert m["ratio"] == pytest.approx(1.0) and not m["merged"]
    # the same B and D, with C/D rising: mixing = sum B C / D over the one-bin estimate
    rising = [(2, 8, 5, 20), (9, 18, 6, 12), (20, 20, 4, 4), (18, 12, 3, 2), (20, 10, 4, 2)]
    cube = make_cube(abcd_entries("PP", rising))
    m = dl.mixing_factor(cube, dl.NOMINAL, "PP", dl.FIVE)
    B, C, D = (np.array([r[i] for r in rising], float) for i in (1, 2, 3))
    assert m["split"] == pytest.approx(np.sum(B * C / D))
    assert m["ratio"] == pytest.approx(np.sum(B * C / D) / (B.sum() * C.sum() / D.sum()))
    assert m["ratio"] > 1.1
    # A was built as B C / D per bin, so the one-bin kappa is the mixing factor
    reg = ps.regions(cube, dl.NOMINAL, "PP")
    assert reg["A"][0] / ps.prediction(reg)[0] == pytest.approx(m["ratio"])
    assert dl.route_closure(cube, dl.NOMINAL, "PP", dl.FIVE)[0] == pytest.approx(1.0)


def test_a_bin_with_empty_d_is_merged_into_its_lower_neighbour():
    bins, m = dl.merge_empty(list(dl.FIVE), [(28, 6, 16), (24, 5, 10), (13, 5, 1), (9, 1, 2), (4, 0, 0)])
    assert [lab for lab, _ in bins] == ["150-200", "200-300", "300-400", ">=400"]
    assert m[-1].tolist() == [13, 1, 2]
    bins, m = dl.merge_empty(list(dl.FIVE), [(5, 1, 0), (24, 5, 10), (13, 5, 1), (9, 1, 2), (4, 1, 3)])
    assert [lab for lab, _ in bins][0] == "150-300" and m[0].tolist() == [29, 6, 10]


def test_toys_merge_empty_bins_as_the_data_do():
    # the middle bin's D is empty in the data and in every toy: it is merged, never dropped
    big = [(1e4, 1e4, 1e4), (1e4, 5e3, 0.0), (2e4, 1e4, 1e4)]
    _, m = dl.merge_empty(dl.FIVE[:3], [row for row in big])
    t = dl.merged_toys(big, n_toys=4000, seed=1)
    assert t["kept"] == 1.0
    assert np.median(t["split"]) == pytest.approx(np.sum(m[:, 0] * m[:, 1] / m[:, 2]), rel=0.01)
    # five bins with one D event in two of them: a toy has no finite sum only when every D is drawn as 0
    small = [(20.0, 5.0, 1.0), (10.0, 3.0, 0.0), (8.0, 4.0, 1.0), (4.0, 1.0, 0.0), (2.0, 0.0, 0.0)]
    t = dl.merged_toys(small, n_toys=40000, seed=2)
    assert t["kept"] == pytest.approx(1 - np.exp(-2.0), abs=0.006)
    boot = dl.bootstrap_bins(small, n_toys=40000, seed=2)
    assert boot["kept"] == t["kept"]
    assert boot["groups"] == [[0, 1], [2, 3, 4]]
    assert np.isfinite(boot["sum"][2])          # P(every D = 0) = exp(-2) is below the 16% above the 84th percentile
    # two windows have no merge rule: a toy with D = 0 in a window has no finite sum, as in nominal_lib
    two = [(52.0, 11.0, 26.0), (26.0, 6.0, 3.0)]
    assert dl.merge_rule(dl.FIVE) and not dl.merge_rule(dl.TWO)
    t = dl.merged_toys(two, n_toys=40000, seed=3, merge=False)
    ref = nl.bootstrap_abcd(two, n_toys=40000, seed=3)
    assert t["kept"] == pytest.approx(1 - np.exp(-3.0), abs=0.005)
    assert np.array_equal(t["split"], ref["split"]) and np.array_equal(t["ratio"], ref["ratio"])
    boot = dl.bootstrap_bins(two, n_toys=40000, seed=3, merge=False)
    assert boot["sum"] == nl.interval68(ref["split"])
    assert dl.bootstrap_bins(two, n_toys=40000, seed=3, merge=True)["sum"][2] < boot["sum"][2]
    assert boot["per_bin"][1] == nl.interval68(nl.bootstrap_abcd([two[1]], n_toys=40000, seed=3)["one_bin"])


def test_shape_test_is_zero_for_a_scaled_copy_of_the_reference():
    ref = np.array([(200, 200), (150, 250), (60, 40), (30, 30)], float)
    p = ref[:, 0] / ref.sum(axis=1)
    n = np.array([100, 80, 50, 40], float)
    c = 0.5 * p * n
    r = dl.shape_test(np.c_[c, n - c], ref, n_toys=200)
    assert r["chi2"] == pytest.approx(0.0, abs=1e-12) and r["k"] == pytest.approx(0.5) and r["ndf"] == 3


# ----------------------------------------------------------------------------- the logistic fit

def test_logistic_fit_recovers_known_parameters():
    mass = np.array([dl.centre(*rng) for _, rng in dl.FIVE])
    x = np.log(mass / dl.PIVOT)
    n = np.full(5, 1e6)
    c = n * dl.logistic_a(-0.4, 0.8, mass)
    fit = dl.logistic_fit(c, n, x)
    assert fit["converged"] and fit["alpha"] == pytest.approx(-0.4, abs=1e-8) and fit["slope"] == pytest.approx(0.8, abs=1e-8)
    assert fit["deviance"] == pytest.approx(0.0, abs=1e-6)
    fixed = dl.logistic_fit(c, n, x, slope=0.8)
    assert fixed["alpha"] == pytest.approx(-0.4, abs=1e-8) and fixed["ndf"] == 4
    assert dl.logistic_r(-0.4, 0.8, mass) == pytest.approx(dl.logistic_a(-0.4, 0.8, mass) / (1 - dl.logistic_a(-0.4, 0.8, mass)))


def test_logistic_fit_on_toys_is_unbiased_and_its_error_is_the_spread():
    mass = np.array([dl.centre(*rng) for _, rng in dl.FIVE])
    x = np.log(mass / dl.PIVOT)
    n = np.array([400, 300, 150, 80, 40])
    rng = np.random.default_rng(1)
    c = rng.binomial(n, dl.logistic_a(-0.2, 0.5, mass), size=(4000, 5)).astype(float)
    fit = dl.logistic_fit(c, np.broadcast_to(n, c.shape).astype(float), x)
    assert fit["converged"].all()
    one = dl.logistic_fit(n * dl.logistic_a(-0.2, 0.5, mass), n, x)
    assert np.mean(fit["slope"]) == pytest.approx(0.5, abs=0.02)
    assert np.std(fit["slope"]) == pytest.approx(np.sqrt(one["cov"][1, 1]), rel=0.1)


def test_smooth_route_applies_the_fitted_ratio_to_b():
    rising = [(2, 8, 5, 20), (9, 18, 6, 12), (20, 20, 4, 4), (18, 12, 3, 2), (20, 10, 4, 2)]
    cube = make_cube(abcd_entries("PP", rising) + abcd_entries("DD", rising))
    for plane in ("PP", "DD"):
        r = dl.smooth_route(cube, dl.NOMINAL, plane, n_toys=500)
        B = np.array([b for _, b, _, _ in rising], float)
        assert r["pred"] == pytest.approx(np.sum(B * dl.logistic_r(r["alpha"], r["slope"], np.exp(r["x"]) * dl.PIVOT)))
        lo, _, hi = r["pred_interval"]
        assert lo < r["pred"] < hi
    pp = dl.smooth_route(cube, dl.NOMINAL, "PP", n_toys=500)
    assert pp["kappa"][0] == pytest.approx(sum(a for a, _, _, _ in rising) / pp["pred"])
    shared = dl.smooth_route(cube, dl.NOMINAL, "DD", slope_plane="PP", n_toys=500)
    assert shared["fixed_slope"] == pytest.approx(pp["slope"]) and shared["slope"] == pytest.approx(pp["slope"])


# ----------------------------------------------------------------------------- the flattened cut

def flatten_cube(plane="PD"):
    """Bin 1: a = 1/2 at 0.25 (the anchor).  Bin 2: a = 1/6, 1/3, 1/2, 1/2 at 0.25, 0.3, 0.4, 0.5.
    Bin 3: a = 2/3 at 0.25 and 1/2 at 0.2 and 0.15 (the buffer [edge, 0.25) out of both regions)."""
    e = [(plane, 175.0, 0.12, FAIL_DPHI, 10), (plane, 175.0, 0.6, FAIL_DPHI, 10),
         (plane, 250.0, 0.12, FAIL_DPHI, 10), (plane, 250.0, 0.27, FAIL_DPHI, 10),
         (plane, 250.0, 0.35, FAIL_DPHI, 10), (plane, 250.0, 0.6, FAIL_DPHI, 30),
         (plane, 350.0, 0.12, FAIL_DPHI, 10), (plane, 350.0, 0.22, FAIL_DPHI, 10), (plane, 350.0, 0.6, FAIL_DPHI, 10)]
    for mass in MASSES:
        e += [(plane, mass, 0.6, PASS_DPHI, 20), (plane, mass, 0.12, PASS_DPHI, 5)]
    return make_cube(e)


def test_choose_edges_picks_the_edges_that_flatten_a():
    cube = flatten_cube()
    bins = dl.FIVE[:3]
    cd = dl.odds_table(cube, dl.NOMINAL, "PD", bins)
    choice = dl.choose_edges(cd)
    assert choice["target"] == pytest.approx(0.5)
    assert choice["edges"] == (0.25, 0.4, 0.2)          # ties go to the edge nearest 0.25 in the list
    after = dl.flatness(dl.counts(dl.flat_regions(cube, dl.NOMINAL, "PD", bins, choice["edges"]))[:, 2:])
    before = dl.flatness(dl.counts(dl.flat_regions(cube, dl.NOMINAL, "PD", bins, [0.25] * 3))[:, 2:])
    assert after["a"] == pytest.approx([0.5, 0.5, 0.5]) and after["chi2"] == pytest.approx(0.0)
    assert before["chi2"] > 10


def test_nominal_edges_reproduce_the_one_bin_regions():
    cube = make_cube(abcd_entries("PD", [(2, 8, 5, 20), (9, 18, 6, 12), (20, 20, 4, 4), (18, 12, 3, 2), (20, 10, 4, 2)]))
    flat = dl.summed(dl.flat_regions(cube, dl.NOMINAL, "PD", dl.FIVE, [0.25] * 5))
    one = ps.regions(cube, dl.NOMINAL, "PD")
    assert {x: flat[x][0] for x in "ABCD"} == {x: one[x][0] for x in "ABCD"}


# ----------------------------------------------------------------------------- the hidden box

def blind_entries():
    """DD with events in the hidden box, in the buffer [0.2, 0.25) and in the open sidebands."""
    e = []
    for mass in MASSES:
        e += [("DD", mass, 0.12, PASS_DPHI, 50), ("DD", mass, 0.22, PASS_DPHI, 7), ("DD", mass, 0.6, PASS_DPHI, 20),
              ("DD", mass, 0.12, FAIL_DPHI, 6), ("DD", mass, 0.22, FAIL_DPHI, 3), ("DD", mass, 0.6, FAIL_DPHI, 9)]
    return e


def blind_cube():
    return make_cube(blind_entries())


def test_a_cut_below_the_box_edge_keeps_the_search_plane_sidebands_open():
    cube = blind_cube()
    regs = dl.flat_regions(cube, dl.NOMINAL, "DD", dl.FIVE, [0.25, 0.2, 0.15, 0.3, 0.1])
    for r in regs:
        assert np.isnan(r["A"][0])
        assert all(np.isfinite(r[x][0]) for x in "BCD")
    assert regs[1]["B"][0] == 20 and regs[1]["D"][0] == 9 and regs[1]["C"][0] == 6     # the buffer is in neither
    no_buffer = ps.regions(cube, ps.Config(wp0=0.2, fail0=0.2, dphi_fail=2.4, mass=(200.0, 300.0)), "DD")
    assert np.isnan(no_buffer["B"][0])                 # what the buffer avoids: B would reach into the box


def test_the_dd_routes_never_read_a_of_the_search_plane():
    cube = blind_cube()
    assert np.isnan(dl.pass_table(cube, dl.NOMINAL, ("DD",))["A"]).all()
    s = dl.smooth_route(cube, dl.NOMINAL, "DD", n_toys=200)
    assert np.isfinite(s["pred"]) and np.isnan(s["A"]) and np.isnan(s["kappa"][0])
    f = dl.flat_route(cube, dl.NOMINAL, "DD", dl.FIVE, [0.25, 0.2, 0.15, 0.3, 0.1], n_toys=200)
    assert np.isfinite(f["pred"][0]) and np.isnan(f["A"]) and np.isnan(f["kappa"][0])
    assert np.isnan(dl.route_closure(cube, dl.NOMINAL, "DD", dl.FIVE)[0])
    t = dl.stratified_table(cube, dl.NOMINAL, dl.TWO, n_toys=200)
    assert np.isfinite(t["estimate"]).all()
    # the prediction is the same whatever the box holds: fill the box differently, blind, compare
    more = make_cube([("DD", m, 0.12, PASS_DPHI, 500) for m in MASSES] +
                     [e for e in blind_entries() if not (e[2] < 0.25 and e[3] == PASS_DPHI)])
    assert dl.smooth_route(more, dl.NOMINAL, "DD", n_toys=200)["pred"] == pytest.approx(s["pred"])


# ----------------------------------------------------------------------------- the rule

def test_rule_takes_the_largest_reading_measured_well_enough():
    readings = [("x", (1.10, 0.05, 0.05)), ("y", (0.80, 0.30, 0.30)), ("z", (0.95, 0.12, 0.12))]
    assert dl.rule(readings) == (pytest.approx(0.12), "z")       # y is imprecise; z's error beats x's |k - 1|
    assert dl.rule([("y", (0.80, 0.30, 0.30))]) == (pytest.approx(np.nan, nan_ok=True), None)


def test_stratified_table_sums_the_bins_and_reads_kappa_per_bin():
    per = [(4, 8, 10, 20), (9, 18, 6, 12), (10, 20, 2, 4), (6, 12, 1, 2), (5, 10, 1, 2)]
    cube = make_cube(abcd_entries("PP", per) + abcd_entries("PD", per) + abcd_entries("DD", per))
    t = dl.stratified_table(cube, dl.NOMINAL, dl.FIVE, n_toys=500)
    body = t[t["bin"] != "sum"]
    assert list(body["bin"]) == [lab for lab, _ in dl.FIVE]
    assert t.iloc[-1]["estimate"] == pytest.approx(sum(b * c / d for _, b, c, d in per))
    assert body["kappa PP"].to_numpy() == pytest.approx(1.0)
