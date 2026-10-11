"""Tests of the per-bin estimate on synthetic 8D cubes: the A/B(M) points and fits, the binned B
route against the one-bin route, the decision rule, the toy expected limit, the choice among the
binnings, and the hidden box."""
import numpy as np
import pandas as pd
import pytest

import abcd6d_io as io
import binning_lib as bl
import estimate_lib as el
import plane_search_lib as ps
import transfer_lib as tl
from test_abcd6d_io import make_hist8d
from test_plane_search_lib import CORNERS

NOMINAL = ps.Config(fail0=0.25, dphi_fail=2.4)
CENTRES = [175.0, 250.0, 350.0, 500.0, 700.0, 900.0, 1750.0]     # one mass per bin of bl.FIT_BINS, at its centre
TREND_A = [113, 110, 106, 100, 92, 84, 50]                        # 100 x (1.2 - 0.0004 M) at the centres
PP, PD, DD = (3, 0), (3, 1), (1, 1)                               # (pixel hits, lost hits) placing an event in each plane


def fill(h, n, region, plane, mass, ndsa0=0):
    """``n`` unit-weight events in one region of a plane at one mass."""
    if n:
        iso0, dphi = CORNERS[region]
        values = dict(iso0=iso0, iso1=0.1, disp0=plane[0], disp1=plane[1], abs_dphi=dphi, ljlj_mass=mass,
                      lj0_mass=1.0, ndsa0=ndsa0)
        h.fill(channel="ch", **{v: np.full(n, values[v], float) for v in h.axes.name[1:]})


def trend_cube(dd_b=(2, 2, 4, 4, 6, 6, 6), dd_b1=(1, 1, 2, 2, 3, 3, 3), poor=False):
    """PD with A/B = 1.2 - 0.0004 M at the bin centres without a DSA muon (B = 200 per bin) and half
    of it with one (B = 400), C = D = 50 per bin; the search plane's B per bin (``dd_b`` without a
    DSA muon, ``dd_b1`` with one), C = D = 3, and five A events per bin, which blinding hides.
    ``poor``: PD's DSA class holds one A and two B events in the last bin."""
    h = make_hist8d("2mu2e")
    for i, (mass, a) in enumerate(zip(CENTRES, TREND_A)):
        last = poor and i == len(CENTRES) - 1
        fill(h, 2 * a, "A", PD, mass, ndsa0=0)
        fill(h, 200, "B", PD, mass, ndsa0=0)
        fill(h, 1 if last else 2 * a, "A", PD, mass, ndsa0=1)
        fill(h, 2 if last else 400, "B", PD, mass, ndsa0=1)
        for region in "CD":
            fill(h, 50, region, PD, mass)
            fill(h, 3, region, DD, mass)
        fill(h, dd_b[i], "B", DD, mass, ndsa0=0)
        fill(h, dd_b1[i], "B", DD, mass, ndsa0=1)
        fill(h, 5, "A", DD, mass)
    return io.blind(io.cube_from_hist(h, "2mu2e"))


FLAT_B = (3, 5, 2, 4, 1, 0, 2)
FLAT_B1 = (2, 1, 3, 0, 2, 1, 1)


def flat_cube():
    """PD with A/B 2.0 without a DSA muon and 0.5 with one in every bin (A/C 20 and undefined), PP
    with half of PD's A/B; the search plane's B varies from bin to bin (``FLAT_B``, ``FLAT_B1``)."""
    h = make_hist8d("2mu2e")
    for i, mass in enumerate(CENTRES):
        for plane, a0, a1 in ((PD, 200, 100), (PP, 100, 50)):
            fill(h, a0, "A", plane, mass, ndsa0=0)
            fill(h, 100, "B", plane, mass, ndsa0=0)
            fill(h, a1, "A", plane, mass, ndsa0=1)
            fill(h, 200, "B", plane, mass, ndsa0=1)
        fill(h, 10, "C", PD, mass)
        fill(h, 30, "D", PD, mass)
        fill(h, FLAT_B[i], "B", DD, mass, ndsa0=0)
        fill(h, FLAT_B1[i], "B", DD, mass, ndsa0=1)
        fill(h, 2, "C", DD, mass)
        fill(h, 2, "D", DD, mass)
    return io.blind(io.cube_from_hist(h, "2mu2e"))


def total(routes, column="measured"):
    return routes.set_index("bin").loc[bl.TOTAL, column]


def test_ab_points_and_the_linear_fit_recover_a_known_trend():
    cube = trend_cube()
    points = bl.ab_points(cube, NOMINAL, bl.FIT_BINS, "PD", "dsa0")
    assert np.allclose(points["x"], CENTRES) and np.allclose(points["A/B"], np.array(TREND_A) / 100)
    assert points["lower"].isna().all()
    fit = bl.fit_ab(points, "linear")
    (a, b), (ea, eb) = fit["params"], np.sqrt(np.diag(fit["cov"]))
    assert fit["n"] == 7 and fit["ndof"] == 5 and fit["deviance"] < 1e-6
    assert abs(a - 1.2) < 1e-5 and abs(b + 4e-4) < 1e-8 and ea > 0 and eb > 0
    value, err = fit["f"](500.0)
    assert np.isclose(value, 1.0) and 0 < err < 0.05 and isinstance(value, float)
    values, errs = fit["f"]([175.0, 500.0])
    assert values.shape == errs.shape == (2,) and np.isclose(values[1], value) and np.isclose(errs[1], err)
    merged = bl.ab_points(cube, NOMINAL, bl.BINNINGS["three bins"], "PD", "dsa1+").set_index("bin")
    assert np.isclose(merged.loc["300-600", "x"], 425.0) and np.isclose(merged.loc["300-600", "A/B"], 0.515)
    assert np.isclose(merged.loc[">=600", "x"], (700 + 900 + 1750) / 3)
    table = bl.ab_fit_table(cube, NOMINAL, bl.FIT_BINS, planes=("PD",), classes=("dsa0",), forms=("linear",))
    assert table["bin"].iloc[0] == "fit" and table["deviance / ndof"].iloc[0] == "0.0 / 5"
    assert table["residual"].iloc[1:].isin(["+0.0", "-0.0"]).all()


def counts_points(y, b=10**6):
    """Points with A/B = ``y`` per bin of CENTRES from B = ``b`` events and A = y B (rounded)."""
    x = np.array(CENTRES)
    a = np.round(np.asarray(y, float) * b)
    r = [tl.ratio((ai, ai), (b, b)) for ai in a]
    return pd.DataFrame({"bin": [str(m) for m in x], "x": x, "A": a, "B": float(b),
                         "A/B": [v[0] for v in r], "minus": [v[1] for v in r], "plus": [v[2] for v in r],
                         "sigma": [0.5 * (v[1] + v[2]) for v in r], "lower": np.nan})


def test_exponential_fit_recovers_its_scale_and_uses_the_bins_without_a_count():
    points = counts_points(1.2 * np.exp(-np.array(CENTRES) / 800.0))
    fit = bl.fit_ab(points, "exponential")
    assert np.allclose(fit["params"], [np.log(1.2), -1 / 800.0], rtol=1e-4) and fit["deviance"] < 1e-3
    assert np.isclose(fit["f"](400.0)[0], 1.2 * np.exp(-0.5), rtol=1e-4)
    assert bl.fit_text(fit).startswith("a = 1.200") and "M0 = 800 " in bl.fit_text(fit)
    assert fit["n"] == 7 and fit["ndof"] == 5 and set(fit["residuals"]) == set(points["bin"])
    # a bin without an A event is an upper limit on A/B and enters both forms; one without a B event a lower limit
    few = counts_points(1.2 * np.exp(-np.array(CENTRES) / 800.0), b=20)
    few.loc[6, ["A", "A/B", "minus", "plus", "sigma"]] = [0.0, 0.0, 0.0, tl.ratio((0.0, 0.0), (20.0, 20.0))[2], np.nan]
    few.loc[5, ["B", "A/B", "minus", "plus", "sigma"]] = [0.0, np.nan, np.nan, np.nan, np.nan]
    few.loc[5, "lower"] = few.loc[5, "A"] / bl.ZERO_UPPER
    for form in bl.FORMS:
        fit = bl.fit_ab(few, form)
        assert fit["n"] == 7 and fit["ndof"] == 5 and np.isfinite(fit["deviance"])
        assert fit["residuals"][few.loc[5, "bin"]] > 0 > fit["residuals"][few.loc[6, "bin"]]
    # the lower limit pulls the fit up at its mass against the same points without that bin
    with_it, without = bl.fit_ab(few, "exponential"), bl.fit_ab(few.drop(index=5), "exponential")
    assert with_it["f"](900.0)[0] > without["f"](900.0)[0] and without["n"] == 6
    empty = few.iloc[:1]
    assert np.isnan(bl.fit_ab(empty)["params"]).all() and np.isnan(bl.fit_ab(empty)["f"](500.0)[0])


def test_binned_routes_sum_to_the_one_bin_route_when_the_factor_is_flat():
    cube = flat_cube()
    routes = {name: bl.binned_routes(cube, NOMINAL, bins) for name, bins in bl.BINNINGS.items()}
    one = total(routes["one bin"])
    assert np.isclose(one, sum(FLAT_B) * 2.0 + sum(FLAT_B1) * 0.5)
    assert el.routes_table(cube, [("M >= 150", NOMINAL)])["B x A/B per class"].item().startswith(f"{one:.1f}")
    for name, df in routes.items():
        assert np.isclose(total(df), one) and np.isclose(total(df, "B"), sum(FLAT_B) + sum(FLAT_B1)), name
        assert (df.loc[df["class"] != bl.SUM, "choice"] == "measured").all(), name
    sums = bl.sums_table(routes).set_index("binning")
    assert (sums["selected / one bin (measured)"] == "1.000").all()
    ratio = bl.plane_ratio_table({(p, c): bl.ab_points(cube, NOMINAL, bl.FIT_BINS, p, c)
                                  for p in ("PP", "PD") for c in ("dsa0", "dsa1+")}, classes=("dsa0", "dsa1+"))
    assert ratio["weighted mean"].str.startswith("0.500").all() and (ratio["chi2 / ndof against it"] == "0.0 / 6").all()


def test_open_test_compares_the_route_between_open_planes_with_the_observed_a():
    cube = flat_cube()
    fits = {("PP", c): bl.fit_ab(bl.ab_points(cube, NOMINAL, bl.FIT_BINS, "PP", c), "linear") for c in bl.CLASSES}
    table = bl.open_test_table(cube, NOMINAL, bl.BINNINGS["cube bins"], fits=fits, ref_plane="PP", search="PD")
    assert list(table["bin"]) == [b for b, _ in bl.BINNINGS["cube bins"]] + [bl.TOTAL]
    assert np.allclose(table["ratio"], 2.0)                         # PP's A/B is half of PD's: the route predicts half of A
    assert (table["choice"].iloc[:-1] == "measured").all() and np.isfinite(table["syst"]).all() and (table["syst"] > 1.0).all()
    assert table["observed A PD"].iloc[-1] == str(7 * (200 + 100))      # A of PD per bin: 200 without a DSA muon, 100 with one
    syst = bl.open_test_syst({"cube bins": table})["cube bins"]
    assert syst.shape == (5,) and np.allclose(syst, table["syst"].to_numpy()[:-1])
    hidden = bl.open_test_table(cube, NOMINAL, bl.BINNINGS["two windows"], ref_plane="PD", search="DD")
    assert hidden["ratio"].isna().all() and (hidden["observed A DD"] == "blind").all()
    rows = bl.signal_regions_table("", [], "2mu2e", NOMINAL, bl.BINNINGS["two windows"], data=cube)
    assert list(rows["region"]) == ["A PD:dsa0", "B PD:dsa0", "A PD:dsa1+", "B PD:dsa1+", "B DD"]
    assert rows.set_index("region").loc["B DD", "sum"] == str(sum(FLAT_B) + sum(FLAT_B1))
    assert rows.set_index("region").loc["A PD:dsa0", "150-300"] == "400"


def test_binned_routes_follow_the_factor_of_each_bin_when_it_falls():
    dd_b, dd_b1 = (2, 2, 4, 4, 6, 6, 6), (1, 1, 2, 2, 3, 3, 3)
    cube = trend_cube(dd_b, dd_b1)
    ab0 = np.array(TREND_A) / 100

    def expected(bins):
        out = 0.0
        for _, (lo, hi) in bins:
            idx = [i for i, m in enumerate(CENTRES) if m >= lo and (hi is None or m < hi)]
            out += sum(dd_b[i] for i in idx) * ab0[idx].mean() + sum(dd_b1[i] for i in idx) * ab0[idx].mean() / 2
        return out

    for name, bins in bl.BINNINGS.items():
        assert np.isclose(total(bl.binned_routes(cube, NOMINAL, bins)), expected(bins)), name
    one, fine = expected(bl.BINNINGS["one bin"]), expected(bl.BINNINGS["cube bins"])
    assert one - fine > 2.0                           # B of DD sits higher in mass than PD's: the one-bin factor is too high
    assert el.routes_table(cube, [("M >= 150", NOMINAL)])["B x A/B per class"].item().startswith(f"{one:.1f}")


def test_decision_rule_takes_the_fitted_factor_where_pd_is_imprecise():
    cube = trend_cube(dd_b=(4, 10, 10, 10, 10, 10, 1), dd_b1=(0, 6, 6, 6, 6, 6, 2), poor=True)
    fits = {("PD", c): bl.fit_ab(bl.ab_points(cube, NOMINAL, bl.FIT_BINS, "PD", c), "linear") for c in bl.CLASSES}
    routes = bl.binned_routes(cube, NOMINAL, bl.FIT_BINS, fits=fits)
    rows = routes[routes["class"] != bl.SUM].set_index(["bin", "class"])
    poor = rows.loc[(">=1000", "dsa1+")]
    assert poor["choice"] == "fitted" and poor["A/B rel"] > bl.MAX_REL_ERR
    assert np.isclose(poor["x"], 1750.0) and np.isclose(poor["A/B fit"], fits[("PD", "dsa1+")]["f"](1750.0)[0])
    assert np.isclose(poor["selected"], poor["fitted"]) and np.isclose(poor["selected"], 2 * poor["A/B fit"])
    assert np.isclose(poor["shape"], abs(1.0 - poor["fitted"])) and poor["shape carried"] == poor["shape"] > 0
    others = rows.drop((">=1000", "dsa1+"))
    assert (others["choice"] == "measured").all() and np.allclose(others["selected"], others["measured"])
    assert (others["shape carried"] == 0).all()
    last = routes[(routes["class"] == bl.SUM) & (routes["bin"] == ">=1000")].iloc[0]
    assert last["choice"] == "dsa0 measured, dsa1+ fitted" and np.isclose(last["shape carried"], poor["shape"])
    table = bl.decision_table({"fit bins": routes}).iloc[0]
    assert table["B >= 5 or estimate < 3"].split(" | ") == ["no"] + ["yes"] * 6      # 4 B events, 4.5 events
    assert not table["every bin passes"] and table["fitted A/B taken in"] == ">=1000 dsa1+"
    assert table["bins cover M >= 150"] and table["largest shape systematic carried"] != "0%"
    view = bl.routes_view(routes)
    assert view["choice"].iloc[-1] == "" and not view.isna().any().any()


def test_expected_limit_reproduces_reference_values():
    # the reference limits come from a separate single-bin implementation of the same asymptotic
    # formula (q_A on the Asimov set solved by bisection with one log-normal nuisance), kept outside
    # the repository; the two agree to better than 1%
    for b, rel, events in ((19.0, 0.33, 13.1), (61.0, 0.28, 29.3), (1.7, 1.5, 4.4), (19.0, 0.0, 9.9), (61.0, 0.0, 16.6)):
        mu, limit = bl.expected_limit([2.0], [b], [rel], 0.0)
        assert limit == pytest.approx(events, rel=0.02) and np.isclose(limit, 2.0 * mu)
        assert bl.expected_limit([1.0], [b], [0.0], rel)[1] == pytest.approx(events, rel=0.02)   # one nuisance either way
    s, b, stat = [0.0, 1.0], [19.0, 61.0], [0.22, 0.2]
    single = bl.expected_limit([1.0], [61.0], [0.2], 0.0)[1]
    assert np.isclose(bl.expected_limit(s, b, stat, 0.0)[1], single, rtol=1e-4)    # a signal-free bin adds nothing
    constrained = bl.expected_limit(s, b, stat, 0.25)[1]
    assert single <= constrained < bl.expected_limit([1.0], [61.0], [0.2], 0.25)[1]    # bin 1 constrains the common nuisance
    assert np.isclose(bl.expected_limit([1.0], [0.0], [0.0], 0.0)[1], 1.96**2 / 2, rtol=1e-4)
    with pytest.raises(ValueError):
        bl.expected_limit([1.0], [np.nan], [0.1], 0.3)


def test_hidden_a_of_the_search_plane_stays_hidden_in_every_mass_bin():
    cube = trend_cube()
    for _, (lo, hi) in bl.FIT_BINS + [b for bins in bl.BINNINGS.values() for b in bins]:
        reg = ps.regions(cube, bl.bin_cfg(NOMINAL, lo, hi), "DD")
        assert np.isnan(reg["A"][0]) and np.isnan(ps.kappa(reg)[0]) and np.isnan(tl.ratio(reg["A"], reg["B"])[0])
    assert ps.regions(cube, bl.bin_cfg(NOMINAL, 100.0, 150.0), "DD")["A"][0] == 0.0     # the box starts at 150 GeV
    table = bl.counts_table(cube, NOMINAL, bl.FIT_BINS, planes=("DD", "PD"))
    dd, pd_ = table[table["plane"] == "DD"], table[table["plane"] == "PD"]
    assert (dd[["A", "A/B", "kappa"]] == "blind").all().all() and (dd["B"] != "blind").all()
    assert not (pd_[["A", "A/B", "kappa"]] == "blind").any().any()
    points = bl.ab_points(cube, NOMINAL, bl.FIT_BINS, "DD", "dsa0")
    assert points["A/B"].isna().all() and np.isfinite(points["x"]).all()
    assert bl.fit_ab(points)["n"] == 0 and np.isnan(bl.fit_ab(points)["params"]).all()
    fits = bl.ab_fit_table(cube, NOMINAL, bl.FIT_BINS, planes=("DD",), classes=("dsa0", None))
    assert (fits.loc[fits["bin"] != "fit", "A/B measured"] == "blind").all()
    assert np.isnan(bl.mean_mass(cube, NOMINAL, "DD", 150.0, None, region="A"))
    routes = bl.binned_routes(cube, NOMINAL, bl.BINNINGS["cube bins"])
    assert np.isfinite(routes["measured"]).all()          # read from B of DD and A, B of PD only
    assert "blind" not in bl.routes_view(routes).to_string()


def signal_cube(masses):
    """A signal cube (never hidden) with A events of the DD plane at the given masses and a B event."""
    h = make_hist8d("2mu2e")
    for mass, n in masses.items():
        fill(h, n, "A", DD, mass)
    fill(h, 3, "B", DD, 500.0)
    fill(h, 1, "A", DD, 100.0)                       # below the search window
    return io.cube_from_hist(h, "2mu2e")


def test_signal_yields_and_the_limit_table(monkeypatch):
    cubes = {"light": signal_cube({250.0: 8, 500.0: 3, 1750.0: 1}), "heavy": signal_cube({250.0: 0, 500.0: 4, 1750.0: 8})}
    monkeypatch.setattr(tl, "sample_cubes", lambda base, kind, names, topologies, prefix: {names[0]: {"2mu2e": cubes[names[0]]}})
    profiles = bl.signal_profiles("/nowhere", list(cubes), "2mu2e", NOMINAL)
    yields = {}
    for name, bins in bl.BINNINGS.items():
        yields[name] = bl.signal_yields("/nowhere", list(cubes), "2mu2e", NOMINAL, bins, profiles=profiles)
        loaded = bl.signal_yields("/nowhere", list(cubes), "2mu2e", NOMINAL, bins)
        assert np.allclose(yields[name]["yield"], loaded["yield"])
        assert np.allclose(yields[name].groupby("point", sort=False)["yield"].sum(), [12.0, 12.0])
    three = yields["three bins"][yields["three bins"]["point"] == "heavy"].set_index("bin")["yield"]
    assert list(three) == [0.0, 4.0, 8.0]
    assert list(bl.yields_view(yields["three bins"]).columns) == ["point", "150-300", "300-600", ">=600", "sum"]
    routes = {name: bl.binned_routes(flat_cube(), NOMINAL, bins) for name, bins in bl.BINNINGS.items()}
    limits = bl.limit_table(routes, yields)
    one = limits[(limits["binning"] == "one bin") & (limits["factor"] == "measured")].set_index("point")
    assert np.allclose(one["relative to one bin"], 1.0)
    assert np.isclose(one.loc["light", "limit [events]"], one.loc["heavy", "limit [events]"])  # one bin: shape-blind
    two = routes["two windows"]
    bins = two[(two["class"] == bl.SUM) & (two["bin"] != bl.TOTAL)]
    rel = 0.5 * (bins["measured minus"] + bins["measured plus"]) / bins["measured"]
    s = yields["two windows"].query("point == 'heavy'")["yield"]
    got = limits.query("binning == 'two windows' and factor == 'measured' and point == 'heavy'")["limit [events]"].item()
    assert np.isclose(got, bl.expected_limit(s, bins["measured"], rel, 0.33)[1])
    assert limits.loc[limits["factor"] == "fitted", "limit [events]"].isna().all()     # no fit given
    view = bl.limit_view(limits, "measured").set_index("point")
    assert (view["one bin"] == "1.000").all() and list(view.columns)[0] == "one bin, measured A/B [events]"
    separate = bl.limit_table(routes, yields, factors=("measured",), common=False).set_index(["binning", "point"])
    shared = limits[limits["factor"] == "measured"].set_index(["binning", "point"])
    assert np.allclose(separate.loc["one bin", "limit [events]"], shared.loc["one bin", "limit [events]"], rtol=1e-4)
    assert separate.loc[("two windows", "heavy"), "limit [events]"] > shared.loc[("two windows", "heavy"), "limit [events]"]


def test_rule_table_counts_each_mass_once_and_skips_ineligible_binnings():
    names = ["2Mu2E_500GeV_0p25GeV_0p4mm", "2Mu2E_800GeV_0p25GeV_0p25mm", "2Mu2E_800GeV_1p2GeV_1p2mm",
             "2Mu2E_800GeV_0p25GeV_1p2mm", "2Mu2E_200GeV_0p25GeV_1p0mm"]
    relative = {"one bin": (1.0, 1.0, 1.0, 1.0, 1.0), "coarse": (0.5, 0.9, 0.9, 0.9, 0.1),
                "fine": (0.8, 0.7, 0.7, 0.7, 2.0), "finest": (0.3, 0.3, 0.3, 0.3, 0.3)}
    limits = pd.DataFrame([{"point": n, "decision point": n in bl.DECISION_POINTS["2mu2e"], "binning": b, "factor": f,
                            "relative to one bin": v if f == "selected" else 0.01}
                           for b, values in relative.items() for n, v in zip(names, values) for f in bl.FACTORS])
    decision = pd.DataFrame({"binning": list(relative), "bins cover M >= 150": [True] * 4,
                             "every bin passes": [True, True, True, False]})
    table = bl.rule_table(limits, decision).set_index("binning")
    assert list(table.columns) == ["eligible", "500 GeV", "800 GeV", "mean over the masses", "chosen"]
    assert np.isclose(table.loc["coarse", "800 GeV"], 0.9) and np.isclose(table.loc["coarse", "mean over the masses"], 0.7)
    assert np.isclose(table.loc["fine", "mean over the masses"], 0.75)       # 0.725 averaged per point, below coarse's 0.8
    assert list(table.index[table["chosen"]]) == ["coarse"] and not table.loc["finest", "eligible"]
    assert bl.mediator_mass("4Mu_1000GeV_1p2GeV_0p96mm") == 1000.0
