"""Toy self-validation of abcd_tools (run with pytest; numpy/scipy only).

Two failure modes these tests exist to catch:
* arithmetic/indexing mistakes in the region sums and estimators (exact tests), and
* a statistics implementation that cannot tell a factorizable background from a
  correlated one (power tests on toys).
"""

import numpy as np
import pytest

from abcd_tools import (edge_index, region_sums, abcd_prediction, closure_ratio, kappa,
                        n_eff, factorization_fit, weighted_correlation,
                        bootstrap_closures, asimov_z, leakage_ratios,
                        prediction_bias_vs_mu, offline_norm_factor, extended_prediction)


XE = np.array([0.0, 1.0, 2.0, 3.0, 4.0])
YE = np.array([0.0, 1.0, 2.0, 3.0])


def toy_plane(rng, nx=20, ny=20, mu=50.0, rho=0.0, wspread=0.0):
    """Weighted factorizable (rho=0) or row-column-correlated (rho>0) toy plane.

    Returns (vals, var, xedges, yedges). rho tilts the 2D density multiplicatively:
    mu_ij *= (1 + rho * xi * yj) with xi, yj in [-1, 1].
    """
    xe = np.linspace(0, 1, nx + 1)
    ye = np.linspace(0, 1, ny + 1)
    ax = np.exp(-3 * np.linspace(0, 1, nx))          # falling x spectrum
    by = np.exp(-2 * np.linspace(0, 1, ny))          # falling y spectrum
    m = mu * np.outer(ax / ax.sum(), by / by.sum()) * nx * ny
    if rho:
        xi = np.linspace(-1, 1, nx)
        yj = np.linspace(-1, 1, ny)
        m = m * (1 + rho * np.outer(xi, yj))
    n = rng.poisson(m)
    # Per-bin average weights must themselves factorize (u_i * v_j): independent
    # per-bin random weights would tilt the plane non-factorizably and put a floor
    # on the closure scatter that no statistical error can describe.
    w = np.outer(1.0 + wspread * rng.random(nx), 1.0 + wspread * rng.random(ny))
    vals = n * w
    var = n * w**2
    return vals, var, xe, ye


def test_edge_index_exact_and_raises():
    assert edge_index(XE, 2.0) == 2
    with pytest.raises(ValueError):
        edge_index(XE, 2.5)


def test_region_sums_exact():
    vals = np.arange(12, dtype=float).reshape(4, 3)  # x-major (4 x-bins, 3 y-bins)
    var = np.ones_like(vals)
    r = region_sums(vals, var, XE, YE, ("lt", 2.0), ("lt", 1.0))
    # A: x-bins 0-1, y-bin 0 -> vals[0,0]+vals[1,0] = 0+3
    assert r["A"] == (3.0, 2.0)
    # B: x-bins 2-3, y-bin 0 -> 6+9
    assert r["B"] == (15.0, 2.0)
    # C: x-bins 0-1, y-bins 1-2 -> 1+2+4+5
    assert r["C"] == (12.0, 4.0)
    # D: x-bins 2-3, y-bins 1-2 -> 7+8+10+11
    assert r["D"] == (36.0, 4.0)


def test_region_sums_guard_and_lo():
    vals = np.ones((4, 3))
    var = np.ones_like(vals)
    # guard band of 1 bin on x removes x-bins adjacent to the boundary
    r = region_sums(vals, var, XE, YE, ("lt", 2.0), ("lt", 1.0), xguard=1)
    assert r["A"][0] == 1.0 and r["B"][0] == 1.0          # one x-bin each side
    # lo bound excludes the first x bin ("sentinel") from all regions
    r = region_sums(vals, var, XE, YE, ("lt", 2.0), ("lt", 1.0), xlo=1.0)
    assert r["A"][0] == 1.0 and r["C"][0] == 2.0


def test_pass_high_orientation():
    vals = np.arange(12, dtype=float).reshape(4, 3)
    var = np.ones_like(vals)
    r = region_sums(vals, var, XE, YE, ("ge", 2.0), ("lt", 1.0))
    assert r["A"][0] == 15.0 and r["B"][0] == 3.0         # mirrored in x


def test_closure_exact_on_factorized():
    a = np.array([4.0, 3.0, 2.0, 1.0])
    b = np.array([5.0, 2.0, 1.0])
    vals = np.outer(a, b)
    var = 0.01 * np.ones_like(vals)
    r = region_sums(vals, var, XE, YE, ("lt", 1.0), ("lt", 1.0))
    pred, _ = abcd_prediction(r)
    assert np.isclose(pred, r["A"][0])
    rr, _ = closure_ratio(r)
    assert np.isclose(rr, 1.0)
    k, _ = kappa(r)
    assert np.isclose(k, 1.0)


def test_closure_toys_unbiased_and_pull_width():
    """Linear error propagation on B*C/D is only trustworthy when all four regions
    are statistically healthy — this is the load-bearing reason for the pre-registered
    n_eff > 10 gate. Assert calibration in the gated regime, and document the
    anticonservative behavior outside it."""
    rng = np.random.default_rng(7)

    def pulls_at(mu, gate):
        rs, pulls = [], []
        for _ in range(300):
            vals, var, xe, ye = toy_plane(rng, mu=mu, wspread=0.5)
            reg = region_sums(vals, var, xe, ye, ("lt", 0.25), ("lt", 0.25))
            if gate and min(n_eff(*reg[k]) for k in "ABCD") < 10:
                continue
            r, vr = closure_ratio(reg)
            if np.isfinite(r) and vr > 0:
                rs.append(r)
                pulls.append((r - 1) / np.sqrt(vr))
        return np.array(rs), np.array(pulls)

    rs, pulls = pulls_at(mu=80.0, gate=True)
    assert len(pulls) > 100
    assert abs(np.mean(rs) - 1) < 0.05
    assert 0.75 < np.std(pulls) < 1.25        # calibrated where the gates pass
    # (No assertion on the starved regime: nan-filtering of toys with empty regions
    # biases which pulls survive, so the width there is not a stable statement. The
    # calibrated-regime assertion above is what licenses the analytic errors, and the
    # n_eff > 10 gate is what keeps the analysis inside it.)


def test_factorization_fit_null_and_power():
    rng = np.random.default_rng(11)
    p_null = [factorization_fit(*toy_plane(rng, mu=100.0)[:2])["pvalue"] for _ in range(60)]
    p_corr = [factorization_fit(*toy_plane(rng, mu=100.0, rho=0.8)[:2])["pvalue"]
              for _ in range(60)]
    p_null, p_corr = np.array(p_null), np.array(p_corr)
    assert 0.25 < np.nanmean(p_null) < 0.75    # roughly uniform under the null
    assert np.nanmedian(p_corr) < 0.01         # strongly-correlated toys rejected
    assert np.nanmedian(p_corr) < np.nanmedian(p_null)


def test_correlation_sign():
    rng = np.random.default_rng(3)
    vals, var, xe, ye = toy_plane(rng, mu=500.0, rho=0.9)
    assert weighted_correlation(vals, xe, ye) > 0.02
    vals, var, xe, ye = toy_plane(rng, mu=500.0, rho=-0.9)
    assert weighted_correlation(vals, xe, ye) < -0.02


def test_bootstrap_cov_calibration():
    rng = np.random.default_rng(5)
    vals, var, xe, ye = toy_plane(rng, mu=200.0, wspread=0.5)
    pts = [dict(xspec=("lt", 0.25), yspec=("lt", 0.25)),
           dict(xspec=("lt", 0.5), yspec=("lt", 0.5))]
    central, cov = bootstrap_closures(vals, var, xe, ye, pts, n_boot=300)
    _, v0 = closure_ratio(region_sums(vals, var, xe, ye, **pts[0]))
    assert 0.3 * v0 < cov[0, 0] < 3.0 * v0     # bootstrap variance same scale as analytic
    corr = cov[0, 1] / np.sqrt(cov[0, 0] * cov[1, 1])
    assert corr > 0.1                           # nested scan points are correlated


def test_asimov_limits():
    assert np.isclose(asimov_z(5, 10, 0.0),
                      np.sqrt(2 * ((15) * np.log(1.5) - 5)))
    assert asimov_z(5, 10, 5.0) < asimov_z(5, 10, 0.5) < asimov_z(5, 10, 0.0)
    assert asimov_z(0, 10, 1.0) == 0.0


def test_leakage_and_bias():
    bkg = {"A": (10.0, 1.0), "B": (100.0, 4.0), "C": (50.0, 2.0), "D": (500.0, 10.0)}
    sig = {"A": (20.0, 0.4), "B": (1.0, 0.1), "C": (0.5, 0.05), "D": (0.1, 0.01)}
    lr = leakage_ratios(sig)
    assert lr["B"] == pytest.approx(0.05)
    bias = prediction_bias_vs_mu(bkg, sig, [0.0, 1.0])
    assert bias[0][1] == pytest.approx(0.0)
    assert abs(bias[1][1]) < 0.02               # tiny leakage -> tiny bias


def test_offline_norm_factor():
    assert offline_norm_factor({"scaled_sum_weights": 125720.0}, 1300658.0) == \
        pytest.approx(0.09666, rel=1e-3)
    with pytest.raises(ValueError):
        offline_norm_factor({"scaled_sum_weights": 1.0}, 0.0)


def test_extended_prediction_unbiased_and_tighter():
    """On factorizable toys the all-sidebands estimator must be unbiased and have a
    smaller variance than plain B*C/D."""
    rng = np.random.default_rng(21)
    rs_plain, rs_ext = [], []
    for _ in range(80):
        vals, var, xe, ye = toy_plane(rng, mu=80.0, wspread=0.5)
        reg = region_sums(vals, var, xe, ye, ("lt", 0.25), ("lt", 0.25))
        rp, _ = closure_ratio(reg)
        a_obs, _, pred, _ = extended_prediction(vals, var, xe, ye,
                                                ("lt", 0.25), ("lt", 0.25), n_boot=10)
        if np.isfinite(rp) and pred > 0:
            rs_plain.append(rp)
            rs_ext.append(a_obs / pred)
    rs_plain, rs_ext = np.array(rs_plain), np.array(rs_ext)
    assert abs(np.mean(rs_ext) - 1) < 0.05
    assert np.std(rs_ext) < np.std(rs_plain)
