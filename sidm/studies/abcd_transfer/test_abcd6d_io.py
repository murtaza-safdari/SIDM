"""Tests of the 6D cube loader, blinding and projections on synthetic histograms."""
import hist
import numpy as np
import pytest

import abcd6d_io as io


def make_hist(topology):
    axes = [hist.axis.StrCategory(["ch"], name="channel")]
    for variable in io.VARIABLES:
        axes.append(hist.axis.Variable(io.axis_edges(topology, variable), name=variable,
                                       underflow=True, overflow=True))
    return hist.Hist(*axes, storage=hist.storage.Weight())


def fill(h, weight=1.0, **values):
    h.fill(channel="ch", weight=weight, **{v: np.atleast_1d(values[v]) for v in io.VARIABLES})


def event(h, iso0, iso1, disp0, disp1, dphi=3.0, mass=400.0, weight=1.0):
    fill(h, weight=weight, iso0=iso0, iso1=iso1, disp0=disp0, disp1=disp1,
         abs_dphi=dphi, ljlj_mass=mass)


def test_cube_shape_and_flow():
    h = make_hist("2mu2e")
    event(h, 0.1, 0.1, 0, 1)                 # displaced plane, A corner
    event(h, 7.0, 0.1, 5, 0, weight=2.0)     # isolation overflow, prompt plane
    event(h, 0.1, 0.1, 0, 1, mass=3000.0)    # mass overflow
    cube = io.cube_from_hist(h, "2mu2e")
    assert cube.shape == tuple(len(io.axis_edges("2mu2e", v)) + 1 for v in io.VARIABLES)
    assert io.project(cube) == (4.0, 6.0)
    assert io.project(cube, iso0=(5, None)) == (2.0, 4.0)
    assert io.project(cube, ljlj_mass=(2500, None)) == (1.0, 1.0)
    assert io.project(cube, ljlj_mass=(None, 2500)) == (3.0, 5.0)


def test_displacement_ranges_follow_the_analysis_definitions():
    h = make_hist("2mu2e")
    event(h, 0.1, 0.1, 2, 1)   # mu displaced (2 pixel hits), egm displaced (1 lost hit)
    event(h, 0.1, 0.1, 3, 0)   # both prompt
    event(h, 0.1, 0.1, 2, 0)   # mu displaced, egm prompt
    cube = io.cube_from_hist(h, "2mu2e")
    assert io.project(cube, **io.plane("2mu2e", "displaced", "displaced"))[0] == 1.0
    assert io.project(cube, **io.plane("2mu2e", "prompt", "prompt"))[0] == 1.0
    assert io.project(cube, **io.plane("2mu2e", "displaced", "prompt"))[0] == 1.0
    assert io.project(cube, **io.plane("2mu2e", "prompt", "displaced"))[0] == 0.0
    h4 = make_hist("4mu")
    event(h4, 0.1, 0.1, 2, 2)
    event(h4, 0.1, 0.1, 2, 3)
    cube4 = io.cube_from_hist(h4, "4mu")
    assert io.project(cube4, **io.plane("4mu", "displaced", "displaced"))[0] == 1.0
    assert io.project(cube4, **io.plane("4mu", "displaced", "prompt"))[0] == 1.0


def test_sentinels_land_in_the_displaced_category():
    h = make_hist("2mu2e")
    event(h, 0.1, 0.1, 0, 999)    # standalone-only muon LJ (0 pixel hits), photon-only egamma LJ
    event(h, 0.1, 0.1, 1, 0)      # one pixel hit, electron with no lost hits
    cube = io.cube_from_hist(h, "2mu2e")
    assert io.project(cube, disp1=io.displacement("2mu2e", 1, "displaced"))[0] == 1.0
    assert io.project(cube, disp1=(9.5, None))[0] == 1.0          # the overflow bin holds it
    assert io.project(cube, **io.plane("2mu2e", "displaced", "displaced"))[0] == 1.0
    assert io.project(cube, **io.plane("2mu2e", "displaced", "prompt"))[0] == 1.0


def test_blind_hides_exactly_the_search_corner():
    h = make_hist("2mu2e")
    event(h, 0.1, 0.1, 0, 1, mass=400.0)   # search corner: blind
    event(h, 0.4, 0.4, 0, 1, mass=400.0)   # inside the 0.5 margin: blind
    event(h, 0.1, 0.1, 0, 1, dphi=1.0)     # isolated but |dphi| < 2: open (C of the dphi plane)
    event(h, 0.1, 0.1, 0, 1, mass=100.0)   # below the mass threshold: open
    event(h, 0.1, 0.1, 5, 0, mass=400.0)   # prompt plane: open
    event(h, 0.1, 0.1, 0, 0, mass=400.0)   # one object displaced: open in 2mu2e
    event(h, 0.6, 0.1, 0, 1, mass=400.0)   # region B with the buffer: open
    event(h, 0.1, 0.6, 0, 1, mass=400.0)   # region C with the buffer: open
    cube = io.blind(io.cube_from_hist(h, "2mu2e"))
    assert cube.blinded
    assert np.isnan(cube.sumw).sum() == np.isnan(cube.sumw2).sum() > 0
    sr = io.plane("2mu2e", "displaced", "displaced")
    box = dict(iso0=(None, 0.5), iso1=(None, 0.5), ljlj_mass=(150, None), abs_dphi=(2, None))
    assert np.isnan(io.project(cube, **box, **sr)[0])
    assert np.isnan(io.project(cube)[0])
    assert io.project(cube, ljlj_mass=(None, 150), **sr)[0] == 1.0
    assert io.project(cube, abs_dphi=(None, 2), **sr)[0] == 1.0
    assert io.project(cube, **io.plane("2mu2e", "prompt", "prompt"))[0] == 1.0
    assert io.project(cube, **io.plane("2mu2e", "displaced", "prompt"))[0] == 1.0
    assert io.project(cube, iso0=(0.5, None), **sr)[0] == 1.0
    assert io.project(cube, iso1=(0.5, None), **sr)[0] == 1.0
    reg = io.regions(cube, 0.25, 0.25, fail0=0.5, fail1=0.5, ljlj_mass=(150, None),
                     abs_dphi=(2, None), **sr)
    assert np.isnan(reg["A"][0]) and reg["B"][0] == 1.0 and reg["C"][0] == 1.0
    total = io.add_cubes([cube, cube])
    assert total.blinded and io.project(total, **io.plane("2mu2e", "prompt", "prompt"))[0] == 2.0


def test_blind_hides_the_one_displaced_corners_in_4mu():
    h = make_hist("4mu")
    event(h, 0.1, 0.1, 1, 1, mass=400.0)   # search corner: blind
    event(h, 0.1, 0.1, 1, 5, mass=400.0)   # leading displaced only, isolated: blind in 4mu
    event(h, 0.1, 0.1, 5, 1, mass=400.0)   # subleading displaced only, isolated: blind in 4mu
    event(h, 0.1, 0.1, 5, 5, mass=400.0)   # prompt plane: open
    event(h, 0.6, 0.1, 1, 5, mass=400.0)   # leading displaced, region B: open
    event(h, 0.1, 0.1, 1, 5, mass=100.0)   # leading displaced, low mass: open
    cube = io.blind(io.cube_from_hist(h, "4mu"))
    box = dict(iso0=(None, 0.5), iso1=(None, 0.5), ljlj_mass=(150, None), abs_dphi=(2, None))
    for plane in (("displaced", "displaced"), ("displaced", "prompt"), ("prompt", "displaced")):
        assert np.isnan(io.project(cube, **box, **io.plane("4mu", *plane))[0])
    assert io.project(cube, **box, **io.plane("4mu", "prompt", "prompt"))[0] == 1.0
    dp = io.plane("4mu", "displaced", "prompt")
    assert io.project(cube, iso0=(0.5, None), **dp)[0] == 1.0
    assert io.project(cube, ljlj_mass=(None, 150), **dp)[0] == 1.0
    assert io.blind_signature() != io.blind_signature(planes={"2mu2e": (), "4mu": ()})


def test_factorising_plane_closes():
    rng = np.random.default_rng(1)
    h = make_hist("4mu")
    n = 20000
    iso0 = rng.exponential(0.4, n)
    iso1 = rng.exponential(0.6, n)
    fill(h, iso0=iso0, iso1=iso1, disp0=np.full(n, 1.0), disp1=np.full(n, 1.0),
         abs_dphi=np.full(n, 3.0), ljlj_mass=np.full(n, 300.0))
    cube = io.cube_from_hist(h, "4mu")
    reg = io.regions(cube, 0.25, 0.25, **io.plane("4mu", "displaced", "displaced"))
    (k, k_err), (pred, _) = io.kappa(reg)
    assert abs(k - 1.0) < 3 * k_err
    assert pred > 0
    value, err = io.ratio(reg["B"], reg["D"])
    assert err > 0 and abs(value - reg["B"][0] / reg["D"][0]) < 1e-12


def test_profile_sums_the_other_axes():
    h = make_hist("2mu2e")
    event(h, 0.1, 0.1, 0, 1)
    event(h, 0.6, 0.1, 0, 1, weight=3.0)
    cube = io.cube_from_hist(h, "2mu2e")
    sumw, sumw2 = io.profile(cube, "iso0", iso1=(None, 0.25))
    assert sumw.shape == (len(io.EDGES["iso"]) + 1,)
    assert sumw.sum() == 4.0 and sumw2.sum() == 10.0


def test_non_edge_boundary_is_refused():
    h = make_hist("2mu2e")
    cube = io.cube_from_hist(h, "2mu2e")
    with pytest.raises(ValueError):
        io.project(cube, iso0=(None, 0.22))
