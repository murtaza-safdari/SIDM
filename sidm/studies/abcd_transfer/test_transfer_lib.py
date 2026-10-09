"""Tests of the transfer-study library on synthetic cubes: blinding of the data path, regions,
statistics and the table shapes."""
import numpy as np
import pytest

import abcd6d_io as io
import transfer_lib as tl
from test_abcd6d_io import event, make_hist


def synthetic_cube(topology="2mu2e"):
    """A cube with events in every region of every plane and one event in the search corner."""
    h = make_hist(topology)
    disp = {"prompt": (5, 0) if topology == "2mu2e" else (5, 5),
            "displaced": (1, 1) if topology == "2mu2e" else (1, 1)}
    for s0 in ("prompt", "displaced"):
        for s1 in ("prompt", "displaced"):
            d0, d1 = disp[s0][0], disp[s1][1]
            for mass in (100.0, 400.0):
                for iso0, iso1, n in ((0.1, 0.1, 6), (0.6, 0.1, 4), (0.1, 0.6, 3), (0.6, 0.6, 2), (0.3, 0.1, 5)):
                    for _ in range(n):
                        event(h, iso0, iso1, d0, d1, dphi=3.0, mass=mass)
                for _ in range(2):                       # non-back-to-back, isolated: C of the dphi plane
                    event(h, 0.1, 0.1, d0, d1, dphi=1.0, mass=mass)
                    event(h, 0.6, 0.1, d0, d1, dphi=1.0, mass=mass)
    return io.cube_from_hist(h, topology)


def fake_inputs(tmp_path, monkeypatch, cubes):
    """A fake input directory whose single data file loads as ``cubes``."""
    (tmp_path / "data").mkdir()
    (tmp_path / "data" / "Run2018.coffea").write_bytes(b"")
    (tmp_path / "MANIFEST.txt").write_text("1 data/Run2018.coffea\n")
    monkeypatch.setattr(io, "load_cubes", lambda path, topologies=tl.TOPOLOGIES, sample=None: dict(cubes))
    return str(tmp_path)


def test_data_is_only_available_blinded(tmp_path, monkeypatch):
    cubes = {t: synthetic_cube(t) for t in tl.TOPOLOGIES}
    base = fake_inputs(tmp_path, monkeypatch, cubes)
    data = tl.summed_cubes(base, "data")
    for t in tl.TOPOLOGIES:
        assert data[t].blinded
        assert np.isnan(tl.regions(data[t], "DD")["A"][0])
        assert np.isnan(tl.search_region(data[t])[0])
        assert not np.isnan(tl.regions(data[t], "PP")["A"][0])
    assert np.isnan(tl.regions(data["4mu"], "DP")["A"][0]) and np.isnan(tl.regions(data["4mu"], "PD")["A"][0])
    assert not np.isnan(tl.regions(data["2mu2e"], "DP")["A"][0])
    cached = np.load(tmp_path / "summed" / "data.npz")
    assert str(cached["blind_box"]) == io.blind_signature()
    assert np.isnan(cached["2mu2e_sumw"]).any()
    again = tl.summed_cubes(base, "data")                 # served from the cache, still blinded
    assert again["2mu2e"].blinded and np.isnan(tl.regions(again["2mu2e"], "DD")["A"][0])
    with pytest.raises(ValueError):
        tl.sample_cubes(base, "data")
    with pytest.raises(ValueError):
        tl.group_cubes(base, "data")


def test_changed_box_is_not_served_from_an_older_cache(tmp_path, monkeypatch):
    cubes = {t: synthetic_cube(t) for t in tl.TOPOLOGIES}
    base = fake_inputs(tmp_path, monkeypatch, cubes)
    (tmp_path / "summed").mkdir()
    np.savez(tmp_path / "summed" / "data.npz", manifest=tl._manifest_hash(base), blinded=False)   # a cache without a box
    tl.summed_cubes(base, "data")
    assert "blind_box" in np.load(tmp_path / "summed" / "data.npz").files
    monkeypatch.setattr(io, "blind_signature", lambda *a, **k: "another box")
    calls = []
    original = io.blind
    monkeypatch.setattr(io, "blind", lambda c, *a, **k: calls.append(1) or original(c, *a, **k))
    tl.summed_cubes(base, "data")
    assert calls, "the cache was reused although the hidden box changed"


def test_buffered_regions_leave_the_strip_out():
    cube = synthetic_cube()
    reg = tl.regions(cube, "PP", "M>=150")
    assert (reg["A"][0], reg["B"][0], reg["C"][0], reg["D"][0]) == (6.0, 4.0, 3.0, 2.0)
    dreg = tl.dphi_regions(cube, "PP", "M>=150")
    assert (dreg["A"][0], dreg["B"][0], dreg["C"][0], dreg["D"][0]) == (6.0, 4.0, 2.0, 2.0)
    assert io.project(cube, iso0=(0.25, 0.5), **tl.selection("2mu2e", "PP"))[0] == 5.0


def test_clopper_pearson_ratio_of_disjoint_counts():
    v, minus, plus = tl.ratio((43.0, 43.0), (18.0, 18.0))
    assert np.isclose(v, 43 / 18)
    assert np.isclose(minus, 0.648, atol=0.002) and np.isclose(plus, 0.906, atol=0.002)
    v, minus, plus = tl.ratio((0.0, 0.0), (18.0, 18.0))
    assert v == 0 and minus == 0 and plus > 0
    assert np.isnan(tl.ratio((np.nan, np.nan), (18.0, 18.0))[0])
    v, minus, plus = tl.ratio_of_ratios((1.0, 0.9, 2.0), (1.0, 0.9, 2.0))     # wide intervals: the minus side stops at zero
    assert v == 1.0 and minus == 1.0 and plus > 1.0


def test_pooled_shape_chi2():
    chi2, ndf = tl.shape_chi2([10, 10], [1, 1])
    assert chi2 == 0 and ndf == 1
    chi2, ndf = tl.shape_chi2([47, 53], [0, 5])
    assert np.isclose(chi2, 55225 / 23500 + 55225 / 29000) and ndf == 1
    with pytest.raises(ValueError):
        tl.shape_chi2([1.5, 2.0], [1.0, 1.0])


def test_prediction_from_an_empty_count_has_an_upper_error():
    assert tl._pred((0.0, 0.0), (2.0, 0.1, 0.1)) == (0.0, 0.0, tl.POISSON_UPPER_ZERO * 2.0)
    v, m, p = tl._pred((4.0, 4.0), (2.0, 0.2, 0.4))
    assert np.isclose(v, 8.0) and m < p
    v, m, p = tl._pred((1.0, 1.0), (0.875, 0.42, 0.79))
    assert m == v and p > v


def test_tables_cover_every_factor_and_displacement_change():
    cube = io.blind(synthetic_cube())
    assert set(tl.factors(tl.regions(cube, "PP"))) == {"r1", "ac", "r0", "ab", "kappa", "pred"}
    assert len(tl.transfer_table(cube)) == len(tl.TRANSFER_PAIRS) == 8
    assert len(tl.dphi_transfer_table(cube)) == 16
    assert tl.transfer_table(cube).iloc[3]["value, displaced"] == "n/a"      # A/C in DD is hidden
    sr = tl.sr_prediction_table(cube)
    assert sr["prediction of A (DD, M>=150)"].str.contains("assumes").all()
    assert len(tl.dphi_shape_table(cube)) == 6
    assert not (tl.closure_table(cube)["observed / predicted"] == "n/a").any()
    assert (tl.dphi_closure_table(cube)["plane"] == "DD").sum() == 10      # 4 factors x 2 sources + 2 in-plane rows


def test_contamination_table_handles_empty_data_regions():
    signal = {"s": {"2mu2e": synthetic_cube()}}
    h = make_hist("2mu2e")
    event(h, 0.6, 0.6, 5, 0, mass=400.0)           # one data event, in D of the prompt plane only
    data = io.blind(io.cube_from_hist(h, "2mu2e"))
    table = tl.contamination_table(signal, data, "2mu2e")
    pp = table[(table["plane kind"] == "iso x iso") & (table["mass"] == "M>=150") & (table["plane"] == "PP")]
    assert pp.set_index("region").loc["D", "max S/N"] != "n/a"
    assert pp.set_index("region").loc["A", "max S/N"] == "n/a"
    assert float(pp.set_index("region").loc["A", "max signal"]) == 6.0
    dd = table[(table["plane kind"] == "dphi x iso0") & (table["plane"] == "DD")].set_index("region")
    assert dd.loc["A", "data"] == "blind" and dd.loc["A", "max S/N"] == "n/a" and dd.loc["A", "s"] == "100.0"
