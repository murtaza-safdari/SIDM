"""Tests of binned_cards: the cards say what the frames say, and the runner can read them."""
import importlib.util
import math
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

import binned_cards as bc
import binning_lib as bl

BINNING = [("150-400", (150.0, 400.0)), (">=400", (400.0, None))]
POINT = "2Mu2E_800GeV_0p25GeV_0p25mm"


# per bin and class: B of the search plane, A and B of the reference plane, the measured A/B with its
# interval, the rule's choice, the fitted A/B with its error, the shape systematic carried
SPEC = {"150-400": {"dsa0": (35, 1155, 1083, 1.07, 0.05, 0.05, "measured", 1.05, 0.04, 0.0),
                    "dsa1+": (30, 84, 177, 0.47, 0.06, 0.07, "measured", 0.48, 0.06, 0.0)},
        ">=400": {"dsa0": (9, 98, 120, 0.82, 0.11, 0.13, "measured", 0.94, 0.10, 0.0),
                  "dsa1+": (4, 6, 21, 0.29, 0.13, 0.21, "fitted", 0.26, 0.10, 0.12)}}


def routes_frame():
    """Two bins, two classes: the first bin measured in both classes, the second with dsa1+ fitted
    and carrying a shape systematic, and no dsa1+ B event in it would be the edge case below."""
    rows = []
    for label, (lo, hi) in BINNING:
        per = []
        for cls, (b, a_ref, b_ref, f, fm, fp, choice, ffit, ferr, shape) in SPEC[label].items():
            est = b * (f if choice == "measured" else ffit)
            rel = (fm + fp) / 2 / f if choice == "measured" else ferr / ffit
            err = est * math.hypot(1 / math.sqrt(b), rel)
            row = {"bin": label, "lo": lo, "hi": hi, "class": cls, "B": float(b), "A ref": float(a_ref),
                   "B ref": float(b_ref), "A/B": f, "A/B minus": fm, "A/B plus": fp, "A/B fit": ffit,
                   "A/B fit err": ferr, "choice": choice, "selected": est, "selected minus": err,
                   "selected plus": err, "shape carried": shape}
            per.append(row)
        total = sum(r["selected"] for r in per)
        rows.extend(per + [{"bin": label, "lo": lo, "hi": hi, "class": bl.SUM, "B": sum(r["B"] for r in per),
                            "A ref": sum(r["A ref"] for r in per), "B ref": sum(r["B ref"] for r in per),
                            "selected": total, "selected minus": math.hypot(*(r["selected minus"] for r in per)),
                            "selected plus": math.hypot(*(r["selected plus"] for r in per)),
                            "shape carried": math.hypot(*(r["shape carried"] for r in per))}])
    rows.append({"bin": bl.TOTAL, "lo": 150.0, "hi": None, "class": bl.SUM, "selected": 60.0})
    df = pd.DataFrame(rows)
    df.attrs.update(ref_plane="PD", search="DD")
    return df


# the signal in the regions the estimate reads from, per bin: (A PD:dsa0, B PD:dsa0, A PD:dsa1+, B PD:dsa1+, B DD)
CONTROL = {"150-400": (0.03, 0.0, 0.0, 0.0, 0.0), ">=400": (17.79, 0.07, 4.68, 0.0, 0.18)}


def control_frame():
    rows = []
    for label, (lo, hi) in BINNING:
        for (plane, region), v in zip(bl.ESTIMATE_REGIONS, CONTROL[label]):
            rows.append({"point": POINT, "plane": plane, "region": region, "label": f"{region} {plane}", "bin": label,
                         "lo": lo, "hi": hi, "yield": v, "error": 0.1 * v})
    return pd.DataFrame(rows)


def yields_frame(first=0.10, second=58.74):
    return pd.DataFrame([{"point": POINT, "bin": "150-400", "lo": 150.0, "hi": 400.0, "yield": first, "error": 0.3},
                         {"point": POINT, "bin": ">=400", "lo": 400.0, "hi": None, "yield": second, "error": 1.2}])


def parse(text):
    """The card's rate line, the first and second bin lines and the nuisance rows."""
    lines = [l for l in text.splitlines() if l and not l.startswith("#") and not l.startswith("-")]
    fields = {}
    bins = []
    nuis = {}
    for l in lines:
        f = l.split()
        if f[0] == "bin":
            bins.append(f[1:])
        elif f[0] in ("observation", "process", "rate"):
            fields.setdefault(f[0], []).append(f[1:])
        elif f[0] in ("imax", "jmax", "kmax"):
            fields[f[0]] = f[1]
        else:
            nuis[f[0]] = f[1:]
    return fields, bins, nuis


def test_lnn_card_mirrors_the_selected_estimate():
    text = bc.card_text(POINT, routes_frame(), yields_frame(), "split at 400", "lnN", syst_rel=0.33)
    fields, bins, nuis = parse(text)
    assert fields["imax"] == "2" and fields["jmax"] == "1" and bins[0] == ["m150to400", "m400up"]
    assert fields["process"][0] == ["sig", "bkg", "sig", "bkg"] and fields["process"][1] == ["0", "1", "0", "1"]
    rates = [float(v) for v in fields["rate"][0]]
    assert rates[0] == pytest.approx(0.10) and rates[2] == pytest.approx(58.74)
    assert rates[1] == pytest.approx(35 * 1.07 + 30 * 0.47, rel=1e-5)
    assert rates[3] == pytest.approx(9 * 0.82 + 4 * 0.26, rel=1e-5)
    obs = [float(v) for v in fields["observation"][0]]
    assert obs == pytest.approx([rates[1], rates[3]])
    # the systematic is one nuisance on every background column and none on the signal
    assert nuis["transfer"] == ["lnN", "-", "1.33", "-", "1.33"]
    # the shape systematic sits in the second bin only, as |measured - fitted| over the estimate
    assert "shape_m150to400" not in nuis
    assert float(nuis["shape_m400up"][4]) == pytest.approx(1 + 0.12 / rates[3], rel=1e-4)
    # the statistical log-normal is asymmetric from the interval
    down, up = (float(v) for v in nuis["stat_m400up"][4].split("/"))
    row = routes_frame()
    sel = row[(row["bin"] == ">=400") & (row["class"] == bl.SUM)].iloc[0]
    assert down == pytest.approx((sel["selected"] - sel["selected minus"]) / sel["selected"], abs=1e-4)
    assert up == pytest.approx((sel["selected"] + sel["selected plus"]) / sel["selected"], abs=1e-4)
    assert nuis["lumi_13TeV"] == ["lnN", "1.025", "-", "1.025", "-"]
    assert "theory cross section" in text


def test_gmn_card_counts_the_sideband_events():
    text = bc.card_text(POINT, routes_frame(), yields_frame(), "split at 400", "gmN", syst_rel=0.33)
    fields, bins, nuis = parse(text)
    assert fields["process"][0] == ["sig", "bkg_dsa0", "bkg_dsa1p", "sig", "bkg_dsa0", "bkg_dsa1p"]
    assert fields["process"][1] == ["0", "1", "2", "0", "1", "2"] and fields["jmax"] == "2"
    rates = [float(v) for v in fields["rate"][0]]
    # the rate is N times the alpha as written
    assert nuis["nB_m400up_dsa1p"][0] == "gmN" and nuis["nB_m400up_dsa1p"][1] == "4"
    alpha = float(nuis["nB_m400up_dsa1p"][7])
    assert rates[5] == pytest.approx(4 * alpha, rel=1e-6) and alpha == pytest.approx(0.26)
    assert nuis["nB_m150to400_dsa0"][1] == "35" and float(nuis["nB_m150to400_dsa0"][3]) == pytest.approx(1.07)
    # the fitted factor's error is symmetric, the measured one's is the interval
    assert nuis["factor_m400up_dsa1p"][6] == f"{(0.26 - 0.10) / 0.26:.4f}/{(0.26 + 0.10) / 0.26:.4f}"
    assert nuis["factor_m400up_dsa0"][5] == f"{(0.82 - 0.11) / 0.82:.4f}/{(0.82 + 0.13) / 0.82:.4f}"
    assert float(nuis["shape_m400up_dsa1p"][6]) == pytest.approx(1 + 0.12 / (4 * 0.26), rel=1e-4)
    assert nuis["transfer"] == ["lnN", "-", "1.33", "1.33", "-", "1.33", "1.33"]


def test_zero_counts_drop_processes_and_bins():
    routes = routes_frame()
    routes.loc[(routes["bin"] == ">=400") & (routes["class"] == "dsa1+"), "B"] = 0.0
    text = bc.card_text(POINT, routes, yields_frame(), "split at 400", "gmN")
    fields, bins, nuis = parse(text)
    assert fields["process"][0] == ["sig", "bkg_dsa0", "bkg_dsa1p", "sig", "bkg_dsa0"]
    assert fields["process"][1] == ["0", "1", "2", "0", "1"] and fields["jmax"] == "2"
    assert "omitted: m400up bkg_dsa1p" in text
    # a process name keeps its number when it is the only one left in a bin
    routes.loc[(routes["bin"] == ">=400") & (routes["class"] == "dsa0"), "B"] = 0.0
    routes.loc[(routes["bin"] == ">=400") & (routes["class"] == "dsa1+"), "B"] = 4.0
    fields, bins, nuis = parse(bc.card_text(POINT, routes, yields_frame(), "split at 400", "gmN"))
    assert fields["process"][0][-1] == "bkg_dsa1p" and fields["process"][1][-1] == "2"
    # a bin with neither signal nor background is dropped
    routes.loc[(routes["bin"] == ">=400") & (routes["class"] != "dsa1+"), ["B", "selected"]] = 0.0
    text = bc.card_text(POINT, routes, yields_frame(second=0.0), "split at 400", "lnN")
    fields, bins, nuis = parse(text)
    assert fields["imax"] == "1" and bins[0] == ["m150to400"] and "dropped: m400up" in text
    with pytest.raises(ValueError):
        bc.card_text(POINT, routes, yields_frame(first=0.0, second=0.0), "split at 400", "lnN")


def test_cr_card_puts_the_control_region_in_the_fit():
    text = bc.card_text(POINT, routes_frame(), yields_frame(), "split at 400", "cr", syst_rel=0.33, control=control_frame())
    fields, bins, nuis = parse(text)
    # the search bins first, then one control channel per bin and class, observed A of the reference plane
    assert bins[0] == ["m150to400", "m400up", "cr_m150to400_dsa0", "cr_m150to400_dsa1p", "cr_m400up_dsa0", "cr_m400up_dsa1p"]
    assert fields["imax"] == "6" and fields["jmax"] == "2"
    obs = [float(v) for v in fields["observation"][0]]
    assert obs[2:] == [1155.0, 84.0, 98.0, 6.0]
    assert fields["process"][0] == ["sig", "bkg_dsa0", "bkg_dsa1p"] * 2 + ["sig", "bkg_dsa0", "sig", "bkg_dsa1p"] * 2
    assert fields["process"][1] == ["0", "1", "2"] * 2 + ["0", "1", "0", "2"] * 2
    rates = [float(v) for v in fields["rate"][0]]
    # the factor is measured in every bin of the cr model: the search bin's rate is B x the measured A/B ...
    assert rates[4] == pytest.approx(9 * 0.82) and rates[5] == pytest.approx(4 * 0.29)
    # ... the control channel's background is B of the reference plane x the same factor, its signal the leak
    assert rates[10] == pytest.approx(17.79) and rates[11] == pytest.approx(120 * 0.82)
    assert rates[12] == pytest.approx(4.68) and rates[13] == pytest.approx(21 * 0.29)
    assert nuis["nB_m400up_dsa0"][:2] == ["gmN", "9"] and nuis["nBref_m400up_dsa0"][:2] == ["gmN", "120"]
    assert float(nuis["nBref_m400up_dsa0"][2 + 11]) == pytest.approx(0.82)
    # the factor is one free parameter per bin and class, on the search bin and its control channel
    lines = [l.split() for l in text.splitlines() if " rateParam " in l and not l.startswith("#")]
    assert [l[0] for l in lines].count("f_m400up_dsa0") == 2
    assert [(l[2], l[3]) for l in lines if l[0] == "f_m400up_dsa0"] == [("m400up", "bkg_dsa0"), ("cr_m400up_dsa0", "bkg_dsa0")]
    assert all(l[4] == "1.0" and l[5] == bc.RATE_PARAM_RANGE for l in lines) and len(lines) == 8
    # the transfer systematic sits on the search bins only, the luminosity on every signal column
    assert nuis["transfer"] == ["lnN", "-", "1.33", "1.33", "-", "1.33", "1.33"] + ["-"] * 8
    assert nuis["lumi_13TeV"][1:] == ["1.025", "-", "-"] * 2 + ["1.025", "-"] * 4
    assert "shape_m400up_dsa1p" not in nuis and "factor_m400up_dsa1p" not in nuis
    assert "B DD >=400 0.18" in text and "theory cross section" in text
    with pytest.raises(ValueError):
        bc.card_text(POINT, routes_frame(), yields_frame(), "split at 400", "cr")
    # a class without a sideband event in either plane has no control channel
    routes = routes_frame()
    routes.loc[(routes["bin"] == ">=400") & (routes["class"] == "dsa1+"), "B"] = 0.0
    fields, bins, nuis = parse(bc.card_text(POINT, routes, yields_frame(), "split at 400", "cr", control=control_frame()))
    assert "cr_m400up_dsa1p" not in bins[0] and fields["imax"] == "5"
    # without an A event in the reference plane the factor starts from the fitted value
    routes = routes_frame()
    routes.loc[(routes["bin"] == ">=400") & (routes["class"] == "dsa1+"), ["A ref", "A/B"]] = [0.0, 0.0]
    fields, bins, nuis = parse(bc.card_text(POINT, routes, yields_frame(), "split at 400", "cr", control=control_frame()))
    assert float(fields["rate"][0][5]) == pytest.approx(4 * 0.26)


def test_card_names_parse_both_ways(tmp_path):
    paths = bc.write_cards({"split at 400": routes_frame()}, {"split at 400": yields_frame()}, tmp_path,
                           meta={"note": "test"}, control_by_binning={"split at 400": control_frame()})
    assert sorted(p.name for p in paths) == [f"datacard_SR_2mu2e_split400_cr_{POINT}.txt",
                                             f"datacard_SR_2mu2e_split400_gmn_{POINT}.txt",
                                             f"datacard_SR_2mu2e_split400_lnn_{POINT}.txt"]
    assert (tmp_path / "datacards.meta.yaml").read_text().startswith("note: test")
    assert bc.parse_card_name(paths[2].stem)["model"] == "cr"
    paths = [p for p in paths if "_lnn_" in p.name] + [p for p in paths if "_lnn_" not in p.name]
    info = bc.parse_card_name(paths[0].stem)
    assert info == {"topology": "2mu2e", "binning": "split at 400", "model": "lnN", "point": POINT}
    # the runner of limit_plotting reads the channel and the signal point out of the same name
    runner = Path(__file__).resolve().parents[2] / "scripts" / "run_combine_limits.py"
    spec = importlib.util.spec_from_file_location("run_combine_limits", runner)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    parsed = mod.parse_card_name(paths[0].stem)
    assert parsed["channel"] == "SR_2mu2e_split400_lnn" and parsed["signal"] == POINT
    assert parsed["m_mediator"] == 800.0 and parsed["ctau"] == 0.25
    assert mod.read_normalisation(paths[0]) == "theory"
    signal, background = mod.read_rates(paths[0])
    assert signal == pytest.approx(0.10) and background == pytest.approx(35 * 1.07 + 30 * 0.47, rel=1e-5)


def test_limits_comparison_is_relative_to_the_same_model(tmp_path):
    csv = tmp_path / "limits.csv"
    rows = [("onebin", "lnn", 0.40), ("split400", "lnn", 0.10), ("onebin", "gmn", 0.44), ("split400", "gmn", 0.11)]
    csv.write_text("datacard,exp_m2,exp_m1,exp,exp_p1,exp_p2,obs\n" + "".join(
        f"datacard_SR_2mu2e_{slug}_{model}_{POINT}.txt,{0.5 * r},{0.7 * r},{r},{1.4 * r},{2 * r},\n"
        for slug, model, r in rows))
    limits = bc.read_limits(csv)
    assert len(limits) == 4 and set(limits["model"]) == {"lnN", "gmN"}
    toy = pd.DataFrame([{"point": POINT, "binning": "one bin", "factor": "selected", "limit on mu": 0.39,
                         "relative to one bin": 1.0},
                        {"point": POINT, "binning": "split at 400", "factor": "selected", "limit on mu": 0.098,
                         "relative to one bin": 0.251}])
    table = bc.comparison(limits, toy)
    split = table[table["binning"] == "split at 400"].iloc[0]
    assert split["relative lnN"] == pytest.approx(0.25) and split["relative gmN"] == pytest.approx(0.25)
    assert split["r lnN"] == pytest.approx(0.10) and split["relative toy"] == pytest.approx(0.251)
    view = bc.comparison_view(table)
    assert view.loc[view["binning"] == "split at 400", "relative lnN"].iloc[0] == "0.250"
    assert np.isfinite(table["r toy"]).all()
