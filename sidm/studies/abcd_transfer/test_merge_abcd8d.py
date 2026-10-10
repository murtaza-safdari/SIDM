"""The chunk combination of merge_abcd8d: simulated chunks, each scaled to the luminosity with
its own sum of weights, combine to the histogram scaled once with the whole sample's; data
chunks are summed as they are."""
import hist
import numpy as np
import pytest

import merge_abcd8d as mg


def chunk(values, sumw, is_data=False, name="abcd8d_2mu2e"):
    h = hist.Hist(hist.axis.Regular(3, 0, 3, name="x"), storage=hist.storage.Weight())
    for v, w in values:
        h.fill(v, weight=w)
    return {"hists": {name: h, "other_hist": h.copy()},
            "metadata": {"scaled_sum_weights": sumw, "is_data": {is_data}, "unweighted_hist": {False}}}


def test_simulated_chunks_are_weighted_by_their_sum_of_weights():
    # chunk 1: 1 raw event scaled by L xs / 10 -> weight 100 (L xs = 1000); chunk 2: 3 raw events, sumw 30 -> weight 33.3 each
    lumi_xs = 1000.0
    c1 = chunk([(0.5, lumi_xs / 10.0)], 10.0)
    c2 = chunk([(0.5, lumi_xs / 30.0)] * 3, 30.0)
    hists, meta = mg.combine([c1, c2], "s")
    h = hists["abcd8d_2mu2e"]
    assert set(hists) == {"abcd8d_2mu2e"}                         # only the campaign's histograms
    assert meta["scaled_sum_weights"] == 40.0
    # the whole sample: 4 raw events, sum of weights 40 -> each weighs L xs / 40 = 25 -> 100 in the bin
    assert h.values()[0] == pytest.approx(100.0)
    assert h.variances()[0] == pytest.approx(4 * 25.0 ** 2)
    assert h.values()[1:].sum() == 0


def test_data_chunks_are_summed_unscaled():
    c1 = chunk([(0.5, 1.0)] * 5, 5.0, is_data=True)
    c2 = chunk([(1.5, 1.0)] * 2, 2.0, is_data=True)
    hists, meta = mg.combine([c1, c2], "d")
    h = hists["abcd8d_2mu2e"]
    assert h.values().tolist() == [5.0, 2.0, 0.0] and h.variances().tolist() == [5.0, 2.0, 0.0]
    assert meta["scaled_sum_weights"] == 7.0


def test_chunks_with_different_histogram_sets_are_refused():
    c1 = chunk([(0.5, 1.0)], 10.0)
    c2 = chunk([(0.5, 1.0)], 10.0)
    c2["hists"]["abcd8d_4mu"] = c2["hists"]["abcd8d_2mu2e"].copy()     # one more histogram than c1
    with pytest.raises(ValueError, match="different histogram sets"):
        mg.combine([c1, c2], "s")
    with pytest.raises(ValueError, match="no chunk"):
        mg.combine([], "s")


def test_plan_merges_only_listed_chunks_and_rescues_the_rest():
    expected = {"a": {"0": "a 0 x\n", "1": "a 1 x\n", "2": "a 2 x\n"}, "b": {"0": "b 0 x\n"}}
    found = {"a": {"0", "1", "7"}, "b": {"0"}}                      # a_7 is not in the job list; a_2 is missing
    todo, rescue, extras = mg.plan(found, expected, ["a", "b"], allow_missing=False)
    assert todo == [("b", ["0"])] and rescue == ["a 2 x\n"] and extras == {"a": ["7"]}
    todo, rescue, extras = mg.plan(found, expected, ["a", "b"], allow_missing=True)
    assert todo == [("a", ["0", "1"]), ("b", ["0"])] and rescue == ["a 2 x\n"]
