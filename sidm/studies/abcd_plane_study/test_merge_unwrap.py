"""Regression test: the merge renormalization must engage on WRAPPED chunk files.

condor/run_sidm_chunk.py saves coffea Runner.run()'s wrapped shape
{"out": {sample: ...}, "processed": set, "exception": int}. The original
normalization fix was validated on synthetic BARE {sample: ...} dicts, so its
engage-condition silently never fired on real chunk files and merged yields
inflated by ~the chunk count. These tests pin the wrapped path.
"""

import sys
from pathlib import Path

import hist
import pytest
from coffea import processor

REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO))

from sidm.tools.cutflow import SimpleCutflow  # noqa: E402
from sidm.scripts.merge_coffea_chunks_eos import (  # noqa: E402
    unwrap_chunk, prescale_chunk, finalize_merged, combine_normalized_chunks)

LUMIXS = 1000.0  # arbitrary common lumi*xs factor baked into the per-chunk scaling


def make_chunk(raw, sumw, wrapped=True):
    """One per-chunk output as postprocess left it: scaled by LUMIXS/sumw_chunk."""
    scale = LUMIXS / sumw
    h = hist.Hist(hist.axis.Regular(1, 0, 1, name="x"), storage="weight")
    h.fill(x=[0.5], weight=[raw * scale])
    cf = SimpleCutflow()
    cf.add_row("all", raw, raw * scale)
    out = {"SAMPLE": {
        "hists": {"h": h},
        "cutflow": {"ch": cf},
        "counters": {},
        "metadata": {
            "n_evts": raw,
            "scaled_sum_weights": sumw,
            "year": processor.set_accumulator(["2018"]),
            "is_data": processor.set_accumulator([False]),
            "unweighted_hist": processor.set_accumulator([False]),
        },
    }}
    if wrapped:
        return {"out": out, "processed": set(), "exception": 0}
    return out


def _correct_value(chunks):
    """lumi*xs * sum(raw) / sum(sumw) — the right merged normalization."""
    raws = [c[0] for c in chunks]
    sumws = [c[1] for c in chunks]
    return LUMIXS * sum(raws) / sum(sumws)


@pytest.mark.parametrize("wrapped", [True, False])
def test_combine_normalized_chunks_engages(wrapped):
    spec = [(2.0, 10.0), (9.0, 30.0)]
    outputs = [make_chunk(r, s, wrapped=wrapped) for r, s in spec]
    merged = combine_normalized_chunks(outputs)
    got = merged["SAMPLE"]["hists"]["h"].sum(flow=True).value
    assert got == pytest.approx(_correct_value(spec), rel=1e-9)
    # cutflow weighted column renormalized identically
    cf = merged["SAMPLE"]["cutflow"]["ch"]
    weighted = list(cf.rows.values())[-1]["weighted"]
    assert weighted == pytest.approx(_correct_value(spec), rel=1e-9)


def test_plain_accumulate_would_inflate():
    """Documents the failure mode the unwrap fix prevents."""
    spec = [(2.0, 10.0), (9.0, 30.0)]
    outputs = [make_chunk(r, s, wrapped=True) for r, s in spec]
    plain = processor.accumulate(outputs)["out"]
    got = plain["SAMPLE"]["hists"]["h"].sum(flow=True).value
    correct = _correct_value(spec)
    assert got == pytest.approx(LUMIXS * (2 / 10 + 9 / 30), rel=1e-9)
    assert got / correct > 1.5  # inflated vs the correct answer


def test_incremental_path_matches_batch():
    """The memory-safe incremental loop in main() must equal the batch combine."""
    spec = [(2.0, 10.0), (9.0, 30.0), (5.0, 20.0)]
    outputs = [make_chunk(r, s, wrapped=True) for r, s in spec]
    batch = combine_normalized_chunks([make_chunk(r, s, wrapped=True) for r, s in spec])

    merged = None
    for o in outputs:
        o = unwrap_chunk(o)
        prescale_chunk(o)
        merged = o if merged is None else processor.accumulate([merged, o])
    merged = finalize_merged(merged)

    for m in (batch, merged):
        assert m["SAMPLE"]["hists"]["h"].sum(flow=True).value == \
            pytest.approx(_correct_value(spec), rel=1e-9)
