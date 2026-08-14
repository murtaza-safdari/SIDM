#!/usr/bin/env python3
"""Standalone brute-force closure test for lj_vertex_chi2 (no network, no ntuples).

    python sidm/tools/test_lj_vertex_chi2.py

Builds synthetic jagged events (LJ constituent-index lists plus the three dimuon-vertex
tables) and compares lj_best_vertex_chi2 against a plain triple-loop reference. Also
covers the large-normChi2 regime: real PatMuonVertex rows reach ~8e7, so the helper must
return those values unchanged and reserve -1.0 for "no stored within-LJ vertex".
"""
import os
import sys

import numpy as np
import awkward as ak

sys.path.insert(0, os.path.dirname(os.path.dirname(
    os.path.dirname(os.path.abspath(__file__)))))
from sidm.tools.lj_vertex_chi2 import lj_best_vertex_chi2  # noqa: E402

_P = _F = 0

# vertex rows are (idx1, idx2, isValid, normChi2[, isDSAMuon1, isDSAMuon2])
TABLES = (("dsa_vtx", "dsa"), ("pat_vtx", "pat"), ("mix_vtx", "mix"))
N_MU = 6            # size of the PF- and DSA-muon index pools
CHI2_BIG = 2.3e7    # above the old 8e5/9e5 filler, which turned it into -1
CHI2_MID = 7.9e5    # just below that filler, the boundary case
CHI2_SMALL = 3.0


def check(name, cond):
    """Record one assertion, printing only the failures."""
    global _P, _F
    if cond:
        _P += 1
    else:
        _F += 1
        print("FAIL:", name)


def build(events):
    """events: list of dicts with keys pf, dsa, dsa_vtx, pat_vtx, mix_vtx."""
    pf_pool = ak.Array([e["pf"] for e in events])
    dsa_pool = ak.Array([e["dsa"] for e in events])
    tables = []
    for key, kind in TABLES:
        rows = []
        for e in events:
            ev = []
            for r in e[key]:
                d = {"originalMuonIdx1": float(r[0]), "originalMuonIdx2": float(r[1]),
                     "isValid": float(r[2]), "normChi2": float(r[3])}
                if kind == "mix":
                    d["isDSAMuon1"], d["isDSAMuon2"] = float(r[4]), float(r[5])
                ev.append(d)
            rows.append(ev)
        tables.append((ak.Array(rows), kind))
    return pf_pool, dsa_pool, tables


def brute_force(events):
    """Plain triple loop over events, LJs and vertex rows. Same semantics, no awkward."""
    out = []
    for e in events:
        row = []
        for i_lj in range(len(e["pf"])):
            pf = {float(x) for x in e["pf"][i_lj]}
            dsa = {float(x) for x in e["dsa"][i_lj]}
            best = None
            for key, kind in TABLES:
                for r in e[key]:
                    if float(r[2]) != 1.0:      # isValid
                        continue
                    i1, i2 = float(r[0]), float(r[1])
                    if kind == "dsa":
                        ok = i1 in dsa and i2 in dsa
                    elif kind == "pat":
                        ok = i1 in pf and i2 in pf
                    else:
                        ok = (i1 in (dsa if float(r[4]) == 1.0 else pf)
                              and i2 in (dsa if float(r[5]) == 1.0 else pf))
                    if ok and (best is None or float(r[3]) < best):
                        best = float(r[3])
            row.append(-1.0 if best is None else best)
        out.append(row)
    return out


def run(events):
    """Return (module output, brute-force reference) for one event list."""
    pf_pool, dsa_pool, tables = build(events)
    got = ak.to_list(lj_best_vertex_chi2(pf_pool, dsa_pool, tables))
    return got, brute_force(events)


def random_events(n_events, seed=20260813):
    """Seeded synthetic events: jagged LJ constituent lists plus the three vertex tables."""
    rng = np.random.default_rng(seed)
    chi2_pool = [CHI2_SMALL, CHI2_MID, CHI2_BIG, 0.4, 1.7, 4.9, 12.0, 480.0]
    events = []
    for _ in range(n_events):
        n_lj = int(rng.integers(0, 4))
        pf, dsa = [], []
        for _ in range(n_lj):
            n_pf, n_dsa = int(rng.integers(0, 4)), int(rng.integers(0, 4))
            pf.append(sorted(rng.choice(N_MU, size=n_pf, replace=False).tolist()))
            dsa.append(sorted(rng.choice(N_MU, size=n_dsa, replace=False).tolist()))
        e = {"pf": pf, "dsa": dsa}
        for key, kind in TABLES:
            rows = []
            for _ in range(int(rng.integers(0, 5))):
                r = [int(rng.integers(0, N_MU)), int(rng.integers(0, N_MU)),
                     int(rng.integers(0, 2) if rng.random() < 0.25 else 1),
                     float(rng.choice(chi2_pool))]
                if kind == "mix":
                    r += [int(rng.integers(0, 2)), int(rng.integers(0, 2))]
                rows.append(r)
            e[key] = rows
        events.append(e)
    return events


def test_brute_force_closure():
    """Compare the helper against the reference over a large seeded sample."""
    events = random_events(3000)
    got, ref = run(events)
    n_lj = sum(len(r) for r in ref)
    check("same jagged structure as the reference",
          [len(r) for r in got] == [len(r) for r in ref])
    mism = [(i, j, got[i][j], ref[i][j])
            for i in range(len(ref)) for j in range(len(ref[i])) if got[i][j] != ref[i][j]]
    check(f"brute-force closure on {n_lj} synthetic LJs ({len(mism)} mismatches)", not mism)
    if mism:
        print("  first mismatches:", mism[:5])
    flat = [v for r in got for v in r]
    # the random sample must actually exercise both regimes
    check("random sample contains no-vertex LJs (-1.0)", any(v == -1.0 for v in flat))
    check("random sample contains large-normChi2 LJs (> 1e5)", any(v > 1e5 for v in flat))
    check("every value is either exactly -1.0 or >= 0 (no other sentinel leaks through)",
          all(v == -1.0 or v >= 0 for v in flat))
    print(f"  checked {len(events)} events / {n_lj} LJs")


def test_large_chi2_and_no_vertex():
    """The regime the 9e5/8e5 filler used to corrupt: huge but genuine normChi2."""
    def ev(pf, dsa, dsa_vtx=(), pat_vtx=(), mix_vtx=()):
        return {"pf": pf, "dsa": dsa, "dsa_vtx": list(dsa_vtx),
                "pat_vtx": list(pat_vtx), "mix_vtx": list(mix_vtx)}

    events = [
        ev([[0, 1]], [[0, 1]], dsa_vtx=[(0, 1, 1, CHI2_BIG)]),                    # 0
        ev([[0, 1]], [[0, 1]], dsa_vtx=[(0, 1, 1, CHI2_MID)]),                    # 1
        ev([[0, 1]], [[0, 1]]),                                                   # 2 no vertices
        ev([[0, 1]], [[0, 1]], dsa_vtx=[(0, 1, 1, CHI2_BIG), (0, 1, 1, CHI2_SMALL)]),  # 3
        ev([[0]], [[0]], dsa_vtx=[(0, 1, 1, CHI2_SMALL)]),                        # 4 single muon
        ev([[0, 1]], [[0, 1]], dsa_vtx=[(0, 1, 0, CHI2_SMALL)]),                  # 5 invalid only
        ev([[2, 3]], [[]], pat_vtx=[(2, 3, 1, CHI2_MID)]),                        # 6 pat table
        ev([[4]], [[5]], mix_vtx=[(4, 5, 1, CHI2_BIG, 0, 1)]),                    # 7 mixed table
    ]
    got, ref = run(events)
    flat = [r[0] for r in got]
    check("genuine normChi2 = 2.3e7 comes back as 2.3e7, not -1", flat[0] == CHI2_BIG)
    check("genuine normChi2 = 7.9e5 comes back as 7.9e5, not -1", flat[1] == CHI2_MID)
    check("LJ with no stored vertex is exactly -1.0",
          flat[2] == -1.0 and isinstance(flat[2], float))
    check("min still wins when a huge and a small vertex both match", flat[3] == CHI2_SMALL)
    check("single-muon LJ is exactly -1.0", flat[4] == -1.0)
    check("LJ whose only vertex has isValid = 0 is exactly -1.0", flat[5] == -1.0)
    check("pat table matched on PF indices", flat[6] == CHI2_MID)
    check("mixed table matched on per-leg muon type, huge value preserved", flat[7] == CHI2_BIG)
    check("targeted cases agree with the brute-force reference", got == ref)


if __name__ == "__main__":
    test_large_chi2_and_no_vertex()
    test_brute_force_closure()
    print(f"\n{_P} passed, {_F} failed")
    print("FAIL" if _F else "PASS")
    sys.exit(1 if _F else 0)
