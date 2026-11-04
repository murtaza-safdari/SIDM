#!/usr/bin/env python3
"""Diagnose the real-mode 2Mu2E excess (Run 3 - 2018) with variants of the e/gamma LJ sources
(run3_lj_eff.EGM_VARIANTS), at the worst grid point and one near the mean, for 2022 and 2023BPix,
on the first R3_N Run 3 chunks and the first V10_N 2018 v10 files. Writes lj_eff_real_diag.json:
{variant: {point: {era: {...}}}}. The committed lj_eff_real_*.json are not touched.
Run from the worktree root with the notebook venv."""
import json, os, sys
sys.argv = [sys.argv[0]] + [a for a in sys.argv[1:] if a != "loose"]      # real mode
_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _DIR)
import run3_lj_eff as L
import run3_signal_validation as V

POINTS = {"worst": ("SIDM_BsTo2DpTo2Mu2E_MBs-100_MDp-5p0_ctau-40p0", "2Mu2E_100GeV_5p0GeV_40p0mm"),
          "mid": ("SIDM_BsTo2DpTo2Mu2E_MBs-200_MDp-1p2_ctau-4p8", "2Mu2E_200GeV_1p2GeV_4p8mm")}
ERAS = ["2022", "2023BPix"]
R3_N, V10_N = 10, 10
OUT = os.path.join(_DIR, "lj_eff_real_diag.json")


def main():
    commit = V.code_commit()
    files = {p: {era: None for era in ERAS} for p in POINTS}
    for p, (name, key) in POINTS.items():
        for era in ERAS:
            L.ERA = era
            files[p][era] = L.run3_files(name, R3_N)
        files[p]["2018"] = L.v10_files(key, "2mu2e", V10_N)
    # positive control: the default path and V0 must agree exactly
    d0 = L.lj_counts(files["mid"]["2022"][:1], True)
    dv = L.lj_counts(files["mid"]["2022"][:1], True, egm_variant="V0")
    assert d0["incl_acc"] == dv["incl_acc"] and d0["excl_acc"] == dv["excl_acc"], "V0 differs from the default"
    out = {}
    for var in L.EGM_VARIANTS:
        out[var] = {}
        for p, (name, key) in POINTS.items():
            ref = L.lj_counts(files[p]["2018"], True, egm_variant=var)
            out[var][p] = {"point": key, "definition": L.EGM_VARIANTS[var]}
            for era in ERAS:
                r3 = L.lj_counts(files[p][era], True, egm_variant=var)
                a, b = r3["incl_acc"], ref["incl_acc"]
                out[var][p][era] = dict(num=a["k"], den=a["n"], eff=a["p"],
                                        ref2018=dict(num=b["k"], den=b["n"], eff=b["p"]),
                                        diff_pct=100 * (a["p"] - b["p"]),
                                        r3_chunks=len(files[p][era]), v10_files=len(files[p]["2018"]),
                                        diag=r3["diag"], diag2018=ref["diag"], code_commit=commit)
                print(f"{var:4s} {p:5s} {era:8s} Run3 {a['p']:.4f} ({a['k']}/{a['n']})  2018 {b['p']:.4f} "
                      f"({b['k']}/{b['n']})  diff {100 * (a['p'] - b['p']):+.2f}%", flush=True)
    json.dump(out, open(OUT, "w"), indent=1)
    print(f"wrote {OUT}")


if __name__ == "__main__":
    main()
