#!/usr/bin/env python3
"""Merge the per-job outputs into lj_eff_<mode>_<era>.json, in the point order of the existing file.
usage: merge.py WORKDIR STUDYDIR NPARTS"""
import json, os, sys
work, study, nparts = sys.argv[1], sys.argv[2], int(sys.argv[3])
for mode in ("real", "geom"):
    for era in ("2022", "2022EE", "2023", "2023BPix"):
        rows = [r for p in range(nparts) for r in json.load(open(os.path.join(work, f"lj_{mode}_{era}_{p}.json")))]
        ref = [r["key"] for r in json.load(open(os.path.join(study, f"lj_eff_geom_{era}.json")))]
        byk = {r["key"]: r for r in rows}
        assert sorted(byk) == sorted(ref) and len(rows) == 36, (mode, era, len(rows))
        assert len({r["code_commit"] for r in rows}) == 1, "jobs ran different commits"
        for r in rows:   # the submit-time commit must be the one stamped into the tarball
            if "tarball_commit" in r:
                assert r["tarball_commit"] == r["code_commit"], (r["key"], r["tarball_commit"], r["code_commit"])
        json.dump([byk[k] for k in ref], open(os.path.join(study, f"lj_eff_{mode}_{era}.json"), "w"), indent=1)
        print(f"wrote lj_eff_{mode}_{era}.json: 36 points")
