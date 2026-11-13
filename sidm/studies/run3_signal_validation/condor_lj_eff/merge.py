#!/usr/bin/env python3
"""Merge the per-job outputs into lj_eff_<mode><tag>_<era>.json (tag '' for v1, '_v2' for v2), in the point
order of the existing v1 file of the era (v2: of 2022, for every era).
usage: merge.py WORKDIR STUDYDIR NPARTS [PRODUCTION (v1|v2), default v1]"""
import json, os, sys
work, study, nparts = sys.argv[1], sys.argv[2], int(sys.argv[3])
production = sys.argv[4] if len(sys.argv) > 4 else "v1"
tag = "" if production == "v1" else f"_{production}"
eras = ("2022", "2022EE", "2023", "2023BPix") + (("2024",) if production == "v2" else ())
for mode in ("real", "geom"):
    for era in eras:
        rows = [r for p in range(nparts) for r in json.load(open(os.path.join(work, f"lj_{mode}_{era}_{p}.json")))]
        ref = [r["key"] for r in json.load(open(os.path.join(study, f"lj_eff_geom_{era if production == 'v1' else '2022'}.json")))]
        byk = {r["key"]: r for r in rows}
        assert sorted(byk) == sorted(ref) and len(rows) == 36, (mode, era, len(rows))
        assert len({r["code_commit"] for r in rows}) == 1, "jobs ran different commits"
        assert {r.get("production", "v1") for r in rows} == {production}, "jobs ran another production"
        for r in rows:   # the submit-time commit must be the one stamped into the tarball
            if "tarball_commit" in r:
                assert r["tarball_commit"] == r["code_commit"], (r["key"], r["tarball_commit"], r["code_commit"])
        json.dump([byk[k] for k in ref], open(os.path.join(study, f"lj_eff_{mode}{tag}_{era}.json"), "w"), indent=1)
        print(f"wrote lj_eff_{mode}{tag}_{era}.json: 36 points")
