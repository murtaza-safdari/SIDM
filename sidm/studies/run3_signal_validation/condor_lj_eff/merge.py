#!/usr/bin/env python3
"""Merge the per-job outputs into lj_eff_<mode>_<era>.json, in the point order of the existing file.
usage: merge.py WORKDIR STUDYDIR"""
import json, os, sys
work, study = sys.argv[1:3]
for mode in ("real", "geom"):
    for era in ("2022", "2022EE", "2023", "2023BPix"):
        rows = [r for p in range(6) for r in json.load(open(os.path.join(work, f"lj_{mode}_{era}_{p}.json")))]
        ref = [r["key"] for r in json.load(open(os.path.join(study, f"lj_eff_geom_{era}.json")))]
        byk = {r["key"]: r for r in rows}
        assert sorted(byk) == sorted(ref) and len(rows) == 36, (mode, era, len(rows))
        json.dump([byk[k] for k in ref], open(os.path.join(study, f"lj_eff_{mode}_{era}.json"), "w"), indent=1)
        print(f"wrote lj_eff_{mode}_{era}.json: 36 points")
