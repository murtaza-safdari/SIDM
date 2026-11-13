#!/usr/bin/env python3
"""One condor job of the LJ-efficiency scan: run3_lj_eff.lj_counts on a subset of grid points of one
era, for Run 3 (first R3_N chunks) and the 2018 v10 reference (first V10_N files), in the same
record format as run3_lj_eff.py writes.
usage: lj_eff_job.py MODE(real|geom) ERA NAME[,NAME...] R3_N V10_N OUT.json COMMIT [PRODUCTION(v1|v2)]"""
import json, os, re, sys

mode, era, names, r3n, v10n, out, commit = sys.argv[1:8]
production = sys.argv[8] if len(sys.argv) > 8 and sys.argv[8] else "v1"
# the commit the tarball was built from (make_tarball.sh); must equal the submit-time argument
stamp = open(os.path.join(os.path.dirname(os.path.abspath(__file__)), "TARBALL_COMMIT")).read().strip()
sys.argv = ["run3_lj_eff.py"] + (["loose"] if mode == "geom" else [])   # LOOSE is read at import
os.environ["RUN3_ERA"] = era
os.environ["RUN3_PRODUCTION"] = production
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
import run3_lj_eff as L

rows = []
for name in names.split(","):
    key, chan = L.r3_to_key(name); is2e = chan == "2Mu2E"
    m = re.match(r"SIDM_BsTo2DpTo(\w+?)_MBs-(\d+)_MDp-([\dp]+)_ctau-([\dp]+)", name)
    r3 = L.lj_counts(L.run3_files(name, int(r3n)), is2e)
    v10 = L.lj_counts(L.v10_files(key, chan.lower(), int(v10n)), is2e)
    rows.append(dict(chan=chan, mbs=int(m.group(2)), mdp=float(m.group(3).replace("p", ".")),
                     ctau=float(m.group(4).replace("p", ".")), key=key, run3=r3, v10=v10,
                     production=production, r3_chunks=int(r3n), v10_files=int(v10n), code_commit=commit, tarball_commit=stamp,
                     r3_chunks_read=r3["n_files_read"], v10_files_read=v10["n_files_read"]))
    print(f"{key:28s} R3 n={r3['n']} incl_acc={r3['incl_acc']['p']:.4f} | v10 n={v10['n']} "
          f"incl_acc={v10['incl_acc']['p']:.4f}", flush=True)
json.dump(rows, open(out, "w"), indent=1)
