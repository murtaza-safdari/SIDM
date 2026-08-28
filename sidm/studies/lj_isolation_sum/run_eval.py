#!/usr/bin/env python3
"""MC evaluation run for the LJ jet-sum isolation study (run ON LPC, in the venv)."""
import json
import sys
import time

WT = "/uscms_data/d3/murtazas/SIDM-wt-ljiso"
OUT = "/uscms_data/d3/murtazas/ljiso_study"
sys.path.insert(0, WT)

import coffea.util
from coffea import processor

from sidm.tools import sidm_processor, llpnanoaodschema, utilities

# ---------------------------------------------------------------- sample plan
# MBs = 500 GeV throughout; for each MDp the shortest and the longest ctau on the grid.
SIG_POINTS = [
    ("0p25GeV", "0p004mm", "short"), ("0p25GeV", "4p0mm", "long"),
    ("1p2GeV", "0p019mm", "short"), ("1p2GeV", "19p0mm", "long"),   # 1p2/19p0 = reference point
    ("5p0GeV", "0p08mm", "short"), ("5p0GeV", "80p0mm", "long"),
]
SIG_FILES = 2

BKG = {  # skimmed backgrounds; file counts chosen for usable yields, not uniformity
    "QCD_Pt80To120": 100,
    "QCD_Pt300To470": 60,
    "QCD_Pt1000": 20,
    "DYJetsToMuMu_M50": 8,
    "TTJets": 8,
}

CHANNELS = ["base_ljObjCut", "4mu", "2mu2e"]

fileset = {}
for mdp, ctau, _tag in SIG_POINTS:
    for chan, cfg in (("4Mu", "signal_4mu_v10.yaml"), ("2Mu2E", "signal_2mu2e_v10.yaml")):
        name = "%s_500GeV_%s_%s" % (chan, mdp, ctau)
        fileset = utilities.make_fileset([name], "llpNanoAOD_v2", max_files=SIG_FILES,
                                         location_cfg=cfg, replace_xcache=True,
                                         fileset=fileset, census_skip=None)
for s, n in BKG.items():
    fileset = utilities.make_fileset([s], "skimmed_llpNanoAOD_v2", max_files=n,
                                     location_cfg="backgrounds.yaml", replace_xcache=True,
                                     fileset=fileset,
                                     census_skip="backgrounds_skimmed.skip.json")

n_files = sum(len(v["files"]) for v in fileset.values())
print("samples: %d, files: %d" % (len(fileset), n_files), flush=True)
for k, v in fileset.items():
    print("   %-32s %4d files" % (k, len(v["files"])), flush=True)

p = sidm_processor.SidmProcessor(CHANNELS, ["ljiso_study"], unweighted_hist=True)
runner = processor.Runner(
    executor=processor.FuturesExecutor(workers=6),
    schema=llpnanoaodschema.LLPNanoAODSchema,
    skipbadfiles=True,
    chunksize=50_000,
    xrootdtimeout=300,
)

t0 = time.time()
out = runner.run(fileset, treename="Events", processor_instance=p)
wall = time.time() - t0
print("WALL TIME: %.1f s" % wall, flush=True)

coffea.util.save(out, OUT + "/ljiso_eval.coffea")
print("saved", OUT + "/ljiso_eval.coffea", flush=True)

summary = {"wall_time_s": wall, "channels": CHANNELS, "n_files_requested": n_files,
           "n_files_processed": len(out.get("processed", [])), "samples": {}}
for s, res in out["out"].items():
    summary["samples"][s] = {
        "n_evts": int(res["metadata"]["n_evts"]),
        "n_files_requested": len(fileset[s]["files"]),
    }
    print("%-32s n_evts=%d" % (s, res["metadata"]["n_evts"]), flush=True)
with open(OUT + "/run_summary.json", "w") as f:
    json.dump(summary, f, indent=2)
print("DONE")
