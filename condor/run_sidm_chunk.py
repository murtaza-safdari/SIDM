#!/usr/bin/env python3

import argparse
import os
import sys
from pathlib import Path

from coffea import processor
import coffea.util

# Make sure parent of sidm/ is importable
repo_parent = Path.cwd()
if str(repo_parent) not in sys.path:
    sys.path.insert(0, str(repo_parent))

from sidm.tools import sidm_processor
from sidm.tools import llpnanoaodschema


def read_filelist(path):
    with open(path) as f:
        return [
            line.strip()
            for line in f
            if line.strip() and not line.strip().startswith("#")
        ]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--sample", required=True)
    parser.add_argument("--filelist", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--treename", default="Events")
    parser.add_argument("--chunksize", type=int, default=50_000)
    parser.add_argument("--workers", type=int, default=1)

    parser.add_argument("--channels", default="base")
    parser.add_argument("--hist-collections", default="muon_base")
    parser.add_argument("--unweighted-hist", action="store_true")
    parser.add_argument("--is-data", action="store_true",
                        help="Process as real data: enables the blinding "
                        "interlock (only SR-blinded channels + data-safe "
                        "collections allowed) and unweighted running.")

    args = parser.parse_args()

    files = read_filelist(args.filelist)
    # SidmProcessor.process reads events.metadata["is_data"] / ["skim_factor"] /
    # ["year"] without .get() defaults, so a bare {sample: files} fileset crashes
    # with KeyError. Supply MC-sample defaults; data submissions should override.
    fileset = {
        args.sample: {
            "files": files,
            "metadata": {
                "is_data": args.is_data,
                "skim_factor": 1.0,
                "year": "2018",
            },
        }
    }

    print("Repo parent:", repo_parent)
    print("Sample:", args.sample)
    print("Number of files:", len(files))
    print("Files:")
    for f in files:
        print("  ", f)

    runner = processor.Runner(
        executor=processor.FuturesExecutor(workers=args.workers),
        schema=llpnanoaodschema.LLPNanoAODSchema,
        skipbadfiles=True,
        chunksize=args.chunksize,
    )

    channels = [x.strip() for x in args.channels.split(",") if x.strip()]
    hist_collections = [x.strip() for x in args.hist_collections.split(",") if x.strip()]

    # ---- BLINDING INTERLOCK (Phase 3) ----------------------------------------
    # Real data may ONLY be histogrammed through channels that carry the SR
    # blind-box veto and through data-safe (reco-only, no gen branches) hist
    # collections. Verify the veto is actually present in each channel's resolved
    # event cuts (name-independent: catches a typo'd or veto-stripped channel) and
    # that every collection is on the data-safe allowlist. Fail loud before any
    # event is read, so a misrouted data submission can never expose the SR.
    if args.is_data:
        from sidm import BASE_DIR
        from sidm.tools import utilities
        BLIND_VETO = "ABCD SR blind box veto (data)"
        # The cosmic control region intentionally omits the blind box; its safety rests on
        # the cosmic tag, which is signal-DEPLETED (verified in MC before any data run).
        # A data channel is allowed if it carries EITHER protection.
        COSMIC_TAG = "ABCD cosmic CR tag (data)"
        DATA_SAFE_COLLECTIONS = {"abcd_data"}
        sel_menu = utilities.load_yaml(f"{BASE_DIR}/configs/selections.yaml")
        bad = []
        for ch in channels:
            spec = sel_menu.get(ch)
            evtc = utilities.flatten(spec["evt_cuts"]) if spec and "evt_cuts" in spec else []
            if BLIND_VETO not in evtc and COSMIC_TAG not in evtc:
                bad.append(f"channel '{ch}' carries neither the SR blind-box veto nor "
                           f"the cosmic-CR tag")
        for hc in hist_collections:
            if hc not in DATA_SAFE_COLLECTIONS:
                bad.append(f"collection '{hc}' is not data-safe "
                           f"(allowed: {sorted(DATA_SAFE_COLLECTIONS)})")
        if bad:
            raise SystemExit(
                "BLINDING INTERLOCK TRIPPED (is_data=True) -- refusing to run:\n  - "
                + "\n  - ".join(bad)
                + "\nData may only run through SR-blinded channels with data-safe "
                  "collections. Aborting before any event is read.")
        print(f"Blinding interlock OK: channels={channels} each carry the SR blind-box "
              f"veto or the cosmic-CR tag; collections={hist_collections} data-safe.")

    p = sidm_processor.SidmProcessor(
        channels,
        hist_collections,
        unweighted_hist=args.unweighted_hist,
    )

    try:
        output = runner.run(
            fileset,
            treename=args.treename,
            processor_instance=p,
        )
    except TypeError as e:
        # coffea's Runner unpacks the executor result without guarding the empty case:
        # when every file in this chunk has 0 events (legitimate after a skim), the
        # preprocessing produces zero work items, the executor returns None, and
        # `wrapped_out, e = executor(...)` raises "cannot unpack non-iterable NoneType
        # object". That is not a failure of this chunk -- it simply has nothing to
        # process -- so emit a valid EMPTY output (the additive identity for the merge's
        # processor.accumulate) and exit cleanly instead of crashing the whole job.
        if "unpack non-iterable NoneType" not in str(e):
            raise
        print("All files in this chunk have 0 events; saving an empty output.")
        output = {"out": {}, "processed": set(), "exception": 0}

    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
    coffea.util.save(output, args.output)

    print("Saved:", args.output)


if __name__ == "__main__":
    main()