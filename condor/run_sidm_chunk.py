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


def read_filelist_metadata(path):
    """Parse the optional "# metadata: year=... is_data=... skim_factor=..." header
    written by make_job_args.py. Returns MC-sample defaults (with a loud warning)
    for filelists that predate the header."""
    meta = {"is_data": False, "skim_factor": 1.0, "year": "2018"}
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line.startswith("# metadata:"):
                for tok in line[len("# metadata:"):].split():
                    key, _, val = tok.partition("=")
                    if key == "is_data":
                        meta["is_data"] = val in ("1", "True", "true")
                    elif key == "skim_factor":
                        meta["skim_factor"] = float(val)
                    elif key == "year":
                        meta["year"] = val
                return meta
    print("WARNING: filelist has no metadata header; assuming MC defaults "
          f"{meta} -- WRONG for skimmed or data samples")
    return meta


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

    args = parser.parse_args()

    files = read_filelist(args.filelist)
    # SidmProcessor.process reads events.metadata["is_data"] / ["skim_factor"] /
    # ["year"] without .get() defaults. Real values come from the filelist
    # metadata header (written by make_job_args.py from the location yaml);
    # a missing header falls back to MC defaults with a warning.
    meta = read_filelist_metadata(args.filelist)
    fileset = {
        args.sample: {
            "files": files,
            "metadata": dict(meta),
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
        # match the shape process() returns so processor.accumulate keeps the
        # sample key even when EVERY chunk of a sample is empty (the merge
        # script skips sumw==0 entries, so the zeros are inert there)
        output = {args.sample: {
            "cutflow": {},
            "hists": {},
            "counters": {},
            "metadata": {
                "n_evts": 0,
                "scaled_sum_weights": 0.0,
                "year": processor.set_accumulator([meta["year"]]),
                "is_data": processor.set_accumulator([meta["is_data"]]),
            },
        }}

    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
    coffea.util.save(output, args.output)

    print("Saved:", args.output)


if __name__ == "__main__":
    main()