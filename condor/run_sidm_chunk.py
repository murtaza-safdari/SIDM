#!/usr/bin/env python3

import argparse
import os
import sys
from pathlib import Path

from coffea import processor
import coffea.util
import yaml

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
    parser.add_argument("--backgrounds-yaml", default="sidm/configs/ntuples/backgrounds.yaml")
    parser.add_argument("--backgrounds-section", default="skimmed_llpNanoAOD_v2")

    args = parser.parse_args()

    files = read_filelist(args.filelist)

    def find_sample_cfg(sample, primary_yaml, primary_section):
        """Find sample metadata.

        Background samples are normally stored in backgrounds.yaml.
        Signal samples may live in signal_*.yaml.  For chunk processing we only
        need lightweight metadata: is_data, skim_factor, and year.
        """
        checked = []

        def check_yaml(yaml_path, preferred_section=None):
            with open(yaml_path) as f:
                cfg = yaml.safe_load(f) or {}

            sections = []
            if preferred_section is not None and preferred_section in cfg:
                sections.append(preferred_section)
            sections.extend([k for k in cfg.keys() if k not in sections])

            for section in sections:
                block = cfg.get(section, {})
                samples = block.get("samples", {})
                if sample in samples:
                    return samples[sample], str(yaml_path), section

            return None

        primary_yaml = Path(primary_yaml)
        checked.append(f"{primary_yaml}:{primary_section}")
        found = check_yaml(primary_yaml, primary_section)
        if found is not None:
            return found

        ntuple_dir = Path("sidm/configs/ntuples")

        # Force signal samples to use the intended signal ntuple YAMLs.
        # Do not allow broad fallback to pick up overlapping names from ffntuples.yaml.
        preferred_yamls = []
        if sample.startswith("4Mu_"):
            preferred_yamls.append(ntuple_dir / "signal_4mu_v10.yaml")
        elif sample.startswith("2Mu2E_"):
            preferred_yamls.append(ntuple_dir / "signal_2mu2e_v10.yaml")

        for yaml_path in preferred_yamls:
            if not yaml_path.exists():
                continue
            if yaml_path.resolve() == primary_yaml.resolve():
                continue
            checked.append(str(yaml_path))
            found = check_yaml(yaml_path, primary_section)
            if found is not None:
                return found

        # Last-resort fallback for unusual non-signal samples only.
        # For normal 4Mu/2Mu2E signals, reaching here means the intended signal YAML
        # does not contain the sample, so fail instead of silently using ffntuples.yaml.
        if sample.startswith(("4Mu_", "2Mu2E_")):
            raise KeyError(
                f"Could not find signal sample={sample!r} in preferred YAML(s): "
                f"{', '.join(str(x) for x in preferred_yamls)}"
            )

        for yaml_path in sorted(ntuple_dir.glob("*.yaml")):
            if yaml_path.resolve() == primary_yaml.resolve():
                continue
            checked.append(str(yaml_path))
            found = check_yaml(yaml_path, primary_section)
            if found is not None:
                return found

        raise KeyError(
            f"Could not find sample={sample!r} in any ntuple yaml. "
            f"Checked: {', '.join(checked)}"
        )

    sample_cfg, sample_cfg_path, sample_cfg_section = find_sample_cfg(
        args.sample,
        args.backgrounds_yaml,
        args.backgrounds_section,
    )

    metadata = {
        "is_data": bool(sample_cfg.get("is_data", False)),
        "skim_factor": float(sample_cfg.get("skim_factor", 1.0)),
        "year": str(sample_cfg.get("year", "2018")),
    }

    fileset = {
        args.sample: {
            "files": files,
            "metadata": metadata,
        }
    }

    print("Repo parent:", repo_parent)
    print("Sample:", args.sample)
    print("Number of files:", len(files))
    print("Metadata source:", f"{sample_cfg_path}:{sample_cfg_section}")
    print("Metadata:", metadata)
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
        output = {"out": {}, "processed": set(), "exception": 0}

    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
    coffea.util.save(output, args.output)

    print("Saved:", args.output)


if __name__ == "__main__":
    main()
