#!/usr/bin/env python3
"""Task lists (VERSION ERA SAMPLE URL per line) for xclean_rows.py, from the location YAMLs.

v1: sidm/configs/ntuples/signal_{4mu,2mu2e}_run3.yaml (key = era); v2: signal_run3_v2_<era>.yaml (key
exoNanoAOD_v2). URLs as make_fileset(..., replace_xcache=True) gives them.

  --stage first   the files of the first comparison: displaced = <chan>_200GeV_1p2GeV_4p8mm j040-j059 (20 x 1000 events),
                  prompt-like = the six PROMPT points j000-j003 (4 x 500 events each); eras 2022, 2023
  --stage full    every listed file of the 36 column samples (v2_census/production/reduced_points.txt) and of
                  the six prompt-like points per channel; eras 2022, 2022EE, 2023, 2023BPix

Usage: make_tasks.py --stage {first,full} > tasks.txt
"""
import argparse
import os

import yaml

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.abspath(os.path.join(HERE, "..", "..", "..", ".."))
NT = os.path.join(REPO, "sidm/configs/ntuples")
ERAS = ["2022", "2022EE", "2023", "2023BPix"]
CHANS = ["4Mu", "2Mu2E"]
DISPLACED = "{chan}_200GeV_1p2GeV_4p8mm"
PROMPT = ["{chan}_100GeV_1p2GeV_0p096mm", "{chan}_150GeV_1p2GeV_0p064mm", "{chan}_200GeV_1p2GeV_0p048mm",
          "{chan}_500GeV_1p2GeV_0p019mm", "{chan}_800GeV_1p2GeV_0p012mm", "{chan}_1000GeV_1p2GeV_0p0096mm"]
COLUMN = [s.strip() for s in open(os.path.join(HERE, "..", "v2_census/production/reduced_points.txt")) if s.strip()]


def files(version, era, sample):
    if version == "v1":
        chan = sample.split("_")[0].lower()
        block = yaml.safe_load(open(f"{NT}/signal_{chan}_run3.yaml"))[era]
    else:
        block = yaml.safe_load(open(f"{NT}/signal_run3_v2_{era}.yaml"))["exoNanoAOD_v2"]
    s = block["samples"][sample]
    base = (block["path"] + s["path"]).replace("root://xcache//", "root://cmseos.fnal.gov//")
    return [base + f for f in s["files"]]


def chunk(url):
    return int(url.rsplit("_j", 1)[1].split(".")[0])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", choices=["first", "full"], required=True)
    args = ap.parse_args()
    for version in ["v1", "v2"]:
        if args.stage == "first":
            for era in ["2022", "2023"]:
                for chan in CHANS:
                    for u in files(version, era, DISPLACED.format(chan=chan)):
                        if 40 <= chunk(u) < 60:
                            print(version, era, DISPLACED.format(chan=chan), u)
                    for p in PROMPT:
                        for u in files(version, era, p.format(chan=chan)):
                            if chunk(u) < 4:
                                print(version, era, p.format(chan=chan), u)
        else:
            for era in ERAS:
                samples = COLUMN + [p.format(chan=c) for c in CHANS for p in PROMPT]
                for s in samples:
                    for u in files(version, era, s):
                        print(version, era, s, u)


if __name__ == "__main__":
    main()
