#!/bin/bash
# code shipped to the jobs: the scan module, the driver, and the v10 location YAMLs it reads.
# usage: make_tarball.sh OUTDIR   (writes OUTDIR/lj_eff_code.tar.gz; keep it out of the repo)
OUT=$(realpath "$1")
cd "$(dirname "$0")/../../../.."
tar czf "$OUT/lj_eff_code.tar.gz" \
  sidm/studies/run3_signal_validation/run3_lj_eff.py \
  sidm/studies/run3_signal_validation/gen_requirement.py \
  sidm/studies/run3_signal_validation/condor_lj_eff/lj_eff_job.py \
  sidm/configs/ntuples/signal_4mu_v10.yaml sidm/configs/ntuples/signal_2mu2e_v10.yaml
