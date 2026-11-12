#!/bin/bash
# Condor worker for identity_files.py: one job list in, identity_<job>.csv out (generator branches; files copied to scratch).
# Environment: setup_env.sh.
set -euo pipefail
JOB=$1
cd "${_CONDOR_SCRATCH_DIR}"
source setup_env.sh
mkdir -p tmp
python identity_files.py --filelist "job_${JOB}.txt" --out "identity_${JOB}.csv" --ctau-module ctau_all_samples.py --tmp "${_CONDOR_SCRATCH_DIR}/tmp"
echo "rows $(($(wc -l < identity_${JOB}.csv) - 1)) errors $(awk -F, 'NR>1 && $2!=""' identity_${JOB}.csv | wc -l)"
date
