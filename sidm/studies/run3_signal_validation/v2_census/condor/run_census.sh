#!/bin/bash
# Condor worker for census_files.py: one job list in, census_<job>.csv and branches_<job>.json out.
# Environment: setup_env.sh.
set -euo pipefail
JOB=$1
cd "${_CONDOR_SCRATCH_DIR}"
source setup_env.sh
mkdir -p tmp
python census_files.py --filelist "job_${JOB}.txt" --out "census_${JOB}.csv" --branches "branches_${JOB}.json" \
    --fingerprint era_fingerprint.py --tmp "${_CONDOR_SCRATCH_DIR}/tmp"
echo "rows $(($(wc -l < census_${JOB}.csv) - 1)) errors $(awk -F, 'NR>1 && $10!=""' census_${JOB}.csv | wc -l)"
date
