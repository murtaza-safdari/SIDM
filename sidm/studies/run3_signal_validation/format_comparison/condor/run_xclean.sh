#!/bin/bash
# Condor worker for xclean_rows.py: task list job_<N>.txt in, rows_<N>.tar (the per-file .npz) out.
# Environment as condor/run_job.sh builds it: CMSSW_14_1_0_pre4 python, venv in scratch from the repository's
# condor/requirements.txt and constraints.txt.
set -euo pipefail
JOB=$1
cd "${_CONDOR_SCRATCH_DIR}"
echo "host=$(hostname) $(date)"
source /cvmfs/cms.cern.ch/cmsset_default.sh
export SCRAM_ARCH=el9_amd64_gcc12
scramv1 project CMSSW CMSSW_14_1_0_pre4 >/dev/null
cd CMSSW_14_1_0_pre4/src; eval `scramv1 runtime -sh`; cd "${_CONDOR_SCRATCH_DIR}"
python3 -m venv xclean_venv --system-site-packages
source xclean_venv/bin/activate
export PIP_CACHE_DIR="${_CONDOR_SCRATCH_DIR}/pip_cache"
python -m pip install --quiet --no-cache-dir -r requirements.txt -c constraints.txt
tar -xzf xclean_code.tar.gz
python sidm/studies/run3_signal_validation/format_comparison/xclean_rows.py rows "job_${JOB}.txt" 1 > "rows_${JOB}.log"
tail -3 "rows_${JOB}.log"
echo "written $(find rows -name '*.npz' | wc -l) of $(wc -l < job_${JOB}.txt); failed $(grep -c FAILED rows_${JOB}.log || true)"
tar -cf "rows_${JOB}.tar" rows "rows_${JOB}.log"
date
