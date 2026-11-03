#!/bin/bash
# condor wrapper: CMSSW python + a scratch venv from the repo's condor/requirements.txt
set -euo pipefail
MODE=$1; ERA=$2; NAMES=$3; R3N=$4; V10N=$5; OUT=$6; COMMIT=$7
cd "${_CONDOR_SCRATCH_DIR}"
source /cvmfs/cms.cern.ch/cmsset_default.sh
export SCRAM_ARCH=el9_amd64_gcc12
scramv1 project CMSSW CMSSW_14_1_0_pre4 >/dev/null
cd CMSSW_14_1_0_pre4/src; eval `scramv1 runtime -sh`; cd "${_CONDOR_SCRATCH_DIR}"
python3 -m venv sidm_venv --system-site-packages; source sidm_venv/bin/activate
export PIP_CACHE_DIR="${_CONDOR_SCRATCH_DIR}/pip_cache"
python -m pip install --quiet --upgrade pip setuptools wheel
python -m pip install --quiet --no-cache-dir -r requirements.txt
tar xzf lj_eff_code.tar.gz
python sidm/studies/run3_signal_validation/condor_lj_eff/lj_eff_job.py "$MODE" "$ERA" "$NAMES" "$R3N" "$V10N" "$OUT" "$COMMIT"
