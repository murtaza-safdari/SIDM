# Sourced by run_census.sh and run_identity.sh in the condor scratch directory: Python environment as
# condor/run_job.sh builds it (CMSSW_14_1_0_pre4 python, venv in scratch), with only the uproot/awkward/numpy
# pins of condor/requirements.txt installed (the census needs nothing else).
echo "host=$(hostname) $(date)"
source /cvmfs/cms.cern.ch/cmsset_default.sh
export SCRAM_ARCH=el9_amd64_gcc12
scramv1 project CMSSW CMSSW_14_1_0_pre4 >/dev/null
cd CMSSW_14_1_0_pre4/src; eval `scramv1 runtime -sh`; cd "${_CONDOR_SCRATCH_DIR}"
python3 -m venv census_venv --system-site-packages
source census_venv/bin/activate
export PIP_CACHE_DIR="${_CONDOR_SCRATCH_DIR}/pip_cache"
python -m pip install --quiet --no-cache-dir $(grep -E '^(uproot|awkward|numpy)==' requirements.txt)
python -c "import uproot, awkward, numpy; print('uproot', uproot.__version__, 'awkward', awkward.__version__, 'numpy', numpy.__version__)"
