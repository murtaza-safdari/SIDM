#!/bin/bash
set -euo pipefail
set -x

SAMPLE=$1
CHUNK=$2
FILELIST=$3
EOS_OUTDIR=$4
# Optional positional args 5/6: comma-separated channels + hist collections.
# Default to the historical "base" / "muon_base" so older submit.sub files
# that don't pass them keep working.
CHANNELS=${5:-base}
HIST_COLLECTIONS=${6:-muon_base}
# Optional positional arg 7: "weighted" keeps genWeight-weighted hists
# (omits --unweighted-hist). Default preserves the historical unweighted behavior.
HIST_WEIGHT_MODE=${7:-unweighted}
# Optional positional arg 8: "data" enables real-data processing and the blinding
# interlock (only SR-blinded channels + data-safe collections are accepted).
IS_DATA_MODE=${8:-mc}

echo "Host:"
hostname
date

echo "CONDOR scratch:"
echo "${_CONDOR_SCRATCH_DIR:-undefined}"

cd "${_CONDOR_SCRATCH_DIR}"

echo "Initial files:"
ls -lah

# ------------------------------------------------------------
# 1. Setup CMSSW base environment
# ------------------------------------------------------------
source /cvmfs/cms.cern.ch/cmsset_default.sh
export SCRAM_ARCH=el9_amd64_gcc12

scramv1 project CMSSW CMSSW_14_1_0_pre4
cd CMSSW_14_1_0_pre4/src
eval `scramv1 runtime -sh`
cd "${_CONDOR_SCRATCH_DIR}"

echo "Base Python:"
which python3
python3 --version

# ------------------------------------------------------------
# 2. Create venv inside Condor scratch
# ------------------------------------------------------------
python3 -m venv sidm_venv --system-site-packages
source sidm_venv/bin/activate

echo "Venv Python:"
which python
python --version

# Keep pip cache inside scratch
export PIP_CACHE_DIR="${_CONDOR_SCRATCH_DIR}/pip_cache"

python -m pip install --upgrade pip setuptools wheel
python -m pip install --no-cache-dir -r requirements.txt

# ------------------------------------------------------------
# 3. Verify dependencies before running analysis
# ------------------------------------------------------------
python - <<'PY'
import coffea
import awkward
import uproot
import fastjet
import vector
import yaml
print("coffea:", coffea.__version__)
print("All required imports worked")
PY

# ------------------------------------------------------------
# 4. Unpack SIDM code
# ------------------------------------------------------------
tar -xzf sidm_code.tar.gz

export PYTHONPATH="${_CONDOR_SCRATCH_DIR}:${PYTHONPATH:-}"

echo "After unpacking:"
ls -lah
ls -lah sidm
ls -lah condor

FILELIST_BASENAME=$(basename "${FILELIST}")
OUTFILE="${SAMPLE}_${CHUNK}.coffea"

echo "Using filelist:"
cat "${FILELIST_BASENAME}"

# ------------------------------------------------------------
# 5. Run processor using venv python
# ------------------------------------------------------------
UNWEIGHTED_FLAG="--unweighted-hist"
if [[ "${HIST_WEIGHT_MODE}" == "weighted" ]]; then
    UNWEIGHTED_FLAG=""
fi
IS_DATA_FLAG=""
if [[ "${IS_DATA_MODE}" == "data" ]]; then
    IS_DATA_FLAG="--is-data"
fi

python condor/run_sidm_chunk.py \
    --sample "${SAMPLE}" \
    --filelist "${FILELIST_BASENAME}" \
    --output "${OUTFILE}" \
    --chunksize 50000 \
    --workers 1 \
    --channels "${CHANNELS}" \
    --hist-collections "${HIST_COLLECTIONS}" \
    ${UNWEIGHTED_FLAG} ${IS_DATA_FLAG}

echo "Output:"
ls -lh "${OUTFILE}"

# ------------------------------------------------------------
# 6. Copy output to EOS
# ------------------------------------------------------------
echo "Copying to EOS:"
echo "${EOS_OUTDIR}/${OUTFILE}"

xrdcp -f "${OUTFILE}" "${EOS_OUTDIR}/${OUTFILE}"

XRDEXIT=$?
if [[ ${XRDEXIT} -ne 0 ]]; then
    echo "xrdcp failed with exit code ${XRDEXIT}"
    exit ${XRDEXIT}
fi

rm -f "${OUTFILE}"

date
echo "Done"