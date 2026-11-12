#!/bin/bash
# chain_job_v2.sh  MBs  MDp  CHAN  CTAU  ERA  NEV  JOBIDX  [TEST]
# v2 Run 3 SIDM signal chain on a Condor worker (el8 image, self-contained in scratch):
#   xrdcp gridpack -> GEN-SIM -> premix -> RECO/AODSIM -> PAT,NANO:@EXO (EXONanoAOD, CMSSW_16_0_10)
#   -> flatten (central NANO merge: Merge.mergeProcess(mergeNANO=True)) -> xrdcp --posc to EOS outputs_v2/<era>/.
# Releases, GTs, HLT menus, premix libraries, beamspots and procModifiers are the McM campaign definitions
# as of 2026-11-02 (all updated 2026-07-01); see PROVENANCE.md "v2". The GEN fragment, gridpacks and
# sample names are those of v1 (chain_job.sh).
# TEST=1 (test jobs only): the nano step also writes MINIAODSIM, both go to outputs_v2/test/<era>/.
# Only the EXONano file is kept per job. Writes <JOBNAME>.cksum (adler32 + size of the uploaded nano),
# the only file condor transfers back. Workers never touch /uscms_data; inputs come via xrdcp from EOS.
MBs=$1; MDp=$2; CHAN=$3; CTAU=$4; ERA=$5; NEV=$6; JOBIDX=$7; TEST=${8:-0}
MDpT=$(echo $MDp | sed 's/\./p/'); CTAUT=$(echo $CTAU | sed 's/\./p/')
NAME=SIDM_BsTo2DpTo${CHAN}_MBs-${MBs}_MDp-${MDpT}_ctau-${CTAUT}     # per-point (ctau in name -> output only)
GPBASE=SIDM_BsTo2DpTo${CHAN}_MBs-${MBs}_MDp-${MDpT}                 # per-cell (ctau-less -> gridpack)
EOS=root://cmseos.fnal.gov//store/group/lpcmetx/SIDM/run3_samplegen
EOSP=/store/group/lpcmetx/SIDM/run3_samplegen
if [ "$TEST" = 1 ]; then ODIR=outputs_v2/test/$ERA/$NAME; else ODIR=outputs_v2/$ERA/$NAME; fi
OUTNAME=${NAME}_${ERA}_${JOBIDX}.root
CKSUM=${NAME}_${ERA}_${JOBIDX}.cksum
FAILLOG=${NAME}_${ERA}_${JOBIDX}.faillog.gz
WORKDIR=$(pwd)
echo "######## CHAIN-v2 $NAME  ERA=$ERA NEV=$NEV JOBIDX=$JOBIDX TEST=$TEST ######## $(date)"
echo "host=$(hostname)  os=$(cat /etc/redhat-release 2>/dev/null)  scratch=$WORKDIR"
source /cvmfs/cms.cern.ch/cmsset_default.sh

statsize(){ xrdfs root://cmseos.fnal.gov stat "$1" 2>/dev/null | grep -oE 'Size:[[:space:]]*[0-9]+' | grep -oE '[0-9]+'; }
# scratch: condor removes the execute directory at job exit; the big files go earlier, on every exit path
cleanup(){ cd $WORKDIR 2>/dev/null && rm -f *.root *.tar.xz */src/*.root 2>/dev/null; }
trap cleanup EXIT
# condor transfers back the .cksum and the .faillog.gz; both always exist (a missing one would hold the job):
# .cksum says FAILED until the upload; .faillog.gz is empty unless a step fails, then it holds that step's whole
# log (gzipped, capped at the last 20 MB of text), so the exception block is readable even when TAUOLA's
# Fortran output flushes after it.
echo "$OUTNAME FAILED start" > $WORKDIR/$CKSUM
: | gzip > $WORKDIR/$FAILLOG
fail(){ echo "### FAIL at $1 (exit $2) ### $(date)"; echo "$OUTNAME FAILED $1 exit=$2" > $WORKDIR/$CKSUM
  if [ -n "$3" ] && [ -f "$3" ]; then
    sed -n '/Begin Fatal Exception/,/End Fatal Exception/p' "$3" | head -60
    tail -c 20000000 "$3" | gzip > $WORKDIR/$FAILLOG; echo "--- whole $(basename $3) in $FAILLOG ($(stat -c%s $WORKDIR/$FAILLOG) bytes gz)"
  fi; exit $2; }
# one summary line per cmsRun step, from its framework job report: wall time, events, peak RSS, output size
stepstat(){ # name fjr outfile
  python3 - "$@" <<'PYEOF'
import sys, os, re
name, fjr, out = sys.argv[1:4]
x = open(fjr).read() if os.path.exists(fjr) else ""
def v(k):
    m = re.search(r'Name="%s" Value="([^"]+)"' % k, x); return float(m.group(1)) if m else float("nan")
m = re.findall(r"<File>.*?<TotalEvents>(\d+)</TotalEvents>", x, re.S); nev = int(m[0]) if m else -1   # first output file
wall, rss = v("TotalJobTime"), v("PeakValueRss")
size = os.path.getsize(out) if os.path.exists(out) else -1
print("STEPSTAT step=%s nevents=%d wall_s=%.0f s_per_event=%.2f peakRSS_MB=%.0f out_bytes=%d bytes_per_event=%.0f" % (
    name, nev, wall, wall / nev if nev > 0 else float("nan"), rss, size, size / nev if nev > 0 and size > 0 else float("nan")))
PYEOF
}

# ---- idempotency: skip if a good output already exists on EOS ----
EXIST=$(statsize $EOSP/$ODIR/$OUTNAME)
if [ -n "$EXIST" ] && [ "$EXIST" -gt 100000 ]; then
  echo "### SKIP: output already exists ($EXIST bytes) -> $EOSP/$ODIR/$OUTNAME ###"
  echo "$OUTNAME skip size=$EXIST" > $WORKDIR/$CKSUM; exit 0
fi

# ---- per-era config: McM campaign definitions (read 2026-11-02; updated 2026-07-01) ----
# REL/ARCH: GEN-SIM and DR release; COND: GS/DR global tag; BS: beamspot; HLT: premix HLT menu;
# PU: premix library; ERAMOD: --era; PREMIXMOD/RECOMOD: procModifiers; NANOCOND: GT of the era's EXONano flow.
NANOREL=CMSSW_16_0_10; NANOARCH=el8_amd64_gcc13
case $ERA in
  2022)     REL=CMSSW_12_4_26; ARCH=el8_amd64_gcc10; COND=124X_mcRun3_2022_realistic_v12
            BS=Realistic25ns13p6TeVEarly2022Collision; HLT=2022v12; ERAMOD=Run3
            PREMIXMOD=premix_stage2,siPixelQualityRawToDigi; RECOMOD=siPixelQualityRawToDigi
            PU="dbs:/Neutrino_E-10_gun/Run3Summer21PrePremix-Summer22_124X_mcRun3_2022_realistic_v11-v2/PREMIX"
            NANOCOND=130X_mcRun3_2022_realistic_v5; FIX124=1;;
  2022EE)   REL=CMSSW_12_4_26; ARCH=el8_amd64_gcc10; COND=124X_mcRun3_2022_realistic_postEE_v1
            BS=Realistic25ns13p6TeVEarly2022Collision; HLT=2022v14; ERAMOD=Run3
            PREMIXMOD=premix_stage2,siPixelQualityRawToDigi; RECOMOD=siPixelQualityRawToDigi
            PU="dbs:/Neutrino_E-10_gun/Run3Summer21PrePremix-Summer22_124X_mcRun3_2022_realistic_v11-v2/PREMIX"
            NANOCOND=130X_mcRun3_2022_realistic_postEE_v6; FIX124=1;;
  2023)     REL=CMSSW_13_0_25; ARCH=el8_amd64_gcc11; COND=130X_mcRun3_2023_realistic_v15
            BS=Realistic25ns13p6TeVEarly2023Collision; HLT=2023v12; ERAMOD=Run3_2023
            PREMIXMOD=premix_stage2; RECOMOD=
            PU="dbs:/Neutrino_E-10_gun/Run3Summer21PrePremix-Summer23_130X_mcRun3_2023_realistic_v13-v1/PREMIX"
            NANOCOND=130X_mcRun3_2023_realistic_v15;;
  2023BPix) REL=CMSSW_13_0_25; ARCH=el8_amd64_gcc11; COND=130X_mcRun3_2023_realistic_postBPix_v6
            BS=Realistic25ns13p6TeVEarly2023Collision; HLT=2023v12; ERAMOD=Run3_2023
            PREMIXMOD=premix_stage2; RECOMOD=
            PU="dbs:/Neutrino_E-10_gun/Run3Summer21PrePremix-Summer23BPix_130X_mcRun3_2023_realistic_postBPix_v1-v1/PREMIX"
            NANOCOND=130X_mcRun3_2023_realistic_postBPix_v6;;
  2024)     REL=CMSSW_14_0_25; ARCH=el8_amd64_gcc12; COND=140X_mcRun3_2024_realistic_v26
            BS=DBrealistic; HLT=2024v14; ERAMOD=Run3_2024
            PREMIXMOD=premix_stage2; RECOMOD=
            PU="dbs:/Neutrino_E-10_gun/RunIIISummer24PrePremix-Premixlib2024_140X_mcRun3_2024_realistic_v26-v1/PREMIX"
            NANOCOND=150X_mcRun3_2024_realistic_v2;;
  *) echo "FATAL: unknown ERA=$ERA"; exit 2;;
esac
RECOPM=""; [ -n "$RECOMOD" ] && RECOPM="--procModifiers $RECOMOD"
MON="--customise Configuration/DataProcessing/Utils.addMonitoring"
# 2022/2022EE only (12_4_X AODSIM read by 16_0_10 EXONano):
#  RECOOUT: keep the DT 1D / CSC 2D muon rec hits in AODSIM (13_0_X+ AOD content has them by default; the EXONano
#           dt/cscMDSshower tables need them: ProductNotFound dt1DRecHits otherwise);
#  NANOFIX: FixMissingStreamerInfos for the 12_4_X HLT scouting classes (cms-sw/cmssw#41348, #45888;
#           FatalRootError on vector<Run3ScoutingVertex> otherwise).
RECOOUT=(); NANOFIX=""
if [ "${FIX124:-0}" = 1 ]; then
  RECOOUT=(--outputCommands "keep *_dt1DRecHits_*_*,keep *_csc2DRecHits_*_*")
  NANOFIX="--customise IOPool/Input/fixReading_12_4_X_Files.fixReading_12_4_X_Files"
fi
echo "era-config: fix124=${FIX124:-0} REL=$REL/$ARCH GT=$COND BS=$BS HLT=$HLT era=$ERAMOD premixmod=$PREMIXMOD recomod=${RECOMOD:-none} nano=$NANOREL/$NANOARCH nanoGT=$NANOCOND"

# ---- fetch gridpack (ctau-less, per cell) ----
GPNAME=${GPBASE}_gridpack.tar.xz
xrdcp -f -N $EOS/gridpacks/$GPNAME $WORKDIR/$GPNAME || { echo "### FAIL gridpack xrdcp ($GPNAME) ###"; exit 3; }
GP=$WORKDIR/$GPNAME

# RNG: populate() draws an independent OS-entropy seed per job -> statistically independent chunks
RNG='--customise_commands=from IOMC.RandomEngine.RandomServiceHelper import RandomNumberServiceHelper as _R; _R(process.RandomNumberGeneratorService).populate()'

# ================= STEP 1: GEN-SIM (fragment identical to v1) =================
export SCRAM_ARCH=$ARCH
cd $WORKDIR; scram project $REL >/dev/null 2>&1 || { echo "FAIL cmsrel $REL"; exit 4; }
cd $REL/src; eval `scram runtime -sh`
mkdir -p Configuration/GenProduction/python
cat > Configuration/GenProduction/python/${NAME}_cff.py <<EOF
import FWCore.ParameterSet.Config as cms
externalLHEProducer = cms.EDProducer("ExternalLHEProducer", args=cms.vstring('$GP'),
    nEvents=cms.untracked.uint32($NEV), numberOfParameters=cms.uint32(1),
    outputFile=cms.string('cmsgrid_final.lhe'),
    scriptName=cms.FileInPath('GeneratorInterface/LHEInterface/data/run_generic_tarball_cvmfs.sh'))
from Configuration.Generator.Pythia8CommonSettings_cfi import *
from Configuration.Generator.MCTunesRun3ECM13p6TeV.PythiaCP5Settings_cfi import *
from Configuration.Generator.PSweightsPythia.PythiaPSweightsSettings_cfi import *
generator = cms.EDFilter("Pythia8HadronizerFilter", maxEventsToPrint=cms.untracked.int32(0),
    pythiaPylistVerbosity=cms.untracked.int32(0), filterEfficiency=cms.untracked.double(1.0),
    pythiaHepMCVerbosity=cms.untracked.bool(False), comEnergy=cms.double(13600.),
    PythiaParameters=cms.PSet(pythia8CommonSettingsBlock, pythia8CP5SettingsBlock, pythia8PSweightsSettingsBlock,
        processParameters=cms.vstring('ParticleDecays:tau0Max = 1000.1','LesHouches:setLifetime = 2',
            'SLHA:useDecayTable = off','32:tau0 = $CTAU','32:tauCalc = off','32:mayDecay = on'),
        parameterSets=cms.vstring('pythia8CommonSettings','pythia8CP5Settings','pythia8PSweightsSettings','processParameters')))
ProductionFilterSequence = cms.Sequence(generator)
EOF
scram b -j4 > $WORKDIR/scramb1.log 2>&1 || fail STEP1_SCRAMB $? $WORKDIR/scramb1.log
echo "### STEP1 GEN-SIM n=$NEV ### $(date)"
cmsDriver.py Configuration/GenProduction/python/${NAME}_cff.py --python_filename gs_cfg.py \
  --eventcontent RAWSIM,LHE --datatier GEN-SIM,LHE --fileout file:step1_gensim.root \
  --conditions $COND --beamspot $BS --step LHE,GEN,SIM --geometry DB:Extended --era $ERAMOD $MON \
  "$RNG" --mc -n $NEV --nThreads 4 --no_exec > $WORKDIR/gs_drv.log 2>&1 || fail STEP1_CMSDRIVER $? $WORKDIR/gs_drv.log
cmsRun -j $WORKDIR/fjr1.xml gs_cfg.py > $WORKDIR/step1.log 2>&1 || fail STEP1_GENSIM $? $WORKDIR/step1.log
GS1=$(pwd)/step1_gensim.root; [ -f $GS1 ] || { echo "FAIL no gensim out"; exit 5; }
stepstat GS $WORKDIR/fjr1.xml $GS1

# ================= STEP 2: premix =================
echo "### STEP2 premix ### $(date)"
cmsDriver.py step1 --mc --eventcontent PREMIXRAW --datatier GEN-SIM-RAW --conditions $COND \
  --step DIGI,DATAMIX,L1,DIGI2RAW,HLT:$HLT --procModifiers $PREMIXMOD --datamix PreMix \
  --pileup_input "$PU" --era $ERAMOD --geometry DB:Extended $MON --nThreads 4 \
  --filein file:$GS1 --fileout file:step2_premix.root "$RNG" \
  --python_filename premix_cfg.py --no_exec -n -1 > $WORKDIR/premix_drv.log 2>&1 || fail STEP2_CMSDRIVER $? $WORKDIR/premix_drv.log
cmsRun -j $WORKDIR/fjr2.xml premix_cfg.py > $WORKDIR/step2.log 2>&1 || fail STEP2_PREMIX $? $WORKDIR/step2.log
[ -f step2_premix.root ] || { echo "FAIL no premix out"; exit 7; }
stepstat PREMIX $WORKDIR/fjr2.xml step2_premix.root
rm -f $GS1 step1_gensim_inLHE.root

# ================= STEP 3: RECO -> AODSIM =================
echo "### STEP3 RECO->AODSIM ### $(date)"
cmsDriver.py step2 --mc --eventcontent AODSIM --datatier AODSIM --conditions $COND \
  --step RAW2DIGI,L1Reco,RECO,RECOSIM $RECOPM --era $ERAMOD --geometry DB:Extended $MON "${RECOOUT[@]}" \
  --nThreads 4 --filein file:step2_premix.root --fileout file:step3_aodsim.root \
  --python_filename reco_cfg.py --no_exec -n -1 > $WORKDIR/reco_drv.log 2>&1 || fail STEP3_CMSDRIVER $? $WORKDIR/reco_drv.log
cmsRun -j $WORKDIR/fjr3.xml reco_cfg.py > $WORKDIR/step3.log 2>&1 || fail STEP3_RECO $? $WORKDIR/step3.log
AOD=$(pwd)/step3_aodsim.root; [ -f $AOD ] || { echo "FAIL no aodsim out"; exit 8; }
stepstat RECO $WORKDIR/fjr3.xml $AOD
rm -f step2_premix.root

# ================= STEP 4: PAT,NANO:@EXO (EXONanoAOD) in CMSSW_16_0_10 =================
echo "### STEP4 EXONano ### $(date)"
export SCRAM_ARCH=$NANOARCH
cd $WORKDIR; scram project $NANOREL >/dev/null 2>&1 || { echo "FAIL cmsrel $NANOREL"; exit 9; }
cd $NANOREL/src; eval `scram runtime -sh`
if [ "$TEST" = 1 ]; then EC=MINIAODSIM,NANOEDMAODSIM1; DT=MINIAODSIM,NANOAODSIM; FO=file:step4_mini.root
else EC=NANOEDMAODSIM1; DT=NANOAODSIM; FO=file:step4_nanoedm.root; fi
cmsDriver.py step3 --scenario pp --mc --step PAT,NANO:@EXO --datatier $DT --eventcontent $EC \
  --era $ERAMOD --conditions $NANOCOND $MON $NANOFIX --nThreads 4 --filein file:$AOD --fileout $FO \
  --python_filename nano_cfg.py --no_exec -n -1 > $WORKDIR/nano_drv.log 2>&1 || fail STEP4_CMSDRIVER $? $WORKDIR/nano_drv.log
cmsRun -j $WORKDIR/fjr4.xml nano_cfg.py > $WORKDIR/step4.log 2>&1 || fail STEP4_NANO $? $WORKDIR/step4.log
# the NANOEDM output: step4_nanoedm.root (production) or the second output of the test job (step4_mini_inNANOEDMAODSIM.root)
NEDM=$(ls step4_nanoedm.root step4_mini_inNANOEDMAODSIM*.root 2>/dev/null | head -1)
[ -n "$NEDM" ] || { echo "FAIL no NANOEDM out"; ls -la *.root; exit 11; }
stepstat PATNANO $WORKDIR/fjr4.xml $NEDM
[ "$TEST" = 1 ] && stepstat MINIAOD $WORKDIR/fjr4.xml step4_mini.root
rm -f $AOD

# ---- flatten the EDM nano exactly as the central NANO merge step does (NanoAODOutputModule) ----
python3 - "$NEDM" <<'PYEOF' > $WORKDIR/merge_drv.log 2>&1 || fail STEP5_MERGECFG $? $WORKDIR/merge_drv.log
import sys
from Configuration.DataProcessing.Merge import mergeProcess
p = mergeProcess("file:" + sys.argv[1], output_file="step5_nano.root", mergeNANO=True)
open("merge_cfg.py", "w").write(p.dumpPython())
PYEOF
cmsRun -j $WORKDIR/fjr5.xml merge_cfg.py > $WORKDIR/step5.log 2>&1 || fail STEP5_MERGE $? $WORKDIR/step5.log
NANO=$(pwd)/step5_nano.root; [ -f $NANO ] || { echo "FAIL no flat nano"; exit 11; }
stepstat FLATNANO $WORKDIR/fjr5.xml $NANO

# ---- sanity: event count = GEN-SIM events, required branches; HARD FAIL otherwise (no silent upload) ----
NGS=$(stepstat GS $WORKDIR/fjr1.xml /dev/null | sed -n 's/.*nevents=\([0-9-]*\).*/\1/p')
python3 - "$NANO" "$NGS" > $WORKDIR/sanity.txt 2>&1 <<'PYEOF'
import sys, ROOT
f = ROOT.TFile.Open(sys.argv[1]); ngs = int(sys.argv[2])
if not f or f.IsZombie(): print("SANITY FAIL: cannot open"); sys.exit(1)
t = f.Get("Events"); r = f.Get("Runs")
if not t or not r: print("SANITY FAIL: no Events/Runs tree"); sys.exit(1)
bs = {b.GetName() for b in t.GetListOfBranches()}
need = ["DSAMuon_pt", "DSAMuon_muonMatch1idx", "DSAMuon_nSegments", "DSAMuon_dxyPVSigned", "Muon_pt",
        "Electron_pt", "GenPart_vx", "GenPart_pdgId", "LHEPart_pt"]
miss = [b for b in need if b not in bs]; nhlt = sum(b.startswith("HLT_") for b in bs)
miss += [p + "*" for p in ("cscMDSCluster_", "dtMDSCluster_") if not any(b.startswith(p) for b in bs)]
print("SANITY nEvents=%d GSevents=%d branches=%d HLT_=%d missing=%s" % (t.GetEntries(), ngs, len(bs), nhlt, miss))
sys.exit(0 if (t.GetEntries() == ngs and ngs > 0 and not miss and nhlt > 0) else 1)
PYEOF
SRC=$?
cat $WORKDIR/sanity.txt
[ "$SRC" -eq 0 ] || fail STEP5_SANITY "$SRC" $WORKDIR/sanity.txt

# ================= STEP 6: xrdcp to EOS (persist-on-successful-close + size verify) =================
up(){ # local remote_dir remote_name
  xrdfs root://cmseos.fnal.gov mkdir -p $EOSP/$2 2>/dev/null
  xrdcp -f -N --posc "$1" $EOS/$2/$3 || { echo "FAIL xrdcp $3"; xrdfs root://cmseos.fnal.gov rm $EOSP/$2/$3 2>/dev/null; exit 12; }
  local L=$(stat -c%s "$1") R=$(statsize $EOSP/$2/$3)
  [ "$L" = "$R" ] || { echo "FAIL size mismatch $3 local=$L remote=$R"; xrdfs root://cmseos.fnal.gov rm $EOSP/$2/$3 2>/dev/null; exit 13; }
  echo "### uploaded $EOSP/$2/$3 ($R bytes) ###"
}
[ "$TEST" = 1 ] && up step4_mini.root $ODIR ${NAME}_${ERA}_${JOBIDX}_MINIAODSIM.root
up $NANO $ODIR $OUTNAME
echo "$OUTNAME adler32=$(xrdadler32 $NANO 2>/dev/null | awk '{print $1}') size=$(stat -c%s $NANO)" > $WORKDIR/$CKSUM
cat $WORKDIR/$CKSUM
echo "######## CHAIN-v2 DONE -> $EOSP/$ODIR/$OUTNAME ######## $(date)"
