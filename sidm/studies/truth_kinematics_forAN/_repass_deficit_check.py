"""Decompose the gen-filter re-pass deficit on a low-mass v10 sample.

The genFilterEmulation cutflow passes 0.84-1.00 of the (already-filtered) v10
events, lowest at the lightest masses. This script re-applies the filter's cut
string to the stored GenPart record of one mXX = 100 GeV sample and, for the
failing events, identifies which cut is responsible and where the failing
leptons sit. Result quoted in truth_kinematics_forAN.ipynb: all four signal
leptons are present in every failing event, the pT > 5 GeV requirement accounts
for 100% of the failures, and the failing leptons cluster at 4.3-4.7 GeV, with
eta, vertex, and statusFlags playing no role. Run from the repo root:
    python sidm/studies/truth_kinematics_forAN/_repass_deficit_check.py
"""
import subprocess
import numpy as np
import uproot
import awkward as ak

BASE = ("/store/group/lpcmetx/SIDM/ULSignalSamples/2018_v10/BsTo2DpTo2Mu2e/"
        "CutDecayFalse_SIDM_BsTo2DpTo2Mu2e_MBs-100_MDp-1p2_ctau-9p6_v3/LLPnanoAODv2/")

ls = subprocess.run(["xrdfs", "cmseos.fnal.gov", "ls", BASE],
                    capture_output=True, text=True).stdout.split()
fn = "root://cmseos.fnal.gov/" + [x for x in ls if x.endswith(".root")][0]
t = uproot.open(fn + ":Events", timeout=120)
a = t.arrays(["GenPart_pdgId", "GenPart_status", "GenPart_statusFlags",
              "GenPart_pt", "GenPart_eta", "GenPart_vx", "GenPart_vy", "GenPart_vz"],
             entry_stop=6000)

pid, st, sf = abs(a["GenPart_pdgId"]), a["GenPart_status"], a["GenPart_statusFlags"]

def bit(x, b):
    return (x & (1 << b)) > 0

lep = ((pid == 11) | (pid == 13)) & (st == 1)
flags = bit(sf, 0) & bit(sf, 8) & bit(sf, 13)
rho = np.sqrt(a["GenPart_vx"]**2 + a["GenPart_vy"]**2)
kin = ((a["GenPart_pt"] > 5) & (abs(a["GenPart_eta"]) < 2.4)
       & (rho < 740) & (abs(a["GenPart_vz"]) < 960))

full = lep & flags & kin
npass = ak.sum(full, axis=1)
fail = npass < 4
print(f"events: {len(npass)}, re-pass = {1 - ak.mean(fail):.4f}")

nf = ak.sum(lep & flags, axis=1)[fail]
print("failing events: flagged-lepton count distribution:",
      np.bincount(nf.to_numpy(), minlength=6)[:6])

sub = lep & flags & ~kin
for name, m in [("pt<=5", a["GenPart_pt"] <= 5),
                ("|eta|>=2.4", abs(a["GenPart_eta"]) >= 2.4),
                ("rho>=740", rho >= 740),
                ("|vz|>=960", abs(a["GenPart_vz"]) >= 960)]:
    frac = ak.sum((sub & m)[fail]) / max(ak.sum(sub[fail]), 1)
    print(f"  failing leptons killed by {name}: {frac:.3f}")

pt_fail = ak.flatten(a["GenPart_pt"][sub][fail]).to_numpy()
print("failing-lepton pT percentiles 25/50/75:",
      np.percentile(pt_fail, [25, 50, 75]).round(2), "GeV")
