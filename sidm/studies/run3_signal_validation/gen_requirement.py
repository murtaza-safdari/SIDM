"""The 2018 v10 generator-level filter, as stored in the GEN process PSet of the v10 files:

  genfilter = GenParticleSelector, src genParticlesForFilter, cut
    (abs(pdgId)==11 || abs(pdgId)==13) && abs(eta)<2.4 && (vertex.rho<740. && abs(vertex.Z)<960.)
    && pt>4. && isLastCopy() && isPromptFinalState() && fromHardProcessFinalState()
  gencount  = CandViewCountFilter, src genfilter, minNumber 4

Emulated on the nano GenPart table: status 1 with the statusFlags bits isPrompt (0), fromHardProcess (8)
and isLastCopy (13), i.e. the post-FSR final-state copies; the vertex is the stored (smeared) one,
while the filter read genParticlesForFilter. Shared by era_fingerprint.py, notebook_figures.py and
run3_lj_eff.py (no dependencies beyond numpy/awkward, so it ships to condor jobs as is).
"""
import numpy as np
import awkward as ak

PT, ETA, RHO, VZ, NMIN = 4.0, 2.4, 740.0, 960.0, 4          # GeV, -, cm, cm, leptons
BRANCHES = ["GenPart_pdgId", "GenPart_status", "GenPart_statusFlags", "GenPart_pt", "GenPart_eta",
            "GenPart_vx", "GenPart_vy", "GenPart_vz"]
TEXT = r"$\geq$4 prompt final-state e/$\mu$, $|\eta|$ < 2.4, $p_{T}$ > 4 GeV"


def leptons(a):
    """Per particle: a final-state lepton of the kind the filter counts, before the kinematic cuts."""
    pid, fl = abs(a["GenPart_pdgId"]), a["GenPart_statusFlags"]
    return (((pid == 11) | (pid == 13)) & (a["GenPart_status"] == 1) & ((fl & 1) > 0)
            & ((fl & (1 << 8)) > 0) & ((fl & (1 << 13)) > 0))


def passes(a):
    """Per event: the 2018 filter decision (numpy bool array)."""
    sel = leptons(a) & (a["GenPart_pt"] > PT) & (abs(a["GenPart_eta"]) < ETA) \
        & (np.hypot(a["GenPart_vx"], a["GenPart_vy"]) < RHO) & (abs(a["GenPart_vz"]) < VZ)
    return ak.to_numpy(ak.sum(sel, axis=1) >= NMIN)
