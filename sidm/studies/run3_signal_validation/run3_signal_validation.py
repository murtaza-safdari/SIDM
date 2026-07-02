"""Helpers for the Run 3 signal validation notebook.

Gen-level lifetime + kinematics on the produced Run 3 LLPNanoAOD, read straight from the
GenPart branches (schema-robust; no reco-object dependence). Proper cτ for a dark photon
(pdgId 32) = |decay vertex - production vertex| / (p/m), where the decay vertex is its lepton
daughter's production vertex. This is the same quantity the analysis's lifetime study measures.
"""
import subprocess
import numpy as np
import awkward as ak
import uproot
from sidm.tools import utilities

BR = ["GenPart_pdgId", "GenPart_vx", "GenPart_vy", "GenPart_vz",
      "GenPart_pt", "GenPart_eta", "GenPart_phi", "GenPart_mass", "GenPart_genPartIdxMother"]


def nominal_mm(sample):
    """Nominal proper cτ in mm parsed from the sample name (e.g. '..._0p48mm' -> 0.48)."""
    return float(sample.split("_")[-1].replace("mm", "").replace("p", "."))


def analyze(sample, era, location_cfg, nfiles=4):
    """Return per-dark-photon gen-level quantities for one sample:
    proper_mm, dpmass, lepton pt/eta, and the two-lepton opening angle dR (collimation)."""
    fs = utilities.make_fileset([sample], era, max_files=nfiles, location_cfg=location_cfg)
    proper, dpmass, lpt, leta, dR = [], [], [], [], []
    nmu = nel = 0
    for f in fs[sample]["files"]:
        a = uproot.open(f)["Events"].arrays(BR)
        pid, vx, vy, vz = a["GenPart_pdgId"], a["GenPart_vx"], a["GenPart_vy"], a["GenPart_vz"]
        pt, eta, phi, mass, mom = (a["GenPart_pt"], a["GenPart_eta"], a["GenPart_phi"],
                                   a["GenPart_mass"], a["GenPart_genPartIdxMother"])
        mom_pid = pid[ak.where(mom >= 0, mom, 0)]
        sel = ((abs(pid) == 11) | (abs(pid) == 13)) & (mom >= 0) & (abs(mom_pid) == 32)  # leptons from a dark photon
        midx = mom[sel]
        dl = np.sqrt((vx[sel] - vx[midx])**2 + (vy[sel] - vy[midx])**2 + (vz[sel] - vz[midx])**2)  # cm
        bg = pt[midx] * np.cosh(eta[midx]) / mass[midx]
        proper.append(ak.to_numpy(ak.flatten(dl / bg)) * 10.0)   # cm -> mm
        lpt.append(ak.to_numpy(ak.flatten(pt[sel]))); leta.append(ak.to_numpy(ak.flatten(abs(eta[sel]))))
        dpmass.append(ak.to_numpy(ak.flatten(mass[abs(pid) == 32])))
        # opening angle of the two leptons of each dark photon (collimation)
        le = eta[sel]; lp = phi[sel]
        de = le[:, ::2] - le[:, 1::2] if False else None  # (kept simple below)
        # pair leptons per event by dark-photon mother
        for ev_eta, ev_phi, ev_mom in zip(le, lp, midx):
            ev_eta = ak.to_numpy(ev_eta); ev_phi = ak.to_numpy(ev_phi); ev_mom = ak.to_numpy(ev_mom)
            for m in np.unique(ev_mom):
                idx = np.where(ev_mom == m)[0]
                if len(idx) == 2:
                    d_eta = ev_eta[idx[0]] - ev_eta[idx[1]]
                    d_phi = np.arctan2(np.sin(ev_phi[idx[0]] - ev_phi[idx[1]]),
                                       np.cos(ev_phi[idx[0]] - ev_phi[idx[1]]))
                    dR.append(np.hypot(d_eta, d_phi))
        nmu += int(ak.sum(sel & (abs(pid) == 13))); nel += int(ak.sum(sel & (abs(pid) == 11)))
    proper = np.concatenate(proper); proper = proper[np.isfinite(proper)]
    return dict(sample=sample, era=era, proper_mm=proper, dpmass=np.concatenate(dpmass),
                lep_pt=np.concatenate(lpt), lep_abseta=np.concatenate(leta), dR=np.array(dR),
                nmu=nmu, nel=nel, meas=proper.mean(), nom=nominal_mm(sample))


def load_grid_scan(eos_dir="/store/group/lpcmetx/SIDM/run3_samplegen/validation"):
    """Load the full 180-point GEN-level scan RESULT lines from EOS into a list of dicts."""
    import re, tempfile, os
    tmp = tempfile.mkdtemp()
    subprocess.run(["xrdcp", "-r", "-f", f"root://cmseos.fnal.gov/{eos_dir}", tmp],
                   capture_output=True)
    rows = []
    root = os.path.join(tmp, os.path.basename(eos_dir))
    for fn in os.listdir(root) if os.path.isdir(root) else []:
        txt = open(os.path.join(root, fn)).read()
        m = re.search(r"RESULT (\S+) .*dpMass=([\d.-]+) psMass=([\d.-]+) .*meas/nom=([\d.-]+)", txt)
        if not m:
            continue
        name = m.group(1); p = name.split("_")
        rows.append(dict(chan=p[1].replace("BsTo2DpTo", ""), mbs=int(float(p[2][4:])),
                         mdp=float(p[3][4:].replace("p", ".")), ctau=float(p[4][5:].replace("p", ".")),
                         dpmass=float(m.group(2)), psmass=float(m.group(3)), meas_over_nom=float(m.group(4))))
    return rows
