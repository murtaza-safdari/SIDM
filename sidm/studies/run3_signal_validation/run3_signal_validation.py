"""Helpers for the Run 3 signal validation notebook.

Gen-level lifetime + kinematics on the produced Run 3 LLPNanoAOD, read straight from the
GenPart branches (schema-robust; no reco-object dependence). Proper cτ for a dark photon
(pdgId 32) = |decay vertex - production vertex| / (p/m), where the decay vertex is its lepton
daughter's production vertex. This is the same quantity the analysis's lifetime study measures.
"""
import subprocess
import tempfile
import os
import re
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
    """Return gen-level quantities for one sample:
    proper_mm and dR (ONE entry per dark photon), plus dpmass / lep_pt / lep_abseta (per lepton)."""
    fs = utilities.make_fileset([sample], era, max_files=nfiles, location_cfg=location_cfg,
                                replace_xcache=True)
    proper, dR, dpmass, lpt, leta = [], [], [], [], []
    nmu = nel = 0
    for f in fs[sample]["files"]:
        a = uproot.open(f)["Events"].arrays(BR)
        pid, vx, vy, vz = a["GenPart_pdgId"], a["GenPart_vx"], a["GenPart_vy"], a["GenPart_vz"]
        pt, eta, phi, mass, mom = (a["GenPart_pt"], a["GenPart_eta"], a["GenPart_phi"],
                                   a["GenPart_mass"], a["GenPart_genPartIdxMother"])
        mom_pid = pid[ak.where(mom >= 0, mom, 0)]
        sel = ((abs(pid) == 11) | (abs(pid) == 13)) & (mom >= 0) & (abs(mom_pid) == 32)  # leptons from a dark photon
        # per-lepton kinematics (correct as per-lepton)
        dpmass.append(ak.to_numpy(ak.flatten(mass[abs(pid) == 32])))
        lpt.append(ak.to_numpy(ak.flatten(pt[sel]))); leta.append(ak.to_numpy(ak.flatten(abs(eta[sel]))))
        nmu += int(ak.sum(sel & (abs(pid) == 13))); nel += int(ak.sum(sel & (abs(pid) == 11)))
        # per-dark-photon quantities: pair the two lepton daughters by their common mother index
        for ev_eta, ev_phi, ev_mom, ev_vx, ev_vy, ev_vz, ev_apt, ev_aeta, ev_am, ev_avx, ev_avy, ev_avz in zip(
                eta[sel], phi[sel], mom[sel], vx[sel], vy[sel], vz[sel],
                pt, eta, mass, vx, vy, vz):
            ev_eta = ak.to_numpy(ev_eta); ev_phi = ak.to_numpy(ev_phi); ev_mom = ak.to_numpy(ev_mom)
            ev_vx = ak.to_numpy(ev_vx); ev_vy = ak.to_numpy(ev_vy); ev_vz = ak.to_numpy(ev_vz)
            ev_apt = ak.to_numpy(ev_apt); ev_aeta = ak.to_numpy(ev_aeta); ev_am = ak.to_numpy(ev_am)
            ev_avx = ak.to_numpy(ev_avx); ev_avy = ak.to_numpy(ev_avy); ev_avz = ak.to_numpy(ev_avz)
            for m in np.unique(ev_mom):                       # one iteration per dark photon
                idx = np.where(ev_mom == m)[0]
                if len(idx) < 1:
                    continue
                # decay vertex = a daughter's production vertex; displacement from the dark photon's vertex
                dl = np.sqrt((ev_vx[idx[0]] - ev_avx[m])**2 + (ev_vy[idx[0]] - ev_avy[m])**2
                             + (ev_vz[idx[0]] - ev_avz[m])**2)  # cm
                bg = ev_apt[m] * np.cosh(ev_aeta[m]) / ev_am[m]
                if bg > 0:
                    proper.append(dl / bg * 10.0)             # cm -> mm
                if len(idx) == 2:
                    d_phi = np.arctan2(np.sin(ev_phi[idx[0]] - ev_phi[idx[1]]),
                                       np.cos(ev_phi[idx[0]] - ev_phi[idx[1]]))
                    dR.append(np.hypot(ev_eta[idx[0]] - ev_eta[idx[1]], d_phi))
    proper = np.array(proper); proper = proper[np.isfinite(proper)]
    return dict(sample=sample, era=era, proper_mm=proper, dpmass=np.concatenate(dpmass),
                lep_pt=np.concatenate(lpt), lep_abseta=np.concatenate(leta), dR=np.array(dR),
                nmu=nmu, nel=nel, meas=proper.mean(), nom=nominal_mm(sample))


def load_grid_scan(eos_dir="/store/group/lpcmetx/SIDM/run3_samplegen/validation", expected=180):
    """Load the full GEN-level scan RESULT lines from EOS into a list of dicts. Raises on an
    xrdcp failure or a short read, so a partial/failed copy cannot masquerade as the full grid."""
    with tempfile.TemporaryDirectory() as tmp:
        r = subprocess.run(["xrdcp", "-r", "-f", f"root://cmseos.fnal.gov/{eos_dir}", tmp],
                           capture_output=True, text=True)
        if r.returncode != 0:
            raise RuntimeError(f"xrdcp of {eos_dir} failed (valid proxy/kerberos?): {r.stderr.strip()}")
        root = os.path.join(tmp, os.path.basename(eos_dir))
        rows = []
        for fn in sorted(os.listdir(root)) if os.path.isdir(root) else []:
            m = re.search(r"RESULT (\S+) .*dpMass=([\d.-]+) psMass=([\d.-]+) .*meas/nom=([\d.-]+)",
                          open(os.path.join(root, fn)).read())
            if not m:
                continue
            name = m.group(1); p = name.split("_")
            rows.append(dict(chan=p[1].replace("BsTo2DpTo", ""), mbs=int(float(p[2][4:])),
                             mdp=float(p[3][4:].replace("p", ".")), ctau=float(p[4][5:].replace("p", ".")),
                             dpmass=float(m.group(2)), psmass=float(m.group(3)), meas_over_nom=float(m.group(4))))
    if len(rows) < expected:
        raise RuntimeError(f"grid scan returned {len(rows)} points, expected {expected} "
                           f"-- partial copy or missing validation/ files on EOS")
    return rows
