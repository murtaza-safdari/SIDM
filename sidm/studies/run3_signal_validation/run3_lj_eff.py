#!/usr/bin/env python3
# Analysis-faithful signal-efficiency: reproduce the SidmProcessor lepton-jet reconstruction
# (build_lepton_jets: anti-kt R=0.4 over muons+DSAmuons+electrons+photons), split into
#   mu_ljs  = jets with >=1 muon/DSA constituent      (yesMu)
#   egm_ljs = jets with 0 muon constituents           (noMu; electron and/or photon)
#
# SCOPE: applies the SIDM LJ-source object selection (selections.yaml obj_ljsource_cuts): PF muon
# looseID+pT>5, DSA pT>10+displacedID + segment-match cross-cleaning (frac<0.34 of DSA segments
# shared with a PF muon -> removed), electron pT>10, photon pT>20+cutBased-loose. (Electron
# MVANonIsoWPL and the custom-cutbased photon ID are approximated; the egm leg is gen-matched.)
# This collapses the mu-LJ multiplicity to the expected count in BOTH eras and makes the exact SR
# LJ-count meaningful. Residual per-era object-reco differences (2018 vs 2022 detector) are expected
# and are what per-era efficiencies/SFs handle in the analysis; the definitive SR yield is the full
# SidmProcessor. Applied identically to Run3 and v10.
#
# Efficiencies (per point), all counts stored so binomial (Clopper-Pearson) errors are exact:
#   INCLUSIVE (clustering-step): 4Mu P(>=2 mu_lj); 2Mu2E P(>=1 mu_lj & >=1 egm_lj)
#   EXACT-COUNT (matches the real SR "=2 muLJs" / "=1 muLJ & =1 egmLJ"): 4Mu (n_mu==2 & n_egm==0);
#       2Mu2E (n_mu==1 & n_egm==1 & the egm_lj gen-matched dR<0.4 to a gen electron from a dark
#       photon -- rejects soft/pileup photons the real SR would not accept).
#   LJ multiplicity distributions (n_mu, n_egm) per point -> extra-LJ migration is visible.
# Denominators: raw (all gen events) and acc (BOTH dark-photon lepton pairs in |eta|<2.4 & pT>1;
#   the same gen acceptance the v10 production applied, imposed identically on both eras).
import sys, subprocess, os, re, json, numpy as np, awkward as ak, uproot, vector, fastjet
from collections import Counter
vector.register_awkward()

MU_PT, EG_PT, ETA, EL_ETA, GEN_PT, DR_MATCH = 3.0, 1.0, 2.4, 2.5, 1.0, 0.4
# LOOSE = clustering-GEOMETRY mode (muon/dsa pT>3, e/gamma pT>1, no object ID, no DSA cross-clean):
# isolates whether the GENERATED final-state leptons cluster into the expected lepton jets the same
# way in Run3 and Run2 (a sample property, driven by collimation/kinematics, not detector ID).
# Default (not LOOSE) = the full SIDM LJ-source object selection + DSA segment-match cross-cleaning.
LOOSE = "loose" in sys.argv
JETDEF = fastjet.JetDefinition(fastjet.antikt_algorithm, 0.4)
MU_M, EL_M = 0.10566, 0.000511
GBR = ["GenPart_pdgId", "GenPart_pt", "GenPart_eta", "GenPart_phi", "GenPart_genPartIdxMother"]

def gen_acc_mask(a):
    """per-event: BOTH dark-photon lepton pairs in |eta|<2.4 & pT>1 (>=2 dark photons with >=2
    in-acceptance daughters). Channel-symmetric (4Mu: both mu pairs; 2Mu2E: mu pair AND ee pair)."""
    pid, gpt, geta, mom = a["GenPart_pdgId"], a["GenPart_pt"], a["GenPart_eta"], a["GenPart_genPartIdxMother"]
    mom_pid = pid[ak.where(mom >= 0, mom, 0)]
    is_lep = (abs(pid) == 11) | (abs(pid) == 13)
    sel = is_lep & (mom >= 0) & (abs(mom_pid) == 32) & (gpt > GEN_PT) & (abs(geta) < ETA)
    out = np.zeros(len(pid), dtype=bool)
    for i, moms in enumerate(ak.to_list(mom[sel])):
        if len(moms) < 4:
            continue
        out[i] = sum(1 for v in Counter(moms).values() if v >= 2) >= 2
    return out

def jet_etaphi(jets):
    px, py, pz = jets.px, jets.py, jets.pz
    p = np.sqrt(px**2 + py**2 + pz**2)
    eta = np.arctanh(ak.where((p > 0) & (abs(pz) < p), pz / p, 0.0))
    phi = np.arctan2(py, px)
    return eta, phi

def lj_counts(files, is2e):
    NMU = []; NEGM = []; ACC = []; MATCH = []; egm_e = []; egm_g = []
    n = 0
    for f in files:
        try:
            ev = uproot.open(f)["Events"]
        except Exception:
            continue
        keys = set(ev.keys())
        br = ["Muon_pt","Muon_eta","Muon_phi","Muon_looseId","DSAMuon_pt","DSAMuon_eta","DSAMuon_phi",
              "DSAMuon_displacedID","DSAMuon_muonMatch1","DSAMuon_nSegments",
              "Electron_pt","Electron_eta","Electron_phi","Photon_pt","Photon_eta","Photon_phi",
              "Photon_cutBased"] + GBR
        a = ev.arrays([b for b in br if b in keys])
        ne = len(a); n += ne
        def blk(coll, m, tid):
            # Apply the SIDM LJ-source object selection (selections.yaml obj_ljsource_cuts).
            if f"{coll}_pt" not in keys:
                z = ak.Array([[]] * ne)
                return ak.zip({"pt": z, "eta": z, "phi": z, "mass": z, "tid": z}, with_name="Momentum4D")
            pt, eta, phi = a[f"{coll}_pt"], a[f"{coll}_eta"], a[f"{coll}_phi"]
            if LOOSE:                               # clustering-geometry mode: loose kinematic preselection only
                keep = (pt > (MU_PT if tid in (3, 8) else EG_PT)) & (abs(eta) < (ETA if tid in (3, 8) else EL_ETA))
            elif coll == "Muon":                    # PF muons: looseID, pT>5, |eta|<2.4
                keep = (pt > 5) & (abs(eta) < ETA) & (a["Muon_looseId"] == 1)
            elif coll == "DSAMuon":                 # DSA: pT>10, |eta|<2.4, displacedID, + segment-match cross-cleaning
                keep = (pt > 10) & (abs(eta) < ETA) & (a["DSAMuon_displacedID"] > 0)
                ns = a["DSAMuon_nSegments"]; frac = ak.where(ns > 0, a["DSAMuon_muonMatch1"] / ns, 0.0)
                keep = keep & (frac < 0.34)         # remove DSA that share >=34% segments with a PF muon (SR "all" veto term)
            elif coll == "Electron":                # e: pT>10, |eta|<2.4 (SR adds MVANonIsoWPL; egm leg is gen-matched)
                keep = (pt > 10) & (abs(eta) < ETA)
            else:                                    # Photon: pT>20, |eta|<2.5, cutBased loose (SR uses a custom cutbased ID)
                keep = (pt > 20) & (abs(eta) < EL_ETA) & (a["Photon_cutBased"] >= 1)
            pt, eta, phi = pt[keep], eta[keep], phi[keep]
            return ak.zip({"pt": pt, "eta": eta, "phi": phi,
                           "mass": ak.zeros_like(pt) + m, "tid": ak.zeros_like(pt) + tid}, with_name="Momentum4D")
        inp = ak.concatenate([blk("Muon", MU_M, 3), blk("DSAMuon", MU_M, 8),
                              blk("Electron", EL_M, 2), blk("Photon", 0.0, 4)], axis=1)
        cs = fastjet.ClusterSequence(inp, JETDEF)
        consts = cs.constituents(); jets = cs.inclusive_jets()
        tid = consts.tid
        muon_n = ak.sum((tid == 3) | (tid == 8), axis=-1)
        ele_n = ak.sum(tid == 2, axis=-1); pho_n = ak.sum(tid == 4, axis=-1)
        is_mu = muon_n > 0; is_egm = muon_n == 0
        NMU.append(ak.to_numpy(ak.sum(is_mu, axis=-1)))
        NEGM.append(ak.to_numpy(ak.sum(is_egm, axis=-1)))
        ACC.append(gen_acc_mask(a))
        egm_e.append(ak.to_numpy(ak.flatten(ele_n[is_egm]))); egm_g.append(ak.to_numpy(ak.flatten(pho_n[is_egm])))
        # gen-match the egm leg (2Mu2E): any egm_lj within dR<0.4 of a gen electron from a dark photon
        if is2e:
            jeta, jphi = jet_etaphi(jets)
            egm_eta, egm_phi = jeta[is_egm], jphi[is_egm]
            pid, gpt, geta, gphi, mom = (a["GenPart_pdgId"], a["GenPart_pt"], a["GenPart_eta"],
                                         a["GenPart_phi"], a["GenPart_genPartIdxMother"])
            mom_pid = pid[ak.where(mom >= 0, mom, 0)]
            ge = (abs(pid) == 11) & (mom >= 0) & (abs(mom_pid) == 32)
            pair = ak.cartesian({"j": ak.zip({"eta": egm_eta, "phi": egm_phi}),
                                 "g": ak.zip({"eta": geta[ge], "phi": gphi[ge]})}, axis=1)
            dphi = (pair.j.phi - pair.g.phi + np.pi) % (2 * np.pi) - np.pi
            dR = np.sqrt((pair.j.eta - pair.g.eta) ** 2 + dphi ** 2)
            MATCH.append(ak.to_numpy(ak.fill_none(ak.any(dR < DR_MATCH, axis=1), False)))
        else:
            MATCH.append(np.zeros(ne, dtype=bool))
    if n == 0:
        return None
    nmu = np.concatenate(NMU); negm = np.concatenate(NEGM); acc = np.concatenate(ACC); match = np.concatenate(MATCH)
    # INCLUSIVE (clustering-geometry, robust to extra fake LJs): 4Mu >=2 muLJ; 2Mu2E >=1 muLJ &
    # >=1 gen-matched egmLJ (egm gen-matched to the gen ee dark photon -> rejects soft/PU photons).
    incl = (nmu >= 2) if not is2e else ((nmu >= 1) & (negm >= 1) & match)
    # EXACT-COUNT (the literal SR; NOT a faithful proxy here -- dominated by un-cleaned DSA muLJs the
    # real SR removes via DSA ID + segment-match veto + PF overlap removal, none reproduced). Kept
    # only so the extra-LJ multiplicity migration is quantified, not as a validated number.
    excl = ((nmu == 2) & (negm == 0)) if not is2e else ((nmu == 1) & (negm == 1) & match)
    def eff(mask, denom):
        k = int((mask & denom).sum()); nn = int(denom.sum())
        return dict(k=k, n=nn, p=(k / nn if nn else 0.0))
    allm = np.ones(n, dtype=bool)
    d = dict(n=int(n), n_acc=int(acc.sum()),
             incl_raw=eff(incl, allm), incl_acc=eff(incl, acc),
             excl_raw=eff(excl, allm), excl_acc=eff(excl, acc),
             nmu_hist=[int((nmu == k).sum()) for k in range(4)] + [int((nmu >= 4).sum())],
             negm_hist=[int((negm == k).sum()) for k in range(4)] + [int((negm >= 4).sum())])
    ee = np.concatenate(egm_e) if egm_e else np.array([]); gg = np.concatenate(egm_g) if egm_g else np.array([])
    if is2e and len(ee):
        d["egm_photon_only_frac"] = float(((ee == 0) & (gg > 0)).mean())
    return d

def run3_files(name, nchunks):
    REDIR = "root://cmseos.fnal.gov/"; d = "/store/group/lpcmetx/SIDM/run3_samplegen/outputs/2022/" + name
    fl = subprocess.run(["xrdfs","root://cmseos.fnal.gov","ls",d],capture_output=True,text=True,timeout=60).stdout.split()
    fl = [REDIR + f for f in fl if f.endswith(".root")]
    return fl[:nchunks] if nchunks else fl

def v10_files(key, chan, nfiles):
    import yaml
    cfg = f"/uscms_data/d3/murtazas/SIDM-wt-run3val/sidm/configs/ntuples/signal_{chan}_v10.yaml"
    blk = yaml.safe_load(open(cfg))["llpNanoAOD_v2"]; entry = blk["samples"][key]
    urls = [blk["path"] + entry.get("path", "") + f for f in entry["files"]]
    urls = [u.replace("root://xcache//", "root://cmseos.fnal.gov//") for u in urls]
    return urls[:nfiles] if nfiles else urls

def r3_to_key(name):
    m = re.match(r"SIDM_BsTo2DpTo(\w+?)_MBs-(\d+)_MDp-([\dp]+)_ctau-([\dp]+)", name)
    return f"{m.group(1)}_{m.group(2)}GeV_{m.group(3)}GeV_{m.group(4)}mm", m.group(1)

if __name__ == "__main__":
    mode = sys.argv[1] if len(sys.argv) > 1 else "grid"
    R3_N = int(sys.argv[2]) if len(sys.argv) > 2 else 0        # 0 = ALL 120 chunks (100k events)
    V10_N = int(sys.argv[3]) if len(sys.argv) > 3 else 25
    if mode == "smoke":
        nm = "SIDM_BsTo2DpTo2Mu2E_MBs-200_MDp-1p2_ctau-4p8"; key = "2Mu2E_200GeV_1p2GeV_4p8mm"
        r3 = lj_counts(run3_files(nm, 6), True); v10 = lj_counts(v10_files(key, "2mu2e", 3), True)
        for lab, d in [("R3", r3), ("v10", v10)]:
            print(f"{lab}: n={d['n']} nacc={d['n_acc']} incl_acc={d['incl_acc']['p']*100:.1f}% "
                  f"excl_acc={d['excl_acc']['p']*100:.1f}% (k={d['excl_acc']['k']}/{d['excl_acc']['n']}) "
                  f"negm_hist={d['negm_hist']} gmatch_in_excl={'yes'}")
        sys.exit(0)
    BASE = "/store/group/lpcmetx/SIDM/run3_samplegen/outputs/2022"
    subdirs = subprocess.run(["xrdfs","root://cmseos.fnal.gov","ls",BASE],capture_output=True,text=True).stdout.split()
    ready = []
    for d in subdirs:
        if "/SIDM_" not in d:
            continue
        nc = len([x for x in subprocess.run(["xrdfs","root://cmseos.fnal.gov","ls",d],capture_output=True,text=True).stdout.split() if x.endswith(".root")])
        if nc >= 120:
            ready.append(d.split("/")[-1])
    out = []
    for name in sorted(ready):
        key, chan = r3_to_key(name); is2e = chan == "2Mu2E"; cl = chan.lower()
        m = re.match(r"SIDM_BsTo2DpTo(\w+?)_MBs-(\d+)_MDp-([\dp]+)_ctau-([\dp]+)", name)
        mbs = int(m.group(2)); mdp = float(m.group(3).replace("p", ".")); ctau = float(m.group(4).replace("p", "."))
        r3 = lj_counts(run3_files(name, R3_N), is2e)
        try:
            v10 = lj_counts(v10_files(key, cl, V10_N), is2e)
        except Exception as e:
            v10 = None; print(f"  v10 fail {key}: {e}")
        out.append(dict(chan=chan, mbs=mbs, mdp=mdp, ctau=ctau, key=key, run3=r3, v10=v10))
        v = f"v10 excl_acc={v10['excl_acc']['p']*100:5.1f}%" if v10 else "v10 --"
        print(f"{key:28s} R3 incl_acc={r3['incl_acc']['p']*100:5.1f}% excl_acc={r3['excl_acc']['p']*100:5.1f}% "
              f"(n_acc={r3['n_acc']:6d}) | {v}")
    OUT = "/uscms_data/d3/murtazas/review_out/lj_eff_%s_2022.json" % ("geom" if LOOSE else "real")
    json.dump(out, open(OUT, "w"), indent=1)
    print(f"\nwrote {OUT}: {len(out)} points")
