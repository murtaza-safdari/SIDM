#!/usr/bin/env python3
"""Per-DSA-muon rows for the v1 / v2 comparison of the DSA-PF segment-match veto ("all", "all + charge").

The veto is evaluated with the cut lambdas of sidm/definitions/cuts.py on a good_matched_muons collection built
as sidm_processor.py builds it: the PF muons in the five DSAMuon_muonMatchKidx slots, ak.drop_none, then the
"muons" cuts of ljsource_cuts (selections.yaml) applied in order. Only "pre" DSA muons are written (pT > 10 GeV,
abs(eta) < 2.4, displaced ID: the ljsource dsaMuons cuts before the veto).

nSegments variants (the veto's denominator):
  stored   DSAMuon_nSegments as stored (correct in v1; twice nDTSegments in v2)
  formula  nDTSegments + ceil(trkNumCSCHits / 6) (the v2 rebuild)
  plus1    formula + 1 for DSA muons with trkNumCSCHits > 0 (bound on the rebuild's undercount)

Per file one .npz: per "pre" DSA muon the kinematics, the segment and hit counts, the five raw slot counts, the
number of good matched PF muons, the veto decisions, and the generated Lxy (cm) and dR of the nearest gen muon
from a dark-photon decay; per event the number of DSA muons, of "pre" DSA muons and of PF muons passing the
ljsource muon cuts.

Usage: xclean_rows.py OUTDIR TASKFILE [NPROC]
  TASKFILE lines: VERSION ERA SAMPLE URL  (see make_tasks.py); output OUTDIR/VERSION/ERA/SAMPLE/<file stem>.npz
"""
import os
import sys
from multiprocessing import Pool

import awkward as ak
import numpy as np
import uproot
import yaml

REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..", ".."))
sys.path.insert(0, REPO)
from sidm.definitions.cuts import obj_cut_defs  # noqa: E402
from sidm.tools.utilities import dR_outer  # noqa: E402

SEL = yaml.safe_load(open(os.path.join(REPO, "sidm/configs/selections.yaml")))
MU_CUTS = SEL["ljsource_cuts"]["obj_cuts"]["muons"]
DSA_F = ["pt", "eta", "phi", "charge", "displacedID", "nSegments", "nDTSegments", "nCSCSegments", "trkNumCSCHits",
         "trkNumDTHits", "outerEta", "outerPhi"]
SLOTS = [f"muonMatch{k}" for k in range(1, 6)] + [f"muonMatch{k}idx" for k in range(1, 6)]
MU_F = ["pt", "eta", "phi", "charge", "looseId", "trkNumDTHits", "trkNumCSCHits", "outerEta", "outerPhi"]
GEN_F = ["pdgId", "genPartIdxMother", "eta", "phi", "vx", "vy"]


def good_matched(dsa, mu):
    """good_matched_muons: (events, dsa, <=5) PF-muon records with numMatch."""
    per = []
    mup = ak.pad_none(mu, 1, axis=1)          # events without PF muons index the padding
    for k in range(1, 6):
        idx = ak.values_astype(dsa[f"muonMatch{k}idx"], np.int64)
        valid = (idx >= 0) & (idx < ak.num(mu, axis=1))
        pf = mup[ak.where(valid, idx, 0)]
        pf = ak.with_field(pf, dsa[f"muonMatch{k}"], "numMatch")
        per.append(ak.mask(pf, valid)[:, :, None])
    nested = ak.drop_none(ak.concatenate(per, axis=2))
    for cut in MU_CUTS:
        nested = nested[obj_cut_defs["muons"][cut](None, nested)]
    return nested


def dphi(a, b):
    return (a - b + np.pi) % (2 * np.pi) - np.pi


def gen_match(a, dsa):
    """dR to and production-vertex Lxy (cm) of the nearest gen muon from a dark-photon (pdgId 32) decay."""
    mom = a.GenPart_genPartIdxMother
    mp = ak.where(mom >= 0, a.GenPart_pdgId[ak.where(mom >= 0, mom, 0)], 0)
    g = (abs(a.GenPart_pdgId) == 13) & (abs(mp) == 32)
    gen = ak.zip({"eta": a.GenPart_eta[g], "phi": a.GenPart_phi[g], "lxy": np.hypot(a.GenPart_vx, a.GenPart_vy)[g]})
    pair = ak.cartesian({"d": ak.zip({"eta": dsa.eta, "phi": dsa.phi}), "g": gen}, nested=True)
    dr = np.sqrt((pair.d.eta - pair.g.eta) ** 2 + dphi(pair.d.phi, pair.g.phi) ** 2)
    best = ak.argmin(dr, axis=2, keepdims=True)
    return (ak.fill_none(ak.flatten(dr[best], axis=2), np.inf),
            ak.fill_none(ak.flatten(pair.g.lxy[best], axis=2), np.nan))


def one(url):
    a = uproot.open(url, timeout=300)["Events"].arrays(
        [f"DSAMuon_{x}" for x in DSA_F + SLOTS] + [f"Muon_{x}" for x in MU_F] + [f"GenPart_{x}" for x in GEN_F])
    dsa = ak.zip({x: a[f"DSAMuon_{x}"] for x in DSA_F + SLOTS}, depth_limit=2)
    mu = ak.zip({x: a[f"Muon_{x}"] for x in MU_F}, depth_limit=2)
    gm = good_matched(dsa, mu)
    good_pf = mu
    for cut in MU_CUTS:
        good_pf = good_pf[obj_cut_defs["muons"][cut](None, good_pf)]
    formula = dsa.nDTSegments + np.ceil(dsa.trkNumCSCHits / 6)
    variants = {"stored": dsa.nSegments, "formula": formula,
                "plus1": ak.where(dsa.trkNumCSCHits > 0, formula + 1, formula)}
    pre = (dsa.pt > 10) & (abs(dsa.eta) < 2.4) & (dsa.displacedID > 0)
    gdr, glxy = gen_match(a, dsa)

    flat = lambda x: ak.to_numpy(ak.flatten(x[pre]))
    out = {x: flat(dsa[x]).astype(np.float32) for x in ["pt", "eta", "phi"]}
    for x in ["charge", "nSegments", "nDTSegments", "nCSCSegments", "trkNumCSCHits", "trkNumDTHits"]:
        out[x] = flat(dsa[x]).astype(np.int16)
    for k in range(1, 6):
        out[f"slotN{k}"] = flat(dsa[f"muonMatch{k}"]).astype(np.int16)
    out["n_good_matched"] = flat(ak.num(gm, axis=2)).astype(np.int16)
    out["gen_dr"] = flat(gdr).astype(np.float32)
    out["gen_lxy"] = flat(glxy).astype(np.float32)
    with np.errstate(divide="ignore", invalid="ignore"):
        for name, ns in variants.items():
            d = ak.with_field(dsa, ns, "nSegments")
            d = ak.with_field(d, gm, "good_matched_muons")
            out[f"all_{name}"] = flat(obj_cut_defs["dsaMuons"]["all"](None, d))
            out[f"allq_{name}"] = flat(obj_cut_defs["dsaMuons"]["all + charge"](None, d))
    out["evt"] = flat(ak.broadcast_arrays(ak.local_index(dsa.pt, axis=0), dsa.pt)[0]).astype(np.int32)
    out["ev_n_dsa"] = ak.to_numpy(ak.num(dsa.pt)).astype(np.int16)
    out["ev_n_pre"] = ak.to_numpy(ak.sum(pre, axis=1)).astype(np.int16)
    out["ev_n_good_pf"] = ak.to_numpy(ak.num(good_pf)).astype(np.int16)
    return out, len(a)


def run(task):
    outdir, version, era, sample, url = task
    dest = os.path.join(outdir, version, era, sample, os.path.basename(url).replace(".root", ".npz"))
    if os.path.exists(dest):
        return dest, "exists"
    os.makedirs(os.path.dirname(dest), exist_ok=True)
    for attempt in range(3):
        try:
            out, n = one(url)
            break
        except Exception as e:  # xrootd hiccups: retry, then report
            err = f"{type(e).__name__}: {e}"
    else:
        return dest, "FAILED " + err
    tmp = dest + ".tmp.npz"
    np.savez_compressed(tmp, n_events=n, url=url, **out)
    os.replace(tmp, dest)
    return dest, f"{n} events, {len(out['pt'])} pre"


def main():
    outdir, taskfile = sys.argv[1:3]
    nproc = int(sys.argv[3]) if len(sys.argv) > 3 else 8
    tasks = [(outdir, *line.split()) for line in open(taskfile) if line.strip()]
    with Pool(nproc) as pool:
        for dest, msg in pool.imap_unordered(run, tasks):
            print(dest, msg, flush=True)


if __name__ == "__main__":
    main()
