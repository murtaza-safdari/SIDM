#!/usr/bin/env python3
"""Era fingerprint of the private Run 3 SIDM signal production (AN appendix "Signal sample
validation"): shows from the file content alone that the samples are at 13.6 TeV and that each
era (2022, 2022EE, 2023, 2023BPix) carries its own conditions.

Per era, over the 200 GeV / 1.2 GeV / 4.8 mm point of both channels (4Mu, 2Mu2E):
  * sqrt(s) = m(GenPart pdgId 35, the bound state) / sqrt(Generator_x1 * Generator_x2), and the
    generator's comEnergy read from the stored configuration;
  * Pileup_nTrueInt, GenVtx x/y/z, PV_z;
  * the number of HLT_ branches and the presence of two 2023-menu displaced-dimuon paths;
  * global tag and CMSSW release of every processing step, read back from the nano file itself:
    the ParameterSets tree holds every PSet blob (keyed by its MD5 = ParameterSetID), the MetaData
    tree the ProcessHistory (process name, PSet id, release). uproot cannot deserialize the
    memberwise-streamed ProcessHistory, so its raw basket bytes are scanned; the PSet ids are
    matched there to the process PSets, which fixes the order of the release strings.
Two detector maps expose the era-specific dead regions: the 2022EE ECAL endcap (e/gamma
reconstruction efficiency of gen electrons, 2Mu2E) and the 2023BPix pixel sector (mean
nTrackerLayers of reco muons matched to gen muons).

Writes era_fingerprint.csv (one row per era), era_fingerprint.npz (histograms, so --replot redraws
the figures without EOS) and the figures (vector PDF via an_style.save). Run from the worktree root
with the notebook venv:  python sidm/studies/run3_signal_validation/era_fingerprint.py --eras ...
"""
import argparse, csv, hashlib, os, re, sys
from concurrent.futures import ThreadPoolExecutor
import numpy as np
from scipy import ndimage
import awkward as ak
import uproot
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
import mplhep as hep

_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _DIR)
sys.path.insert(0, os.path.join(_DIR, "..", "truth_kinematics_forAN"))
import run3_signal_validation as V
import an_style

an_style.FIG_DIR = os.path.join(_DIR, "figures")
ERAS = ["2022", "2022EE", "2023", "2023BPix"]
CHANS = ["4Mu", "2Mu2E"]
POINT = "{chan}_200GeV_1p2GeV_4p8mm"
REF = ("4Mu_200GeV_1p2GeV_4p8mm", "llpNanoAOD_v2", "signal_4mu_v10.yaml")   # 2018 v10, 13 TeV
PATHS = ["HLT_DoubleL2Mu_L3Mu16NoVtx_VetoL3Mu0DxyMax0p1cm",
         "HLT_DoubleL2Mu10NoVtx_2Cha_CosmicSeed_VetoL3Mu0DxyMax1cm"]
ECOL = {"2022": "#1f77b4", "2022EE": "#2ca02c", "2023": "#d62728", "2023BPix": "#9467bd",
        "2018": "#000000"}
ELS = {"2022": "-", "2022EE": "--", "2023": "-", "2023BPix": "--", "2018": "-"}

# detector-map binning and selections
EE_ETA = np.round(np.arange(-3.0, 3.0001, 0.1), 2)
BP_ETA = np.round(np.arange(-2.4, 2.4001, 0.1), 2)
PHI = np.linspace(-np.pi, np.pi, 37)
EE_RHO, EE_Z = 110.0, 290.0        # gen electron vertex before the ECAL face (EB r~129, EE |z|~315 cm)
BP_RHO = 10.0                      # gen muon vertex inside BPix layer 3 (r~10.9 cm): crosses L3, L4
DR_E, DR_MU = 0.1, 0.05


# ---------------------------------------------------------------- provenance read-back
def _raw(branch):
    return b"".join(branch.basket(i).data.tobytes() for i in range(branch.num_baskets))


def _blobs(branch):
    """Every PSet blob of the ParameterSets tree, keyed by its ParameterSetID (= MD5 of the blob).
    Each entry is a pair<Hash, ParameterSetBlob>; the blob is the entry's trailing ROOT std::string
    (1-byte length, or 0xff + 4-byte length, then '<...>'), located with the basket entry offsets."""
    out = {}
    for i in range(branch.num_baskets):
        bk = branch.basket(i)
        d, off = bk.data.tobytes(), bk.byte_offsets
        for lo, hi in zip(off[:-1], off[1:]):
            e = d[lo:hi]
            for m in re.finditer(rb"<", e):
                j = m.start()
                if (j >= 1 and e[j - 1] != 255 and e[j - 1] == len(e) - j) or \
                   (j >= 5 and e[j - 5] == 255 and int.from_bytes(e[j - 4:j], "big") == len(e) - j):
                    out[hashlib.md5(e[j:]).hexdigest()] = e[j:]
                    break
            else:
                raise RuntimeError("ParameterSets entry without a recognisable blob")
    if len(out) != branch.num_entries:
        raise RuntimeError(f"{len(out)} distinct PSet blobs for {branch.num_entries} entries")
    return out


def _get(blob, key):
    """Value of one parameter: strings hex-decoded, PSet references returned as the id. Keys of
    ES sources/producers in a process PSet carry a type prefix ('PoolDBESSource@GlobalTag')."""
    m = re.search(rb"[<;](?:\w+@)?" + re.escape(key.encode()) + rb"=\+(\w+)\(([^)]*)\)", blob)
    if not m:
        return None
    typ, val = m.group(1).decode(), m.group(2).decode()
    if typ in ("S", "s"):
        return bytes.fromhex(val).decode()
    if typ == "vS":
        return [bytes.fromhex(x).decode() for x in val.strip("{}").split(",") if x]
    return val


def provenance(url):
    """Process history of one nano file: [(process, release, global tag)], plus the HLT menu,
    generator comEnergy and vertex-smearing parameters. (The premix library is not recoverable:
    the input fileNames of the PreMixingModule are untracked, so absent from the stored PSet.)"""
    f = uproot.open(url)
    blobs = _blobs(f["ParameterSets"]["IdToParameterSetsBlobs"])
    ph = _raw(f["MetaData"]["ProcessHistory"])
    procs = []
    for h, b in blobs.items():
        name = _get(b, "@process_name")
        if name is not None and bytes.fromhex(h) in ph:
            procs.append((ph.find(bytes.fromhex(h)), name, h))
    procs.sort()
    # a merged file stores one history per input job (the 2018 v10 files: one GEN PSet per job);
    # keep each process name once, in the order of its first appearance (the first history)
    seen = {}
    for pos, name, h in procs:
        seen.setdefault(name, (pos, name, h))
    procs = list(seen.values())
    rels = [r.decode() for r in re.findall(rb"CMSSW_\d+_\d+_\d+(?:_patch\d+|_pre\d+|_UL)?", ph)]
    if not procs or len(rels) % len(procs):
        raise RuntimeError(f"{url}: {len(procs)} processes but {len(rels)} release strings")
    rels = rels[:len(procs)]
    out = dict(steps=[], hlt_menu=None, com=None, vtx=None)
    for (_, name, h), rel in zip(procs, rels):
        p = blobs[h]
        gt = _get(blobs[_get(p, "GlobalTag")], "globaltag") if _get(p, "GlobalTag") else None
        out["steps"].append((name, rel, gt))
        if _get(p, "HLTConfigVersion"):
            out["hlt_menu"] = _get(blobs[_get(p, "HLTConfigVersion")], "tableName")
        if _get(p, "generator"):
            out["com"] = float(_get(blobs[_get(p, "generator")], "comEnergy"))
        if _get(p, "VtxSmeared"):
            v = blobs[_get(p, "VtxSmeared")]
            out["vtx"] = {k: float(_get(v, k)) for k in ("X0", "Y0", "Z0", "SigmaZ", "BetaStar")
                          if _get(v, k) is not None}
    return out


# ---------------------------------------------------------------- event content
SCAL = ["Generator_x1", "Generator_x2", "Pileup_nTrueInt", "GenVtx_x", "GenVtx_y", "GenVtx_z", "PV_z",
        "GenPart_pdgId", "GenPart_mass", "LHEPart_incomingpz", "LHEPart_status"]
GEN = ["GenPart_pt", "GenPart_eta", "GenPart_phi", "GenPart_vx", "GenPart_vy", "GenPart_vz",
       "GenPart_genPartIdxMother"]


def _gen_from_zd(a, pdg):
    pid, mom = a["GenPart_pdgId"], a["GenPart_genPartIdxMother"]
    mpid = pid[ak.where(mom >= 0, mom, 0)]
    return (abs(pid) == pdg) & (mom >= 0) & (abs(mpid) == 32)


def _match(eta, phi, reco):
    """For each gen particle, the closest reco object of the event (a record of the fields of
    `reco`, None if the event has none) and its Delta R (inf if none)."""
    pair = ak.cartesian({"g": ak.zip({"eta": eta, "phi": phi}), "r": reco}, nested=True)
    dphi = (pair.g.phi - pair.r.phi + np.pi) % (2 * np.pi) - np.pi
    dr = np.sqrt((pair.g.eta - pair.r.eta) ** 2 + dphi ** 2)
    im = ak.argmin(dr, axis=-1, keepdims=True)
    return ak.firsts(pair.r[im], axis=-1), ak.fill_none(ak.firsts(dr[im], axis=-1), np.inf)


def read_file(url, maps):
    """Per-event scalars and, if asked, the map histograms of one file."""
    f = uproot.open(url)
    t = f["Events"]
    keys = t.keys()
    br = SCAL + (GEN if maps else [])
    if maps == "ee":
        br += ["Electron_eta", "Electron_phi", "Photon_eta", "Photon_phi"]
    if maps == "bpix":
        br += ["Muon_eta", "Muon_phi", "Muon_nTrackerLayers", "Muon_isPFcand", "Muon_isTracker"]
    a = t.arrays([b for b in br if b in keys])
    m35 = ak.firsts(a["GenPart_mass"][a["GenPart_pdgId"] == 35])
    sqrts = ak.to_numpy(ak.fill_none(m35, np.nan)) / np.sqrt(ak.to_numpy(a["Generator_x1"] * a["Generator_x2"]))
    # independent estimate: 2 E_beam, E_beam = p_z / x of the higher-x incoming LHE parton (x1 <-> +z).
    # m/sqrt(x1 x2) is exact only for massless partons; for c/b-initiated events with x ~ 1e-3 the
    # parton mass is not negligible against its momentum and that estimator falls below sqrt(s).
    if "LHEPart_incomingpz" in a.fields:
        pz = a["LHEPart_incomingpz"][a["LHEPart_status"] == -1]
        x1, x2 = ak.to_numpy(a["Generator_x1"]), ak.to_numpy(a["Generator_x2"])
        pp, pm = ak.to_numpy(ak.max(pz, axis=1)), -ak.to_numpy(ak.min(pz, axis=1))
        sqrts_lhe = 2 * np.where(x1 >= x2, pp / x1, pm / x2)
    else:
        sqrts_lhe = np.full(len(a), np.nan)
    out = dict(n=len(a), sqrts=sqrts, sqrts_lhe=sqrts_lhe, hlt={k for k in keys if k.startswith("HLT_")})
    for k in ("Pileup_nTrueInt", "GenVtx_x", "GenVtx_y", "GenVtx_z", "PV_z"):
        out[k] = ak.to_numpy(a[k]) if k in a.fields else np.full(len(a), np.nan)
    if maps == "ee":
        s = _gen_from_zd(a, 11) & (a["GenPart_pt"] > 10) & (abs(a["GenPart_eta"]) > 1.5) \
            & (abs(a["GenPart_eta"]) < 3.0) & (np.hypot(a["GenPart_vx"], a["GenPart_vy"]) < EE_RHO) \
            & (abs(a["GenPart_vz"]) < EE_Z)
        eta, phi = a["GenPart_eta"][s], a["GenPart_phi"][s]
        _, de = _match(eta, phi, ak.zip({"eta": a["Electron_eta"], "phi": a["Electron_phi"]}))
        _, dg = _match(eta, phi, ak.zip({"eta": a["Photon_eta"], "phi": a["Photon_phi"]}))
        ok = ak.flatten((de < DR_E) | (dg < DR_E))
        e, p = ak.to_numpy(ak.flatten(eta)), ak.to_numpy(ak.flatten(phi))
        out["ee_den"] = np.histogram2d(e, p, [EE_ETA, PHI])[0]
        out["ee_num"] = np.histogram2d(e[ak.to_numpy(ok)], p[ak.to_numpy(ok)], [EE_ETA, PHI])[0]
    if maps == "bpix":
        s = _gen_from_zd(a, 13) & (a["GenPart_pt"] > 5) & (abs(a["GenPart_eta"]) < 2.4) \
            & (np.hypot(a["GenPart_vx"], a["GenPart_vy"]) < BP_RHO)
        eta, phi = a["GenPart_eta"][s], a["GenPart_phi"][s]
        mu, dr = _match(eta, phi, ak.zip({"eta": a["Muon_eta"], "phi": a["Muon_phi"],
                                          "lay": a["Muon_nTrackerLayers"], "trk": a["Muon_isTracker"],
                                          "pf": a["Muon_isPFcand"]}))
        has = dr < DR_MU
        lay = ak.fill_none(mu.lay, 0)
        trk = ak.fill_none(mu.trk, False)
        pf = ak.fill_none(mu.pf, False)
        e, p = ak.to_numpy(ak.flatten(eta)), ak.to_numpy(ak.flatten(phi))
        has, trk, pf = (ak.to_numpy(ak.flatten(x)) for x in (has, has & trk, has & pf))
        lay = ak.to_numpy(ak.flatten(lay)).astype(float)
        out["bp_den"] = np.histogram2d(e, p, [BP_ETA, PHI])[0]
        out["bp_pf"] = np.histogram2d(e[pf], p[pf], [BP_ETA, PHI])[0]
        out["bp_ntrk"] = np.histogram2d(e[trk], p[trk], [BP_ETA, PHI])[0]
        out["bp_lay"] = np.histogram2d(e[trk], p[trk], [BP_ETA, PHI], weights=lay[trk])[0]
    return out


def collect(urls, maps, workers, map_nfiles=0):
    """Read all urls; the detector maps (if any) only from the first map_nfiles (0 = all)."""
    nm = map_nfiles if map_nfiles > 0 else len(urls)
    with ThreadPoolExecutor(workers) as ex:
        parts = list(ex.map(lambda iu: read_file(iu[1], maps if iu[0] < nm else None), enumerate(urls)))
    out = dict(n=sum(p["n"] for p in parts), nhlt=sorted({len(p["hlt"]) for p in parts}),
               paths={q: sorted({q in p["hlt"] for p in parts}) for q in PATHS})
    for k in ("sqrts", "sqrts_lhe", "Pileup_nTrueInt", "GenVtx_x", "GenVtx_y", "GenVtx_z", "PV_z"):
        out[k] = np.concatenate([p[k] for p in parts])
    out["map_files"] = sum(1 for p in parts if any(k.startswith(("ee_", "bp_")) for k in p))
    for k in parts[0]:
        if k.startswith(("ee_", "bp_")):
            out[k] = np.sum([p[k] for p in parts], axis=0)
    return out


# ---------------------------------------------------------------- figures
def _lab(axes, com):
    an_style.cms_sim_labels(axes, com=com)


def _finish(fig, axes, name, com=13.6):
    _lab(axes, com)
    return an_style.save(fig, name)


def fig_sqrt_s(H):
    fig, ax = plt.subplots(figsize=(10, 7.5), layout="constrained")
    edges = H["sqrts_edges"]
    for era in [e for e in ERAS if f"sqrts_{e}" in H] + ["2018"]:
        h = H[f"sqrts_{era}"]
        lab = f"{era} (Run 3)" if era != "2018" else "2018 v10 (Run 2 reference)"
        ax.stairs(h / h.sum(), edges / 1e3, color=ECOL[era], ls=ELS[era], lw=2.5, label=lab)
    ax.set_xlabel(r"$\sqrt{s} = m(B_s) / \sqrt{x_1 x_2}$  [TeV]")
    ax.set_ylabel("Fraction of events")
    ax.set_yscale("log"); ax.set_ylim(1e-4, 3e2)   # headroom: legend above the spikes
    ax.legend(loc="upper left", fontsize=15, ncol=2)
    an_style.cms_sim_labels([ax], com="13 / 13.6 TeV")
    return an_style.save(fig, "era_sqrt_s")


def fig_pileup(H):
    fig, ax = plt.subplots(figsize=(10, 7.5), layout="constrained")
    edges = H["pu_edges"]
    for era in [e for e in ERAS if f"pu_{e}" in H]:
        h = H[f"pu_{era}"]
        ax.stairs(h / h.sum(), edges, color=ECOL[era], ls=ELS[era], lw=2.5,
                  label=f"{era}: mean {H[f'pu_mean_{era}']:.1f}")
    ax.set_xlabel("Pileup_nTrueInt (true pileup interactions)"); ax.set_ylabel("Fraction of events")
    ax.set_xlim(edges[0], edges[-1]); ax.set_ylim(0, None)
    ax.legend(loc="upper right", fontsize=14)
    return _finish(fig, [ax], "era_pileup")


def fig_beamspot(H):
    fig, axes = plt.subplots(1, 3, figsize=(21, 7.5), layout="constrained")
    for ax, k, xl in zip(axes, ("x", "y", "z"), ("generated vertex x  [cm]", "generated vertex y  [cm]",
                                                  "generated vertex z  [cm]")):
        edges = H[f"vtx{k}_edges"]
        for era in [e for e in ERAS if f"vtx{k}_{e}" in H]:
            h = H[f"vtx{k}_{era}"]
            ax.stairs(h / h.sum(), edges, color=ECOL[era], ls=ELS[era], lw=2.5, label=era)
        ax.set_xlabel(xl); ax.set_xlim(edges[0], edges[-1]); ax.set_ylim(0, None)
    axes[0].set_ylabel("Fraction of events")
    axes[0].legend(loc="upper left", fontsize=20)
    return _finish(fig, axes, "era_beamspot")


def _map_panel(ax, z, xedges, cmap, vmin, vmax, mask_eta=None):
    zz = np.ma.masked_invalid(z.T)
    m = ax.pcolormesh(xedges, PHI, zz, cmap=cmap, vmin=vmin, vmax=vmax, edgecolors="face", linewidth=0)
    if mask_eta is not None:
        ax.axvspan(-mask_eta, mask_eta, color="0.85", hatch="//", ec="0.6", lw=0)
        ax.text(0, 0, "barrel\n(not shown)", ha="center", va="center", fontsize=18, color="0.3")
    ax.set_xlabel(r"gen $\eta$"); ax.set_xlim(xedges[0], xedges[-1]); ax.set_ylim(-np.pi, np.pi)
    return m


def _tag(ax, text):
    ax.text(0.03, 0.96, text, transform=ax.transAxes, ha="left", va="top", fontsize=22,
            bbox=dict(facecolor="white", edgecolor="none", alpha=0.92, pad=3))


def _eff_maps(fig, axes, num_a, den_a, num_b, den_b, xedges, eras, zlab, mask_eta, dlab, dlim, vlim):
    with np.errstate(invalid="ignore", divide="ignore"):
        ea, eb = num_a / den_a, num_b / den_b
    for ax, z, era in zip(axes[:2], (ea, eb), eras):
        m = _map_panel(ax, z, xedges, "viridis", *vlim, mask_eta)
        _tag(ax, era)
    fig.colorbar(m, ax=axes[:2], label=zlab, pad=0.01)
    md = _map_panel(axes[2], eb - ea, xedges, "RdBu", -dlim, dlim, mask_eta)
    _tag(axes[2], dlab)
    fig.colorbar(md, ax=axes[2], label="difference", pad=0.01)
    axes[0].set_ylabel(r"gen $\phi$")


def fig_ecal_ee(H):
    fig, axes = plt.subplots(1, 3, figsize=(22, 7.5), layout="constrained", sharey=True)
    _eff_maps(fig, axes, H["ee_num_2022"], H["ee_den_2022"], H["ee_num_2022EE"], H["ee_den_2022EE"],
              EE_ETA, ("2022", "2022EE"), r"e/$\gamma$ reco efficiency", 1.5, "2022EE $-$ 2022",
              0.6, (0.4, 1.0))
    return _finish(fig, axes, "era_ecal_ee")


def fig_bpix(H):
    fig, axes = plt.subplots(1, 3, figsize=(22, 7.5), layout="constrained", sharey=True)
    _eff_maps(fig, axes, H["bp_lay_2023"], H["bp_ntrk_2023"], H["bp_lay_2023BPix"], H["bp_ntrk_2023BPix"],
              BP_ETA, ("2023", "2023BPix"), r"mean $\mu$ nTrackerLayers", None,
              "2023BPix $-$ 2023", 3.0, (8, 16))
    return _finish(fig, axes, "era_bpix")


def fig_reco_eff():
    """The two committed per-era figures of run3_study_plots, rendered to one two-page PDF. Their
    suptitles are dropped (the AN caption carries them) and in-figure text raised to >= 7 pt at
    7.5 in print width."""
    import run3_study_plots as P
    path = os.path.join(an_style.FIG_DIR, "era_reco_eff.pdf")
    os.makedirs(an_style.FIG_DIR, exist_ok=True)
    with PdfPages(path) as pdf:
        for fig in (P.fig_geometry_eras(), P.fig_kinematics_eras()):
            minfs = 8 / (7.5 / fig.get_figwidth())
            if fig._suptitle is not None:
                fig._suptitle.set_visible(False)
            for ax in fig.axes:
                for t in ax.texts:
                    if t.get_fontsize() < minfs and "CMS" not in t.get_text():
                        t.set_fontsize(minfs)
                leg = ax.get_legend()
                if leg is not None:
                    for t in leg.get_texts() + [leg.get_title()]:
                        t.set_fontsize(max(t.get_fontsize(), minfs))
            fig.tight_layout()
            pdf.savefig(fig)
            plt.close(fig)
    print(f"saved {path}")
    return path


# ---------------------------------------------------------------- main
def _stats(x):
    x = x[np.isfinite(x)]
    return x


def hole(H, kind, a, b, thr):
    """Bins where era b falls below era a by more than thr (efficiency or mean layers), with
    their eta/phi extent and the integrated values inside that region."""
    if kind == "ee":
        na, da, nb, db, xe = H[f"ee_num_{a}"], H[f"ee_den_{a}"], H[f"ee_num_{b}"], H[f"ee_den_{b}"], EE_ETA
    else:
        na, da, nb, db, xe = H[f"bp_lay_{a}"], H[f"bp_ntrk_{a}"], H[f"bp_lay_{b}"], H[f"bp_ntrk_{b}"], BP_ETA
    with np.errstate(invalid="ignore", divide="ignore"):
        d = nb / db - na / da
    sel = (d < -thr) & (da >= 20) & (db >= 20)
    if not sel.any():
        return None
    # keep the largest 8-connected cluster of such bins (phi wraps; not needed for these holes),
    # so isolated statistical fluctuations do not stretch the quoted region
    lab, n = ndimage.label(sel, structure=np.ones((3, 3)))
    sel = lab == 1 + np.argmax(ndimage.sum(sel, lab, range(1, n + 1)))
    ie, ip = np.where(sel)
    return dict(nbins=int(sel.sum()), eta=(xe[ie.min()], xe[ie.max() + 1]), phi=(PHI[ip.min()], PHI[ip.max() + 1]),
                a=na[sel].sum() / da[sel].sum(), b=nb[sel].sum() / db[sel].sum(),
                a_out=na[~sel].sum() / da[~sel].sum(), b_out=nb[~sel].sum() / db[~sel].sum(),
                den_a=int(da[sel].sum()), den_b=int(db[sel].sum()))


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--eras", default=",".join(ERAS))
    ap.add_argument("--channels", default=",".join(CHANS))
    ap.add_argument("--nfiles", type=int, default=0, help="files per sample (0 = all 120)")
    ap.add_argument("--map-nfiles", type=int, default=0, help="files per sample for the maps (0 = all)")
    ap.add_argument("--ref-nfiles", type=int, default=10, help="files of the 2018 v10 reference")
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--csv", default=os.path.join(_DIR, "era_fingerprint.csv"))
    ap.add_argument("--npz", default=os.path.join(_DIR, "era_fingerprint.npz"))
    ap.add_argument("--no-maps", action="store_true", help="skip the detector maps")
    ap.add_argument("--no-figures", action="store_true")
    ap.add_argument("--replot", action="store_true", help="redraw figures from --npz, no EOS access")
    args = ap.parse_args()
    eras, chans = args.eras.split(","), args.channels.split(",")
    an_style.set_style()

    if args.replot:
        H = dict(np.load(args.npz, allow_pickle=True))
    else:
        H = dict(sqrts_edges=np.linspace(12500, 14000, 151), pu_edges=np.arange(0, 101, 1.0),
                 vtxx_edges=np.linspace(0.035, 0.11, 151), vtxy_edges=np.linspace(-0.025, -0.008, 69),
                 vtxz_edges=np.linspace(-15, 15, 61))
        rows = []
        for era in eras:
            evs, prov = [], {}
            for chan in chans:
                urls = V._fileset_files(POINT.format(chan=chan), era, f"signal_{chan.lower()}_run3.yaml",
                                        args.nfiles)
                maps = None
                if not args.no_maps:
                    maps = "ee" if (chan == "2Mu2E" and era in ("2022", "2022EE")) else \
                           "bpix" if era in ("2023", "2023BPix") else None
                evs.append(collect(urls, maps, args.workers, args.map_nfiles))
                pa, pb = provenance(urls[0]), provenance(urls[-1])
                if pa != pb:
                    print(f"WARNING {era} {chan}: provenance differs between {urls[0]} and {urls[-1]}")
                prov[chan] = pa
                print(f"{era} {chan}: {len(urls)} files, {evs[-1]['n']} events, maps from "
                      f"{evs[-1]['map_files']} files", flush=True)
            c = {k: np.concatenate([e[k] for e in evs]) for k in ("sqrts", "sqrts_lhe", "Pileup_nTrueInt",
                                                                 "GenVtx_x", "GenVtx_y", "GenVtx_z", "PV_z")}
            for k in {k for e in evs for k in e if k.startswith(("ee_", "bp_"))}:
                H[f"{k}_{era}"] = np.sum([e[k] for e in evs if k in e], axis=0)
            s = _stats(c["sqrts"]); pu = c["Pileup_nTrueInt"]
            H[f"sqrts_{era}"] = np.histogram(s, H["sqrts_edges"])[0]
            H[f"pu_{era}"] = np.histogram(pu, H["pu_edges"])[0]; H[f"pu_mean_{era}"] = pu.mean()
            for k in "xyz":
                H[f"vtx{k}_{era}"] = np.histogram(c[f"GenVtx_{k}"], H[f"vtx{k}_edges"])[0]
            p0 = prov[chans[0]]
            same = all(prov[ch]["steps"] == p0["steps"] for ch in chans)
            nhlt = sorted({n for e in evs for n in e["nhlt"]})
            q16, q50, q84 = np.percentile(s, [16, 50, 84])
            row = dict(era=era, channels="+".join(chans), files_per_sample=len(urls),
                       events=sum(e["n"] for e in evs),
                       sqrt_s_median_GeV=round(q50, 2), sqrt_s_68pct_halfwidth_GeV=round((q84 - q16) / 2, 3),
                       sqrt_s_frac_below_13p5TeV=round(float((s < 13500).mean()), 4),
                       sqrt_s_lhe_median_GeV=round(float(np.nanmedian(c["sqrts_lhe"])), 2),
                       gen_comEnergy_GeV=p0["com"],
                       pileup_nTrueInt_mean=round(pu.mean(), 2), pileup_nTrueInt_std=round(pu.std(), 2),
                       genvtx_x_mean_cm=round(c["GenVtx_x"].mean(), 4), genvtx_y_mean_cm=round(c["GenVtx_y"].mean(), 4),
                       genvtx_z_std_cm=round(c["GenVtx_z"].std(), 3), pv_z_std_cm=round(c["PV_z"].std(), 3),
                       n_hlt_branches="/".join(map(str, nhlt)))
            for q in PATHS:
                v = sorted({x for e in evs for x in e["paths"][q]})
                row[q] = "yes" if v == [True] else "no" if v == [False] else "mixed"
            for name, rel, gt in p0["steps"]:
                row[f"{name}_release"] = rel; row[f"{name}_globaltag"] = gt
            row.update(hlt_menu=p0["hlt_menu"], vtxsmear_X0_Y0_SigmaZ_cm=
                       f"{p0['vtx']['X0']:.6g}/{p0['vtx']['Y0']:.6g}/{p0['vtx']['SigmaZ']:.4g}" if p0["vtx"] else "",
                       provenance_same_both_channels=same)
            rows.append(row)
            print(row, flush=True)
        # 2018 v10 reference (13 TeV)
        ref = V._fileset_files(REF[0], REF[1], REF[2], args.ref_nfiles)
        r = collect(ref, None, args.workers)
        H["sqrts_2018"] = np.histogram(_stats(r["sqrts"]), H["sqrts_edges"])[0]
        pr = provenance(ref[0])
        print(f"2018 v10 reference: {len(ref)} files, {r['n']} events, sqrt(s) median "
              f"{np.median(_stats(r['sqrts'])):.1f} GeV (LHE: {np.nanmedian(r['sqrts_lhe']):.1f}), comEnergy {pr['com']}, steps {pr['steps']}")
        H["ref_info"] = np.array(repr(dict(files=len(ref), events=r["n"], median=float(np.median(_stats(r["sqrts"]))),
                                           com=pr["com"], steps=pr["steps"])))
        keys = list(dict.fromkeys(k for row in rows for k in row))
        with open(args.csv, "w", newline="") as fh:
            w = csv.DictWriter(fh, fieldnames=keys); w.writeheader(); w.writerows(rows)
        print(f"wrote {args.csv}")
        np.savez_compressed(args.npz, **H)
        print(f"wrote {args.npz}")

    have = {e for e in ERAS if f"sqrts_{e}" in H}
    for kind, a, b, thr in (("ee", "2022", "2022EE", 0.3), ("bp", "2023", "2023BPix", 1.0)):
        if f"{kind}_den_{a}" in H or f"{kind}_ntrk_{a}" in H:
            if {a, b} <= have:
                h = hole(H, kind, a, b, thr)
                print(f"hole {kind} {b} vs {a} (drop > {thr}):",
                      h if h else "NONE FOUND")
    if args.no_figures:
        return
    fig_sqrt_s(H); fig_pileup(H); fig_beamspot(H)
    if {"2022", "2022EE"} <= have and "ee_den_2022" in H:
        fig_ecal_ee(H)
    if {"2023", "2023BPix"} <= have and "bp_ntrk_2023" in H:
        fig_bpix(H)
    fig_reco_eff()


if __name__ == "__main__":
    main()
