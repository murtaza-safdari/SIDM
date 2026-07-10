"""Shared setup for the ABCD plane-choice notebooks: samples, normalization, planes.

Everything sample- or plane-specific is declared HERE, once, keyed by name (never by
list position). The notebooks import from this module so they cannot drift apart.
"""

import json
import os
import subprocess

import coffea.util

import abcd_tools as at

# ---------------------------------------------------------------------------
# where things live
# ---------------------------------------------------------------------------
EOS_MERGED = "root://cmseos.fnal.gov//store/group/lpcmetx/SIDM/coffea_outputs/murtazas/abcd_plane_study"
WORKDIR = "/uscms_data/d3/murtazas/abcd_study_local"       # local staging (coffea.util.load is local-open only)
STUDY_DIR = os.path.dirname(os.path.abspath(__file__))

# census summaries: pre-skim / pre-filter sum of gen weights per sample
CENSUS_SIGNAL_2MU2E = os.path.join(STUDY_DIR, "../../configs/census/signal_2mu2e_v10.census.summary.json")
CENSUS_SIGNAL_4MU = os.path.join(STUDY_DIR, "../../configs/census/signal_4mu_v10.census.summary.json")
CENSUS_BKG_UNSKIMMED = "/uscms_data/d3/murtazas/abcd_census/backgrounds_unskimmed.census.json"

# ---------------------------------------------------------------------------
# samples (name -> process; never index-ordered)
# ---------------------------------------------------------------------------
QCD_BINS = ["15To20", "20To30", "30To50", "50To80", "80To120", "120To170",
            "170To300", "300To470", "470To600", "600To800", "800To1000", "1000"]
BACKGROUNDS = {f"QCD_Pt{b}": "QCD" for b in QCD_BINS}
BACKGROUNDS.update({"DYJetsToMuMu_M10to50": "DY", "DYJetsToMuMu_M50": "DY",
                    "TTJets": "TTJets", "WW": "Diboson", "WZ": "Diboson", "ZZ": "Diboson"})
PROCESSES = ["QCD", "DY", "TTJets", "Diboson"]

_CTAUS = {"100GeV_1p2GeV": ["0p096mm", "9p6mm", "96p0mm"],
          "500GeV_1p2GeV": ["0p019mm", "1p9mm", "19p0mm"],
          "1000GeV_1p2GeV": ["0p0096mm", "0p96mm", "9p6mm"],
          "500GeV_0p25GeV": ["0p4mm", "4p0mm"],
          "500GeV_5p0GeV": ["8p0mm", "80p0mm"]}
SIGNALS_2MU2E = [f"2Mu2E_{m}_{c}" for m, cs in _CTAUS.items() for c in cs]
SIGNALS_4MU = [f"4Mu_{m}_{c}" for m, cs in _CTAUS.items() for c in cs]

CHANNELS = {"2mu2e": "2mu2e_abcd_scan", "4mu": "4mu_abcd_scan"}

# ---------------------------------------------------------------------------
# normalization
# ---------------------------------------------------------------------------
FW = json.load(open(os.path.join(STUDY_DIR, "processed_fraction.json")))


def fetch(sample):
    """xrdcp the merged output locally (once) and load it."""
    os.makedirs(WORKDIR, exist_ok=True)
    local = os.path.join(WORKDIR, f"{sample}.coffea")
    if not os.path.exists(local):
        subprocess.run(["xrdcp", "-s", f"{EOS_MERGED}/{sample}.coffea", local], check=True)
    out = coffea.util.load(local)
    out = out["out"] if isinstance(out, dict) and "out" in out else out
    return out[sample]


def load_normalized(sample, sumw_pre_map, ttjets_nnlo=False):
    """Merged output with hists scaled to lumi*xs/sumw_pre (and f_w corrected).

    Returns (hists_dict, merged_output). Cutflows are NOT corrected — relative only.
    """
    o = fetch(sample)
    factor = at.offline_norm_factor(o["metadata"], sumw_pre_map[sample])
    factor /= FW.get(sample, 1.0)
    if ttjets_nnlo and sample == "TTJets":
        factor *= at.ttjets_xsec_rescale()
    return {n: h * factor for n, h in o["hists"].items()}, o


def accumulate_normalized(samples, sumw_pre_map, keep_prefix="abcd_scan", ttjets_nnlo=False):
    """Memory-safe sums of normalized hists: (total, by_process).

    Loads ONE sample at a time, keeps only hists whose name starts with keep_prefix
    (the dense scan hists decompress to ~200 MB per sample — holding all 44 samples
    OOMs the interactive node), and frees each sample before the next.
    """
    import gc
    total, by_process = {}, {p: {} for p in PROCESSES}
    for s in samples:
        hists, _ = load_normalized(s, sumw_pre_map, ttjets_nnlo=ttjets_nnlo)
        proc = BACKGROUNDS.get(s)
        for n, h in hists.items():
            if keep_prefix and not n.startswith(keep_prefix):
                continue
            total[n] = h if n not in total else total[n] + h
            if proc is not None:
                bp = by_process[proc]
                bp[n] = h if n not in bp else bp[n] + h
        del hists
        gc.collect()
    return total, by_process


def sum_process(hists_by_sample, samples):
    """Sum one hist name across samples: {name: summed hist} for the given samples."""
    out = {}
    for s in samples:
        for n, h in hists_by_sample[s].items():
            out[n] = h if n not in out else out[n] + h
    return out


# ---------------------------------------------------------------------------
# candidate planes
# ---------------------------------------------------------------------------
# Each entry: hist name, plane axes (x, y) with pass orientation, and the SR values of
# every OTHER scan axis (applied via project_plane sel= when the plane is evaluated).
# Displacement conventions on the axes:
#   mudisp  [-1.5,-0.5,2.5,3.5,100]: pass = ("lt", 2.5)  (no-pix-info + <=2 hits)
#   mudisp N=3 variant: pass = ("lt", 3.5)
#   egmdisp [-0.5,0.5,20,1000]: pass = ("ge", 0.5)  (>=1 lost hit, or photon-only)
#   mupix   [-1.5..5.5,100] fine axis; egmlost [-0.5..2.5,20,1000] fine axis
# Iso quirk prescriptions when iso is an EVENT CUT (coarse iso3 axes,
# edges [-0.02, 0, WP, 1000]):
#   (i) sentinel passes -> ("lt", WP); (ii) sentinel dropped -> ("window", 0.0, WP).
SR = {"dphi": ("ge", 2.0), "mjj": ("ge", 150.0)}

PLANES = {
    "2mu2e": {
        "P1_iso_iso": dict(hist="abcd_scan_2mu2e_iso_iso", x="muiso", y="egmiso",
                           xspec=("lt", 0.25), yspec=("lt", 0.10),
                           cuts={"dphi": SR["dphi"], "mjj": SR["mjj"],
                                 "mudisp": ("lt", 2.5), "egmdisp": ("ge", 0.5)}),
        "P2_muiso_dphi": dict(hist="abcd_scan_2mu2e_iso_iso", x="muiso", y="dphi",
                              xspec=("lt", 0.25), yspec=("ge", 2.0),
                              cuts={"egmiso": ("lt", 0.10), "mjj": SR["mjj"],
                                    "mudisp": ("lt", 2.5), "egmdisp": ("ge", 0.5)}),
        "P3_egmiso_dphi": dict(hist="abcd_scan_2mu2e_iso_iso", x="egmiso", y="dphi",
                               xspec=("lt", 0.10), yspec=("ge", 2.0),
                               cuts={"muiso": ("lt", 0.25), "mjj": SR["mjj"],
                                     "mudisp": ("lt", 2.5), "egmdisp": ("ge", 0.5)}),
        "P4_muiso_mjj": dict(hist="abcd_scan_2mu2e_iso_iso", x="muiso", y="mjj",
                             xspec=("lt", 0.25), yspec=("ge", 150.0),
                             cuts={"egmiso": ("lt", 0.10), "dphi": SR["dphi"],
                                   "mudisp": ("lt", 2.5), "egmdisp": ("ge", 0.5)}),
        "P5_muiso_mupix": dict(hist="abcd_scan_2mu2e_muiso_mupix", x="muiso", y="mupix",
                               xspec=("lt", 0.25), yspec=("lt", 2.5),
                               cuts={"egmiso3": ("lt", 0.10), "egmdisp": ("ge", 0.5),
                                     "dphi": SR["dphi"], "mjj": SR["mjj"]}),
        "P6_mupix_dphi": dict(hist="abcd_scan_2mu2e_muiso_mupix", x="mupix", y="dphi",
                              xspec=("lt", 2.5), yspec=("ge", 2.0),
                              cuts={"muiso": ("lt", 0.25), "egmiso3": ("lt", 0.10),
                                    "egmdisp": ("ge", 0.5), "mjj": SR["mjj"]}),
        "P7_egmlost_dphi": dict(hist="abcd_scan_2mu2e_egmiso_egmlost", x="egmlost", y="dphi",
                                xspec=("ge", 0.5), yspec=("ge", 2.0),
                                cuts={"egmiso": ("lt", 0.10), "muiso3": ("lt", 0.25),
                                      "mudisp2": ("lt", 2.5), "mjj": SR["mjj"]}),
        "P8_dphi_mjj": dict(hist="abcd_scan_2mu2e_iso_iso", x="dphi", y="mjj",
                            xspec=("ge", 2.0), yspec=("ge", 150.0),
                            cuts={"muiso": ("lt", 0.25), "egmiso": ("lt", 0.10),
                                  "mudisp": ("lt", 2.5), "egmdisp": ("ge", 0.5)}),
    },
    "4mu": {
        "Q1_iso_iso": dict(hist="abcd_scan_4mu_iso_iso", x="muiso0", y="muiso1",
                           xspec=("lt", 0.25), yspec=("lt", 0.25),
                           cuts={"dphi": SR["dphi"], "mjj": SR["mjj"],
                                 "mudisp0": ("lt", 2.5), "mudisp1": ("lt", 2.5)}),
        "Q2_iso0_dphi": dict(hist="abcd_scan_4mu_iso_iso", x="muiso0", y="dphi",
                             xspec=("lt", 0.25), yspec=("ge", 2.0),
                             cuts={"muiso1": ("lt", 0.25), "mjj": SR["mjj"],
                                   "mudisp0": ("lt", 2.5), "mudisp1": ("lt", 2.5)}),
        "Q3_iso0_pix0": dict(hist="abcd_scan_4mu_iso_pix_lead", x="muiso0", y="mupix0",
                             xspec=("lt", 0.25), yspec=("lt", 2.5),
                             cuts={"muiso3_1": ("lt", 0.25), "mudisp2_1": ("lt", 2.5),
                                   "dphi": SR["dphi"], "mjj": SR["mjj"]}),
        "Q4_pix_pix": dict(hist="abcd_scan_4mu_pix_pix", x="mupix0", y="mupix1",
                           xspec=("lt", 2.5), yspec=("lt", 2.5),
                           cuts={"muiso3_0": ("lt", 0.25), "muiso3_1": ("lt", 0.25),
                                 "dphi": SR["dphi"], "mjj": SR["mjj"]}),
        "Q5_dphi_mjj": dict(hist="abcd_scan_4mu_iso_iso", x="dphi", y="mjj",
                            xspec=("ge", 2.0), yspec=("ge", 150.0),
                            cuts={"muiso0": ("lt", 0.25), "muiso1": ("lt", 0.25),
                                  "mudisp0": ("lt", 2.5), "mudisp1": ("lt", 2.5)}),
    },
}

# planes whose axes are quasi-boolean: screened by gates 1-2 only (no closure trend)
SCREENED_ONLY = {"2mu2e": ["P5_muiso_mupix", "P6_mupix_dphi", "P7_egmlost_dphi"],
                 "4mu": ["Q3_iso0_pix0", "Q4_pix_pix"]}


def plane_arrays(hists, channel, plane, parity=None, prescription="i"):
    """(vals, var, xedges, yedges) for one candidate plane.

    parity: None (both), 0 (selection half) or 1 (confirmation half).
    prescription: iso-quirk handling — 'i' sentinel included, 'ii' sentinel events
    dropped wherever iso appears (axes handled by the caller via xlo/ylo in
    region_sums; event-cut iso axes handled here via window).
    """
    spec = PLANES[channel][plane]
    h = at.get_channel(hists[spec["hist"]], CHANNELS[channel])
    sel = dict(spec["cuts"])
    if prescription == "ii":
        for ax, s in list(sel.items()):
            if "iso" in ax and s[0] == "lt":
                sel[ax] = ("window", 0.0, s[1])
    if parity is not None:
        sel["parity"] = ("bin", parity)
    return at.project_plane(h, spec["x"], spec["y"], sel)
