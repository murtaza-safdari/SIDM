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
# member-isolation / mother-composition / cosmic-input campaign
EOS_MERGED_MEMBER = EOS_MERGED + "_member"
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

# --- DY generator-weight pathology (Phase-V review finding; see README) ---------
# A handful of unskimmed powheg DY files carry per-event weights up to ~1e13x the
# sample median (36 files listed in the study notes). For M10to50 the SKIMMED events
# are clean (sumw_proc matches the rogue-free denominator), so its denominator is
# repaired here. For M50 the rogue events contaminate the skims themselves
# (sumw_proc ~3e4x the sane value), so the sample is EXCLUDED from weighted results
# and bounded by counts in notebook 01. Both samples are superseded by the team's
# in-progress DYJetsToLL migration.
SUMW_PRE_OVERRIDES = {"DYJetsToMuMu_M10to50": 6.79123e10}  # rogue-free census sum
EXCLUDED_BACKGROUNDS = {
    "DYJetsToMuMu_M50": "rogue generator weights contaminate the skimmed events",
}
ANALYSIS_BACKGROUNDS = {s: p for s, p in BACKGROUNDS.items()
                        if s not in EXCLUDED_BACKGROUNDS}


def fetch(sample, eos_dir=None, tag=""):
    """xrdcp the merged output locally (once) and load it.

    eos_dir/tag select the campaign: default = the plane-choice scan; pass
    eos_dir=EOS_MERGED_MEMBER, tag=".member" for the member campaign
    (the tag keeps the two local caches apart).
    """
    os.makedirs(WORKDIR, exist_ok=True)
    local = os.path.join(WORKDIR, f"{sample}{tag}.coffea")
    if not os.path.exists(local):
        subprocess.run(["xrdcp", "-s", f"{eos_dir or EOS_MERGED}/{sample}.coffea", local],
                       check=True)
    out = coffea.util.load(local)
    out = out["out"] if isinstance(out, dict) and "out" in out else out
    return out[sample]


def load_normalized(sample, sumw_pre_map, ttjets_nnlo=False, eos_dir=None, tag=""):
    """Merged output with hists scaled to lumi*xs/sumw_pre (and f_w corrected).

    Returns (hists_dict, merged_output). Cutflows are NOT corrected — relative only.
    """
    o = fetch(sample, eos_dir=eos_dir, tag=tag)
    factor = at.offline_norm_factor(o["metadata"], sumw_pre_map[sample])
    factor /= FW.get(sample, 1.0)
    if ttjets_nnlo and sample == "TTJets":
        factor *= at.ttjets_xsec_rescale()
    return {n: h * factor for n, h in o["hists"].items()}, o


def accumulate_normalized(samples, sumw_pre_map, keep_prefix="abcd_scan", ttjets_nnlo=False,
                          eos_dir=None, tag=""):
    """Memory-safe sums of normalized hists: (total, by_process).

    Loads ONE sample at a time, keeps only hists whose name starts with keep_prefix
    (the dense scan hists decompress to ~200 MB per sample — holding all 44 samples
    OOMs the interactive node), and frees each sample before the next.
    """
    import gc
    total, by_process = {}, {p: {} for p in PROCESSES}
    for s in samples:
        hists, _ = load_normalized(s, sumw_pre_map, ttjets_nnlo=ttjets_nnlo,
                                   eos_dir=eos_dir, tag=tag)
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
        "P9_egmiso_mjj": dict(hist="abcd_scan_2mu2e_iso_iso", x="egmiso", y="mjj",
                              xspec=("lt", 0.10), yspec=("ge", 150.0),
                              cuts={"muiso": ("lt", 0.25), "dphi": SR["dphi"],
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
        "Q6_iso0_mjj": dict(hist="abcd_scan_4mu_iso_iso", x="muiso0", y="mjj",
                            xspec=("lt", 0.25), yspec=("ge", 150.0),
                            cuts={"muiso1": ("lt", 0.25), "dphi": SR["dphi"],
                                  "mudisp0": ("lt", 2.5), "mudisp1": ("lt", 2.5)}),
    },
}

# planes whose axes are quasi-boolean: screened by gates 1-2 only (no closure trend)
SCREENED_ONLY = {"2mu2e": ["P5_muiso_mupix", "P6_mupix_dphi", "P7_egmlost_dphi"],
                 "4mu": ["Q3_iso0_pix0", "Q4_pix_pix"]}


def plane_arrays(hists, channel, plane, parity=None, prescription="i", cuts_override=None):
    """(vals, var, xedges, yedges) for one candidate plane.

    parity: None (both), 0 (selection half) or 1 (confirmation half).
    prescription: iso-quirk handling — 'i' sentinel included, 'ii' sentinel events
    dropped wherever iso appears (axes handled by the caller via xlo/ylo in
    region_sums; event-cut iso axes handled here via window).
    cuts_override: replace the SR event-cut menu (dict axis->spec; {} = presel,
    axes not listed are integrated) — used by the staged-tightness ladder.
    """
    spec = PLANES[channel][plane]
    h = at.get_channel(hists[spec["hist"]], CHANNELS[channel])
    sel = dict(spec["cuts"]) if cuts_override is None else dict(cuts_override)
    for ax in spec["cuts"]:
        sel.setdefault(ax, "sum")
    if prescription == "ii":
        for ax, s in list(sel.items()):
            if "iso" in ax and s[0] == "lt":
                sel[ax] = ("window", 0.0, s[1])
    if parity is not None:
        sel["parity"] = ("bin", parity)
    return at.project_plane(h, spec["x"], spec["y"], sel)


def snap_edge(edges, v, tie="low"):
    """Nearest bin edge to v (boundary scales must land on exact edges).

    tie: which edge wins an exact tie ("low" or "high"). The ladder passes the
    LOOSENING direction of the axis: a loosened 'ge' boundary (150/2 = 75 on the
    50-GeV mJJ grid) snaps to 50, a loosened 'lt' boundary snaps up — so the
    loose->tight ladder stays monotone loose. The resolved boundary is what the
    stage label's t means; notebook 04 documents the resolved values.
    """
    import numpy as _np
    e = _np.asarray(edges)
    d = _np.abs(e - v)
    i = int(_np.argmin(d))          # argmin returns the LOWER edge on a tie
    if tie == "high" and i + 1 < len(e) and abs(d[i + 1] - d[i]) < 1e-9:
        i += 1
    return float(e[i])


def stage_menu(channel, plane):
    """The staged loose->tight event-cut menu for one plane (see staged_points).

    Returns [(stage_label, cuts_override_dict)] loosest first. Shared so notebook
    04 can rebuild the anchor stage (for the odd-half look and the plateau check)
    with cuts identical to the notebook-02 ladder.
    """
    spec = PLANES[channel][plane]
    cuts = spec["cuts"]
    disp = {k: v for k, v in cuts.items()
            if "disp" in k or "pix" in k or "lost" in k or "iso3" in k
            or k in ("muiso", "egmiso", "muiso0", "muiso1")}
    stages = [("presel", {}), ("+disp/iso cuts", dict(disp))]
    if "dphi" in cuts:
        d2 = dict(disp)
        d2["dphi"] = cuts["dphi"]
        stages.append(("+dphi", d2))
    if "mjj" in cuts:
        d3 = dict(stages[-1][1])
        d3["mjj"] = cuts["mjj"]
        stages.append(("+mjj (SR)", d3))
    return stages


def staged_points(hists, channel, plane, prescription="i", parity=0):
    """Closure R along the staged loose->tight ladder (notebooks 02 and 04).

    Stages relax BOTH the event cuts and the plane boundaries (loosest first):
    the stage_menu() event-cut stages, each at boundary scales t = 2.0, 1.5, 1.0.
    Prescription 'iii' excludes the isolation sentinel bin from the plane axes.
    """
    import numpy as _np
    spec = PLANES[channel][plane]
    stages = stage_menu(channel, plane)
    pts = []
    for si, (slabel, scuts) in enumerate(stages):
        vals, var, xe, ye = plane_arrays(hists, channel, plane, parity=parity,
                                         cuts_override=scuts)
        xlo = 0.0 if (prescription == "iii" and "iso" in spec["x"] and xe[0] < 0) else None
        ylo = 0.0 if (prescription == "iii" and "iso" in spec["y"] and ye[0] < 0) else None
        for t in (2.0, 1.5, 1.0):
            xtie = "high" if spec["xspec"][0] == "lt" else "low"   # loosening side
            ytie = "high" if spec["yspec"][0] == "lt" else "low"
            xc = snap_edge(xe[1:-1], spec["xspec"][1] * (t if spec["xspec"][0] == "lt" else 1 / t), tie=xtie)
            yc = snap_edge(ye[1:-1], spec["yspec"][1] * (t if spec["yspec"][0] == "lt" else 1 / t), tie=ytie)
            reg = at.region_sums(vals, var, xe, ye, (spec["xspec"][0], xc),
                                 (spec["yspec"][0], yc), xlo=xlo, ylo=ylo)
            # health per the REGISTERED rule: n_eff in B, C, D (A is the
            # prediction target, reported separately, not gated)
            ne = min(at.n_eff(*reg[k]) for k in "BCD")
            r, vr = at.closure_ratio(reg)
            pts.append(dict(stage=si, label=f"{slabel} t={t}", R=r,
                            err=float(_np.sqrt(max(vr, 0))), neff=float(ne),
                            neff_A=float(at.n_eff(*reg["A"]))))
    return pts


# ---------------------------------------------------------------------------
# derived / discrete planes (built OFFLINE from the scan hists; user-directed
# phase-space expansion). Each builder returns (vals, var, xedges, yedges) with
# the categorical x axis encoded as integer bin edges [0, 1, ..., n].
# ---------------------------------------------------------------------------
import numpy as np


def _grid3(hists, channel, hname, xa, ya, za, cuts, parity):
    """3D (xa, ya, za) arrays from a scan hist with the remaining axes selected."""
    import hist as _h  # noqa
    h = at.get_channel(hists[hname], CHANNELS[channel])
    sel = dict(cuts)
    if parity is not None:
        sel["parity"] = ("bin", parity)
    axes = [a.name for a in h.axes]
    out = h
    for ax in axes:
        if ax in (xa, ya, za):
            continue
        spec = sel.pop(ax, "sum")
        if spec == "sum":
            out = out[{ax: slice(None, None, sum)}]
        elif spec[0] == "lt":
            out = out[{ax: slice(0, at.edge_index(np.asarray(out.axes[ax].edges), spec[1]), sum)}]
        elif spec[0] == "ge":
            e = np.asarray(out.axes[ax].edges)
            out = out[{ax: slice(at.edge_index(e, spec[1]), len(e) - 1, sum)}]
        elif spec[0] == "window":
            e = np.asarray(out.axes[ax].edges)
            out = out[{ax: slice(at.edge_index(e, spec[1]), at.edge_index(e, spec[2]), sum)}]
        elif spec[0] == "bin":
            out = out[{ax: slice(spec[1], spec[1] + 1, sum)}]
    order = [a.name for a in out.axes]
    v = out.view()["value"]
    w = out.view()["variance"]
    perm = [order.index(a) for a in (xa, ya, za)]
    v = np.transpose(v, perm); w = np.transpose(w, perm)
    edges = [np.asarray(out.axes[a].edges) for a in (xa, ya, za)]
    return v, w, edges


def derived_plane_arrays(hists, channel, kind, parity=None, cuts_override=None):
    """Derived-plane (vals, var, xedges, yedges) built from the scan hists.

    kinds (2mu2e unless noted):
      ntight_dphi    x = N(tight-iso LJs) in {0,1,2}, y = |dphi|      [from H1]
      ntight_mjj     x = N(tight-iso LJs),            y = mJJ         [from H1]
      jetmatch_dphi  x = N(jet-matched LJs) in {0,1,2}, y = |dphi|    [from H1]
      photononly_dphi x = egm-LJ category {e-containing, photon-only}, y = |dphi| [from H3]
      ntight_dphi_4mu x = N(tight) of the two mu-LJs (WP 0.25),  y = |dphi| [4mu H1]
    Default event cuts (overridable): mjj >= 150 + standard displacement; iso cuts
    are consumed by the derived axis where applicable.
    """
    cuts = cuts_override
    if kind in ("ntight_dphi", "ntight_mjj", "jetmatch_dphi"):
        xa, ya = ("muiso", "egmiso")
        za = "dphi" if kind != "ntight_mjj" else "mjj"
        if cuts is None:
            cuts = {"mudisp": ("lt", 2.5), "egmdisp": ("ge", 0.5)}
            cuts["mjj" if kind != "ntight_mjj" else "dphi"] =                 SR["mjj"] if kind != "ntight_mjj" else SR["dphi"]
        v, w, (ex, ey, ez) = _grid3(hists, "2mu2e", "abcd_scan_2mu2e_iso_iso",
                                    xa, ya, za, cuts, parity)
        if kind == "jetmatch_dphi":
            xm = ex[:-1] >= 0.0        # True = real iso (jet matched); sentinel bin < 0
            ym = ey[:-1] >= 0.0
            cats = (xm[:, None, None].astype(int) + ym[None, :, None].astype(int))
        else:
            wpx, wpy = 0.25, 0.10
            xm = (ex[:-1] >= 0.0) & (ex[1:] <= wpx)   # bins fully below WP, sentinel excluded
            ym = (ey[:-1] >= 0.0) & (ey[1:] <= wpy)
            cats = (xm[:, None, None].astype(int) + ym[None, :, None].astype(int))
        nz = v.shape[2]
        out_v = np.zeros((3, nz)); out_w = np.zeros((3, nz))
        for c in (0, 1, 2):
            m = (cats == c)[:, :, 0]
            out_v[c] = v[m].sum(axis=0)
            out_w[c] = w[m].sum(axis=0)
        return out_v, out_w, np.array([0., 1., 2., 3.]), ez
    if kind == "photononly_dphi":
        if cuts is None:
            cuts = {"egmiso": ("window", 0.0, 0.10), "muiso3": ("lt", 0.25),
                    "mudisp2": ("lt", 2.5), "mjj": SR["mjj"]}
        sel = dict(cuts)
        if parity is not None:
            sel["parity"] = ("bin", parity)
        h = at.get_channel(hists["abcd_scan_2mu2e_egmiso_egmlost"], CHANNELS["2mu2e"])
        v, w, ex, ey = at.project_plane(h, "egmlost", "dphi", sel)
        # egmlost edges [-0.5,0.5,1.5,2.5,20,1000]: photon-only = last bin (999 fill)
        out_v = np.stack([v[:-1].sum(axis=0), v[-1]])
        out_w = np.stack([w[:-1].sum(axis=0), w[-1]])
        return out_v, out_w, np.array([0., 1., 2.]), ey
    if kind == "ntight_dphi_4mu":
        if cuts is None:
            cuts = {"mudisp0": ("lt", 2.5), "mudisp1": ("lt", 2.5), "mjj": SR["mjj"]}
        v, w, (ex, ey, ez) = _grid3(hists, "4mu", "abcd_scan_4mu_iso_iso",
                                    "muiso0", "muiso1", "dphi", cuts, parity)
        wp = 0.25
        xm = (ex[:-1] >= 0.0) & (ex[1:] <= wp)
        ym = (ey[:-1] >= 0.0) & (ey[1:] <= wp)
        cats = (xm[:, None, None].astype(int) + ym[None, :, None].astype(int))
        nz = v.shape[2]
        out_v = np.zeros((3, nz)); out_w = np.zeros((3, nz))
        for c in (0, 1, 2):
            m = (cats == c)[:, :, 0]
            out_v[c] = v[m].sum(axis=0)
            out_w[c] = w[m].sum(axis=0)
        return out_v, out_w, np.array([0., 1., 2., 3.]), ez
    raise ValueError(kind)


DERIVED_PLANES = {
    "2mu2e": {
        "D1_ntight_dphi": dict(kind="ntight_dphi", xspec=("ge", 2.0), yspec=("ge", 2.0),
                               stage_axes=["mjj"]),
        "D2_ntight_mjj": dict(kind="ntight_mjj", xspec=("ge", 2.0), yspec=("ge", 150.0),
                              stage_axes=["dphi"]),
        "D3_jetmatch_dphi": dict(kind="jetmatch_dphi", xspec=("ge", 2.0), yspec=("ge", 2.0),
                                 stage_axes=["mjj"]),
        "D4_photononly_dphi": dict(kind="photononly_dphi", xspec=("ge", 1.0), yspec=("ge", 2.0),
                                   stage_axes=["mjj"]),
    },
    "4mu": {
        "D5_ntight_dphi": dict(kind="ntight_dphi_4mu", xspec=("ge", 2.0), yspec=("ge", 2.0),
                               stage_axes=["mjj"]),
    },
}
