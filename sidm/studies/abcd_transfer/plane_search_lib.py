"""Search for an ABCD configuration whose validation plane is compatible with the search plane.

Every ABCD plane the inclusive 6D histograms allow is a pair of axes among the leading muon
LJ isolation (``iso0``), the other LJ's isolation (``iso1``) and |dphi|, with the third
variable and the LJ-pair mass cut at the event level, evaluated in a displacement plane:
the four planes of :mod:`transfer_lib` (PP, DP, PD, DD) and finer ones that split a plane
by the stored displacement variable (pixel hits of the muon LJ, lost hits of the egamma LJ).

For one :class:`Config` and one displacement plane :func:`regions` returns A/B/C/D with the
convention of ``transfer_lib``: B shares the pass side of the second axis with A (the
isolation fails), C shares the pass side of the first axis (|dphi| or ``iso1`` fails), D
fails both, so that kappa = A D / (B C) and the in-plane estimate of A is B C / D.  In the
search plane (DD at mass >= 150) region A is hidden and every projection touching the
hidden box is NaN, so a configuration whose fail regions reach into the box reports NaN
there instead of a number.

The quantities compared between a candidate validation plane V and DD are the open ratios
B/D and C/D (pulls in log space), the |dphi| shape of the isolation-fail slice and the
isolation shape of the |dphi|-fail slice (pooled two-sample chi2), the closure kappa_V and
its stability against the LJ-pair mass, and the signal leakage into B, C, D of DD.
"""
import dataclasses
import itertools
import os

import numpy as np
import pandas as pd
from scipy import stats

import abcd6d_io as io
import transfer_lib as tl

AXES4 = ("iso0", "iso1", "abs_dphi", "ljlj_mass")

# ----------------------------------------------------------------------------- planes

# Finer displacement planes: label -> (disp0 range, disp1 range) on the stored axes.
# 2mu2e: disp0 = pixel hits of the muon LJ, disp1 = lost hits of the egamma LJ (999 = photon-only).
_MU = {"0": (None, 0.5), "1": (0.5, 1.5), "2": (1.5, 2.5), "3": (2.5, 3.5), "4": (3.5, 4.5), "5+": (4.5, None),
       "0-1": (None, 1.5), "1-2": (0.5, 2.5), "3-4": (2.5, 4.5)}
_EGM = {"e1": (0.5, 1.5), "e2-9": (1.5, 9.5), "e": (0.5, 9.5), "g": (9.5, None)}


def plane_selection(topology, label):
    """Displacement selection of a plane label: ``PP``/``DP``/``PD``/``DD`` or a split such as
    ``PD:mu3`` (muon LJ with 3 pixel hits, egamma LJ displaced), ``DD:g`` (photon-only egamma
    LJ), ``PP:mu3-4``, ``DD:mu0`` or ``PD:mu5+:e`` (two splits)."""
    base, *splits = label.split(":")
    sel = dict(io.plane(topology, *tl.PLANES[base]))
    for s in splits:
        if s.startswith("mu"):
            key, rng = "disp0", _MU[s[2:]]
        elif topology == "2mu2e" and s in _EGM:
            key, rng = "disp1", _EGM[s]
        elif topology == "4mu" and s.startswith("sub"):
            key, rng = "disp1", _MU[s[3:]]
        else:
            raise ValueError(f"unknown split {s!r} for {topology}")
        sel[key] = _intersect(rng, sel[key], label)
    return sel


def _intersect(rng, base, label):
    """Intersection of a split's value range with its plane's range; a split outside its plane
    (such as ``DD:mu3``) is an error rather than a silent change of plane."""
    lo = base[0] if rng[0] is None else (rng[0] if base[0] is None else max(rng[0], base[0]))
    hi = base[1] if rng[1] is None else (rng[1] if base[1] is None else min(rng[1], base[1]))
    if lo is not None and hi is not None and lo >= hi:
        raise ValueError(f"{label}: the split lies outside its plane")
    return (lo, hi)


def plane_contains(topology, label, state0, state1):
    """True if the split plane lies inside the plane with displacement states (state0, state1)."""
    sel = plane_selection(topology, label)
    full = io.plane(topology, state0, state1)

    def inside(r, full_r):
        lo, hi = r
        flo, fhi = full_r
        return (flo is None or (lo is not None and lo >= flo)) and (fhi is None or (hi is not None and hi <= fhi))
    return inside(sel["disp0"], full["disp0"]) and inside(sel["disp1"], full["disp1"])


# ----------------------------------------------------------------------------- cubes

class Cube4:
    """A per-plane 4D cube (iso0, iso1, abs_dphi, ljlj_mass) with flow bins, e.g. an exported
    signal or background sample in one displacement plane.  ``project`` accepts and ignores
    displacement ranges so the region functions work on both cube types."""

    def __init__(self, topology, sumw, sumw2):
        self.topology, self.sumw, self.sumw2 = topology, sumw, sumw2
        self.blinded = False

    def project(self, **ranges):
        index = tuple(io.bin_slice(io.axis_edges(self.topology, v), *ranges.get(v, (None, None))) for v in AXES4)
        return float(np.sum(self.sumw[index])), float(np.sum(self.sumw2[index]))

    def profile(self, variable, **ranges):
        axis = AXES4.index(variable)
        index = [io.bin_slice(io.axis_edges(self.topology, v), *ranges.get(v, (None, None))) for v in AXES4]
        index[axis] = slice(None)
        other = tuple(i for i in range(4) if i != axis)
        return np.sum(self.sumw[tuple(index)], axis=other), np.sum(self.sumw2[tuple(index)], axis=other)


def project(cube, **ranges):
    if isinstance(cube, Cube4):
        return cube.project(**ranges)
    return io.project(cube, **ranges)


def profile(cube, variable, **ranges):
    if isinstance(cube, Cube4):
        return cube.profile(variable, **ranges)
    return io.profile(cube, variable, **ranges)


def to_cube4(cube, plane):
    """The 4D cube of one displacement plane of a full cube (sum over the displacement axes)."""
    index = io.selection_index(cube.topology, **plane_selection(cube.topology, plane))
    disp_axes = tuple(io.VARIABLES.index(v) for v in ("disp0", "disp1"))
    return Cube4(cube.topology, np.sum(cube.sumw[index], axis=disp_axes), np.sum(cube.sumw2[index], axis=disp_axes))


def load_export(base):
    """``{name: {topology: {plane: Cube4}}}`` of the exported signal points and background groups
    (files written with :func:`to_cube4` for each of the four planes)."""
    out = {}
    for path in sorted(os.listdir(os.path.join(base, "export"))):
        if not path.endswith(".npz"):
            continue
        z = np.load(os.path.join(base, "export", path))
        name = path[: -len(".npz")]
        for key in z.files:
            if not key.endswith("_sumw"):
                continue
            t, plane, _ = key.split("_")
            out.setdefault(name, {}).setdefault(t, {})[plane] = Cube4(t, z[key], z[f"{t}_{plane}_sumw2"])
    return out


# ----------------------------------------------------------------------------- configurations

KINDS = ("dphi x iso0", "iso0 x iso1", "dphi x iso1")


@dataclasses.dataclass(frozen=True)
class Config:
    """One ABCD configuration.

    ``kind`` names the two axes.  For ``dphi x iso0`` the isolation ``iso1`` < ``wp1`` is an
    event cut; for ``iso0 x iso1`` |dphi| >= ``dphi_pass`` is; for ``dphi x iso1`` ``iso0`` <
    ``wp0`` is.  Isolation passes below ``wp`` and fails at or above ``fail`` (``fail`` ==
    ``wp`` means no buffer); |dphi| passes at or above ``dphi_pass`` and fails below
    ``dphi_fail``.  ``mass`` is a (low, high) window on the LJ-pair mass.
    """
    kind: str = "dphi x iso0"
    wp0: float = 0.25
    fail0: float = 0.5
    wp1: float = 0.5
    fail1: float = 0.5
    dphi_pass: float = 2.4
    dphi_fail: float = 2.0
    mass: tuple = (150.0, None)

    def label(self):
        m = f"M>={self.mass[0]:g}" if self.mass[1] is None else (
            f"M<{self.mass[1]:g}" if self.mass[0] is None else f"M {self.mass[0]:g}-{self.mass[1]:g}")
        if self.kind == "dphi x iso0":
            return (f"dphi {self.dphi_pass:g}/{self.dphi_fail:g} x iso0 {self.wp0:g}/{self.fail0:g}, "
                    f"iso1<{self.wp1:g}, {m}")
        if self.kind == "iso0 x iso1":
            return (f"iso0 {self.wp0:g}/{self.fail0:g} x iso1 {self.wp1:g}/{self.fail1:g}, "
                    f"dphi>={self.dphi_pass:g}, {m}")
        return (f"dphi {self.dphi_pass:g}/{self.dphi_fail:g} x iso1 {self.wp1:g}/{self.fail1:g}, "
                f"iso0<{self.wp0:g}, {m}")

    def dd_open(self):
        """True if B, C and D of the DD plane lie outside the hidden box, so that the in-plane
        estimate exists (A is hidden whenever the window reaches mass >= 150)."""
        if self.mass[1] is not None and self.mass[1] <= io.BLIND_MASS_MIN:
            return True
        if self.kind == "dphi x iso0":
            return self.fail0 >= io.BLIND_ISO and self.dphi_fail <= io.BLIND_DPHI_MIN
        if self.kind == "iso0 x iso1":
            return self.fail0 >= io.BLIND_ISO and self.fail1 >= io.BLIND_ISO
        return self.fail1 >= io.BLIND_ISO and self.dphi_fail <= io.BLIND_DPHI_MIN

    def nominal_a(self):
        """True if region A is the planned search region or a subset of it."""
        return (self.kind == "dphi x iso0" and self.wp0 <= 0.25 and self.wp1 <= 0.5 and self.dphi_pass >= 2.4
                and self.mass[0] is not None and self.mass[0] >= 150.0)


def region_ranges(cfg):
    """``{region: ranges}`` on the ABCD axes (without the displacement selection)."""
    mass = {"ljlj_mass": cfg.mass}
    if cfg.kind == "dphi x iso0":
        ev = dict(iso1=(None, cfg.wp1), **mass)
        p0, f0 = (None, cfg.wp0), (cfg.fail0, None)
        pd_, fd = (cfg.dphi_pass, None), (None, cfg.dphi_fail)
        return {"A": dict(iso0=p0, abs_dphi=pd_, **ev), "B": dict(iso0=f0, abs_dphi=pd_, **ev),
                "C": dict(iso0=p0, abs_dphi=fd, **ev), "D": dict(iso0=f0, abs_dphi=fd, **ev)}
    if cfg.kind == "iso0 x iso1":
        ev = dict(abs_dphi=(cfg.dphi_pass, None), **mass)
        p0, f0 = (None, cfg.wp0), (cfg.fail0, None)
        p1, f1 = (None, cfg.wp1), (cfg.fail1, None)
        return {"A": dict(iso0=p0, iso1=p1, **ev), "B": dict(iso0=f0, iso1=p1, **ev),
                "C": dict(iso0=p0, iso1=f1, **ev), "D": dict(iso0=f0, iso1=f1, **ev)}
    ev = dict(iso0=(None, cfg.wp0), **mass)
    p1, f1 = (None, cfg.wp1), (cfg.fail1, None)
    pd_, fd = (cfg.dphi_pass, None), (None, cfg.dphi_fail)
    return {"A": dict(iso1=p1, abs_dphi=pd_, **ev), "B": dict(iso1=f1, abs_dphi=pd_, **ev),
            "C": dict(iso1=p1, abs_dphi=fd, **ev), "D": dict(iso1=f1, abs_dphi=fd, **ev)}


def regions(cube, cfg, plane):
    """A/B/C/D (sumw, sumw2) of ``cfg`` in displacement plane ``plane`` (a label, or a
    selection dict)."""
    disp = plane if isinstance(plane, dict) else plane_selection(cube.topology, plane)
    return {r: project(cube, **rng, **disp) for r, rng in region_ranges(cfg).items()}


def regions_weighted(cube, cfg, parts, weights):
    """A/B/C/D of a plane assembled from sub-planes ``parts`` (labels) with fixed ``weights``."""
    out = {r: [0.0, 0.0] for r in "ABCD"}
    for label, w in zip(parts, weights):
        for r, (s, s2) in regions(cube, cfg, label).items():
            out[r][0] += w * s
            out[r][1] += w * w * s2
    return {r: tuple(v) for r, v in out.items()}


# ----------------------------------------------------------------------------- statistics

def kappa(reg):
    """kappa = (A/C)/(B/D) as (value, minus, plus): Clopper-Pearson ratios for unit weights,
    symmetric propagation otherwise; NaN when A is hidden or a region is empty."""
    ac, bd = tl.ratio(reg["A"], reg["C"]), tl.ratio(reg["B"], reg["D"])
    return tl.ratio_of_ratios(ac, bd)


def prediction(reg):
    """B C / D with its uncertainty (value, err) or NaN."""
    _, pred = io.kappa(reg)
    return pred


def log_sigma(r):
    """Symmetrised uncertainty of ln(value) from an asymmetric (value, minus, plus) ratio."""
    v, m, p = r
    if not np.isfinite(v) or v <= 0 or not np.isfinite(p):
        return np.nan
    up = np.log1p(p / v)
    down = -np.log1p(-min(m, 0.95 * v) / v)
    return 0.5 * (up + down)


def pull(r_a, r_b):
    """Pull of ln(r_a / r_b) against its uncertainty."""
    s = np.hypot(log_sigma(r_a), log_sigma(r_b))
    if not np.isfinite(s) or s <= 0 or not (r_a[0] > 0 and r_b[0] > 0):
        return np.nan
    return np.log(r_a[0] / r_b[0]) / s


def shape_p(cube, variable, sel, plane_a, plane_b):
    """Pooled two-sample chi2 between the ``variable`` distributions of two displacement planes
    under the event selection ``sel`` (unit-weight data): (chi2, ndf, p, n_a, n_b).  NaN when a
    distribution touches the hidden box or is empty."""
    t = cube.topology
    n_a = profile(cube, variable, **sel, **plane_selection(t, plane_a))[0]
    n_b = profile(cube, variable, **sel, **plane_selection(t, plane_b))[0]
    if np.isnan(n_a).any() or np.isnan(n_b).any() or n_a.sum() == 0 or n_b.sum() == 0:
        return np.nan, 0, np.nan, float(np.nansum(n_a)), float(np.nansum(n_b))
    chi2, ndf = tl.shape_chi2(n_a, n_b)
    return chi2, ndf, float(stats.chi2.sf(chi2, ndf)) if ndf > 0 else np.nan, float(n_a.sum()), float(n_b.sum())


def fail_slice_selection(cfg, which):
    """Event selection of the slice whose shape is compared: ``"iso"`` = the isolation-fail
    slice (shape of the first axis), ``"first"`` = the first-axis-fail slice (shape of the
    isolation axis)."""
    rng = region_ranges(cfg)
    d = dict(rng["D"])
    if cfg.kind == "dphi x iso0":
        iso, first = "iso0", "abs_dphi"
    elif cfg.kind == "iso0 x iso1":
        iso, first = "iso0", "iso1"
    else:
        iso, first = "iso1", "abs_dphi"
    if which == "iso":
        d.pop(first)
        return first, d
    d.pop(iso)
    return iso, d


MASS_BINS = [("150-200", (150.0, 200.0)), ("200-300", (200.0, 300.0)), ("300-400", (300.0, 400.0)),
             ("400-600", (400.0, 600.0)), (">=600", (600.0, None))]


def kappa_vs_mass(cube, cfg, plane, bins=MASS_BINS):
    """kappa in mass bins inside the configuration's window, and the chi2 of the bins against
    the integrated value, taken in log space with the symmetrised log error of each bin
    (:func:`log_sigma`), over every bin with a finite kappa; ndf = bins used - 1."""
    rows = []
    k_all = kappa(regions(cube, cfg, plane))
    for label, (lo, hi) in bins:
        lo_w, hi_w = cfg.mass
        lo_b = lo if lo_w is None else (lo_w if lo is None else max(lo, lo_w))
        hi_b = hi if hi_w is None else (hi_w if hi is None else min(hi, hi_w))
        if lo_b is not None and hi_b is not None and lo_b >= hi_b:
            continue
        reg = regions(cube, dataclasses.replace(cfg, mass=(lo_b, hi_b)), plane)
        k = kappa(reg)
        rows.append({"bin": label, **{r: reg[r][0] for r in "ABCD"}, "kappa": k[0], "minus": k[1], "plus": k[2],
                     "log sigma": log_sigma(k)})
    df = pd.DataFrame(rows)
    chi2 = ndf = np.nan
    ok = (df["kappa"] > 0) & np.isfinite(df["log sigma"]) if len(df) else pd.Series(dtype=bool)
    if ok.sum() >= 2 and np.isfinite(k_all[0]) and k_all[0] > 0:
        chi2 = float(np.sum((np.log(df.loc[ok, "kappa"] / k_all[0]) / df.loc[ok, "log sigma"]) ** 2))
        ndf = int(ok.sum()) - 1
    return df, k_all, chi2, ndf


def mass_table(cube, cfg, planes=("PP", "PD")):
    """kappa and A/B per mass bin and plane, with the log-space chi2 of the bins against the
    integrated kappa."""
    rows = []
    for plane in planes:
        df, k_all, chi2, ndf = kappa_vs_mass(cube, cfg, plane)
        for _, b in df.iterrows():
            reg = regions(cube, dataclasses.replace(cfg, mass=dict(MASS_BINS)[b["bin"]]), plane)
            rows.append({"plane": plane, "bin": b["bin"], **{r: f"{b[r]:.0f}" for r in "ABCD"},
                         "kappa": tl.fmt((b["kappa"], b["minus"], b["plus"]), 2), "A/B": tl.fmt(tl.ratio(reg["A"], reg["B"]), 2)})
        rows.append({"plane": plane, "bin": "all", "kappa": tl.fmt(k_all, 2),
                     "A/B": f"chi2 of the bins {chi2:.1f} / {ndf}" if np.isfinite(chi2) else "n/a"})
    return pd.DataFrame(rows).fillna("")


# ----------------------------------------------------------------------------- the scan

def config_grid(kind="dphi x iso0", masses=((150.0, None),)):
    """The configurations scanned per axis kind."""
    cfgs = []
    for mass in masses:
        if kind == "dphi x iso0":
            for wp0, fail0, wp1, dp, df in itertools.product((0.2, 0.25, 0.3), (None, 0.5), (0.25, 0.5, 1.0),
                                                              (2.0, 2.2, 2.4, 2.6, 2.8), (None, 2.0, 1.5)):
                f0 = wp0 if fail0 is None else fail0
                d_fail = dp if df is None else df
                if d_fail > dp:
                    continue
                cfgs.append(Config(kind, wp0, f0, wp1, wp1, dp, d_fail, mass))
        elif kind == "iso0 x iso1":
            for wp0, fail0, wp1, fail1, dp in itertools.product((0.2, 0.25, 0.3), (None, 0.5), (0.25, 0.5),
                                                                 (None, 0.5, 0.75, 1.0), (2.0, 2.4, 2.6)):
                f0 = wp0 if fail0 is None else fail0
                f1 = wp1 if fail1 is None else fail1
                if f1 < wp1:
                    continue
                cfgs.append(Config(kind, wp0, f0, wp1, f1, dp, dp, mass))
        else:
            for wp0, wp1, fail1, dp, df in itertools.product((0.25,), (0.5,), (0.5, 0.75, 1.0),
                                                              (2.0, 2.2, 2.4, 2.6, 2.8), (None, 2.0)):
                d_fail = dp if df is None else df
                if d_fail > dp:
                    continue
                cfgs.append(Config(kind, wp0, wp0, wp1, fail1, dp, d_fail, mass))
    return list(dict.fromkeys(cfgs))


VALIDATION_PLANES = {
    "2mu2e": ["PP", "DP", "PD", "PP:mu3", "PP:mu4", "PP:mu5+", "PD:mu3", "PD:mu4", "PD:mu5+",
              "PD:e1", "PD:e2-9", "PD:e", "PD:g", "PD:mu3:e", "PD:mu3:g", "DP:mu0", "DP:mu1-2"],
    "4mu": ["PP", "PP:mu3", "PP:mu4", "PP:mu5+", "PP:sub3", "PP:sub4", "PP:sub5+"],
}
SEARCH_SPLITS = {"2mu2e": ["DD", "DD:e", "DD:g", "DD:mu0", "DD:mu1-2"], "4mu": ["DD"]}


def scan(cube, cfgs, planes=None, search="DD"):
    """One row per (configuration, validation plane): closure in the plane, open-ratio pulls and
    shape p-values against the search plane, the search plane's counts and in-plane estimate."""
    t = cube.topology
    planes = VALIDATION_PLANES[t] if planes is None else planes
    rows = []
    for cfg in cfgs:
        dd = regions(cube, cfg, search)
        dd_f = {n: tl.ratio(dd[a], dd[b]) for n, (a, b) in (("bd", ("B", "D")), ("cd", ("C", "D")), ("bc", ("B", "C")))}
        pred = prediction(dd)
        for plane in planes:
            reg = regions(cube, cfg, plane)
            k = kappa(reg)
            f = {n: tl.ratio(reg[a], reg[b]) for n, (a, b) in (("bd", ("B", "D")), ("cd", ("C", "D")),
                                                               ("bc", ("B", "C")), ("ac", ("A", "C")), ("ab", ("A", "B")))}
            row = {"config": cfg.label(), "kind": cfg.kind, "wp0": cfg.wp0, "fail0": cfg.fail0, "wp1": cfg.wp1,
                   "fail1": cfg.fail1, "dphi_pass": cfg.dphi_pass, "dphi_fail": cfg.dphi_fail,
                   "mass_lo": cfg.mass[0], "mass_hi": cfg.mass[1], "dd_open": cfg.dd_open(), "plane": plane,
                   **{f"V_{r}": reg[r][0] for r in "ABCD"},
                   "kappa": k[0], "kappa_minus": k[1], "kappa_plus": k[2],
                   "kappa_sig": (k[0] - 1) / (k[1] if k[0] > 1 else k[2]) if np.isfinite(k[0]) else np.nan,
                   **{f"V_{n}": f[n][0] for n in f},
                   **{f"DD_{r}": dd[r][0] for r in "ABCD"},
                   **{f"DD_{n}": dd_f[n][0] for n in dd_f},
                   "DD_pred": pred[0], "DD_pred_err": pred[1],
                   "pull_bd": pull(f["bd"], dd_f["bd"]), "pull_cd": pull(f["cd"], dd_f["cd"]),
                   "pull_bc": pull(f["bc"], dd_f["bc"])}
            row["compat_chi2"] = row["pull_bd"] ** 2 + row["pull_cd"] ** 2
            for which, tag in (("iso", "shape_first"), ("first", "shape_iso")):
                var, sel = fail_slice_selection(cfg, which)
                chi2, ndf, p, n_v, n_d = shape_p(cube, var, sel, plane, search)
                row[f"{tag}_p"], row[f"{tag}_chi2"], row[f"{tag}_ndf"] = p, chi2, ndf
                row[f"{tag}_n"] = f"{n_v:.0f} vs {n_d:.0f}"
            row["pred_from_V"] = pred[0] * k[0] if np.isfinite(k[0]) else np.nan
            rows.append(row)
    return pd.DataFrame(rows)


# ----------------------------------------------------------------------------- signal

def _plane_cube(signal, plane):
    """A full 6D cube is used as is; a dict of exported per-plane 4D cubes is indexed by plane."""
    return signal[plane] if isinstance(signal, dict) else signal


def leakage(signal, cfg, plane="DD"):
    """Signal counts in A/B/C/D of ``plane`` and the fractions B/A, C/A, D/A."""
    reg = regions(_plane_cube(signal, plane), cfg, plane)
    a = reg["A"][0]
    return {**{r: reg[r][0] for r in "ABCD"}, **{f"f{r}": (reg[r][0] / a if a > 0 else np.nan) for r in "BCD"}}


def leakage_table(signals, topology, cfg, data_dd, plane="DD"):
    """Per signal point: leakage fractions and the relative bias of the in-plane estimate when
    the signal in A equals the background estimate (first order in the leaked counts)."""
    rows = []
    b, c, d = (data_dd[r][0] for r in "BCD")
    for name, cubes in signals.items():
        if topology not in cubes:
            continue
        lk = leakage(cubes[topology], cfg, plane)
        bias = (lk["fB"] / b if b > 0 else np.nan) + (lk["fC"] / c if c > 0 else np.nan) - (lk["fD"] / d if d > 0 else np.nan)
        pred = prediction(data_dd)[0]
        rows.append({"point": name.replace("signal_", ""), "S_A": lk["A"], "S_B/S_A": lk["fB"], "S_C/S_A": lk["fC"],
                     "S_D/S_A": lk["fD"], "bias of B C/D at S_A = B C/D": bias * pred if np.isfinite(pred) else np.nan})
    return pd.DataFrame(rows)


def acceptance(signal, cfg, reference, plane="DD"):
    """Signal yield in A of ``cfg`` relative to A of ``reference`` (same plane)."""
    cube = _plane_cube(signal, plane)
    a = regions(cube, cfg, plane)["A"][0]
    a_ref = regions(cube, reference, plane)["A"][0]
    return a / a_ref if a_ref > 0 else np.nan


# ----------------------------------------------------------------------------- summary tables

def plane_table(cube, cfg, planes):
    """Counts, the four ratios and kappa of ``cfg`` in each plane (one row per plane)."""
    rows = []
    for plane in planes:
        reg = regions(cube, cfg, plane)
        f = {n: tl.ratio(reg[a], reg[b]) for n, (a, b) in (("A/B", ("A", "B")), ("B/D", ("B", "D")), ("C/D", ("C", "D")),
                                                           ("A/C", ("A", "C")))}
        pred = prediction(reg)
        rows.append({"plane": plane, **{r: tl.fmt_count(reg[r]) for r in "ABCD"}, **{n: tl.fmt(v, 2) for n, v in f.items()},
                     "B C/D": "n/a" if np.isnan(pred[0]) else f"{pred[0]:.1f} ± {pred[1]:.1f}", "kappa": tl.fmt(kappa(reg), 2)})
    return pd.DataFrame(rows)


def compatibility_table(cube, cfg, planes, search="DD"):
    """Open-ratio pulls and fail-slice shape p-values of each validation plane against the search plane."""
    dd = regions(cube, cfg, search)
    dd_f = {n: tl.ratio(dd[a], dd[b]) for n, (a, b) in (("B/D", ("B", "D")), ("C/D", ("C", "D")))}
    rows = []
    for plane in planes:
        reg = regions(cube, cfg, plane)
        f = {n: tl.ratio(reg[a], reg[b]) for n, (a, b) in (("B/D", ("B", "D")), ("C/D", ("C", "D")))}
        row = {"plane": plane, "kappa": tl.fmt(kappa(reg), 2)}
        for n in f:
            row[f"{n} (plane)"] = tl.fmt(f[n], 2)
            row[f"{n} pull vs {search}"] = f"{pull(f[n], dd_f[n]):+.1f}"
        for which, name in (("iso", "p(first-axis shape, iso-fail slice)"), ("first", "p(iso shape, first-axis-fail slice)")):
            var, sel = fail_slice_selection(cfg, which)
            chi2, ndf, p, n_v, n_d = shape_p(cube, var, sel, plane, search)
            row[name] = "n/a" if np.isnan(p) else f"{p:.2f} ({n_v:.0f} vs {n_d:.0f} events)"
        rows.append(row)
    rows.append({"plane": search, **{f"{n} (plane)": tl.fmt(dd_f[n], 2) for n in dd_f}})
    return pd.DataFrame(rows)


MU_CLASSES = (("0", "mu0"), ("1-2", "mu1-2"), ("3", "mu3"), ("4", "mu4"), (">=5", "mu5+"))


def class_table(cube, cfg, planes=("PP", "DP", "PD", "DD")):
    """Counts and ratios per plane and muon pixel-hit class (classes outside a plane are skipped)."""
    rows = []
    for plane in planes:
        for label, split in (("all", None),) + MU_CLASSES:
            try:
                reg = regions(cube, cfg, plane if split is None else f"{plane}:{split}")
            except ValueError:
                continue
            rows.append({"plane": plane, "pixel hits": label, **{r: tl.fmt_count(reg[r]) for r in "ABCD"},
                         "A/B": tl.fmt(tl.ratio(reg["A"], reg["B"]), 2), "B/D": tl.fmt(tl.ratio(reg["B"], reg["D"]), 2),
                         "C/D": tl.fmt(tl.ratio(reg["C"], reg["D"]), 2), "kappa": tl.fmt(kappa(reg), 2)})
    return pd.DataFrame(rows)


EGM_CLASSES = (("1 lost hit", "e1"), ("2-9 lost hits", "e2-9"), ("photon-only", "g"))


def egm_class_table(cube, cfg, planes=("PD", "DD")):
    """Counts per region of the planes with a displaced egamma LJ, split by its class
    (electron with 1 lost hit, electron with 2-9, photon-only), with the photon-only share."""
    rows = []
    for plane in planes:
        for r in "ABCD":
            per = {label: regions(cube, cfg, f"{plane}:{split}")[r][0] for label, split in EGM_CLASSES}
            tot = sum(per.values())
            rows.append({"plane": plane, "region": r, **{k: ("blind" if np.isnan(v) else f"{v:.0f}") for k, v in per.items()},
                         "photon-only share": "blind" if np.isnan(tot) else (f"{per['photon-only'] / tot:.2f}" if tot > 0 else "n/a")})
    return pd.DataFrame(rows)


def variant_table(cube, planes=("PP", "PD"), base=Config()):
    """kappa of the validation planes for the isolation variants of the nominal configuration
    (other-LJ isolation cut, leading-isolation working point, no buffers)."""
    variants = [("nominal", base)]
    variants += [(f"iso1 < {v:g}", dataclasses.replace(base, wp1=v, fail1=v)) for v in (0.25, 1.0)]
    variants += [(f"iso0 pass < {v:g}", dataclasses.replace(base, wp0=v)) for v in (0.2, 0.3)]
    variants += [("no isolation buffer (fail >= 0.25)", dataclasses.replace(base, fail0=base.wp0)),
                 ("no |dphi| buffer (fail < 2.4)", dataclasses.replace(base, dphi_fail=base.dphi_pass))]
    rows = []
    for name, cfg in variants:
        row = {"variant": name, "in-plane estimate exists in DD": "yes" if cfg.dd_open() else "no (fail regions reach the hidden box)"}
        for plane in planes:
            row[f"kappa {plane}"] = tl.fmt(kappa(regions(cube, cfg, plane)), 2)
        rows.append(row)
    return pd.DataFrame(rows)


def _times(count, factor):
    """count (sumw, sumw2) x factor (value, minus, plus) -> (value, minus, plus) via transfer_lib._pred."""
    return tl._pred(count, factor)


def _product(r1, r2):
    """Product of two (value, minus, plus) ratios, relative errors added in quadrature."""
    if not (np.isfinite(r1[0]) and np.isfinite(r2[0])) or r1[0] <= 0 or r2[0] <= 0:
        return np.nan, np.nan, np.nan
    v = r1[0] * r2[0]
    return v, v * np.hypot(r1[1] / r1[0], r2[1] / r2[0]), v * np.hypot(r1[2] / r1[0], r2[2] / r2[0])


def pixel0_step_ratios(cube, cfg):
    """A/B of the pixel-0 class relative to the prompt classes in the two open populations: with
    a prompt egamma LJ in the configuration's window (DP:mu0 against PP) and with a displaced
    egamma LJ below 150 GeV (DD:mu0 against PD).  Returned as {"egamma prompt": r, "egamma displaced,
    M<150": r} with (value, minus, plus)."""
    low = dataclasses.replace(cfg, mass=(None, 150.0))
    out = {}
    for name, c, zero, prompt in (("egamma prompt", cfg, "DP:mu0", "PP"), ("egamma displaced, M<150", low, "DD:mu0", "PD")):
        r0, rp = regions(cube, c, zero), regions(cube, c, prompt)
        out[name] = tl.ratio_of_ratios(tl.ratio(r0["A"], r0["B"]), tl.ratio(rp["A"], rp["B"]))
    return out


def estimate_table(cube, cfg_windows, kappa_plane="PD", ab_plane="PD", step_cfg=None):
    """The background estimates of the search region per mass window and muon class.

    For the 1-2 pixel-hit class: the in-plane B C/D, the same times kappa of ``kappa_plane``, and
    B times A/B of ``ab_plane``.  For the pixel-0 class: B times A/B of ``ab_plane`` times the
    pixel-0/prompt step ratio, for each of the two open pixel-0 populations
    (:func:`pixel0_step_ratios`, evaluated for ``step_cfg``, default the first window).
    """
    rows = []
    steps = pixel0_step_ratios(cube, cfg_windows[0][1] if step_cfg is None else step_cfg)
    for label, cfg in cfg_windows:
        dd12 = regions(cube, cfg, "DD:mu1-2")
        dd0 = regions(cube, cfg, "DD:mu0")
        v_k = regions(cube, cfg, kappa_plane)
        v_ab = regions(cube, cfg, ab_plane)
        k = kappa(v_k)
        ab = tl.ratio(v_ab["A"], v_ab["B"])
        pred = prediction(dd12)
        in_plane = (pred[0], pred[1], pred[1]) if np.isfinite(pred[0]) else (np.nan, np.nan, np.nan)
        if np.isfinite(k[0]) and np.isfinite(pred[0]) and pred[0] > 0:
            v = pred[0] * k[0]
            scaled = (v, v * np.hypot(pred[1] / pred[0], k[1] / k[0]), v * np.hypot(pred[1] / pred[0], k[2] / k[0]))
        else:
            scaled = (np.nan, np.nan, np.nan)
        ab_est = _times(dd12["B"], ab)
        row = {"window": label, "DD 1-2 hits B/C/D": f"{dd12['B'][0]:.0f}/{dd12['C'][0]:.0f}/{dd12['D'][0]:.0f}",
               "in-plane B C/D": tl.fmt(in_plane, 1), f"kappa {kappa_plane}": tl.fmt(k, 2),
               f"x kappa {kappa_plane}": tl.fmt(scaled, 1), f"A/B {ab_plane}": tl.fmt(ab, 2),
               f"B x A/B {ab_plane}": tl.fmt(ab_est, 1), "DD 0 hits B": f"{dd0['B'][0]:.0f}"}
        for name, step in steps.items():
            row[f"0 hits: B x A/B {ab_plane} x step ({name})"] = f"{tl.fmt(_times(dd0['B'], _product(ab, step)), 1)} (step {tl.fmt(step, 2)})"
        rows.append(row)
    return pd.DataFrame(rows)


def muon_step_tests(cube, cfg):
    """The open tests of the muon displacement step for the 1-2 pixel-hit class and the pixel-0 class."""
    low = dataclasses.replace(cfg, mass=(None, 150.0))
    comp = Config("iso0 x iso1", cfg.wp0, cfg.fail0, 0.5, 0.5, 2.0, 2.0, (None, 150.0))
    tests = [("A/B, EGM prompt, M>=150 (analysis plane)", cfg, "PP", "DP:mu1-2", "DP:mu0"),
             ("A/B, EGM displaced, M<150 (analysis plane)", low, "PD", "DD:mu1-2", "DD:mu0"),
             ("A/B, EGM displaced, M<150 (iso x iso plane)", comp, "PD", "DD:mu1-2", "DD:mu0")]
    rows = []
    for name, c, prompt, disp12, disp0 in tests:
        r_p = regions(cube, c, prompt)
        r_12 = regions(cube, c, disp12)
        r_0 = regions(cube, c, disp0)
        ab_p, ab_12, ab_0 = (tl.ratio(r["A"], r["B"]) for r in (r_p, r_12, r_0))
        row = {"test": name, "prompt plane": f"{prompt}: {tl.fmt(ab_p, 2)} ({r_p['A'][0]:.0f}/{r_p['B'][0]:.0f})",
               "1-2 pixel hits": f"{tl.fmt(ab_12, 2)} ({r_12['A'][0]:.0f}/{r_12['B'][0]:.0f})",
               "ratio 1-2 hits / prompt": tl.fmt(tl.ratio_of_ratios(ab_12, ab_p), 2)}
        for split in ("mu3", "mu4", "mu5+"):
            r_s = regions(cube, c, f"{prompt}:{split}")
            row[f"ratio 1-2 hits / prompt {split[2:]} hits"] = tl.fmt(tl.ratio_of_ratios(ab_12, tl.ratio(r_s["A"], r_s["B"])), 2)
        row["0 pixel hits"] = f"{tl.fmt(ab_0, 2)} ({r_0['A'][0]:.0f}/{r_0['B'][0]:.0f})"
        row["ratio 0 hits / prompt"] = tl.fmt(tl.ratio_of_ratios(ab_0, ab_p), 2)
        rows.append(row)
    for name, c, prompt, disp12 in (("kappa, EGM displaced, M<150 (analysis plane)", low, "PD", "DD:mu1-2"),
                                    ("kappa, EGM displaced, M<150 (iso x iso plane)", comp, "PD", "DD:mu1-2")):
        k_p, k_12 = kappa(regions(cube, c, prompt)), kappa(regions(cube, c, disp12))
        row = {"test": name, "prompt plane": f"{prompt}: {tl.fmt(k_p, 2)}", "1-2 pixel hits": tl.fmt(k_12, 2),
               "ratio 1-2 hits / prompt": tl.fmt(tl.ratio_of_ratios(k_12, k_p), 2)}
        for split in ("mu3", "mu4", "mu5+"):
            row[f"ratio 1-2 hits / prompt {split[2:]} hits"] = tl.fmt(tl.ratio_of_ratios(k_12, kappa(regions(cube, c, f"{prompt}:{split}"))), 2)
        rows.append(row)
    return pd.DataFrame(rows).fillna("")


def signal_table(signals, topology, cfg, reference, plane="DD"):
    """Per signal point: A yield at the stored normalisation, acceptance of A relative to ``reference``,
    leakage fractions and the mass fractions of A above 200, 300 and 400 GeV (in ``plane``)."""
    rows = []
    edges = io.EDGES["ljlj_mass"]
    for name, cubes in sorted(signals.items()):
        if topology not in cubes:
            continue
        cube = _plane_cube(cubes[topology], plane)
        lk = leakage(cubes[topology], cfg, plane)
        rng = {k: v for k, v in region_ranges(cfg)["A"].items() if k != "ljlj_mass"}
        if not isinstance(cube, Cube4):
            rng.update(plane_selection(topology, plane))
        prof = profile(cube, "ljlj_mass", **rng)[0]
        tot = prof.sum()
        row = {"point": name.replace("signal_", ""), "A (stored xs)": f"{lk['A']:.1f}",
               "acceptance vs reference A": f"{acceptance(cubes[topology], cfg, reference, plane):.3f}",
               "S_B/S_A": f"{lk['fB']:.4f}", "S_C/S_A": f"{lk['fC']:.4f}", "S_D/S_A": f"{lk['fD']:.4f}"}
        for e in (200.0, 300.0, 400.0):
            i = int(np.nonzero(np.isclose(edges, e))[0][0])
            row[f"A fraction at M>={e:g}"] = f"{prof[i + 1:].sum() / tot:.3f}" if tot > 0 else "n/a"
        rows.append(row)
    return pd.DataFrame(rows)


def signal_class_table(signals, topology, cfg):
    """Per signal point (full 6D cubes): the share of the search-region yield by muon pixel-hit class
    (0 / 1 / 2) and the share with a prompt muon LJ relative to the search region; for 2mu2e also the
    egamma class shares (lost hits 1 / 2-9 / photon-only) and the share with a prompt egamma LJ."""
    rows = []
    a = region_ranges(cfg)["A"]
    for name, cubes in sorted(signals.items()):
        if topology not in cubes:
            continue
        cube = cubes[topology]
        disp1 = io.displacement(topology, 1, "displaced")
        sr = project(cube, disp0=io.MU_DISPLACED, disp1=disp1, **a)[0]
        if not sr > 0:
            continue
        by_mu = [project(cube, disp0=rng, disp1=disp1, **a)[0] / sr for rng in (_MU["0"], _MU["1"], _MU["2"])]
        row = {"point": name.replace("signal_", ""), "search region (stored xs)": f"{sr:.1f}",
               "pixel hits 0 / 1 / 2": " / ".join(f"{v:.2f}" for v in by_mu),
               "prompt muon LJ / search region": f"{project(cube, disp0=io.MU_PROMPT, disp1=disp1, **a)[0] / sr:.2f}"}
        if topology == "2mu2e":
            by_e = [project(cube, disp0=io.MU_DISPLACED, disp1=rng, **a)[0] / sr for rng in (_EGM["e1"], _EGM["e2-9"], _EGM["g"])]
            row["lost hits 1 / 2-9 / photon-only"] = " / ".join(f"{v:.2f}" for v in by_e)
            row["prompt egamma LJ / search region"] = f"{project(cube, disp0=io.MU_DISPLACED, disp1=io.EGM_PROMPT, **a)[0] / sr:.2f}"
        rows.append(row)
    return pd.DataFrame(rows)


def mc_composition_table(groups, data, topology, cfg, planes=("PP", "DP", "PD", "DD")):
    """Weighted simulation yields per process group and region next to the data counts, with the
    effective count and kappa of the summed simulation.  ``groups`` maps a group name to its cubes
    per topology (full 6D cubes or dicts of exported per-plane cubes)."""
    rows = []
    for plane in planes:
        per = {g: regions(_plane_cube(cubes[topology], plane), cfg, plane) for g, cubes in groups.items()}
        tot = {r: (sum(per[g][r][0] for g in per), sum(per[g][r][1] for g in per)) for r in "ABCD"}
        k = io.kappa(tot)[0]
        d = regions(data, cfg, plane)
        for r in "ABCD":
            rows.append({"plane": plane, "region": r, "data": tl.fmt_count(d[r]),
                         "simulation": f"{tot[r][0]:.1f} ± {np.sqrt(tot[r][1]):.1f}", "N_eff": f"{tl.n_eff(tot[r]):.1f}",
                         **{g: f"{per[g][r][0]:.1f}" for g in sorted(per)},
                         "kappa (simulation)": (f"{k[0]:.2f} ± {k[1]:.2f}" if np.isfinite(k[0]) else "n/a") if r == "A" else ""})
    return pd.DataFrame(rows)


def signal_in_validation_table(signals, topology, data, cfg_windows, planes=("PD", "DP")):
    """Per mass window and validation plane: the data count of region A next to the signal yield
    there at the stored cross section, per signal point."""
    rows = []
    for label, cfg in cfg_windows:
        for plane in planes:
            d = regions(data, cfg, plane)
            row = {"window": label, "plane": plane, "data A": tl.fmt_count(d["A"]), "data B": tl.fmt_count(d["B"])}
            largest = 0.0
            for name, cubes in sorted(signals.items()):
                if topology in cubes:
                    s = regions(_plane_cube(cubes[topology], plane), cfg, plane)["A"][0]
                    largest = max(largest, s)
                    row[f"{name.replace('signal_', '')} in A"] = f"{s:.1f}"
            row["largest signal / data A"] = f"{largest / d['A'][0]:.2f}" if d["A"][0] > 0 else "n/a"
            rows.append(row)
    return pd.DataFrame(rows)


def subtracted_table(groups, data, topology, cfg_windows, plane="PD", group="DY"):
    """Per mass window: the data counts of ``plane``, the simulated yield of one process group in
    each region with its effective count and its share of the data, and kappa and A/B of the plane
    with that yield subtracted from the data (the simulation's statistical error propagated)."""
    rows = []
    for label, cfg in cfg_windows:
        d = regions(data, cfg, plane)
        m = regions(_plane_cube(groups[group][topology], plane), cfg, plane)
        sub = {r: (d[r][0] - m[r][0], d[r][1] + m[r][1]) for r in "ABCD"}
        rows.append({"window": label, "data A/B/C/D": "/".join(tl.fmt_count(d[r]) for r in "ABCD"),
                     f"{group} (simulation) A/B/C/D": "/".join(f"{m[r][0]:.1f}" for r in "ABCD"),
                     f"{group} N_eff A/B/C/D": "/".join(f"{tl.n_eff(m[r]):.1f}" for r in "ABCD"),
                     f"{group} / data": "/".join((f"{m[r][0] / d[r][0]:.2f}" if d[r][0] > 0 else "n/a") for r in "ABCD"),
                     "kappa": tl.fmt(kappa(d), 2), f"kappa, {group} subtracted": tl.fmt(kappa(sub), 2),
                     "A/B": tl.fmt(tl.ratio(d["A"], d["B"]), 2), f"A/B, {group} subtracted": tl.fmt(tl.ratio(sub["A"], sub["B"]), 2)})
    return pd.DataFrame(rows)
