"""Validation-region histograms in which a single muon stands in for one muon LJ (the
construction of W. Si's 2020 thesis, section 3.6.2).  A lone muon is a one-constituent LJ
candidate of the clustering (``proxy_mus``: muon_n == 1, pT > 30 GeV); the channels require
no two-muon LJ (2mu2e proxy) or exactly one (4mu proxy), so both regions are disjoint from
both search regions.  2mu2e proxy: role 0 = leading lone muon, role 1 = leading egamma LJ.
4mu proxy: role 0 = the muon LJ, role 1 = leading lone muon.  Axes as ``abcd8d`` without the
LJ-mass axis, plus the number of tight b-tagged jets (pT > 30, |eta| < 2.4, DeepFlavB tight)
farther than dR 0.4 from both objects."""
import awkward as ak
import hist
import numpy as np

from sidm.definitions.abcd6d_pilot import checked_array
from sidm.definitions.abcd8d import numeric_axis as abcd8d_axis
from sidm.tools.histogram import Histogram

CHANNELS = {"2mu2e": "abcd_proxy_2mu2e_eHEM", "4mu": "abcd_proxy_4mu_eHEM"}

VARIABLES = {
    "2mu2e": ("iso0", "iso1", "disp0", "disp1", "abs_dphi", "ljlj_mass", "ndsa0", "nbjets"),
    "4mu": ("iso0", "iso1", "disp0", "disp1", "abs_dphi", "ljlj_mass", "ndsa0", "ndsa1", "nbjets"),
}

NBJETS_EDGES = [-0.5, 0.5, 1.5, 12.5]          # 0, 1, >= 2 tight b-tagged jets away from both objects


def roles(topology, objs):
    if topology == "2mu2e":
        return objs["proxy_mus"][:, 0], objs["egm_ljs"][:, 0]
    return objs["mu_ljs"][:, 0], objs["proxy_mus"][:, 0]


def nonnegative(isolation):
    """LJ isolation clipped at zero: the matched jet's energy fractions can sum to slightly
    more than one, which makes an isolated object's value a small negative number that the
    isolation axis (no underflow bin) would otherwise drop."""
    return ak.where(isolation < 0, 0.0, isolation)


def event_values(topology, objs):
    role0, role1 = roles(topology, objs)
    if topology == "2mu2e":
        disp1 = ak.min(role1.egamma.lostHits, axis=-1)
    else:
        disp1 = ak.max(role1.muons.trkNumPixelHits, axis=-1)
    bjets = objs["bjets"]
    away = (bjets.delta_r(role0) > 0.4) & (bjets.delta_r(role1) > 0.4)
    values = {
        "iso0": nonnegative(role0.isolation),
        "iso1": nonnegative(role1.isolation),
        "disp0": ak.max(role0.muons.trkNumPixelHits, axis=-1),
        "disp1": disp1,
        "abs_dphi": abs(role0.delta_phi(role1)),
        "ljlj_mass": (role0 + role1).mass,
        "ndsa0": role0.dsaMu_n,
        "nbjets": ak.sum(away, axis=-1),
    }
    if topology == "4mu":
        values["ndsa1"] = role1.dsaMu_n
    return values


def numeric_axis(topology, variable):
    if variable == "nbjets":
        return hist.axis.Variable(NBJETS_EDGES, name=variable, label=variable, underflow=False, overflow=False)
    return abcd8d_axis(topology, variable)


class ProxyHistogram(Histogram):
    def __init__(self, topology):
        super().__init__(axes=[], storage="weight")
        self.topology = topology
        self.variables = VARIABLES[topology]
        self.channel = CHANNELS[topology]

    def make_hist(self, name, channels=None, lj_reco_choices=None):
        if channels is None or self.channel not in channels:
            raise ValueError(f"{name}: required channel {self.channel} missing")
        self.name = name
        axes = [hist.axis.StrCategory([self.channel], name="channel", growth=False, overflow=False)]
        self.has_reco_axis = lj_reco_choices is not None
        if self.has_reco_axis:
            axes.append(hist.axis.StrCategory(lj_reco_choices, name="lj_reco", growth=False, overflow=False))
        axes.extend(numeric_axis(self.topology, v) for v in self.variables)
        self.hist = hist.Hist(*axes, storage=hist.storage.Weight())

    def fill(self, objs, evt_weights, verbose=False):
        if objs["ch"] != self.channel:
            return
        nproxy = ak.num(objs["proxy_mus"])
        if self.topology == "2mu2e":
            valid = (nproxy >= 1) & (ak.num(objs["egm_ljs"]) >= 1)
        else:
            valid = (nproxy >= 1) & (ak.num(objs["mu_ljs"]) >= 1)
        if not bool(ak.all(valid)):
            raise ValueError(f"{self.name}: channel contains an event without the proxy pair")
        weights = checked_array(evt_weights, "weight")
        if not np.isfinite(weights).all():
            raise ValueError(f"{self.name}: nonfinite weights")
        if len(weights) == 0:
            return
        values = event_values(self.topology, objs)
        args = {v: checked_array(values[v], v, len(weights)) for v in self.variables}
        args["channel"] = self.channel
        if self.has_reco_axis:
            args["lj_reco"] = objs["lj_reco"]
        self.hist.fill(**args, weight=weights)


def register(hist_defs):
    for topology in CHANNELS:
        name = f"abcd8d_proxy_{topology}"
        if name in hist_defs:
            raise ValueError(f"Histogram already defined: {name}")
        hist_defs[name] = ProxyHistogram(topology)
