"""Inclusive ABCD histograms with two axes the 6D pilot lacks: the mass of the leading muon
LJ and the number of DSA muons in each muon LJ.  Same channels, event selection, object roles
and fill rules as ``abcd6d_pilot``; coarser isolation, |dphi| and pair-mass edges (every
working point and boundary used by the ABCD studies is kept) so that the larger histogram
stays below 0.4 GB in memory."""
import awkward as ak
import hist
import numpy as np

from sidm.definitions.abcd6d_pilot import CHANNELS, checked_array
from sidm.tools.histogram import Histogram

VARIABLES = {
    "2mu2e": ("iso0", "iso1", "disp0", "disp1", "abs_dphi", "ljlj_mass", "lj0_mass", "ndsa0"),
    "4mu": ("iso0", "iso1", "disp0", "disp1", "abs_dphi", "ljlj_mass", "lj0_mass", "ndsa0", "ndsa1"),
}

# (edges, underflow, overflow)
EDGES = {
    "iso": ([0, 0.1, 0.15, 0.2, 0.25, 0.3, 0.4, 0.5, 0.75, 1, 2, 5], False, True),
    "pixel": ([-0.5, 0.5, 1.5, 2.5, 3.5, 4.5, 11.5], False, False),       # 0, 1, 2, 3, 4, >= 5 hits
    "lost": ([-0.5, 0.5, 1.5, 9.5], False, True),                          # 0, 1, 2-9, photon-only (999)
    "abs_dphi": ([0, 1, 1.5, 2, 2.2, 2.4, 2.6, 2.8, 3.0, np.pi], False, True),
    "ljlj_mass": ([0, 100, 150, 200, 300, 400, 600, 800, 1000, 2500], False, True),
    "lj0_mass": {"2mu2e": ([0, 5, 30, 60, 75, 105, 150, 300], False, True),
                 "4mu": ([0, 5, 75, 105, 300], False, True)},
    "ndsa": ([-0.5, 0.5, 1.5, 12.5], False, False),                        # 0, 1, >= 2 DSA muons
}


def nonnegative(isolation):
    """LJ isolation clipped at zero: the matched jet's energy fractions can sum to slightly
    more than one, which makes an isolated object's value a small negative number that the
    isolation axis (no underflow bin) would otherwise drop."""
    return ak.where(isolation < 0, 0.0, isolation)


def event_values(topology, objs):
    mu = objs["mu_ljs"]
    role0 = mu[:, 0]
    if topology == "2mu2e":
        role1 = objs["egm_ljs"][:, 0]
        disp1 = ak.min(role1.egamma.lostHits, axis=-1)
    else:
        role1 = mu[:, 1]
        disp1 = ak.max(role1.muons.trkNumPixelHits, axis=-1)
    values = {
        "iso0": nonnegative(role0.isolation),
        "iso1": nonnegative(role1.isolation),
        "disp0": ak.max(role0.muons.trkNumPixelHits, axis=-1),
        "disp1": disp1,
        "abs_dphi": abs(role0.delta_phi(role1)),
        "ljlj_mass": (role0 + role1).mass,
        "lj0_mass": role0.mass,
        "ndsa0": role0.dsaMu_n,
    }
    if topology == "4mu":
        values["ndsa1"] = role1.dsaMu_n
    return values


def numeric_axis(topology, variable):
    if variable.startswith("iso"):
        key = "iso"
    elif variable == "disp0":
        key = "pixel"
    elif variable == "disp1":
        key = "lost" if topology == "2mu2e" else "pixel"
    elif variable.startswith("ndsa"):
        key = "ndsa"
    else:
        key = variable
    spec = EDGES[key]
    if isinstance(spec, dict):
        spec = spec[topology]
    edges, underflow, overflow = spec
    return hist.axis.Variable(edges, name=variable, label=variable, underflow=underflow, overflow=overflow)


class TopologyHistogram(Histogram):
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
        nmu = ak.num(objs["mu_ljs"])
        valid = (nmu >= 1) & (ak.num(objs["egm_ljs"]) >= 1) if self.topology == "2mu2e" else nmu >= 2
        if not bool(ak.all(valid)):
            raise ValueError(f"{self.name}: channel contains an invalid LJ pair")
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
        name = f"abcd8d_{topology}"
        if name in hist_defs:
            raise ValueError(f"Histogram already defined: {name}")
        hist_defs[name] = TopologyHistogram(topology)
