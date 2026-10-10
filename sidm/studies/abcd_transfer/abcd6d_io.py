"""Load, blind and project the inclusive 6D ABCD histograms.

The ``abcd6d_<topology>`` histograms are filled with no isolation, displacement, |dphi| or
LJ-pair mass requirement; every region used in the study is a projection of the stored
cube.  Axis order in the file: ``channel`` (one category), then ``iso0``, ``iso1``,
``disp0``, ``disp1``, ``abs_dphi``, ``ljlj_mass``, each with underflow and overflow.

* 2mu2e: ``iso0``/``disp0`` belong to the leading muon LJ (isolation, maximum pixel
  hits of its muons), ``iso1``/``disp1`` to the leading egamma LJ (isolation, minimum
  lost hits of its electrons and photons).
* 4mu: object 0 is the leading muon LJ, object 1 the subleading one, both with pixel hits
  as the displacement variable.

The ``abcd8d_<topology>`` histograms of the follow-up campaign add the mass of the leading
muon LJ (``lj0_mass``) and the number of DSA muons in each muon LJ (``ndsa0``; ``ndsa1`` in 4mu),
with coarser edges on the other axes and flow bins only where a value can fall outside the
edges.  The ``abcd8d_proxy_<topology>`` histograms of the same campaign fill the validation
regions in which a single muon (a one-constituent LJ candidate) stands in for one muon LJ:
in 2mu2e the lone muon takes role 0 next to the egamma LJ, in 4mu the muon LJ keeps role 0
and the lone muon is object 1; they carry the ``abcd8d`` axes without the LJ mass, plus the
number of tight b-tagged jets away from both objects (``nbjets``).  A :class:`Cube` therefore
carries its own :class:`AxisSpec` and the histogram family (``prefix``); every projection,
blinding and region function reads the axes from the cube, and the ``abcd6d`` edges below are
the default spec of a cube built without one.

A muon LJ is displaced when its maximum pixel hits are <= 2, an egamma LJ when its minimum
lost hits are >= 1 (the ``displaced`` cuts of ``sidm/definitions/cuts.py``; photon-only
egamma LJs carry the lost-hits sentinel 999 and are displaced by construction).  Prompt is
the complement (max pixel hits > 2; min lost hits == 0), which is how the validation plane
of the 6D campaign inverts the requirement.  The stricter ``reverse_displaced`` cuts of
``cuts.py`` (every muon > 2 pixel hits, every electron and photon with 0 lost hits) are not
used by any selection and not used here.

Blinding: :func:`blind` sets the cells of the hidden box to NaN, so that any projection
touching them is NaN instead of silently wrong; ``transfer_lib.summed_cubes`` applies it to
data before anything is cached or returned.  The search region is the displaced plane at
LJ-pair mass >= 150 with the leading muon LJ isolation < 0.25, the other LJ isolation < 0.5
and |dphi| >= 2.4.  The hidden box is that region plus a margin on the two ABCD axes:
``iso0`` < ``BLIND_ISO`` (0.5) and |dphi| >= ``BLIND_DPHI_MIN`` (2.0), with ``iso1`` below
``BLIND_ISO``, which is the search region's own edge.  In 4mu the isolated corners of the
two one-object-displaced planes hold up to half of the signal's search-region yield at the
stored normalisation, comparable to the background the prompt-plane factors predict there, so
``BLIND_PLANES`` hides them too.

The proxy regions are disjoint from both search regions, but the 4mu proxy region is not
signal-free: a 4mu signal event whose second dark-photon pair loses one muon is a muon LJ plus
a lone muon, and 22% of the 800/1.2 GeV signal lands there against 1% once a b-tagged jet is
required (the thesis construction).  :func:`blind` therefore hides, in the 4mu proxy cube with
no b-tagged jet, the isolated corner (both isolations < ``BLIND_ISO``, |dphi| >= ``BLIND_DPHI_MIN``,
pair mass >= ``BLIND_MASS_MIN``) of every plane with a displaced object; the 2mu2e proxy region
holds 0.7% of the 2mu2e signal's search-channel yield in a sample several times larger and
stays open.
"""
import dataclasses

import numpy as np

VARIABLES = ("iso0", "iso1", "disp0", "disp1", "abs_dphi", "ljlj_mass")

# Stored bin edges of the campaign.  They are checked against each file at load time.
EDGES = {
    "iso": np.array([0, 0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.4, 0.5, 0.75, 1, 2, 5], float),
    "pixel": np.arange(-0.5, 12, 1.0),
    "lost": np.arange(-0.5, 10, 1.0),
    "abs_dphi": np.array([0, 0.5, 1, 1.5, 2, 2.2, 2.4, 2.6, 2.7, 2.8, 2.9, 3, 3.05, 3.1, np.pi]),
    "ljlj_mass": np.array([0, 50, 100, 150, 200, 300, 400, 600, 800, 1000, 1500, 2500], float),
}

# The ``abcd8d`` campaign: (edges, underflow, overflow) per axis kind; ``lj0_mass`` per topology.
VARIABLES8D = {"2mu2e": VARIABLES + ("lj0_mass", "ndsa0"), "4mu": VARIABLES + ("lj0_mass", "ndsa0", "ndsa1")}
VARIABLES_PROXY = {"2mu2e": VARIABLES + ("ndsa0", "nbjets"), "4mu": VARIABLES + ("ndsa0", "ndsa1", "nbjets")}
FAMILIES = {"abcd8d": VARIABLES8D, "abcd8d_proxy": VARIABLES_PROXY}
EDGES8D = {
    "iso": (np.array([0, 0.1, 0.15, 0.2, 0.25, 0.3, 0.4, 0.5, 0.75, 1, 2, 5], float), False, True),
    "pixel": (np.array([-0.5, 0.5, 1.5, 2.5, 3.5, 4.5, 11.5]), False, False),      # 0, 1, 2, 3, 4, >= 5 hits
    "lost": (np.array([-0.5, 0.5, 1.5, 9.5]), False, True),                        # 0, 1, 2-9, photon-only (999)
    "abs_dphi": (np.array([0, 1, 1.5, 2, 2.2, 2.4, 2.6, 2.8, 3.0, np.pi]), False, True),
    "ljlj_mass": (np.array([0, 100, 150, 200, 300, 400, 600, 800, 1000, 2500], float), False, True),
    "lj0_mass": {"2mu2e": (np.array([0, 5, 30, 60, 75, 105, 150, 300], float), False, True),
                 "4mu": (np.array([0, 5, 75, 105, 300], float), False, True)},
    "ndsa": (np.array([-0.5, 0.5, 1.5, 12.5]), False, False),                      # 0, 1, >= 2 DSA muons
    "nbjets": (np.array([-0.5, 0.5, 1.5, 12.5]), False, False),                    # 0, 1, >= 2 tight b-tagged jets
}

BLIND_ISO = 0.5
BLIND_MASS_MIN = 150.0
BLIND_DPHI_MIN = 2.0
# Planes whose isolated corner (both isolations < BLIND_ISO, |dphi| >= BLIND_DPHI_MIN,
# mass >= BLIND_MASS_MIN) is hidden.
BLIND_PLANES = {
    "2mu2e": (("displaced", "displaced"),),
    "4mu": (("displaced", "displaced"), ("displaced", "prompt"), ("prompt", "displaced")),
}

# Displacement selections per object, as (low, high) value ranges on the stored axes.
# ``None`` means open-ended and includes the flow bin on that side.
MU_DISPLACED = (None, 2.5)      # max pixel hits <= 2
MU_PROMPT = (2.5, None)         # max pixel hits > 2
EGM_DISPLACED = (0.5, None)     # min lost hits >= 1
EGM_PROMPT = (None, 0.5)        # min lost hits == 0


def axis_edges(topology, variable):
    if variable in ("iso0", "iso1"):
        return EDGES["iso"]
    if variable == "disp0":
        return EDGES["pixel"]
    if variable == "disp1":
        return EDGES["lost"] if topology == "2mu2e" else EDGES["pixel"]
    return EDGES[variable]


def displacement(topology, obj, state):
    """Value range of ``disp0``/``disp1`` selecting a displaced or prompt object."""
    egm = topology == "2mu2e" and obj == 1
    if state == "displaced":
        return EGM_DISPLACED if egm else MU_DISPLACED
    if state == "prompt":
        return EGM_PROMPT if egm else MU_PROMPT
    raise ValueError(state)


def plane(topology, obj0, obj1):
    """Displacement selection of a plane, e.g. ``plane("2mu2e", "displaced", "prompt")``."""
    return {"disp0": displacement(topology, 0, obj0), "disp1": displacement(topology, 1, obj1)}


@dataclasses.dataclass(frozen=True, eq=False)
class AxisSpec:
    """The numeric axes of a cube: names, stored edges and which flow bins exist."""

    variables: tuple
    edges: tuple
    underflow: tuple
    overflow: tuple

    def __eq__(self, other):
        return (isinstance(other, AxisSpec) and self.variables == other.variables
                and self.underflow == other.underflow and self.overflow == other.overflow
                and all(np.array_equal(a, b) for a, b in zip(self.edges, other.edges)))

    def index(self, variable):
        if variable not in self.variables:
            raise ValueError(f"{variable}: not an axis of this cube ({self.variables})")
        return self.variables.index(variable)

    def edges_of(self, variable):
        return self.edges[self.index(variable)]

    def nbins(self, variable):
        i = self.index(variable)
        return len(self.edges[i]) - 1 + int(self.underflow[i]) + int(self.overflow[i])

    @property
    def shape(self):
        return tuple(self.nbins(v) for v in self.variables)


def spec6d(topology):
    """Axis spec of the ``abcd6d`` campaign: the six variables, flow bins on both sides."""
    return AxisSpec(VARIABLES, tuple(axis_edges(topology, v) for v in VARIABLES),
                    (True,) * len(VARIABLES), (True,) * len(VARIABLES))


def spec8d(topology, prefix="abcd8d"):
    """Axis spec of the ``abcd8d`` campaign (``prefix="abcd8d_proxy"``: its proxy histograms)."""
    variables = FAMILIES[prefix][topology]
    specs = []
    for variable in variables:
        if variable in ("iso0", "iso1"):
            key = "iso"
        elif variable == "disp0":
            key = "pixel"
        elif variable == "disp1":
            key = "lost" if topology == "2mu2e" else "pixel"
        elif variable.startswith("ndsa"):
            key = "ndsa"
        else:
            key = variable
        spec = EDGES8D[key]
        specs.append(spec[topology] if isinstance(spec, dict) else spec)
    return AxisSpec(variables, tuple(e for e, _, _ in specs),
                    tuple(u for _, u, _ in specs), tuple(o for _, _, o in specs))


def spec_proxy(topology):
    """Axis spec of the ``abcd8d_proxy`` histograms."""
    return spec8d(topology, "abcd8d_proxy")


def spec_from_hist(h):
    """Axis spec read from a stored ``hist.Hist`` (numeric axes after the channel axis)."""
    axes = list(h.axes)[1:]
    return AxisSpec(tuple(ax.name for ax in axes), tuple(np.array(ax.edges, float) for ax in axes),
                    tuple(bool(ax.traits.underflow) for ax in axes), tuple(bool(ax.traits.overflow) for ax in axes))


@dataclasses.dataclass(frozen=True)
class Cube:
    """Weighted sums of one stored histogram, flow bins included, channel axis removed."""

    topology: str
    sumw: np.ndarray
    sumw2: np.ndarray
    blinded: bool = False
    spec: AxisSpec = None
    prefix: str = "abcd6d"      # histogram family: abcd6d, abcd8d (search channels) or abcd8d_proxy

    def __post_init__(self):
        if self.spec is None:
            object.__setattr__(self, "spec", spec6d(self.topology))
        if self.sumw.shape != self.spec.shape:
            raise ValueError(f"cube shape {self.sumw.shape} does not match its axes {self.spec.shape}")

    @property
    def shape(self):
        return self.sumw.shape


def cube_from_hist(h, topology):
    """Build a :class:`Cube` from the stored ``hist.Hist`` (or the SIDM wrapper around it)."""
    h = getattr(h, "hist", h)
    names = [ax.name for ax in h.axes]
    if names[0] != "channel" or tuple(names[1:len(VARIABLES) + 1]) != VARIABLES:
        raise ValueError(f"unexpected axes {names}")
    if len(h.axes[0]) != 1:
        raise ValueError("expected a single channel category")
    variables = tuple(names[1:])
    if variables == VARIABLES:                # the 6D campaign: its edges are fixed
        prefix, spec = "abcd6d", spec6d(topology)
        for variable, ax in zip(VARIABLES, h.axes[1:]):
            if not np.allclose(ax.edges, axis_edges(topology, variable)) or not (ax.traits.underflow and ax.traits.overflow):
                raise ValueError(f"{variable}: stored axis differs from the campaign axis")
    else:                                     # the 8D campaign and its proxy histograms: axes fixed too
        matches = [p for p, family in FAMILIES.items() if family[topology] == variables]
        if not matches:
            raise ValueError(f"stored axes {variables} belong to no known histogram family")
        prefix, spec = matches[0], spec_from_hist(h)
        if spec != spec8d(topology, prefix):
            raise ValueError(f"stored axes {spec.variables} differ from the {prefix} campaign axes")
    view = h.view(flow=True)[0]
    if view.shape != spec.shape:
        raise ValueError(f"unexpected cube shape {view.shape}, expected {spec.shape}")
    return Cube(topology, np.array(view["value"], float), np.array(view["variance"], float), spec=spec, prefix=prefix)


def load_cubes(path, topologies=("2mu2e", "4mu"), sample=None, prefix="abcd6d"):
    """Read one merged ``.coffea`` file and return ``{topology: Cube}`` of its ``<prefix>_<topology>``
    histograms (``abcd6d``, ``abcd8d`` or ``abcd8d_proxy``)."""
    import coffea.util  # imported here so the projection code needs no coffea install

    payload = coffea.util.load(path)
    out = payload["out"]
    if sample is None:
        if len(out) != 1:
            raise ValueError(f"{path}: {len(out)} samples, pass sample=")
        (sample,) = out
    hists = out[sample]["hists"]
    return {t: cube_from_hist(hists[f"{prefix}_{t}"], t) for t in topologies}


def load_cube(path, topology, sample=None, prefix="abcd6d"):
    return load_cubes(path, (topology,), sample, prefix)[topology]


def add_cubes(cubes):
    """Sum of cubes of one topology."""
    cubes = list(cubes)
    if (not cubes or len({(c.topology, c.prefix) for c in cubes}) != 1
            or any(c.spec != cubes[0].spec for c in cubes)):
        raise ValueError("add_cubes needs cubes of one topology and family with the same axes")
    return dataclasses.replace(cubes[0], sumw=sum(c.sumw for c in cubes), sumw2=sum(c.sumw2 for c in cubes),
                               blinded=any(c.blinded for c in cubes))


def _edge_index(edges, value):
    hits = np.nonzero(np.isclose(edges, value))[0]
    if len(hits) != 1:
        raise ValueError(f"{value} is not a stored edge")
    return int(hits[0])


def bin_slice(edges, lo, hi, underflow=True, overflow=True):
    """Flow-aware index slice of a value range on an axis with ``len(edges) - 1`` regular bins
    plus the flow bins that exist.

    Index 0 is the underflow bin when there is one.  ``lo``/``hi`` must be stored edges;
    ``None`` extends the range through the flow bin on that side (or to the outer edge when
    the axis has no flow bin there).
    """
    offset = int(underflow)
    start = 0 if lo is None else _edge_index(edges, lo) + offset
    stop = len(edges) - 1 + offset + int(overflow) if hi is None else _edge_index(edges, hi) + offset
    if stop <= start:
        raise ValueError(f"empty range ({lo}, {hi})")
    return slice(start, stop)


def selection_index(cube, **ranges):
    """Tuple of slices selecting ``ranges`` (``variable=(lo, hi)``) on the cube's axes.  A
    topology string stands for the ``abcd6d`` axes of that topology."""
    spec = spec6d(cube) if isinstance(cube, str) else cube.spec
    unknown = set(ranges) - set(spec.variables)
    if unknown:
        raise ValueError(f"{sorted(unknown)}: not axes of this cube ({spec.variables})")
    index = []
    for i, variable in enumerate(spec.variables):
        lo, hi = ranges.get(variable, (None, None))
        index.append(bin_slice(spec.edges[i], lo, hi, spec.underflow[i], spec.overflow[i]))
    return tuple(index)


# Planes of the 4mu proxy region whose isolated corner is hidden when the event has no b-tagged jet.
BLIND_PLANES_PROXY = {
    "2mu2e": (),
    "4mu": (("displaced", "displaced"), ("displaced", "prompt"), ("prompt", "displaced")),
}


def blind_signature(iso_max=BLIND_ISO, mass_min=BLIND_MASS_MIN, dphi_min=BLIND_DPHI_MIN, planes=None,
                    prefix="abcd6d"):
    """String identifying the hidden box, stored next to cached blinded cubes."""
    if planes is None:                                        # resolved at call time, like blind()
        planes = BLIND_PLANES_PROXY if prefix == "abcd8d_proxy" else BLIND_PLANES
    tag = " nbjets=0" if prefix == "abcd8d_proxy" else ""
    return (f"iso<{iso_max:g} mass>={mass_min:g} dphi>={dphi_min:g}{tag} "
            f"planes={sorted((t, p) for t, ps in planes.items() for p in ps)}")


def blind(cube, iso_max=BLIND_ISO, mass_min=BLIND_MASS_MIN, dphi_min=BLIND_DPHI_MIN, planes=None):
    """Return a copy with the hidden cells set to NaN.

    The box is the isolated corner (both isolations below ``iso_max``) at LJ-pair mass
    >= ``mass_min`` and |dphi| >= ``dphi_min`` in each plane of ``planes`` (default:
    ``BLIND_PLANES[cube.topology]``; for a proxy cube ``BLIND_PLANES_PROXY[cube.topology]``,
    restricted to events with no b-tagged jet).
    """
    proxy = cube.prefix == "abcd8d_proxy"
    if planes is None:
        planes = (BLIND_PLANES_PROXY if proxy else BLIND_PLANES)[cube.topology]
    extra = {"nbjets": (None, 0.5)} if proxy else {}
    sumw, sumw2 = cube.sumw.copy(), cube.sumw2.copy()
    for obj0, obj1 in planes:
        index = selection_index(cube, iso0=(None, iso_max), iso1=(None, iso_max),
                                ljlj_mass=(mass_min, None), abs_dphi=(dphi_min, None),
                                **plane(cube.topology, obj0, obj1), **extra)
        sumw[index] = np.nan
        sumw2[index] = np.nan
    return dataclasses.replace(cube, sumw=sumw, sumw2=sumw2, blinded=True)


def project(cube, **ranges):
    """Weighted sum and its variance over the selected ranges; NaN if a blind cell is touched."""
    index = selection_index(cube, **ranges)
    return float(np.sum(cube.sumw[index])), float(np.sum(cube.sumw2[index]))


def profile(cube, variable, **ranges):
    """Per-bin weighted sums along ``variable`` (flow bins included) after selecting ``ranges``."""
    axis = cube.spec.index(variable)
    index = list(selection_index(cube, **ranges))
    index[axis] = slice(None)
    other = tuple(i for i in range(len(cube.spec.variables)) if i != axis)
    sumw = np.sum(cube.sumw[tuple(index)], axis=other)
    sumw2 = np.sum(cube.sumw2[tuple(index)], axis=other)
    return sumw, sumw2


def regions(cube, boundary0, boundary1, fail0=None, fail1=None, **ranges):
    """A/B/C/D sums for an isolation plane with optional buffers between pass and fail.

    An object passes when its isolation is below its boundary and fails when it is at or
    above ``fail`` (default: the boundary itself).  A = pass/pass, B = fail/pass,
    C = pass/fail, D = fail/fail; the first object is ``iso0``.
    """
    fail0 = boundary0 if fail0 is None else fail0
    fail1 = boundary1 if fail1 is None else fail1
    pass0, fail0 = (None, boundary0), (fail0, None)
    pass1, fail1 = (None, boundary1), (fail1, None)
    return {
        "A": project(cube, iso0=pass0, iso1=pass1, **ranges),
        "B": project(cube, iso0=fail0, iso1=pass1, **ranges),
        "C": project(cube, iso0=pass0, iso1=fail1, **ranges),
        "D": project(cube, iso0=fail0, iso1=fail1, **ranges),
    }


def ratio(num, den):
    """Ratio of two (sumw, sumw2) pairs with uncorrelated-propagation uncertainty."""
    (n, n2), (d, d2) = num, den
    if d <= 0 or n < 0:
        return np.nan, np.nan
    value = n / d
    rel2 = (n2 / n**2 if n > 0 else 0.0) + d2 / d**2
    return value, value * np.sqrt(rel2)


def kappa(reg):
    """kappa = A D / (B C) with propagated uncertainty, and the prediction B C / D."""
    a, b, c, d = (reg[k] for k in "ABCD")
    nan = (np.nan, np.nan)
    if any(np.isnan(x[0]) for x in (b, c, d)) or min(b[0], c[0], d[0]) <= 0:
        return nan, nan
    pred = b[0] * c[0] / d[0]
    pred_err = pred * np.sqrt(b[1] / b[0] ** 2 + c[1] / c[0] ** 2 + d[1] / d[0] ** 2)
    if np.isnan(a[0]) or a[0] <= 0:      # blind or empty A: the prediction still exists
        return nan, (pred, pred_err)
    k = a[0] / pred
    k_err = k * np.sqrt(a[1] / a[0] ** 2 + (pred_err / pred) ** 2)
    return (k, k_err), (pred, pred_err)


def n_eff(sumw, sumw2):
    return sumw**2 / sumw2 if sumw2 > 0 else 0.0
