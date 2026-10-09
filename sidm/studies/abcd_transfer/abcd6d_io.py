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


@dataclasses.dataclass(frozen=True)
class Cube:
    """Weighted sums of one 6D histogram, flow bins included, channel axis removed."""

    topology: str
    sumw: np.ndarray
    sumw2: np.ndarray
    blinded: bool = False

    @property
    def shape(self):
        return self.sumw.shape


def cube_from_hist(h, topology):
    """Build a :class:`Cube` from the stored ``hist.Hist`` (or the SIDM wrapper around it)."""
    h = getattr(h, "hist", h)
    names = [ax.name for ax in h.axes]
    if names[0] != "channel" or tuple(names[1:]) != VARIABLES:
        raise ValueError(f"unexpected axes {names}")
    if len(h.axes[0]) != 1:
        raise ValueError("expected a single channel category")
    for variable, ax in zip(VARIABLES, h.axes[1:]):
        if not np.allclose(ax.edges, axis_edges(topology, variable)):
            raise ValueError(f"{variable}: stored edges differ from the campaign edges")
    view = h.view(flow=True)[0]
    expected = tuple(len(axis_edges(topology, v)) + 1 for v in VARIABLES)
    if view.shape != expected:
        raise ValueError(f"unexpected cube shape {view.shape}, expected {expected}")
    return Cube(topology, np.array(view["value"], float), np.array(view["variance"], float))


def load_cubes(path, topologies=("2mu2e", "4mu"), sample=None):
    """Read one merged ``.coffea`` file and return ``{topology: Cube}`` of its 6D histograms."""
    import coffea.util  # imported here so the projection code needs no coffea install

    payload = coffea.util.load(path)
    out = payload["out"]
    if sample is None:
        if len(out) != 1:
            raise ValueError(f"{path}: {len(out)} samples, pass sample=")
        (sample,) = out
    hists = out[sample]["hists"]
    return {t: cube_from_hist(hists[f"abcd6d_{t}"], t) for t in topologies}


def load_cube(path, topology, sample=None):
    return load_cubes(path, (topology,), sample)[topology]


def add_cubes(cubes):
    """Sum of cubes of one topology."""
    cubes = list(cubes)
    if not cubes or len({c.topology for c in cubes}) != 1:
        raise ValueError("add_cubes needs cubes of one topology")
    return Cube(cubes[0].topology, sum(c.sumw for c in cubes), sum(c.sumw2 for c in cubes),
                blinded=any(c.blinded for c in cubes))


def _edge_index(edges, value):
    hits = np.nonzero(np.isclose(edges, value))[0]
    if len(hits) != 1:
        raise ValueError(f"{value} is not a stored edge")
    return int(hits[0])


def bin_slice(edges, lo, hi):
    """Flow-aware index slice of a value range on an axis with ``len(edges) + 1`` bins.

    Index 0 is the underflow bin, index ``len(edges)`` the overflow bin.  ``lo``/``hi`` must
    be stored edges; ``None`` extends the range through the flow bin on that side.
    """
    start = 0 if lo is None else _edge_index(edges, lo) + 1
    stop = len(edges) + 1 if hi is None else _edge_index(edges, hi) + 1
    if stop <= start:
        raise ValueError(f"empty range ({lo}, {hi})")
    return slice(start, stop)


def selection_index(topology, **ranges):
    """Tuple of slices selecting ``ranges`` (``variable=(lo, hi)``) on the cube."""
    index = []
    for variable in VARIABLES:
        lo, hi = ranges.get(variable, (None, None))
        index.append(bin_slice(axis_edges(topology, variable), lo, hi))
    return tuple(index)


def blind_signature(iso_max=BLIND_ISO, mass_min=BLIND_MASS_MIN, dphi_min=BLIND_DPHI_MIN, planes=None):
    """String identifying the hidden box, stored next to cached blinded cubes."""
    planes = BLIND_PLANES if planes is None else planes      # resolved at call time, like blind()
    return (f"iso<{iso_max:g} mass>={mass_min:g} dphi>={dphi_min:g} "
            f"planes={sorted((t, p) for t, ps in planes.items() for p in ps)}")


def blind(cube, iso_max=BLIND_ISO, mass_min=BLIND_MASS_MIN, dphi_min=BLIND_DPHI_MIN, planes=None):
    """Return a copy with the hidden cells set to NaN.

    The box is the isolated corner (both isolations below ``iso_max``) at LJ-pair mass
    >= ``mass_min`` and |dphi| >= ``dphi_min`` in each plane of ``planes`` (default:
    ``BLIND_PLANES[cube.topology]``).
    """
    planes = BLIND_PLANES[cube.topology] if planes is None else planes
    sumw, sumw2 = cube.sumw.copy(), cube.sumw2.copy()
    for obj0, obj1 in planes:
        index = selection_index(cube.topology, iso0=(None, iso_max), iso1=(None, iso_max),
                                ljlj_mass=(mass_min, None), abs_dphi=(dphi_min, None),
                                **plane(cube.topology, obj0, obj1))
        sumw[index] = np.nan
        sumw2[index] = np.nan
    return dataclasses.replace(cube, sumw=sumw, sumw2=sumw2, blinded=True)


def project(cube, **ranges):
    """Weighted sum and its variance over the selected ranges; NaN if a blind cell is touched."""
    index = selection_index(cube.topology, **ranges)
    return float(np.sum(cube.sumw[index])), float(np.sum(cube.sumw2[index]))


def profile(cube, variable, **ranges):
    """Per-bin weighted sums along ``variable`` (flow bins included) after selecting ``ranges``."""
    axis = VARIABLES.index(variable)
    index = list(selection_index(cube.topology, **ranges))
    index[axis] = slice(None)
    other = tuple(i for i in range(len(VARIABLES)) if i != axis)
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
