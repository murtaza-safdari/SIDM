"""Module to define the Histogram and Axis classes"""

# columnar analysis
import hist
import awkward as ak
import copy
from coffea.processor import AccumulatorABC

# --- Helpers for Pickling (Must be top-level) ---

def get_channel_val(objs, mask):
    """Helper to retrieve channel for histogram filling"""
    return objs["ch"]

def get_lj_reco_val(objs, mask):
    """Helper to retrieve lj_reco for histogram filling"""
    return objs["lj_reco"]

def default_evt_mask(objs):
    """Default mask that selects all events"""
    return slice(None)

class SimpleHistFiller:
    """Functor to replace the local 'f' closure in simple_hist for pickling"""
    def __init__(self, obj, attr, absval):
        self.obj = obj
        self.attr = attr
        self.absval = absval

    def __call__(self, objs, mask):
        if self.attr == "n":
            return ak.num(objs[self.obj])
        elif self.absval:
            return abs(getattr(objs[self.obj], self.attr))
        else:
            return getattr(objs[self.obj], self.attr)

# ------------------------------------------------

class Histogram(AccumulatorABC):
    """Class to represent histograms"""

    def __init__(self, axes, storage="weight", evt_mask=None):
        self.axes = axes
        self.storage = storage
        self.evt_mask = default_evt_mask if evt_mask is None else evt_mask
        self.hist = None
        self.name = ""

    def identity(self):
        """Return an empty Histogram object"""
        new_hist = Histogram(self.axes, self.storage, self.evt_mask)
        new_hist.name = self.name
        if self.hist:
            new_hist.hist = copy.deepcopy(self.hist)
            new_hist.hist.reset()
        return new_hist

    def add(self, other):
        """Add another Histogram object to this one"""
        if other.hist:
            if self.hist:
                self.hist += other.hist
            else:
                self.hist = copy.deepcopy(other.hist)

    def scale(self, weight):
        """Apply scale factor"""
        if self.hist is not None:
            self.hist *= weight

    def clean_for_pickle(self):
        """
        Remove functions (lambdas) that cannot be pickled by standard pickle.
        This must be called before returning the accumulator from the processor.
        """
        self.evt_mask = None
        if self.axes:
            for axis in self.axes:
                axis.fill_func = None # Drop the lambda

    @classmethod
    def simple_hist(cls, obj, attr, absval, nbins, xmin, xmax, label):
        f = SimpleHistFiller(obj, attr, absval)
        return cls([
            Axis(hist.axis.Regular(nbins, xmin, xmax, name=f"{obj}_{attr}", label=label), f)
            ])

    def make_hist(self, name, channels=None, lj_reco_choices=None):
        self.name = name
        if channels is not None:
            channel_axis = hist.axis.StrCategory(channels, name="channel", label="Channel")
            self.axes = [Axis(channel_axis, get_channel_val)] + self.axes
        if lj_reco_choices is not None:
            lj_reco_axis = hist.axis.StrCategory(lj_reco_choices, name="lj_reco", label="LJ Reco")
            self.axes = [Axis(lj_reco_axis, get_lj_reco_val)] + self.axes

        axes = [a.axis for a in self.axes]
        self.hist = hist.Hist(*axes, storage=self.storage)

    def fill(self, objs, evt_weights):
        # Create fill args, warning user and skipping hists that cannot be filled
        try:
            fill_args = {a.name: a.fill_func(objs, self.evt_mask(objs)) for a in self.axes}
        except (AttributeError, KeyError, ValueError) as e:
            print(f"Warning: a histogram with the name {self.name} could not be filled and will be skipped")
            return

        masked_weights = evt_weights[self.evt_mask(objs)]
        fill_args["weight"] = masked_weights*ak.ones_like(fill_args[self.axes[-1].name])
        for name in fill_args.keys():
            if name not in ("channel", "lj_reco"):
                fill_args[name] = ak.flatten(fill_args[name], axis=None)

        try:
            self.hist.fill(**fill_args)
        except ValueError:
            print(f"Warning: a histogram with the name {self.name} could not be filled and will be skipped")

class Axis:
    """Class to represent histogram axes"""
    def __init__(self, axis, fill_func):
        self.axis = axis
        self.name = axis.name
        self.fill_func = fill_func