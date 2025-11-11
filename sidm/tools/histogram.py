"""Module to define the Histogram and Axis classes"""

# columnar analysis
import hist
import awkward as ak
import copy ##### MODIFIED: Import copy #####
from coffea.processor import AccumulatorABC ##### MODIFIED: Import AccumulatorABC #####


class Histogram(AccumulatorABC): ##### MODIFIED: Inherit from AccumulatorABC #####
    """Class to represent histograms

    Histogram mostly exists so that hist.Hists and the appropriate filling arguments can be
    defined in one place. In addition to the filling function associated with each Axis, the user
    can optionally provide an event mask that is applied to all object collections used to fill
    the histogram, e.g. to ensure only events with >=2 muons are used to fill dR(mu, mu) hists.
    
    ##### MODIFIED #####
    This class inherits from AccumulatorABC to be a valid coffea accumulator.
    This requires implementing the 'identity()' and 'add()' methods.
    ####################
    """

    def __init__(self, axes, storage="weight", evt_mask=None):
        self.axes = axes
        self.storage = storage
        # Allow all events to pass if no mask is explicitly provided
        self.evt_mask = (lambda objs: slice(None)) if evt_mask is None else evt_mask
        self.hist = None
        self.name = "" ##### MODIFIED: Added name for debugging #####

    ##### START NEW METHOD #####
    def identity(self):
        """Return an empty Histogram object (required by AccumulatorABC)"""
        # Create a new Histogram object with the same properties
        new_hist = Histogram(self.axes, self.storage, self.evt_mask)
        new_hist.name = self.name
        
        # If self.hist exists (i.e., make_hist has been called),
        # create an empty, identical-in-structure histogram.
        if self.hist:
            # For modern 'hist' objects, we deepcopy and then reset.
            new_hist.hist = copy.deepcopy(self.hist)
            new_hist.hist.reset()
        return new_hist
    ##### END NEW METHOD #####

    ##### START NEW METHOD #####
    def add(self, other):
        """Add another Histogram object to this one (required by AccumulatorABC)"""
        if other.hist:  # Only add if the other one has a histogram
            if self.hist:
                # For modern 'hist' objects, accumulation is done with +=
                self.hist += other.hist
            else:
                # If self is empty, just take a copy of the other one
                # This is crucial for the first merge
                self.hist = copy.deepcopy(other.hist)
    ##### END NEW METHOD #####

    @classmethod
    def simple_hist(cls, obj, attr, absval, nbins, xmin, xmax, label):
        """Method to simplify creation of basic obj.attribute hists"""
        # define fill function
        def f(objs, mask):
            if attr == "n":
                return ak.num(objs[obj])
            elif absval:
                return abs(objs[obj].__getattr__(attr))
            else:
                return objs[obj].__getattr__(attr)
        return cls([
            Axis(hist.axis.Regular(nbins, xmin, xmax, name=f"{obj}_{attr}", label=label), f)
            ])

    def make_hist(self, name, channels=None, lj_reco_choices=None):
        self.name = name ##### MODIFIED: Store name #####
        if channels is not None:
            channel_axis = hist.axis.StrCategory(channels, name="channel", label="Channel")
            self.axes = [Axis(channel_axis, lambda objs, mask: objs["ch"])] + self.axes
        if lj_reco_choices is not None:
            lj_reco_axis = hist.axis.StrCategory(lj_reco_choices, name="lj_reco", label="LJ Reco")
            self.axes = [Axis(lj_reco_axis, lambda objs, mask: objs["lj_reco"])] + self.axes

        axes = [a.axis for a in self.axes]
        self.hist = hist.Hist(*axes, storage=self.storage)

    def fill(self, objs, evt_weights):
        """Fill associated hist.Hist"""
        # Create fill args, warning user and skipping hists that cannot be filled
        try:
            fill_args = {a.name: a.fill_func(objs, self.evt_mask(objs)) for a in self.axes}
        except (AttributeError, KeyError, ValueError) as e:
            print(f"Warning: a histogram with the name {self.name} could not be filled and will be skipped")
            return

        # Use last axis to define weight structure to avoid channels axis
        masked_weights = evt_weights[self.evt_mask(objs)]
        fill_args["weight"] = masked_weights*ak.ones_like(fill_args[self.axes[-1].name])
        for name in fill_args.keys():
            if name not in ("channel", "lj_reco"):
                fill_args[name] = ak.flatten(fill_args[name], axis=None)

        # Fill hist, warning user and skipping hists that cannot be filled
        try:
            self.hist.fill(**fill_args)
        except ValueError:
            print(f"Warning: a histogram with the name {self.name} could not be filled and will be skipped")

class Axis:
    """Class to represent histogram axes

    Axis just bundles together hist.axis objects and functions to fill them.
    """

    def __init__(self, axis, fill_func):
        self.axis = axis
        self.name = axis.name
        self.fill_func = fill_func