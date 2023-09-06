"""Module to define the base SIDM processor"""

# python
import copy
import importlib
# columnar analysis
from coffea import processor
import awkward as ak
#local
from sidm.tools import selection, cutflow, histogram, utilities
from sidm.definitions.hists import hist_defs
from sidm.definitions.objects import primary_objs
# always reload local modules to pick up changes during development
importlib.reload(selection)
importlib.reload(cutflow)
importlib.reload(histogram)
importlib.reload(utilities)


class SidmProcessor(processor.ProcessorABC):
    """Class to apply selections, make histograms, and make cutflows

    Accepts NanoEvents records that are assumed to have been produced by FFSchema. Selections are
    chosen by supplying a list of selection names (as defined in selections.yaml), and histograms
    are chosen by providing a list of histogram collection names (as definined in
    hist_collections.yaml).
    """

    def __init__(
        self,
        channel_names,
        hist_collection_names,
        selections_cfg="../configs/selections.yaml",
        histograms_cfg="../configs/hist_collections.yaml"
    ):
        self.channel_names = channel_names
        self.hist_collection_names = hist_collection_names
        self.selections_cfg = selections_cfg
        self.histograms_cfg = histograms_cfg

    def process(self, events):
        """Apply selections, make histograms and cutflow"""

        # create object collections
        objs = {}
        for obj_name, obj_def in primary_objs.items():
            objs[obj_name] = obj_def(events)

            # pt order objects with a pt attribute
            # fixme: would be good to explicitly order other objects as well
            if hasattr(objs[obj_name], "pt"):
                objs[obj_name] = objs[obj_name][ak.argsort(objs[obj_name].pt, ascending=False)]

        # evaluate object selections for all analysis channels
        channels = self.build_analysis_channels(objs)

        # define histograms
        hists = self.build_histograms()

        cutflows = {}
        for channel in channels:
            # apply object selection
            sel_objs = channel.apply_obj_masks(objs)
            # apply event selection
            sel_objs = channel.apply_evt_cuts(sel_objs)

            # fill all hists
            sel_objs["ch"] = channel.name
            evt_weights = events.weightProduct[channel.all_evt_cuts.all(*channel.evt_cuts)]
            for h in hists.values():
                h.fill(sel_objs, evt_weights)

            # make cutflow
            cutflows[channel.name] = cutflow.Cutflow(channel.all_evt_cuts, channel.evt_cuts,
                                                     events.weightProduct)

        out = {
            "cutflow": cutflows,
            "hists": {n: h.hist for n, h in hists.items()}, # output hist.Hists, not Histograms
        }
        return {events.metadata["dataset"]: out}

    def build_analysis_channels(self, objs):
        """Create list of Selection objects that define analysis channels"""
        selection_menu = utilities.load_yaml(self.selections_cfg)

        channels = []
        evaluated_obj_cuts = {}
        for name in self.channel_names:
            cuts = selection_menu[name]
            # flatten object and event cut lists
            for obj, obj_cuts in cuts["obj_cuts"].items():
                cuts["obj_cuts"][obj] = utilities.flatten(obj_cuts)
            cuts["evt_cuts"] = utilities.flatten(cuts["evt_cuts"])

            # build Selection objects
            channel = selection.Selection(name, cuts)
            evaluated_obj_cuts = channel.evaluate_obj_cuts(objs, evaluated_obj_cuts)
            channel.make_obj_masks(evaluated_obj_cuts)
            channels.append(channel)

        return channels

    def build_histograms(self):
        """Create dictionary of Histogram objects"""
        hist_menu = utilities.load_yaml(self.histograms_cfg)

        # build dictionary and create hist.Hist objects
        hists = {}
        for collection in self.hist_collection_names:
            collection = utilities.flatten(hist_menu[collection])
            for hist_name in collection:
                hists[hist_name] = copy.deepcopy(hist_defs[hist_name])
                hists[hist_name].make_hist(self.channel_names)
        return hists

    def postprocess(self, accumulator):
        pass
