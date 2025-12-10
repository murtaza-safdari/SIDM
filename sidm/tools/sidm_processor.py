"""Module to define the base SIDM processor"""

# python
import copy
import numpy as np
# columnar analysis
from coffea import processor
from coffea.nanoevents.methods import nanoaod
from coffea.nanoevents.methods import vector as cvec
import awkward as ak
import fastjet
import vector
# local
from sidm import BASE_DIR
from sidm.tools import selection, cutflow, utilities
from sidm.definitions.hists import hist_defs, counter_defs
from sidm.definitions.objects import preLj_objs, postLj_objs
import coffea.nanoevents.transforms as tr

def _patched_local2global(stack):
    """
    Original: index,target_offsets,!local2global
    Turn jagged local index into global index
    """
    target_offsets = ak.Array(stack.pop())
    index = ak.Array(stack.pop())
    index = index.mask[index >= 0] + target_offsets[:-1]
    index = index.mask[index < target_offsets[1:]]

    out = ak.flatten(ak.fill_none(index, -1), axis=None)
    out = ak.values_astype(out, np.int64)

    stack.append(out)
tr.local2global = _patched_local2global

class SidmProcessor(processor.ProcessorABC):
    """Class to apply selections, make histograms, and make cutflows"""

    def __init__(
        self,
        channel_names,
        hist_collection_names,
        lj_reco_choices=["0.4"],
        selections_cfg="configs/selections.yaml",
        histograms_cfg="configs/hist_collections.yaml",
        unweighted_hist=False,
        skim_mode=False,
        verbose=False,
    ):
        self.channel_names = channel_names
        self.hist_collection_names = hist_collection_names
        self.lj_reco_choices = lj_reco_choices
        self.selections_cfg = selections_cfg
        self.histograms_cfg = histograms_cfg
        self.unweighted_hist = unweighted_hist
        self.skim_mode = skim_mode
        self.obj_defs = preLj_objs
        self.verbose = verbose
        self.year = "2018" 

    def process(self, events):
        """Apply selections, make histograms and cutflow"""
        
        cutflows = processor.dict_accumulator({})
        counters = processor.dict_accumulator({})
        skims = processor.dict_accumulator({}) 

        if "dataset" in events.metadata:
            dataset_name = events.metadata["dataset"]
        else:
            dataset_name = "Skim"
        
        is_data = events.metadata.get("is_data", False)

        # Pre-initialize skims structure
        if self.skim_mode:
            for lj_reco in self.lj_reco_choices:
                str_lj_reco = str(lj_reco)
                skims[str_lj_reco] = processor.dict_accumulator({})
                for channel in self.channel_names:
                    skims[str_lj_reco][channel] = processor.list_accumulator([])

        objs = self.build_objects(events)
        evt_weights =  self.obj_defs["weight"](events)

        hists = self.build_histograms()
        ch_cuts = self.build_cuts()

        for channel, cuts in ch_cuts.items():
            obj_selection = selection.JaggedSelection(cuts["obj"], self.verbose)
            nested_selection = selection.NestedSelection(cuts["obj"], self.verbose)

            for lj_reco in self.lj_reco_choices:
                str_lj_reco = str(lj_reco)

                # Pre-LJ Selection
                sel_objs = obj_selection.apply_obj_cuts(objs)

                # Nested Selection
                try:
                    sel_objs["dsaMuons"]["good_matched_muons"] = nested_selection.apply_obj_cuts(sel_objs, sel_objs["dsaMuons"].matched_muons, "muons" )
                    sel_objs["muons"]["good_matched_dsa_muons"] = nested_selection.apply_obj_cuts(sel_objs, sel_objs["muons"].matched_dsa_muons,"dsaMuons")
                except Exception as e:
                    if self.verbose: print(f"Note: failed to apply nested selections: {e}")

                # Muon Selection
                prelj_selection = selection.JaggedSelection(cuts["preLj_obj"], self.verbose)
                sel_objs = prelj_selection.apply_obj_cuts_preLj(sel_objs)

                # Reconstruct LJs
                sel_objs["ljs"] = self.build_lepton_jets(sel_objs, float(lj_reco))

                # LJ Selection
                lj_selection = selection.JaggedSelection(cuts["lj"], self.verbose)
                sel_objs = lj_selection.apply_obj_cuts(sel_objs)

                # Post-LJ Objects
                for obj in postLj_objs:
                    sel_objs[obj] = postLj_objs[obj](sel_objs)
                
                postLj_selection = selection.JaggedSelection(cuts["postLj_obj"], self.verbose)
                sel_objs = postLj_selection.apply_obj_cuts(sel_objs)

                # Event Selection
                evt_selection = selection.Selection(cuts["evt"], self.verbose)
                sel_objs = evt_selection.apply_evt_cuts(sel_objs)

                sel_objs["ch"] = channel
                sel_objs["lj_reco"] = lj_reco

                # Final Mask
                final_mask = evt_selection.all_evt_cuts.all(*evt_selection.evt_cuts)
                
                # --- SKIM MODE ---
                if self.skim_mode:
                    skimmed_events = events[final_mask]
                    flat_skim = utilities.flatten_for_parquet(skimmed_events)
                    skims[str_lj_reco][channel] = processor.list_accumulator([flat_skim])
                # -----------------

                hist_weights = evt_weights[final_mask]

                # Cutflow
                if str_lj_reco not in cutflows:
                    cutflows[str_lj_reco] = processor.dict_accumulator({}) 
                cutflows[str_lj_reco][channel] = cutflow.Cutflow(evt_selection.all_evt_cuts, evt_selection.evt_cuts, evt_weights)

                # Histograms
                if not self.skim_mode:
                    fill_weights = hist_weights
                    if self.unweighted_hist:
                        fill_weights =  ak.ones_like(hist_weights)
                    for h in hists.values():
                        h.fill(sel_objs, fill_weights)

                # Counters
                if str_lj_reco not in counters:
                    counters[str_lj_reco] = processor.dict_accumulator({})
                if channel not in counters[str_lj_reco]:
                    counters[str_lj_reco][channel] = processor.dict_accumulator({}) 

                for name, counter in counter_defs.items():
                    try:
                        count_value = counter(sel_objs)
                        counters[str_lj_reco][channel][name] = processor.value_accumulator(int, initial=count_value)
                    except (KeyError, AttributeError):
                        if self.verbose: print(f"Warning: cannot fill counter {name}. Skipping.")

        if len(self.lj_reco_choices) == 1:
            cutflows = cutflows[self.lj_reco_choices[0]]

        # Prepare Output
        if "entrystop" in events.metadata and "entrystart" in events.metadata:
            n_evts = events.metadata["entrystop"] - events.metadata["entrystart"]
        else:
            n_evts = len(events)
            
        skim_factor = events.metadata.get("skim_factor", 1.0)
        scaled_sum_weights = ak.sum(evt_weights) / skim_factor

        # FIX: Clean histograms (remove lambdas) before pickling/returning
        for h in hists.values():
            h.clean_for_pickle()

        out = processor.dict_accumulator({
            "cutflow": cutflows,
            "hists": processor.dict_accumulator(hists),
            "counters": counters,
            "skims": skims,
            "metadata": processor.dict_accumulator({
                "n_evts": processor.value_accumulator(int, initial=n_evts),
                "scaled_sum_weights": processor.value_accumulator(float, initial=scaled_sum_weights),
            }),
        })

        return processor.dict_accumulator({dataset_name: out})

    def build_objects(self, events):
        """Create object collections"""
        objs = {}
        for obj_name, obj_def in self.obj_defs.items():
            try:
                obj = obj_def(events)
            except AttributeError:
                print(f"Warning: {obj_name} not found in this sample. Skipping.")
                continue
            objs[obj_name] = obj
            objs[obj_name] = self.order(objs[obj_name])

            if hasattr(obj, "children"):
                objs[obj_name]["lxy"] = utilities.lxy(objs[obj_name])

            if hasattr(obj, "vx") and not hasattr(obj, "dxy") and "bs" in objs:
                objs[obj_name]["dxy"] = utilities.dxy(objs[obj_name], ref=objs["bs"])

            if objs[obj_name].ndim == 1 and "x" in obj.fields:
                counts = ak.ones_like(objs[obj_name].x, dtype=np.int32)
                objs[obj_name] = ak.unflatten(objs[obj_name], counts)

        return objs

    def make_vector(self, objs, collection, fields, type_id=None, mass=None):
        shape = ak.ones_like(objs[collection].pt, dtype=np.dtype(int))
        forms = {f: objs[collection][f] if f in objs[collection].fields else -1*shape for f in fields}
        forms["part_type"] = objs[collection]["type"] if type_id is None else type_id*shape
        forms["mass"] = objs[collection]["mass"] if mass is None else mass*shape
        return vector.zip(forms)

    def make_constituent(self, consts, type_ids, name, fields):
        relevant_consts = consts[ak.any((consts.part_type == x for x in type_ids), axis=0)]
        forms = {f: relevant_consts.__getattr__(f) for f in fields}
        return ak.zip(forms, with_name=name, behavior=nanoaod.behavior)

    def build_lepton_jets(self, objs, lj_reco):
        """Reconstruct lepton jets according to defintion given by lj_reco"""

        collections = ["muons", "dsaMuons", "electrons", "photons"]
        fields = [objs[c].fields for c in collections]

        unsafe_fields = ['muonIdxG','dsaIdxG','good_matched_muons','good_matched_dsa_muons']

        all_fields = list(set().union(*fields))
        for field in unsafe_fields:
            try:
                all_fields.remove(field)
            except ValueError:
                continue

        muon_inputs = self.make_vector(objs, "muons", all_fields,  type_id=3)
        dsa_inputs = self.make_vector(objs, "dsaMuons", all_fields, type_id=8, mass=0.106)
        ele_inputs = self.make_vector(objs, "electrons", all_fields, type_id=2)
        photon_inputs = self.make_vector(objs, "photons", all_fields, type_id=4)
        lj_inputs = ak.concatenate([muon_inputs, dsa_inputs, ele_inputs, photon_inputs], axis=-1)

        distance_param = abs(lj_reco)
        jet_def = fastjet.JetDefinition(fastjet.antikt_algorithm, distance_param)
        cluster = fastjet.ClusterSequence(lj_inputs, jet_def)
        jets = cluster.inclusive_jets()

        ljs = ak.zip(
            {"x": jets.x, "y": jets.y, "z": jets.z, "t": jets.t},
            with_name="LorentzVector",
            behavior=nanoaod.behavior
        )

        consts = cluster.constituents()
        common_fields = list(set(fields[0]).intersection(*fields[1:]))
        ljs["constituents"] = self.make_constituent(consts, [2, 3, 4, 8], "PtEtaPhiMCollection", common_fields)

        objs["dsaMuons"] = ak.with_field(
            objs["dsaMuons"], 
            ak.full_like(objs["dsaMuons"].pt, 0.105712890625), 
            "mass"
        )

        safe_pf_fields = list(objs["muons"].fields)
        safe_dsa_fields = list(objs["dsaMuons"].fields)

        for field in unsafe_fields:
            if field in safe_pf_fields: safe_pf_fields.remove(field)
            if field in safe_dsa_fields: safe_dsa_fields.remove(field)

        muon_fields = list(set(safe_pf_fields).intersection(safe_dsa_fields))

        ljs["muons"] = self.make_constituent(consts, [3, 8], "Muon", muon_fields)
        ljs["pfMuons"] = self.make_constituent(consts, [3], "Muon", safe_pf_fields)
        ljs["dsaMuons"] = self.make_constituent(consts, [8], "DSAMuon", safe_dsa_fields)
        ljs["electrons"] = self.make_constituent(consts, [2], "Electron", objs["electrons"].fields)
        ljs["photons"] = self.make_constituent(consts, [4], "Photon", objs["photons"].fields)

        ljs["pfMu_n"] = ak.num(ljs.pfMuons, axis=-1)
        ljs["dsaMu_n"] = ak.num(ljs.dsaMuons, axis=-1)
        ljs["muon_n"] = ak.num(ljs.muons, axis=-1)
        ljs["electron_n"] = ak.num(ljs.electrons, axis=-1)
        ljs["photon_n"] = ak.num(ljs.photons, axis=-1)

        ljs["dRSpread"] = ak.max(ak.flatten(
            ljs["constituents"].metric_table(ljs["constituents"], axis=2), axis=-1), axis=-1)

        ljs["matched_jet"] = ljs.nearest(objs["jets"], threshold=0.4)       
        ljs["lepton_fraction"] =  ljs["matched_jet"].chEmEF + ljs["matched_jet"].neEmEF + ljs["matched_jet"].muEF
        ljs["isolation"] = ak.fill_none((ljs["matched_jet"].energy / ljs.energy) * (1 - (ljs["lepton_fraction"])), 0)
        ljs["dR_matched_jet"] = ljs.delta_r(ljs["matched_jet"])

        ljs = self.order(ljs)
        return ljs

    def build_cuts(self):
        """ Make list of cuts"""
        selection_menu = utilities.load_yaml(f"{BASE_DIR}/{self.selections_cfg}")
        ch_cuts = {}

        for channel in self.channel_names:
            ch_cuts[channel] = {"obj": {}, "preLj_obj": {}, "lj": {}, "postLj_obj": {}, "evt": {}}
            cuts = selection_menu[channel]
            
            for obj, obj_cuts in cuts["obj_cuts"].items():
                ch_cuts[channel]["obj"][obj] = utilities.flatten(obj_cuts)

            if "preLj_obj_cuts" in cuts:
                for obj, obj_cuts in cuts["preLj_obj_cuts"].items():
                    ch_cuts[channel]["preLj_obj"][obj] = utilities.flatten(obj_cuts)

            if "postLj_obj_cuts" in cuts:
                for obj, obj_cuts in cuts["postLj_obj_cuts"].items():
                    if obj == "ljs":
                        ch_cuts[channel]["lj"][obj] = utilities.flatten(obj_cuts)
                    else:
                        ch_cuts[channel]["postLj_obj"][obj] = utilities.flatten(obj_cuts)

            if "evt_cuts" in cuts:
                ch_cuts[channel]["evt"] = utilities.flatten(cuts["evt_cuts"])

        return ch_cuts

    def build_histograms(self):
        """Create dictionary of Histogram objects"""
        hist_menu = utilities.load_yaml(f"{BASE_DIR}/{self.histograms_cfg}")
        hists = {}
        for collection in self.hist_collection_names:
            collection = utilities.flatten(hist_menu[collection])
            for hist_name in collection:
                hists[hist_name] = copy.deepcopy(hist_defs[hist_name])
                lj_reco_names = self.lj_reco_choices if len(self.lj_reco_choices) > 1 else None
                hists[hist_name].make_hist(hist_name, self.channel_names, lj_reco_names)
        return hists

    def order(self, obj):
        """Explicitly order objects"""
        if hasattr(obj, "pt"):
            obj = obj[ak.argsort(obj.pt, ascending=False)]
        return obj

    def postprocess(self, accumulator):
        """Modify accumulator after process has run on all chunks"""
        for sample, output in accumulator.items():
            # Robustly handle sum_weights (might be accumulator object OR raw number)
            sum_weights_acc = output["metadata"]["scaled_sum_weights"]
            if hasattr(sum_weights_acc, "value"):
                sum_weights = sum_weights_acc.value
            else:
                sum_weights = sum_weights_acc
            
            lumixs_weight = utilities.get_lumixs_weight(sample, self.year, sum_weights)
            
            for name in output["cutflow"]:
                accumulator[sample]["cutflow"][name].scale(lumixs_weight)
            
            if not self.unweighted_hist:
                for name in output["hists"]:
                    h = accumulator[sample]["hists"][name]
                    # Robust check: Is it a Wrapper or a raw Hist?
                    if hasattr(h, "hist"):
                        h.hist *= lumixs_weight
                    else:
                        # Assume it's a raw Hist from Dask reduction
                        h *= lumixs_weight
                        accumulator[sample]["hists"][name] = h