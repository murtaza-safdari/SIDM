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
#local
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
        lj_reco_choices=["0.4"],
        selections_cfg="configs/selections.yaml",
        histograms_cfg="configs/hist_collections.yaml",
        unweighted_hist=False,
        save_intermediate=False, ##### MODIFIED: Added new flag #####
        read_intermediate=False, ##### MODIFIED: Added new flag #####
        verbose=False,
    ):
        self.channel_names = channel_names
        self.hist_collection_names = hist_collection_names
        self.lj_reco_choices = lj_reco_choices
        self.selections_cfg = selections_cfg
        self.histograms_cfg = histograms_cfg
        self.unweighted_hist = unweighted_hist
        self.save_intermediate = save_intermediate ##### MODIFIED: Store flag #####
        self.read_intermediate = read_intermediate ##### MODIFIED: Store flag #####
        self.obj_defs = preLj_objs
        self.verbose = verbose
        self.year = "2018" # fixme: may be better to store as event metadata

    def process(self, item): ##### MODIFIED: Changed 'events' to 'item' #####
        """Apply selections, make histograms and cutflow"""
        
        # --- START NEW SECTION ---
        # Handle the two different input types:
        # 1. 'read_intermediate=True': 'item' is a tuple (ak.Record, dataset_name)
        #    loaded manually from our skim file.
        # 2. 'read_intermediate=False': 'item' is a standard NanoEvents object
        #    from the Dask runner.
        if self.read_intermediate:
            # When calling manually, we pass a tuple: (events, dataset_name)
            try:
                events, dataset_name = item
            except (TypeError, ValueError):
                # Fallback for old/manual calls
                events = item
                dataset_name = "intermediate_skim" # hardcoded fallback
            
            # 'events' is a "dumb" ak.Record from Parquet.
            # We must "re-awaken" the behaviors (like .pt, .eta, .nearest())
            # that were stripped when saving.
            objs = {}
            behavior_map = {
                "muons": "Muon",
                "dsaMuons": "DSAMuon",
                "electrons": "Electron",
                "photons": "Photon",
                "jets": "Jet",
                "genParticles": "GenParticle", 
                "ljs": "LorentzVector",      
            }

            # Use .fields because 'events' is an ak.Record
            for field in events.fields:
                if field in behavior_map:
                    # ak.with_name "tags" the loaded array with its
                    # behavior name (e.g., "Muon"), which links it
                    # to the nanoaod.behavior methods.
                    objs[field] = ak.with_name(
                        events[field],
                        behavior_map[field],
                        behavior=nanoaod.behavior,
                    )
                else:
                    objs[field] = events[field] # e.g., 'weights', 'dataset'
            
            try:
                evt_weights = objs["weights"]
            except KeyError:
                print("Warning: 'weights' not found in intermediate file. Using weight=1.")
                # evt_weights = ak.ones_like(objs[list(objs.keys())[0]])
                evt_weights = ak.ones_like(ak.sum(out[samples[0]]['events']["base"]["bs"]['ntracks'], axis=1)) # FIXME
        else:
            # Standard path: build objects from NanoAOD
            events = item # <--- 'item' is just 'events' in this case
            dataset_name = events.metadata["dataset"]
            objs = self.build_objects(events)
            evt_weights =  self.obj_defs["weight"](events)
        # --- END NEW SECTION ---
        
        # --- START MODIFIED SECTION ---
        # All outputs must be coffea accumulators.
        # Initialize them as empty dict_accumulators.
        cutflows = processor.dict_accumulator({})
        counters = processor.dict_accumulator({})
        intermediate_events = processor.dict_accumulator({}) 
        # --- END MODIFIED SECTION ---

        # define histograms
        # This returns a dict of our 'Histogram' wrapper objects
        hists = self.build_histograms()

        ### define pre-lj object, lj, post-lj obj, and event cuts per channel
        ch_cuts = self.build_cuts()

        # loop through lj reco choices and channels, treating each lj+channel pair as a unique Selection
        for channel, cuts in ch_cuts.items():
            obj_selection = selection.JaggedSelection(cuts["obj"], self.verbose)
            nested_selection = selection.NestedSelection(cuts["obj"], self.verbose)

            for lj_reco in self.lj_reco_choices:
                str_lj_reco = str(lj_reco) # Use string key for safety

                # --- START NEW SECTION --- 
                if self.read_intermediate:
                    # When reading intermediate, 'objs' is already the 
                    # 'sel_objs' from the previous run.
                    # We skip all object building and LJ reconstruction.
                    sel_objs = objs
                    
                    # We *re-apply* all selections. This allows
                    # us to run with new/modified selection configs.
                    sel_objs = obj_selection.apply_obj_cuts(sel_objs)
                    
                    # Re-apply nested selection
                    try:
                        # This 'try' block is CRITICAL.
                        # The 'matched_muons' fields are complex cross-references
                        # that are *DROPPED* by ak.to_parquet.
                        # This block will fail, print a warning, and continue,
                        # which is the expected behavior.
                        sel_objs["dsaMuons"]["good_matched_muons"] = nested_selection.apply_obj_cuts(sel_objs, sel_objs["dsaMuons"].matched_muons, "muons" )
                        sel_objs["muons"]["good_matched_dsa_muons"] = nested_selection.apply_obj_cuts(sel_objs, sel_objs["muons"].matched_dsa_muons,"dsaMuons")
                    except Exception as e:
                        print(f"Note: failed to apply nested selections on skim (this is normal if fields were dropped): {e}")

                    # Re-apply selections to muons
                    prelj_selection = selection.JaggedSelection(cuts["preLj_obj"], self.verbose)
                    sel_objs = prelj_selection.apply_obj_cuts_preLj(sel_objs)

                    # *** DO NOT RECONSTRUCT LJs ***
                    # (LJs are already in sel_objs)

                    # Re-apply obj selection to ljs
                    lj_selection = selection.JaggedSelection(cuts["lj"], self.verbose)
                    sel_objs = lj_selection.apply_obj_cuts(sel_objs)

                    # *** DO NOT RE-ADD POST-LJ OBJS ***
                    # (They are already in sel_objs)

                else:
                    # Standard path (this is the original code)
                    # apply pre-LJ object selection
                    sel_objs = obj_selection.apply_obj_cuts(objs)

                    # apply selections on matched_muons...
                    try:
                        sel_objs["dsaMuons"]["good_matched_muons"] = nested_selection.apply_obj_cuts(sel_objs, sel_objs["dsaMuons"].matched_muons, "muons" )
                        sel_objs["muons"]["good_matched_dsa_muons"] = nested_selection.apply_obj_cuts(sel_objs, sel_objs["muons"].matched_dsa_muons,"dsaMuons")
                    except Exception as e:
                        print(f"Failed to apply selections to the nested matched muon collections. Error message: {e}")

                    # apply selections to muons...
                    prelj_selection = selection.JaggedSelection(cuts["preLj_obj"], self.verbose)
                    sel_objs = prelj_selection.apply_obj_cuts_preLj(sel_objs)

                    # reconstruct lepton jets
                    sel_objs["ljs"] = self.build_lepton_jets(sel_objs, float(lj_reco))

                    # apply obj selection to ljs
                    lj_selection = selection.JaggedSelection(cuts["lj"], self.verbose)
                    sel_objs = lj_selection.apply_obj_cuts(sel_objs)

                    # add post-lj objects to sel_objs
                    for obj in postLj_objs:
                        sel_objs[obj] = postLj_objs[obj](sel_objs)
                # --- END NEW SECTION ---

                # This part now runs for BOTH paths:
                # apply post-lj obj selection
                postLj_selection = selection.JaggedSelection(cuts["postLj_obj"], self.verbose)
                sel_objs = postLj_selection.apply_obj_cuts(sel_objs)

                # build Selection objects and apply event selection
                evt_selection = selection.Selection(cuts["evt"], self.verbose)
                sel_objs = evt_selection.apply_evt_cuts(sel_objs)

                # fill all hists
                sel_objs["ch"] = channel
                sel_objs["lj_reco"] = lj_reco

                # --- START MODIFIED SECTION ---
                
                # Get final event mask
                final_mask = evt_selection.all_evt_cuts.all(*evt_selection.evt_cuts)

                # Define histogram weights
                if self.read_intermediate:
                    # evt_weights was loaded *filtered*. We just applied *new*
                    # cuts via final_mask. We must filter it *again*.
                    hist_weights = evt_weights[final_mask]
                else:
                    # evt_weights is *unfiltered*. Filter it once.
                    hist_weights = evt_weights[final_mask]
                
                # --- END MODIFIED SECTION ---

                # --- START MODIFIED SECTION ---

                # make cutflow
                if str_lj_reco not in cutflows:
                    # Must add a dict_accumulator, not a plain dict
                    cutflows[str_lj_reco] = processor.dict_accumulator({}) 
                cutflows[str_lj_reco][channel] = cutflow.Cutflow(evt_selection.all_evt_cuts, evt_selection.evt_cuts, evt_weights)

                # Store intermediate events if requested
                if self.save_intermediate:
                    if str_lj_reco not in intermediate_events:
                        # Must add a dict_accumulator
                        intermediate_events[str_lj_reco] = processor.dict_accumulator({}) 
                    
                    selected_events = {}
                    for key, value in sel_objs.items():
                        try:
                            # Filter all event-parallel awkward arrays
                            if isinstance(value, ak.Array) and len(value) == len(final_mask):
                                selected_events[key] = value[final_mask]
                            # Keep non-parallel info (like 'ch', 'lj_reco')
                            else:
                                selected_events[key] = value
                        except Exception:
                            selected_events[key] = value
                    
                    # Store the weights for the selected events
                    selected_events["weights"] = hist_weights # Save the *new* filtered weights
                    intermediate_events[str_lj_reco][channel] = selected_events

                # fill histograms for this channel+lj_reco pair
                fill_weights = hist_weights # Use the filtered weights
                if self.unweighted_hist:
                    fill_weights =  ak.ones_like(hist_weights)
                for h in hists.values():
                    # h is our 'Histogram' object, which has a .fill() method
                    h.fill(sel_objs, fill_weights)
                
                # --- END MODIFIED SECTION ---

                # Fill counters
                if str_lj_reco not in counters:
                    # Must add a dict_accumulator
                    counters[str_lj_reco] = processor.dict_accumulator({})
                if channel not in counters[str_lj_reco]:
                    # Must add a dict_accumulator
                    counters[str_lj_reco][channel] = processor.dict_accumulator({}) 

                for name, counter in counter_defs.items():
                    try:
                        # Calculate the counter value
                        count_value = counter(sel_objs)
                        # Must store it as a value_accumulator to be a valid leaf
                        counters[str_lj_reco][channel][name] = processor.value_accumulator(int, initial=count_value)
                    except (KeyError, AttributeError) as e:
                        print(f"Warning: cannot fill counter {name}. Skipping.")

        # lose lj_reco dimension to cutflows if only one reco was run
        if len(self.lj_reco_choices) == 1:
            cutflows = cutflows[self.lj_reco_choices[0]]

        # --- START MODIFIED SECTION ---
        
        # --- START METADATA BLOCK FIX ---
        # Get metadata, handling both NanoAOD and skim inputs
        if self.read_intermediate:
            # 'objs' is a dict. Get length from the first collection.
            first_collection_name = list(objs.keys())[0] 
            n_evts = len(objs[first_collection_name])
            scaled_sum_weights = ak.sum(evt_weights)
            # 'dataset_name' was passed in the 'item' tuple
        else:
            n_evts = events.metadata["entrystop"] - events.metadata["entrystart"]
            scaled_sum_weights = ak.sum(evt_weights)/events.metadata["skim_factor"]
            # 'dataset_name' was already set from events.metadata["dataset"]
        # --- END METADATA BLOCK FIX ---

        # All leaves of the accumulator must be AccumulatorABC objects
        # Wrap all outputs in their accumulator types
        out = processor.dict_accumulator({
            "cutflow": cutflows, # This is now a dict_accumulator
            "hists": processor.dict_accumulator(hists), # 'hists' is a dict of our 'Histogram' accumulators
            "counters": counters, # This is now a dict_accumulator
            "metadata": processor.dict_accumulator({
                # Wrap metadata numbers in value_accumulator
                "n_evts": processor.value_accumulator(int, initial=n_evts),
                "scaled_sum_weights": processor.value_accumulator(float, initial=scaled_sum_weights),
            }),
        })

        if self.save_intermediate:
            # This 'save_intermediate' logic is incompatible with the
            # manual IterativeExecutor call, as 'ak.Array' is not
            # an accumulator leaf. It works with Dask runners.
            if not self.read_intermediate:
                if len(self.lj_reco_choices) == 1:
                     intermediate_events["dataset"] = dataset_name
                else:
                    for lj_reco_key in intermediate_events:
                        intermediate_events[lj_reco_key]["dataset"] = dataset_name
            
            if len(self.lj_reco_choices) == 1:
                intermediate_events = intermediate_events[self.lj_reco_choices[0]]

            out["events"] = intermediate_events

        # The final output MUST be a dict_accumulator
        # to work with the IterativeExecutor.
        return processor.dict_accumulator({dataset_name: out})
        # --- END MODIFIED SECTION ---

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

            # pt order
            objs[obj_name] = self.order(objs[obj_name])


            # add lxy attribute to particles with children
            if hasattr(obj, "children"):
                objs[obj_name]["lxy"] = utilities.lxy(objs[obj_name])

            # add dxy wrt beamspot for all objs that don't already have it
            if hasattr(obj, "vx") and not hasattr(obj, "dxy") and "bs" in objs:
                objs[obj_name]["dxy"] = utilities.dxy(objs[obj_name], ref=objs["bs"])

            # add dimension to one-per-event objects to allow independent obj and evt cuts
            # skip objects with no fields
            if objs[obj_name].ndim == 1 and "x" in obj.fields:
                counts = ak.ones_like(objs[obj_name].x, dtype=np.int32)
                objs[obj_name] = ak.unflatten(objs[obj_name], counts)

        return objs

    def make_vector(self, objs, collection, fields, type_id=None, mass=None):
        shape = ak.ones_like(objs[collection].pt, dtype=np.dtype(int))
        # all objects must have the same fields to later concatenate and cluster them
        # set fields that aren't available for a given object to be -1
        # these additional fields will be removed after clustering anyway
        forms = {f: objs[collection][f] if f in objs[collection].fields else -1*shape for f in fields}
        forms["part_type"] = objs[collection]["type"] if type_id is None else type_id*shape
        forms["mass"] = objs[collection]["mass"] if mass is None else mass*shape
        return vector.zip(forms)

    def make_constituent(self, consts, type_ids, name, fields):
        """Return array of particles of given type_ids, name, and only specified fields"""
        relevant_consts = consts[ak.any((consts.part_type == x for x in type_ids), axis=0)]
        forms = {f: relevant_consts.__getattr__(f) for f in fields}
        return ak.zip(forms, with_name=name, behavior=nanoaod.behavior)

    def build_lepton_jets(self, objs, lj_reco):
        """Reconstruct lepton jets according to defintion given by lj_reco"""

        # Use electron/muon/photon/dsamuon collections with a custom distance parameter
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

        # turn lepton jets back into LorentzVectors that match existing structures
        ljs = ak.zip(
            {"x": jets.x,
             "y": jets.y,
             "z": jets.z,
             "t": jets.t},
            with_name="LorentzVector",
            behavior=nanoaod.behavior
        )

        # add fields to access LJ constituents
        consts = cluster.constituents()
        common_fields = list(set(fields[0]).intersection(*fields[1:]))
        ljs["constituents"] = self.make_constituent(consts, [2, 3, 4, 8], "PtEtaPhiMCollection", common_fields)


    ######
        ## FIX ME! Won't be able to access the dsaMuon matches from the LJ constituent muon, and vice versa
        ## (can only access it from the original muon collection in objects)

        objs["dsaMuons"]["mass"] = ak.full_like(objs["dsaMuons"].pt, 0.105712890625)

        safe_pf_fields = list(objs["muons"].fields)
        safe_dsa_fields = list(objs["dsaMuons"].fields)

        for field in unsafe_fields:
            if field in safe_pf_fields:
                safe_pf_fields.remove(field)
            if field in safe_dsa_fields:
                safe_dsa_fields.remove(field)

        muon_fields = list(set(safe_pf_fields).intersection(safe_dsa_fields))

        ljs["muons"] = self.make_constituent(consts, [3, 8], "Muon", muon_fields)
        ljs["pfMuons"] = self.make_constituent(consts, [3], "Muon", safe_pf_fields)
        ljs["dsaMuons"] = self.make_constituent(consts, [8], "DSAMuon", safe_dsa_fields)
    ######

        ljs["electrons"] = self.make_constituent(consts, [2], "Electron", objs["electrons"].fields)
        ljs["photons"] = self.make_constituent(consts, [4], "Photon", objs["photons"].fields)

        # define LJ-level quantities

        # number of constituents
        ljs["pfMu_n"] = ak.num(ljs.pfMuons, axis=-1)
        ljs["dsaMu_n"] = ak.num(ljs.dsaMuons, axis=-1)
        ljs["muon_n"] = ak.num(ljs.muons, axis=-1)
        ljs["electron_n"] = ak.num(ljs.electrons, axis=-1)
        ljs["photon_n"] = ak.num(ljs.photons, axis=-1)

        # dRSpread (the maximum dR betwen any pair of constituents in each lepton jet)
        # a) for each constituent, find the dR between it and all other constituents in the same LJ
        # b) flatten that into a list of dRs per LJ
        # c) and then take the maximum dR per LJ, leaving us with a single value per LJ
        ljs["dRSpread"] = ak.max(ak.flatten(
            ljs["constituents"].metric_table(ljs["constituents"], axis=2), axis=-1), axis=-1)

        # LJ isolation
        ljs["matched_jet"] = ljs.nearest(objs["jets"], threshold=0.4)       
        ljs["lepton_fraction"] =  ljs["matched_jet"].chEmEF + ljs["matched_jet"].neEmEF + ljs["matched_jet"].muEF
        ljs["isolation"] = ak.fill_none((ljs["matched_jet"].energy / ljs.energy) * (1 - (ljs["lepton_fraction"])), 0)
        ljs["dR_matched_jet"] = ljs.delta_r(ljs["matched_jet"])

        # todo: add LJ displacement

        # pt order the new LJs
        ljs = self.order(ljs)

        # return the new LJ collection
        return ljs

    def build_cuts(self):
        """ Make list of pre-lj object, lj, post-lj obj, and event cuts per channel"""

        selection_menu = utilities.load_yaml(f"{BASE_DIR}/{self.selections_cfg}")

        ch_cuts = {}

        for channel in self.channel_names:
            ch_cuts[channel] = {}
            ch_cuts[channel]["obj"] = {}
            ch_cuts[channel]["preLj_obj"] = {}
            ch_cuts[channel]["lj"] = {}
            ch_cuts[channel]["postLj_obj"] = {}
            ch_cuts[channel]["evt"] = {}

            cuts = selection_menu[channel]
            for obj, obj_cuts in cuts["obj_cuts"].items():
                if obj not in ch_cuts[channel]["obj"]:
                    ch_cuts[channel]["obj"][obj] = []
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
        # build dictionary and create hist.Hist objects
        hists = {}
        for collection in self.hist_collection_names:
            collection = utilities.flatten(hist_menu[collection])
            for hist_name in collection:
                hists[hist_name] = copy.deepcopy(hist_defs[hist_name])
                # Add lj_reco axis only when more than one reco is run
                lj_reco_names = self.lj_reco_choices if len(self.lj_reco_choices) > 1 else None
                # This 'make_hist' is from our Histogram wrapper class
                hists[hist_name].make_hist(hist_name, self.channel_names, lj_reco_names)
        return hists

    def order(self, obj):
        """Explicitly order objects"""
        # pt order objects with a pt attribute
        if hasattr(obj, "pt"):
            obj = obj[ak.argsort(obj.pt, ascending=False)]
        # fixme: would be good to explicitly order other objects as well
        return obj

    def postprocess(self, accumulator):
        """Modify accumulator after process has run on all chunks"""
        # scale cutflow and hists according to lumi*xs
        for sample, output in accumulator.items():
            
            # --- START MODIFIED SECTION ---
            # 'scaled_sum_weights' is now a value_accumulator,
            # so we must access its .value attribute.
            sum_weights = output["metadata"]["scaled_sum_weights"].value
            # --- END MODIFIED SECTION ---

            lumixs_weight = utilities.get_lumixs_weight(sample, self.year, sum_weights)
            for name in output["cutflow"]:
                accumulator[sample]["cutflow"][name].scale(lumixs_weight)
            if not self.unweighted_hist:
                
                # --- START MODIFIED SECTION ---
                # 'hists' now contains our 'Histogram' wrapper objects.
                # We must scale the '.hist' attribute inside them.
                for name in output["hists"]:
                    accumulator[sample]["hists"][name].hist *= lumixs_weight
                # --- END MODIFIED SECTION ---