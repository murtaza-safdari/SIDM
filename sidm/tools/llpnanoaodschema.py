from coffea.nanoevents import NanoAODSchema
from coffea.nanoevents.methods import nanoaod, base, candidate, vector
import awkward as ak
import numpy as np
from dask_awkward import dask_property

# --- 1. Initialize Behaviors (Must happen BEFORE class definitions) ---
# We copy 'Muon' behaviors to 'DSAMuon' and 'LLPMuon' first.
# This ensures we get standard methods (like .nearest), but our 
# class definitions below will OVERRIDE the record constructor to use our custom classes.
nanoaod.behavior.update(ak._util.copy_behaviors("Muon", "DSAMuon", nanoaod.behavior))
nanoaod.behavior.update(ak._util.copy_behaviors("Muon", "LLPMuon", nanoaod.behavior))

# --- 2. Define Mixin Classes ---

@ak.mixin_class(nanoaod.behavior)
class DSAMuon(candidate.PtEtaPhiMCandidate, base.NanoCollection, base.Systematic):
    """LLPNanoAOD DSA muon object"""
    @dask_property
    def matched_muons(self):
        """The matched PF muons (up to 5) as determined by the NanoAOD branch muonMatchNidx)"""
        muon_match_total = ak.concatenate([
            self.muonMatch1[:, :, np.newaxis],
            self.muonMatch2[:, :, np.newaxis],
            self.muonMatch3[:, :, np.newaxis],
            self.muonMatch4[:, :, np.newaxis],
            self.muonMatch5[:, :, np.newaxis],
        ], axis=2) 

        pf_matches = self._events().Muon._apply_global_index(self.muonIdxG)
        concat = ak.with_field(pf_matches, muon_match_total, where="numMatch")
        return concat

    @matched_muons.dask
    def matched_muons(self, dask_array):
        muon_match_total = ak.concatenate([
            dask_array.muonMatch1[:, :, np.newaxis],
            dask_array.muonMatch2[:, :, np.newaxis],
            dask_array.muonMatch3[:, :, np.newaxis],
            dask_array.muonMatch4[:, :, np.newaxis],
            dask_array.muonMatch5[:, :, np.newaxis],
        ], axis=2)

        pf_matches = dask_array._events().Muon._apply_global_index(dask_array.muonIdxG)
        concat = ak.with_field(pf_matches, muon_match_total, where="numMatch")
        return concat

# Register Vector Behaviors for DSAMuon
nanoaod._set_repr_name("DSAMuon")
# Note: ak.mixin_class creates 'DSAMuonArray' in this module's scope
DSAMuonArray.ProjectionClass2D = vector.TwoVectorArray  # noqa: F821
DSAMuonArray.ProjectionClass3D = vector.ThreeVectorArray  # noqa: F821
DSAMuonArray.ProjectionClass4D = DSAMuonArray  # noqa: F821
DSAMuonArray.MomentumClass = vector.LorentzVectorArray  # noqa: F821

@ak.mixin_class(nanoaod.behavior)
class LLPMuon(candidate.PtEtaPhiMCandidate, base.NanoCollection, base.Systematic):
    """LLPNanoAOD Muon object"""
    @dask_property
    def matched_dsa_muons(self):
        """The matched PF muons (up to 5) as determined by the NanoAOD branch muonMatchNidx)"""
        muon_match_total = ak.concatenate([
            self.dsaMatch1[:, :, np.newaxis],
            self.dsaMatch2[:, :, np.newaxis],
            self.dsaMatch3[:, :, np.newaxis],
            self.dsaMatch4[:, :, np.newaxis],
            self.dsaMatch5[:, :, np.newaxis],
        ], axis=2) 

        dsa_matches = self._events().DSAMuon._apply_global_index(self.dsaIdxG)
        concat = ak.with_field(dsa_matches, muon_match_total, where="numMatch")
        return concat

    @matched_dsa_muons.dask
    def matched_dsa_muons(self, dask_array):
        muon_match_total = ak.concatenate([
            dask_array.dsaMatch1[:, :, np.newaxis],
            dask_array.dsaMatch2[:, :, np.newaxis],
            dask_array.dsaMatch3[:, :, np.newaxis],
            dask_array.dsaMatch4[:, :, np.newaxis],
            dask_array.dsaMatch5[:, :, np.newaxis],
        ], axis=2)

        dsa_matches = dask_array._events().DSAMuon._apply_global_index(dask_array.dsaIdxG)
        concat = ak.with_field(dsa_matches, muon_match_total, where="numMatch")
        return concat

# Register Vector Behaviors for LLPMuon
nanoaod._set_repr_name("LLPMuon")
LLPMuonArray.ProjectionClass2D = vector.TwoVectorArray  # noqa: F821
LLPMuonArray.ProjectionClass3D = vector.ThreeVectorArray  # noqa: F821
LLPMuonArray.ProjectionClass4D = LLPMuonArray  # noqa: F821
LLPMuonArray.MomentumClass = vector.LorentzVectorArray  # noqa: F821

# --- 3. Schema Definition ---

class LLPNanoAODSchema(NanoAODSchema):
    """LLPNano schema builder
    LLPNano is an extended NanoAOD format that includes DSA Muons and improved displacement info
    """
    mixins = {
        **NanoAODSchema.mixins,
        "Muon": "LLPMuon", #Adds the matched_dsa_muon property on top of the normal NanoAOD Muon behavior
        "DSAMuon": "DSAMuon",
    }

    all_cross_references = {
        **NanoAODSchema.all_cross_references,
        "Muon_dsaMatch1idx": "DSAMuon",
        "Muon_dsaMatch2idx": "DSAMuon",
        "Muon_dsaMatch3idx": "DSAMuon",
        "Muon_dsaMatch4idx": "DSAMuon",
        "Muon_dsaMatch5idx": "DSAMuon",
        "DSAMuon_muonMatch1idx": "Muon",
        "DSAMuon_muonMatch2idx": "Muon",
        "DSAMuon_muonMatch3idx": "Muon",
        "DSAMuon_muonMatch4idx": "Muon",
        "DSAMuon_muonMatch5idx": "Muon",
    }

    nested_items = {
        **NanoAODSchema.nested_items,
        "Muon_dsaIdxG": [
            "Muon_dsaMatch1idxG",
            "Muon_dsaMatch2idxG",
            "Muon_dsaMatch3idxG",
            "Muon_dsaMatch4idxG",
            "Muon_dsaMatch5idxG",
        ],
        "DSAMuon_muonIdxG": [
            "DSAMuon_muonMatch1idxG",
            "DSAMuon_muonMatch2idxG",
            "DSAMuon_muonMatch3idxG",
            "DSAMuon_muonMatch4idxG",
            "DSAMuon_muonMatch5idxG",
        ],
    }

    @classmethod
    def behavior(cls):
        """Behaviors necessary to implement this schema"""
        # Return the global behavior dict (already updated at import time)
        return nanoaod.behavior