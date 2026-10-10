"""Add the campaign pre-filter to the processor of the campaign code tree (idempotent)."""
import sys

path = (sys.argv[1] if len(sys.argv) > 1 else "/uscms_data/d3/murtazas/abcd8d/code") + "/sidm/tools/sidm_processor.py"
with open(path) as f:
    s = f.read()
MARK = "# abcd8d campaign pre-filter"
if MARK in s:
    print("already patched")
    sys.exit()

old = '''        # create object collections
        # fixme: only include objs used in cuts or hists
        objs = {}
'''
new = '''        # abcd8d campaign pre-filter: every channel of this campaign requires the trigger OR and
        # two muons (PF or DSA) with pT > 26 GeV after the object cuts, so keeping only the
        # events that satisfy a superset of those cuts on the raw collections changes no channel
        # yield; the sum of generator weights used for the normalisation is taken before the
        # filter.  The cutflow's first row counts the filtered events.
        if not is_data:
            scaled_sum_weights = ak.sum(self.obj_defs["weight"](events)) / events.metadata["skim_factor"]
        else:
            scaled_sum_weights = float(len(events)) / events.metadata["skim_factor"]
        trigger_or = (events.HLT.DoubleL2Mu23NoVtx_2Cha | events.HLT.DoubleL2Mu23NoVtx_2Cha_CosmicSeed
                      | events.HLT.DoubleL2Mu25NoVtx_2Cha_Eta2p4 | events.HLT.DoubleL2Mu25NoVtx_2Cha_CosmicSeed_Eta2p4)
        n_high_pt = ak.sum(events.Muon.pt > 26, axis=-1) + ak.sum(events.DSAMuon.pt > 26, axis=-1)
        events = events[trigger_or & (n_high_pt >= 2)]
        # create object collections
        # fixme: only include objs used in cuts or hists
        objs = {}
'''
assert old in s
s = s.replace(old, MARK + "\n" + new, 1)
old2 = '''                "scaled_sum_weights": ak.sum(evt_weights)/events.metadata["skim_factor"],
'''
new2 = '''                "scaled_sum_weights": scaled_sum_weights,      # before the campaign pre-filter
'''
assert s.count(old2) == 1
s = s.replace(old2, new2, 1)
with open(path, "w") as f:
    f.write(s)
print("patched", path)
