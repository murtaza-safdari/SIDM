"""Module to define schema that turns FireFighter ntuples into a awkward arrays"""

# columnar analysis
from coffea.nanoevents.schemas.base import BaseSchema, zip_forms
from coffea.nanoevents.methods import base, vector
from coffea.nanoevents import transforms
# local
from sidm.tools import utilities


def get_offsets(branches, counts_name):
    """Turn counts branches (e.g. muon_n) into offset arrays"""
    if counts_name in branches:
        return transforms.counts2offsets_form(branches.pop(counts_name))
    return None


class FFSchema(BaseSchema):
    """FF schema builder

    The FF schema is built from all branches found in the supplied file, which
    is intended to be a FireFighter ntuple (https://github.com/phylsix/Firefighter).
    Analogous to the procedure in in TreeMakerSchema, array collections are
    generated in three steps:

    1. The branches of vector-like objects are grouped into a single collection
       with the corresponding coordinate variables names mapped to the standard
       variable names for coffea.nanoevents.methods.vector behaviors. For example:
       - muons will behave as LorentzVectors
       - primary vertices will behave as ThreeVectors
       - missing energy will behave as a TwoVector

    2. The branches corresponding to attributes of a subobject are merged into
       a single collection associated with that subobject. For example:
       - all pfjets_pfcands_<attribute> branches are merged into a single collection

    3. The branches corresponding to the attributes and subobjects associated with
       a given object are grouped into a single collection named <object>.
       For example:
       - all pfjet_<attribute> branches are merged into a single collection so
         pfjet attributes can be accessed like pfjet.<attribute>
       - all pfjet_<subobject> branches are merged into a single collection so
         <subobject> attributes can be accessed like pfjet.<subobject>.<attribute>


    All collections are then zipped into one `base.NanoEvents` record and returned.
    """

    def __init__(self, base_form):
        super().__init__(base_form)
        self._form["contents"] = self._build_collections(self._form["contents"])

    def _build_collections(self, branch_forms):
        """Modify branch forms ensure proper object behavior and nesting"""

        # define special cases
        # one-value-per-event branches that contain underscores
        multiword_single_values = [
            b for b in branch_forms if b.startswith(("HLT", "tomatchfilter"))
        ]
        # names of objects that contain underscores
        multiword_objects = [
            "akjet_ak4PFJetsCHS",
        ]
        # trigger objects, which are LorentzVectors with no other attributes
        # names contain underscores and tend to be subsets of other object names
        trigger_objects = [
            b[:-2] for b in branch_forms if b.startswith(("TO", "L1TO")) and b.endswith("_n")
        ]
        # names of count branches that don't fit the normal pattern
        counts_names = {
            "pfjet_pfcand": "pfjet_pfcands_n",
            "pfjet_pfcands": None,
        }

        # Turn any vector-like objects (e.g. LorentzVectors) into the appropriate awkward form
        vector_objects = list(set(b.split("/")[0] for b in branch_forms if "/" in b))

        for obj in vector_objects:
            components = set(k.split('.')[-1] for k in branch_forms if k.startswith(f"{obj}/"))
            # optional fixme: add case for candidates (lorentz+charge)
            # optional fixme: add case for pfjet_pfcand, which could be PtEtaPhiELorentzVector
            # handle lorentz vectors
            if components == {"fX", "fY", "fZ", "fT"}:
                if obj.endswith('_p4'):
                    obj_name = obj[:-3]
                elif obj.endswith('_rawP4'):
                    obj_name = obj[:-6]
                else:
                    obj_name = obj
                form = zip_forms(
                    {
                        "x": branch_forms.pop(f"{obj}/{obj}.fCoordinates.fX"),
                        "y": branch_forms.pop(f"{obj}/{obj}.fCoordinates.fY"),
                        "z": branch_forms.pop(f"{obj}/{obj}.fCoordinates.fZ"),
                        "t": branch_forms.pop(f"{obj}/{obj}.fCoordinates.fT"),
                    },
                    obj_name,
                    "LorentzVector",
                )
                branch_forms[obj_name] = form
                # remove trigger object counts branches
                if obj_name in trigger_objects:
                    branch_forms.pop(f"{obj_name}_n")
            # handle three-vectors
            elif components == {"fX", "fY", "fZ"}:
                form = zip_forms(
                    {
                        "x": branch_forms.pop(f"{obj}/{obj}.fCoordinates.fX"),
                        "y": branch_forms.pop(f"{obj}/{obj}.fCoordinates.fY"),
                        "z": branch_forms.pop(f"{obj}/{obj}.fCoordinates.fZ"),
                    },
                    obj,
                    "ThreeVector",
                )
                branch_forms[obj] = form
            # handle two-vectors
            elif components == {"fX", "fY"}:
                form = zip_forms(
                    {
                        "x": branch_forms.pop(f"{obj}/fCoordinates/fCoordinates.fX"),
                        "y": branch_forms.pop(f"{obj}/fCoordinates/fCoordinates.fY"),
                    },
                    obj,
                    "TwoVector",
                )
                branch_forms[obj] = form
            else:
                raise ValueError(
                    f"Unrecognized class with split branches: {components}"
                )

        # identify object branches (as opposed to one-value-per-event branches)
        # exclude multiword_object and trigger_object branches, which require futher processing
        object_branches = [
            b for b in branch_forms if "_" in b
            and b not in multiword_single_values
            and not b.startswith(tuple(multiword_objects))
            and not b.startswith(tuple(trigger_objects))
        ]

        # identify base objects (e.g. "muon")
        objects = list(set(b.split('_')[0] for b in object_branches))

        # add multiword objects if necessary
        for mw_obj in multiword_objects:
            mw_obj_branches = [b for b in branch_forms if b.startswith(mw_obj)]
            if len(mw_obj_branches) > 0:
                objects.append(mw_obj)
                object_branches += mw_obj_branches

        for obj in objects:
            # identify all object attributes
            # e.g. "pt" from "muon_pt", or "pfcand_pt" from "pfjet_pfcand_pt"
            attributes = [b.split(f"{obj}_")[1] for b in object_branches if b.startswith(f"{obj}_")]

            # identify subobjects (e.g. "pfcand" from "pfjet_pfcand_pt")
            subobjects = [a.split("_")[0] for a in attributes if "_" in a]
            subobjects = list(set(s for s in subobjects if subobjects.count(s) > 1))

            # distinguish between attributes of objects and of subobjects
            attributes, subattributes = utilities.partition_list(
                attributes,
                lambda x: not x.startswith(tuple(subobjects))
            )

            # create subobject jagged arrays
            for subobj in subobjects:
                base_name = f"{obj}_{subobj}"
                counts_name = counts_names.get(base_name, f"{base_name}_n")
                offsets = get_offsets(branch_forms, counts_name)

                branch_forms[base_name] = zip_forms(
                    {
                        a.split(f"{subobj}_")[1]: branch_forms.pop(f"{obj}_{a}")
                        for a in subattributes
                        if a.startswith(f"{subobj}_") and not a.endswith("_n")
                    },
                    base_name,
                    offsets,
                )
                attributes.append(subobj)

            # create object jagged arrays, including nesting of subobjects
            counts_name = counts_names.get(obj, f"{obj}_n")
            offsets = get_offsets(branch_forms, counts_name)
            # handle non-vector objects
            if obj not in branch_forms:
                branch_forms[obj] = zip_forms(
                    {a: branch_forms.pop(f"{obj}_{a}") for a in attributes if a != "n"},
                    obj,
                    offsets=offsets
                )
            # add attributes to existing vector-like objects
            else:
                for attribute in attributes:
                    if attribute == "n":
                        continue
                    branch = branch_forms.pop(f"{obj}_{attribute}")
                    branch_forms[obj]["content"]["contents"][attribute] = branch["content"]

        return branch_forms

    @property
    def behavior(self):
        """Behaviors necessary to implement this schema"""
        behavior = {}
        behavior.update(base.behavior)
        behavior.update(vector.behavior)
        return behavior
