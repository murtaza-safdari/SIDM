""" Tool to add ntuple locations to sidm/configs/ntuple_locations.yaml.

Developed and tested on FNAL LPC. Will be updated after issue
https://github.com/CoffeaTeam/coffea-casa/issues/374 is resolved. Note that cmsenv or equivalent
is needed to import XRootD.

Usage: python add_ntuples.py -o OUTPUT_CONFIG -n NTUPLE_NAME -c NTUPLE_COMMENT -d NTUPLE_ROOT_DIR
"""

from __future__ import print_function
import argparse
import yaml
from XRootD import client


parser = argparse.ArgumentParser()
parser.add_argument("-o", "--output-cfg", dest="cfg", required=True,
                    help="Path to output config, e.g. '../configs/ntuple_locations.yaml'")
parser.add_argument("-n", "--name", dest="name", required=True,
                    help="Name of group of ntuples, e.g. 'ffntuple_v4'")
parser.add_argument("-c", "--comment", dest="comment", required=True,
                    help=("Comment to describe group of ntuples, e.g. "
                          "'Most recent ntuples from Weinan -- only includes 2mu2e'"))
parser.add_argument("-d", "--directory", dest="directory", required=True,
                    help=("Path to ntuple root directory, e.g. "
                    "'root://cmseos.fnal.gov//store/group/lpcmetx/SIDM/ffNtupleV4/2018/'"))
parser.add_argument("-f", "--first-dir", dest="first_dir", action='store_true',
                    help="Choose first option when encountering unexpected directory structure")
# fixme: add option to associate multiple subdirectories with one process name
args = parser.parse_args()


def parse_name(name):
    """Parse sample directory name to produce simplified name
    
    Assumes structure like "SIDM_XXTo2ATo2Mu2E_mXX-100_mA-1p2_ctau-0p096_TuneCP..."
    """

    process_names = {
        "SIDM_XXTo2ATo2Mu2E_mXX": "2Mu2E_",
        "SIDM_XXTo2ATo4Mu_mXX" : "4Mu_",
        "DYJetsToLL_M" : "DYJetsToLL_M",
        "QCD_Pt" : "QCD_Pt",
        "TTJets_TuneCP5_13TeV" : "TTJets",
        "WW_TuneCP5_13TeV" : "WW",
        "WZ_TuneCP5_13TeV" : "WZ",
        "ZZ_TuneCP5_13TeV" : "ZZ",
        # fixme: add backgrounds and data as necessary
    }
    chunks = name.split("-")
    try:
        simplified_name = process_names[chunks[0]] # process name
    except KeyError:
        print("Unrecognized process name. Skipping {}".format(name))
        return None

    # further simplify names as necessary
    if name.startswith("SIDM"):
        simplified_name += chunks[1].replace("_mA", "GeV_") # bound state mass
        simplified_name += chunks[2].replace("_ctau", "GeV_") # dark photon mass
        simplified_name += chunks[3].split("_TuneCP")[0] + "mm" # dark photon ctau
    elif name.startswith("DYJetsToLL_M"):
        simplified_name += chunks[1].split("_")[0] # mass range
    elif name.startswith("QCD_Pt"):
        simplified_name += chunks[1].split("_")[0] # pT range

    return simplified_name

def descend(ntuple_path, sample_path, choose_first_dir=False):
    path = ntuple_path + "/" + sample_path
    dir_contents = xrd_client.dirlist(path)[1]
    num_found = dir_contents.size

    # Handle emtpy directories
    if num_found == 0:
        print("Found zero objects in {}. Skipping.".format(path))
        return None

    # Allow user to choose directory if more than one is found
    if num_found > 1 and not choose_first_dir:
        print("Unexpected directory structure. Found {} objects in {}".format(num_found, path))
        print("Please type the number of the directory you would like to use. Options are:")
        print("S", "SKIP DIRECTORY")
        for i, x in enumerate(dir_contents):
            print(i, x.name)
        dir_ix = raw_input() # fixme: check input
    else:
        dir_ix = 0

    if dir_ix == "S":
        return None

    return sample_path + "/" + dir_contents.dirlist[int(dir_ix)].name


# Set up xrd client
redirector = args.directory.split("//store")[0]
xrd_client = client.FileSystem(redirector)

coffea_casa_dir = args.directory.replace("cmseos.fnal.gov", "xcache")
output = {
    args.name: {
        "path": coffea_casa_dir,
        "samples": {},
    }
}
ntuple_path = args.directory.split(redirector)[1]
samples = xrd_client.dirlist(ntuple_path)[1]

# Traverse ntuple directory and construct output dictionary
# Assumes same structure as root://cmseos.fnal.gov//store/group/lpcmetx/SIDM/ffNtupleV4/2018/
for sample in samples:
    simple_name = parse_name(sample.name)
    if simple_name is None:
        continue
    output[args.name]["samples"][simple_name] = {}
    sample_path = sample.name

    # Descend two layers, expecting to find a single directory at each
    try:
        for _ in range(2):
            sample_path = descend(ntuple_path, sample_path, args.first_dir)
            if sample_path is None:
                raise StopIteration()
    except StopIteration:
        continue

    # If traversal was successful, add path and files to output dictionary
    try:
        files = [f.name for f in xrd_client.dirlist(ntuple_path + sample_path)[1]]
        # Handle cases with additional directory layer
        if len(files) == 1 and "0000" in files:
            sample_path += "/0000"
            files = [f.name for f in xrd_client.dirlist(ntuple_path + sample_path)[1]]
    except TypeError:
        print("Unexpected directory structure. Skipping {}".format(sample_path))
    output[args.name]["samples"][simple_name]["path"] = sample_path + "/"
    output[args.name]["samples"][simple_name]["files"] = files

# Avoid yaml references, a la stackoverflow.com/questions/13518819
yaml.Dumper.ignore_aliases = lambda *args : True

with open(args.cfg, 'a') as out_file:
    out_file.write("\n\n# " + args.comment + "\n")
    yaml.dump(output, out_file, default_flow_style=False)
    out_file.write("\n")

