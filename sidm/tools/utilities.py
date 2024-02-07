"""Module to define miscellaneous helper methods"""

import os
import yaml
import numpy as np
import matplotlib.pyplot as plt
import mplhep as hep


def print_list(l):
    """Print one list element per line"""
    print('\n'.join(l))

def print_debug(name, val, print_mode=True):
    """Print variable name and value"""
    if print_mode:
        print(f"{name}: {val}")

def partition_list(l, condition):
    """Given a single list, return separate lists of elements that pass or fail a condition"""
    passes = []
    fails = []
    for x in l:
        if condition(x):
            passes.append(x)
        else:
            fails.append(x)
    return passes, fails

def flatten(x):
    """Flatten arbitrarily nested list or dict"""
    # https://stackoverflow.com/questions/2158395/
    flattened_list = []
    def loop(sublist):
        if isinstance(sublist, dict):
            sublist = sublist.values()
        for item in sublist:
            if isinstance(item, (dict, list)):
                loop(item)
            else:
                flattened_list.append(item)
    loop(x)
    return flattened_list

def add_unique_and_flatten(flattened_list, x):
    """Flatten arbitrarily nested list or dict, keeping only unique items"""
    # https://stackoverflow.com/questions/2158395/
    def loop(sublist):
        if isinstance(sublist, dict):
            sublist = sublist.values()
        for item in sublist:
            if isinstance(item, (dict, list)):
                loop(item)
            elif item not in flattened_list:
                flattened_list.append(item)
    loop(x)
    return flattened_list

def dR(obj1, obj2):
    """Return dR between obj1 and the nearest obj2"""
    return obj1.nearest(obj2, return_metric=True)[1]

def lxy(obj):
    """Return transverse distance between production and decay vertices"""
    return (obj.dauvtx - obj.vtx).r

def set_plot_style(style='cms', dpi=50):
    """Set plotting style using mplhep"""
    if style == 'cms':
        plt.style.use(hep.style.CMS)
    else:
        raise NotImplementedError
    plt.rcParams['figure.dpi'] = dpi

def plot(hists, skip_label=False, **kwargs):
    """Plot using hep.hist(2d)plot and add cms labels"""
    dim = len(hists[0].axes) if isinstance(hists, list) else len(hists.axes)
    if dim == 1:
        h = hep.histplot(hists, **kwargs)
    elif dim == 2:
        h = hep.hist2dplot(hists, **kwargs)
    else:
        raise NotImplementedError(f"Cannot plot {dim}-dimensional hist")
    if not skip_label:
        hep.cms.label()
    return h

def load_yaml(cfg):
    """Load yaml files and return corresponding dict"""
    cwd = os.path.dirname(os.path.abspath(__file__))
    with open(f"{cwd}/{cfg}", encoding="utf8") as yaml_cfg:
        return yaml.safe_load(yaml_cfg)

def make_fileset(samples, ntuple_version, max_files=-1, location_cfg="../configs/ntuple_locations.yaml"):
    """Make fileset to pass to processor.runner"""
    if ntuple_version not in ["ffntuple_v2", "ffntuple_v4"]:
        raise NotImplementedError("Only ffntuple_v2 and ffntuple_v4 ntuples have been implemented")
    locations = load_yaml(location_cfg)[ntuple_version]
    fileset = {}
    for sample in samples:
        base_path = locations["path"] + locations["samples"][sample]["path"]
        file_list = [base_path + f for f in locations["samples"][sample]["files"]]
        if max_files != -1:
            file_list = file_list[:max_files]
        fileset[sample] = file_list
    return fileset

def check_bit(array, bit_num):
    """Return boolean stored in the bit_numth bit of array"""
    return (array & pow(2, bit_num)) > 0

def get_hist_mean(hist):
    """Return mean of 1D histogram"""
    return np.atleast_1d(hist.profile(axis=0).view())[0].value
