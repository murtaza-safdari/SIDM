"""Shared figure conventions for the truth-kinematics AN figures.

Every figure destined for the analysis note goes through these helpers so
sizing, labels, and output format stay consistent. Truth-level figures carry
the CMS Simulation label; save as vector PDF into figures/.
"""
import os
import matplotlib.pyplot as plt
import mplhep as hep

FIG_DIR = os.path.join(os.path.dirname(__file__), "figures")

# single-column and full-width sizes for the cms-tdr note layout
SINGLE = (10, 8)
WIDE = (16, 8)


def set_style():
    plt.style.use(hep.style.CMS)
    plt.rcParams["savefig.bbox"] = "tight"


def cms_sim_label(ax, com=13):
    """CMS Simulation label + sqrt(s); data=False everywhere at truth level."""
    hep.cms.label(ax=ax, data=False, com=com)


def save(fig, name):
    """Save an AN figure as vector PDF with a deterministic filename."""
    os.makedirs(FIG_DIR, exist_ok=True)
    path = os.path.join(FIG_DIR, f"{name}.pdf")
    fig.savefig(path)
    print(f"saved {path}")
    return path
