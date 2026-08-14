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
    """CMS Simulation label + sqrt(s); data=False everywhere at truth level.

    The canvas is drawn first so constrained-layout geometry is final before
    mplhep computes the label offsets -- otherwise multi-panel figures with
    colorbars render the label pieces overlapping."""
    fig = ax.figure
    fig.canvas.draw()
    if fig.get_layout_engine() is not None:
        # freeze the settled geometry: placing a label on a later panel must
        # not reflow the layout and shift labels already placed on earlier ones
        fig.set_layout_engine("none")
    hep.cms.label(ax=ax, data=False, com=com)


def save(fig, name):
    """Save an AN figure as vector PDF with a deterministic filename."""
    os.makedirs(FIG_DIR, exist_ok=True)
    path = os.path.join(FIG_DIR, f"{name}.pdf")
    fig.savefig(path)
    print(f"saved {path}")
    return path
