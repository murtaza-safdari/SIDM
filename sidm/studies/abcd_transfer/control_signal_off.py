"""The cr cards of ``binned_cards.py`` with the signal of the control channels set to zero.

The cr model puts a bin's control region (A of the prompt-muon plane) into the likelihood with the
point's signal yield there, so that a signal in the control region is part of the hypothesis.
Setting those rates to zero and keeping everything else gives the same likelihood without that
signal; the ratio of the two expected limits is what the signal in the control region costs, and
the ratio of the zeroed card's limit to the gmN card's is what the free per-bin factor costs.
Section 5.1 of ``binning.ipynb`` reads the limits of these cards (``binning_limits/control_signal_off/``).

    python control_signal_off.py --cards binning_cards --outdir <dir> [--points ...] [--binnings cubebins split400]

The output cards carry a header line saying what was changed.  Combine is run on them with
``sidm/scripts/run_combine_limits.py`` as for the others.
"""
import argparse
from pathlib import Path

import binned_cards as bc

HEADER = "# the control channels' signal rates set to 0 (control_signal_off.py): the cr model without signal in the control region"


def zero_control_signal(text):
    """The card text with every ``sig`` rate of a ``cr_*`` channel set to 0; returns (text, number zeroed)."""
    lines = text.split("\n")
    heads = [i for i, l in enumerate(lines) if l.startswith("bin ")]
    if len(heads) != 2:
        raise ValueError("expected two 'bin' lines (channels, then columns)")
    i = heads[1]
    bins, procs = lines[i].split()[1:], lines[i + 1].split()[1:]
    if not lines[i + 1].startswith("process") or not lines[i + 3].startswith("rate"):
        raise ValueError("expected 'process', 'process', 'rate' after the column line")
    rates = lines[i + 3].split()[1:]
    if not len(bins) == len(procs) == len(rates):
        raise ValueError("column count mismatch")
    zeroed = 0
    for k, (b, p) in enumerate(zip(bins, procs)):
        if b.startswith("cr_") and p == "sig" and float(rates[k]) > 0:
            rates[k] = "0"
            zeroed += 1
    width = len(lines[i]) - len("".join(f"{v:<20}" for v in bins))      # the writer's column width and label field
    lines[i + 3] = "rate".ljust(width) + "".join(f"{v:<20}" for v in rates)
    return "\n".join([HEADER] + lines), zeroed


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--cards", default="binning_cards", help="directory of the cr cards")
    ap.add_argument("--outdir", required=True)
    ap.add_argument("--binnings", nargs="+", default=["cubebins", "split400"], help="binning slugs of the card names")
    ap.add_argument("--points", nargs="+", default=list(bc.bl.DECISION_POINTS[bc.TOPOLOGY]), help="the decision points by default")
    args = ap.parse_args(argv)
    out = Path(args.outdir)
    out.mkdir(parents=True, exist_ok=True)
    n = 0
    for slug in args.binnings:
        for point in args.points:
            src = Path(args.cards) / f"datacard_SR_{bc.TOPOLOGY}_{slug}_cr_{point}.txt"
            text, zeroed = zero_control_signal(src.read_text())
            (out / src.name).write_text(text)
            print(f"{src.name}: {zeroed} control-channel signal rates set to 0")
            n += 1
    print(f"{n} cards in {out}")


if __name__ == "__main__":
    main()
