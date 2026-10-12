"""Combine datacards for the 2mu2e background estimate per bin of the LJ-pair mass.

The counting experiment of ``binning.ipynb`` section 5 as a datacard per signal point and
binning, run with ``sidm/scripts/run_combine_limits.py`` (``combine -M AsymptoticLimits --run
blind``) so that the toy expected limits of the notebook can be compared with combine's.  Two
models of the background's statistical uncertainty are written for every binning:

``lnN``
    one background process per mass bin with the selected estimate of :func:`binning_lib.binned_routes`
    as its rate, an asymmetric log-normal from the estimate's 68% interval (the Poisson error of B
    and the factor's error together) and, where the factor is fitted, a log-normal for the shape
    systematic |measured - fitted|; this is the model of :func:`binning_lib.expected_limit` with the
    asymmetry kept;
``gmN``
    one background process per mass bin and DSA class, its rate the sideband count B times the
    selected factor, with B a ``gmN`` nuisance (combine profiles the count with a Gamma
    distribution, as the counting cards of ``sidm/studies/limit_plotting`` do for their simulated
    counts), the factor an asymmetric log-normal and the shape systematic as above;
``cr``
    the ``gmN`` model with the control region in the likelihood: per mass bin and DSA class a
    second channel whose observation is A of the reference plane (an open region), whose
    background is B of that plane times the factor and whose signal is the point's yield there,
    with the factor a free ``rateParam`` shared by the two channels (its nominal the measured A/B
    of the bin), so that a signal in the control region is part of the signal hypothesis instead
    of a bias of the estimate (:func:`binning_lib.injection_table` measures that bias).  The factor
    is measured in every bin, as a free parameter constrained by the bin's own control counts;
    the A/B(M) fit and its shape systematic have no place in it.

All carry the systematic of the estimate (``syst_rel``) as one log-normal common to every search
bin, the signal per bin at the stored cross section (mu = 1 is the cross section of
``cross_sections.yaml``) with its simulation-statistics log-normal and the 2.5% luminosity
uncertainty.  The ``observation`` line of a search bin carries the background (the cards are
blinded; the expected limit never reads it).  The search region's A is never read: everything
here comes from the numeric frames of ``binning_lib``, which read the sidebands only, and the
control channels read the reference plane, which is open.
"""
import argparse
import os
import re
import subprocess
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

import binning_lib as bl

TOPOLOGY = "2mu2e"
MODELS = ("lnN", "gmN", "cr")
RATE_PARAM_RANGE = "[0,50]"     # the range of the free factor of the cr model, a multiple of its nominal
# the binning names as they appear in the card's channel (combine and the runner's filename
# pattern allow word characters only)
SLUGS = {"one bin": "onebin", "two windows": "twowindows", "three bins": "threebins", "cube bins": "cubebins",
         "split at 400": "split400"}
NAMES = {v: k for k, v in SLUGS.items()}
LUMI_UNC = 0.025        # the luminosity uncertainty of the counting cards of limit_plotting
MCSTAT_CAP = 2.0        # the cap on the signal's simulation-statistics log-normal, as there
KAPPA_FLOOR = 0.05      # the lowest downward factor of an asymmetric log-normal (an interval reaching zero)
CARD_NAME = re.compile(r"^datacard_SR_(?P<topology>\w+?)_(?P<slug>\w+?)_(?P<model>lnn|gmn|cr)_(?P<point>(?:2Mu2E|4Mu)_\w+)$")
MODEL_NAMES = {"lnn": "lnN", "gmn": "gmN", "cr": "cr"}


def bin_name(lo, hi):
    """``m150to200``, ``m600up``, ``mto150``: the mass window as a word."""
    lo_s = "" if lo is None or not np.isfinite(lo) else f"{lo:g}"
    hi_s = "up" if hi is None or not np.isfinite(hi) else f"to{hi:g}"
    return f"m{lo_s}{hi_s}"


def _fmt(x):
    return f"{float(x):.6g}"


def _kappas(value, minus, plus):
    """The down/up factors of an asymmetric log-normal from an absolute 68% interval."""
    down = max((value - minus) / value, KAPPA_FLOOR)
    up = (value + plus) / value
    return f"{down:.4f}/{up:.4f}"


def _nominal_factor(row):
    """Where the free factor of the cr model starts: the measured A/B of the bin; without an A event
    in the reference plane the fitted A/B, and without a fit the 68% upper limit of the measured
    one.  The control channel's counts constrain it from there."""
    f = float(row["A/B"])
    if np.isfinite(f) and f > 0:
        return f
    fit = float(row["A/B fit"])
    if np.isfinite(fit) and fit > 0:
        return fit
    return bl.ZERO_UPPER / float(row["B ref"])


def _processes(routes, model):
    """The background processes of one card's search bins: (bin, name, rate, rows of nuisances)
    per entry, where a nuisance row is (name, kind, value) and the rate is the number the nuisance
    rows imply.  A ``rateParam`` row's value is the parameter's nominal."""
    out = []
    for label, part in routes[routes["bin"] != bl.TOTAL].groupby("bin", sort=False):
        lo, hi = part["lo"].iloc[0], part["hi"].iloc[0]
        b = bin_name(lo, hi)
        if model == "lnN":
            row = part[part["class"] == bl.SUM].iloc[0]
            rate, minus, plus = (float(row[c]) for c in ("selected", "selected minus", "selected plus"))
            if not np.isfinite(rate):
                raise ValueError(f"{label}: the selected estimate is not finite")
            if rate <= 0:
                out.append((b, "bkg", 0.0, []))
                continue
            nuis = [(f"stat_{b}", "lnN", _kappas(rate, minus, plus))]
            shape = float(row["shape carried"])
            if shape > 0:
                nuis.append((f"shape_{b}", "lnN", _fmt(1 + shape / rate)))
            out.append((b, "bkg", rate, nuis))
        elif model == "gmN":
            for _, row in part[part["class"] != bl.SUM].iterrows():
                cls = str(row["class"]).replace("+", "p")
                name = f"bkg_{cls}"
                n = int(round(float(row["B"])))
                if n == 0:
                    out.append((b, name, 0.0, []))
                    continue
                if row["choice"] == "measured":
                    f, minus, plus = (float(row[c]) for c in ("A/B", "A/B minus", "A/B plus"))
                else:
                    f = float(row["A/B fit"])
                    minus = plus = float(row["A/B fit err"])
                if not np.isfinite(f) or f <= 0:
                    raise ValueError(f"{label} {row['class']}: no usable factor ({row['choice']})")
                alpha = float(_fmt(f))                 # the rate is N times the alpha as written
                rate = n * alpha
                nuis = [(f"nB_{b}_{cls}", f"gmN {n}", _fmt(alpha)),
                        (f"factor_{b}_{cls}", "lnN", _kappas(f, minus, plus))]
                shape = float(row["shape carried"])
                if shape > 0:
                    nuis.append((f"shape_{b}_{cls}", "lnN", _fmt(1 + shape / rate)))
                out.append((b, name, rate, nuis))
        elif model == "cr":
            for _, row in part[part["class"] != bl.SUM].iterrows():
                cls = str(row["class"]).replace("+", "p")
                name = f"bkg_{cls}"
                n, n_ref = int(round(float(row["B"]))), int(round(float(row["B ref"])))
                if n == 0 or n_ref == 0:       # nothing to scale, or no sideband to measure the factor from
                    out.append((b, name, 0.0, []))
                    continue
                alpha = float(_fmt(_nominal_factor(row)))
                out.append((b, name, n * alpha, [(f"nB_{b}_{cls}", f"gmN {n}", _fmt(alpha)),
                                                 (f"f_{b}_{cls}", "rateParam", "1.0")]))
        else:
            raise ValueError(f"unknown model {model!r}")
    return out


def _control_bins(routes, control, point):
    """The control channels of the cr model: per search bin and class with a sideband event in both
    planes, (bin name, observation, [(process, rate, nuisances)]) with the signal's yield in A of
    the reference plane and the reference plane's B times the nominal factor, the factor's
    ``rateParam`` shared with the search bin, and the reference plane's B count as ``gmN``."""
    ref = routes.attrs.get("ref_plane", "PD")
    sig = control[control["point"] == point].set_index(["label", "bin"])
    out = []
    for label, part in routes[routes["bin"] != bl.TOTAL].groupby("bin", sort=False):
        lo, hi = part["lo"].iloc[0], part["hi"].iloc[0]
        b = bin_name(lo, hi)
        for _, row in part[part["class"] != bl.SUM].iterrows():
            cls = str(row["class"]).replace("+", "p")
            n, n_ref, a_ref = (int(round(float(row[c]))) for c in ("B", "B ref", "A ref"))
            if n == 0 or n_ref == 0:
                continue
            try:
                s = sig.loc[(f"A {ref}:{row['class']}", label)]
            except KeyError as e:
                raise ValueError(f"no signal yield in A {ref}:{row['class']} for {label}") from e
            alpha = float(_fmt(_nominal_factor(row)))
            processes = [("sig", float(s["yield"]), float(s["error"]), []),
                         (f"bkg_{cls}", n_ref * alpha, None, [(f"nBref_{b}_{cls}", f"gmN {n_ref}", _fmt(alpha)),
                                                               (f"f_{b}_{cls}", "rateParam", "1.0")])]
            out.append((f"cr_{b}_{cls}", a_ref, processes))
    return out


def card_text(point, routes, yields, binning_name, model, syst_rel=0.33, topology=TOPOLOGY, lumi_unc=LUMI_UNC,
              control=None):
    """The datacard of one signal point for one binning and background model, as text.  The ``cr``
    model needs ``control``, the :func:`binning_lib.signal_region_yields` of the binning."""
    if model == "cr" and control is None:
        raise ValueError("the cr model needs the signal's yields in the control regions")
    bkg = _processes(routes, model)
    sig = yields[yields["point"] == point].copy()
    if sig.empty:
        raise ValueError(f"no signal yields for {point}")
    sig["name"] = [bin_name(lo, hi) for lo, hi in zip(sig["lo"], sig["hi"])]
    sig = sig.set_index("name")
    bins = list(dict.fromkeys(b for b, _, _, _ in bkg))
    missing = [b for b in bins if b not in sig.index]
    if missing:
        raise ValueError(f"signal yields missing for bins {missing}")
    # a bin with neither signal nor background is dropped
    kept, dropped = [], []
    for b in bins:
        has_bkg = any(rate > 0 for bb, _, rate, _ in bkg if bb == b)
        if float(sig.loc[b, "yield"]) > 0 or has_bkg:
            kept.append(b)
        else:
            dropped.append(b)
    # combine numbers the processes by name, the same number in every bin: jmax counts the names
    ids = {name: k + 1 for k, name in enumerate(dict.fromkeys(name for _, name, rate, _ in bkg if rate > 0))}
    columns = []            # (bin, process, id, rate)
    nuisances = {}          # name -> (kind, {column index: value})
    omitted = []
    for b in kept:
        s = float(sig.loc[b, "yield"])
        columns.append((b, "sig", 0, s))
        if s > 0:
            rel = min(float(sig.loc[b, "error"]) / s, MCSTAT_CAP)
            nuisances.setdefault(f"mcstat_sig_{b}", ("lnN", {}))[1][len(columns) - 1] = _fmt(1 + rel)
        nuisances.setdefault("lumi_13TeV", ("lnN", {}))[1][len(columns) - 1] = _fmt(1 + lumi_unc)
        for bb, name, rate, nuis in bkg:
            if bb != b:
                continue
            if rate <= 0:
                omitted.append(f"{b} {name}")
                continue
            columns.append((b, name, ids[name], rate))
            for nname, kind, value in nuis:
                nuisances.setdefault(nname, (kind, {}))[1][len(columns) - 1] = value
            nuisances.setdefault("transfer", ("lnN", {}))[1][len(columns) - 1] = _fmt(1 + syst_rel)
    if not any(rate > 0 for _, name, _, rate in columns if name == "sig"):
        raise ValueError(f"{point}: no signal in any bin of {binning_name}")
    observation = {b: sum(rate for bb, name, _, rate in columns if bb == b and name != "sig") for b in kept}
    channels = list(kept)
    if model == "cr":
        for cb, obs, processes in _control_bins(routes, control, point):
            channels.append(cb)
            observation[cb] = obs
            for name, rate, error, nuis in processes:
                if name == "sig":
                    columns.append((cb, "sig", 0, rate))
                    if rate > 0:
                        rel = min(error / rate, MCSTAT_CAP)
                        nuisances.setdefault(f"mcstat_sig_{cb}", ("lnN", {}))[1][len(columns) - 1] = _fmt(1 + rel)
                    nuisances.setdefault("lumi_13TeV", ("lnN", {}))[1][len(columns) - 1] = _fmt(1 + lumi_unc)
                else:
                    columns.append((cb, name, ids[name], rate))
                    for nname, kind, value in nuis:
                        nuisances.setdefault(nname, (kind, {}))[1][len(columns) - 1] = value
    rate_params = {n: cells for n, (kind, cells) in nuisances.items() if kind == "rateParam"}
    nuisances = {n: v for n, v in nuisances.items() if v[0] != "rateParam"}
    width = max(16, max(len(n) for _, n, _, _ in columns) + 2, max(len(b) for b in channels) + 2)
    col = lambda vals: "".join(f"{v:<{width}}" for v in vals)
    names = sorted(nuisances, key=lambda n: (n == "transfer", n == "lumi_13TeV", n))
    name_w = max(len(n) for n in list(names) + list(rate_params)) + 2
    kind_w = max(len(kind) for kind, _ in nuisances.values()) + 2
    lines = [
        f"# SIDM {topology} data-driven background per bin of the LJ-pair mass: {binning_name} ({model} model) -- {point}",
        "# background = B of the displaced-displaced plane x A/B of the prompt-muon plane per DSA class, per bin"
        + (" (binning.ipynb, binning_lib.binned_routes, selected factor)" if model != "cr" else
           " (binning.ipynb, binning_lib.binned_routes), the factor a free parameter per bin and class"),
        "# signal at the theory cross section (cross_sections.yaml): mu = 1 is the stored cross section",
        "# observation = the background of the bin (blinded; combine --run blind never reads it)",
    ]
    if model == "cr":
        ref = routes.attrs.get("ref_plane", "PD")
        sig_b = control[(control["point"] == point) & (control["region"] == "B")]
        in_b = [f"{r['label']} {r['bin']} {r['yield']:.2f}" for _, r in sig_b.iterrows() if r["yield"] > 0.005]
        lines += [
            f"# cr_<bin>_<class>: the control channel of the bin and class; observation = A of the {ref} plane (open),"
            f" background = B of {ref} x the factor, signal = the point's yield in that A",
            "# f_<bin>_<class>: the factor, a free rateParam shared by the search bin and its control channel;"
            " nominal = the measured A/B (the fitted one, or the upper limit, where A has no event)",
            "# signal in the B sidebands at mu = 1, not modelled: " + (", ".join(in_b) if in_b else "none above 0.005"),
        ]
    if dropped:
        lines.append(f"# bins without signal or background, dropped: {', '.join(dropped)}")
    if omitted:
        lines.append(f"# background processes without a sideband event, omitted: {', '.join(omitted)}")
    lines += [
        f"imax {len(channels)}  number of bins",
        f"jmax {len(ids)}  number of background processes",
        "kmax *  number of nuisance parameters",
        "-" * 80,
        "bin".ljust(name_w + kind_w) + col(channels),
        "observation".ljust(name_w + kind_w) + col(_fmt(observation[b]) for b in channels),
        "-" * 80,
        "bin".ljust(name_w + kind_w) + col(b for b, _, _, _ in columns),
        "process".ljust(name_w + kind_w) + col(n for _, n, _, _ in columns),
        "process".ljust(name_w + kind_w) + col(str(i) for _, _, i, _ in columns),
        "rate".ljust(name_w + kind_w) + col(_fmt(r) for _, _, _, r in columns),
        "-" * 80,
    ]
    for n in names:
        kind, cells = nuisances[n]
        lines.append(f"{n:<{name_w}}{kind:<{kind_w}}" + col(cells.get(i, "-") for i in range(len(columns))))
    for n, cells in rate_params.items():
        for i, value in cells.items():
            lines.append(f"{n:<{name_w}}rateParam  {columns[i][0]} {columns[i][1]} {value} {RATE_PARAM_RANGE}")
    return "\n".join(lines) + "\n"


def card_path(outdir, point, binning_name, model, topology=TOPOLOGY):
    return Path(outdir) / f"datacard_SR_{topology}_{SLUGS[binning_name]}_{model.lower()}_{point}.txt"


def write_cards(routes_by_binning, yields_by_binning, outdir, models=MODELS, syst_rel=0.33, topology=TOPOLOGY,
                meta=None, control_by_binning=None):
    """One card per signal point, binning and model under ``outdir``; returns the paths.  ``meta``
    (a dict) is written as ``datacards.meta.yaml`` next to them for the runner's provenance;
    ``control_by_binning`` (the signal's yields in the control regions per binning) is needed for
    the ``cr`` model."""
    outdir = Path(outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    paths = []
    for name, routes in routes_by_binning.items():
        yields = yields_by_binning[name]
        control = None if control_by_binning is None else control_by_binning[name]
        for point in dict.fromkeys(yields["point"]):
            for model in models:
                path = card_path(outdir, point, name, model, topology)
                path.write_text(card_text(point, routes, yields, name, model, syst_rel, topology, control=control))
                paths.append(path)
    if meta is not None:
        (outdir / "datacards.meta.yaml").write_text(yaml.safe_dump(meta, sort_keys=False))
    return paths


def parse_card_name(stem):
    m = CARD_NAME.match(stem)
    if not m:
        raise ValueError(f"not a binned card name: {stem}")
    return {"topology": m["topology"], "binning": NAMES[m["slug"]], "model": MODEL_NAMES[m["model"]], "point": m["point"]}


def read_limits(path):
    """The ``limits.csv`` of ``run_combine_limits.py`` over the binned cards as one row per point,
    binning and model: ``r`` the median expected limit on mu (mu = 1 the stored cross section) and
    its 16% and 84% quantiles ``r minus`` and ``r plus``."""
    raw = pd.read_csv(path)
    rows = []
    for _, r in raw.iterrows():
        info = parse_card_name(Path(r["datacard"]).stem)
        rows.append({**info, "r": float(r["exp"]), "r minus": float(r["exp_m1"]), "r plus": float(r["exp_p1"]),
                     "r 2.5%": float(r["exp_m2"]), "r 97.5%": float(r["exp_p2"])})
    return pd.DataFrame(rows)


def comparison(limits, toy, reference="one bin", factor="selected"):
    """Combine against the toy per point and binning: ``r`` per model (the expected limit on mu),
    each relative to the ``reference`` binning of the same model, and the toy's relative limit
    (:func:`binning_lib.limit_table`, ``factor``), whose reference is the one-bin estimate with the
    measured factor."""
    toy_rel = toy[toy["factor"] == factor].set_index(["point", "binning"])["relative to one bin"]
    toy_mu = toy[toy["factor"] == factor].set_index(["point", "binning"])["limit on mu"]
    rows = []
    for point in dict.fromkeys(limits["point"]):
        for binning in dict.fromkeys(limits["binning"]):
            row = {"point": point, "binning": binning}
            for model in dict.fromkeys(limits["model"]):
                sel = limits[(limits["point"] == point) & (limits["binning"] == binning) & (limits["model"] == model)]
                ref = limits[(limits["point"] == point) & (limits["binning"] == reference) & (limits["model"] == model)]
                r = float(sel["r"].iloc[0]) if len(sel) else np.nan
                r0 = float(ref["r"].iloc[0]) if len(ref) else np.nan
                row[f"r {model}"] = r
                row[f"relative {model}"] = r / r0 if r0 > 0 else np.nan
            row["r toy"] = float(toy_mu.get((point, binning), np.nan))
            row["relative toy"] = float(toy_rel.get((point, binning), np.nan))
            rows.append(row)
    return pd.DataFrame(rows)


def comparison_view(table):
    """``comparison`` rounded for display: r to three figures, ratios to three decimals."""
    out = table.copy()
    for c in out.columns:
        if c.startswith("r "):
            out[c] = [f"{v:.3g}" if np.isfinite(v) else "n/a" for v in out[c]]
        elif c.startswith("relative "):
            out[c] = [f"{v:.3f}" if np.isfinite(v) else "n/a" for v in out[c]]
    return out


def main(argv=None):
    """Write the cards from the 8D cubes, with the notebook's configuration."""
    import dataclasses  # noqa: F401  (the notebook's imports, kept for parity)
    import transfer_lib as tl
    import plane_search_lib as ps

    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("--base", default=os.environ.get("ABCD8D_LOCAL", "/uscms_data/d3/murtazas/abcd8d_local"))
    p.add_argument("--outdir", required=True)
    p.add_argument("--syst", type=float, default=0.33)
    p.add_argument("--form", default="exponential")
    args = p.parse_args(argv)

    data = tl.summed_cubes(args.base, "data", prefix="abcd8d")
    d2 = data[TOPOLOGY]
    nominal = ps.Config(fail0=0.25, dphi_fail=2.4)
    points = {c: bl.ab_points(d2, nominal, bl.FIT_BINS, "PD", c) for c in bl.CLASSES}
    fit = {("PD", c): bl.fit_ab(points[c], args.form) for c in bl.CLASSES}
    binnings = {**bl.BINNINGS, **bl.LATER_BINNINGS}
    routes = {name: bl.binned_routes(d2, nominal, bins, fits=fit) for name, bins in binnings.items()}
    profiles = bl.signal_profiles(args.base, bl.SIGNAL_POINTS[TOPOLOGY], TOPOLOGY, nominal)
    yields = {name: bl.signal_yields(args.base, bl.SIGNAL_POINTS[TOPOLOGY], TOPOLOGY, nominal, bins, profiles=profiles)
              for name, bins in binnings.items()}
    # the signal in the control regions, every distinct bin at once (the labels name the windows)
    all_bins = list({label: window for bins in binnings.values() for label, window in bins}.items())
    regions = bl.signal_region_yields(args.base, bl.SIGNAL_POINTS[TOPOLOGY], TOPOLOGY, nominal, all_bins)
    control = {name: regions[regions["bin"].isin([label for label, _ in bins])].reset_index(drop=True)
               for name, bins in binnings.items()}
    try:
        commit = subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True, check=True).stdout.strip()
    except (subprocess.CalledProcessError, FileNotFoundError):
        commit = None
    meta = {"study": "sidm/studies/abcd_transfer/binning.ipynb", "writer": "binned_cards.py", "topology": TOPOLOGY,
            "inputs": args.base, "fit form": args.form, "systematic": args.syst, "models": list(MODELS),
            "binnings": {name: [[label, [lo, hi]] for label, (lo, hi) in bins] for name, bins in binnings.items()},
            "estimate per bin (selected)": {name: {str(r["bin"]): float(r["selected"]) for _, r in
                                                   routes[name][routes[name]["class"] == bl.SUM].iterrows()}
                                            for name in binnings},
            "signal points": bl.SIGNAL_POINTS[TOPOLOGY], "commit": commit}
    paths = write_cards(routes, yields, args.outdir, syst_rel=args.syst, meta=meta, control_by_binning=control)
    print(f"{len(paths)} cards in {args.outdir}")
    for name in binnings:
        est = meta["estimate per bin (selected)"][name]
        print(f"  {name}: " + ", ".join(f"{b} {v:.1f}" for b, v in est.items()))


if __name__ == "__main__":
    main()
