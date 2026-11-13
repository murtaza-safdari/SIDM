#!/usr/bin/env python3
"""Proper-ctau check on every produced Run 3 signal sample (180 points x 4 eras), from the GenPart
vertices kept in LLPNanoAOD, as a check on the produced files themselves (the 180-point GEN scan was
a separate LHE+GEN job). Per sample, the first --nfiles chunks (default 4, ~2000 events, the same for
every point; the 36 high-statistics points have 120): proper ctau of each dark photon (pdgId 32) =
|decay vertex - production vertex| / (p/m), the decay vertex being the production vertex of its
lepton daughter with pdgId > 0 (exactly one per dark photon). Writes ctau_all_samples.csv and
figures/run3_ctau_all.pdf; --replot redraws from the CSV.
--production v2 (EXONanoAOD, five eras) reads no EOS: it draws figures_v2/run3_ctau_all.pdf from the v2
census table v2_census/results/ctau_v2.csv (every chunk of every sample, so 100k events for the 36 column
samples and 2000 for the others). With such unequal statistics the figure shows the per-sample pull,
(mean - nominal) / (std / sqrt(n)), per era against a unit Gaussian; the per-era and overall pull and
ratio statistics are printed and written with --summary.
Run from the worktree root with the notebook venv.
"""
import argparse, csv, json, os, sys
from concurrent.futures import ThreadPoolExecutor
import numpy as np
import awkward as ak
import uproot
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _DIR)
sys.path.insert(0, os.path.join(_DIR, "..", "truth_kinematics_forAN"))
import run3_signal_validation as V
from sidm.tools import utilities
import an_style  # pylint: disable=import-error  # found through the sys.path entry above

an_style.FIG_DIR = os.path.join(_DIR, "figures")
ERAS = ["2022", "2022EE", "2023", "2023BPix"]
ECOL = {"2022": "#1f77b4", "2022EE": "#2ca02c", "2023": "#d62728", "2023BPix": "#9467bd"}
CSV = os.path.join(_DIR, "ctau_all_samples.csv")
BR = ["GenPart_pdgId", "GenPart_genPartIdxMother", "GenPart_pt", "GenPart_eta", "GenPart_mass",
      "GenPart_vx", "GenPart_vy", "GenPart_vz"]


def proper_ctau(url):
    a = uproot.open(url)["Events"].arrays(BR)
    pid, mom = a.GenPart_pdgId, a.GenPart_genPartIdxMother
    m = ak.where(mom >= 0, mom, 0)
    sel = ((pid == 11) | (pid == 13)) & (mom >= 0) & (abs(pid[m]) == 32)      # one daughter per dark photon
    dl = np.sqrt((a.GenPart_vx - a.GenPart_vx[m]) ** 2 + (a.GenPart_vy - a.GenPart_vy[m]) ** 2
                 + (a.GenPart_vz - a.GenPart_vz[m]) ** 2)[sel]
    p = (a.GenPart_pt[m] * np.cosh(a.GenPart_eta[m]))[sel]
    ms = a.GenPart_mass[m][sel]
    x = ak.to_numpy(ak.flatten(dl / (p / ms))) * 10.0                          # cm -> mm
    return x[np.isfinite(x)]


def run(nfiles, workers):
    jobs = []
    for era in ERAS:
        for cfg in ("signal_4mu_run3.yaml", "signal_2mu2e_run3.yaml"):
            for s in utilities.load_yaml(f"{utilities.BASE_DIR}/configs/ntuples/{cfg}")[era]["samples"]:
                jobs.append((era, s, V._fileset_files(s, era, cfg, nfiles)))

    commit = V.code_commit()

    def one(j):
        era, s, urls = j
        x = np.concatenate([proper_ctau(u) for u in urls])
        nom = V.nominal_mm(s)
        return dict(era=era, sample=s, n_files=len(urls), n_decays=len(x), measured_mm=round(x.mean(), 5),
                    stat_err_mm=round(x.std() / np.sqrt(len(x)), 5), nominal_mm=nom, ratio=round(x.mean() / nom, 5),
                    code_commit=commit)
    with ThreadPoolExecutor(workers) as ex:
        rows = list(ex.map(one, jobs))
    with open(CSV, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)
    print(f"wrote {CSV}: {len(rows)} samples")
    return rows


def figure(rows):
    fig, ax = plt.subplots(figsize=(10, 7.5), layout="constrained")
    edges = np.linspace(0.85, 1.15, 31)
    for era in ERAS:
        r = np.array([float(x["ratio"]) for x in rows if x["era"] == era])
        ax.hist(r, bins=edges, histtype="step", lw=2.2, color=ECOL[era],
                label=f"{era}: mean {r.mean():.3f}, std {r.std():.3f}")
    n = np.median([int(x["n_decays"]) for x in rows]); sig = 1 / np.sqrt(n)
    per_era = len(rows) / len(ERAS)
    xx = np.linspace(edges[0], edges[-1], 300)
    ax.plot(xx, per_era * np.diff(edges)[0] * np.exp(-0.5 * ((xx - 1) / sig) ** 2) / (sig * np.sqrt(2 * np.pi)),
            "k--", lw=1.8, label=rf"statistical expectation ($\sigma$ = 1/$\sqrt{{{n:.0f}}}$)")
    ax.set_xlabel(r"measured / nominal proper $c\tau$"); ax.set_ylabel("samples / 0.01")
    ax.set_xlim(edges[0], edges[-1]); ax.set_ylim(0, ax.get_ylim()[1] * 1.6)
    ax.legend(loc="upper left", fontsize=14, title=f"{len(rows)} produced samples, 180 per era",
              title_fontsize=14, frameon=True, facecolor="white", edgecolor="none", framealpha=0.92)
    an_style.cms_sim_labels([ax], com=13.6)
    return an_style.save(fig, "run3_ctau_all")


CSV_V2 = os.path.join(_DIR, "v2_census", "results", "ctau_v2.csv")
ERAS_V2 = V.ERAS_BY_PRODUCTION["v2"]
ECOL_V2 = {**ECOL, "2024": "#ff7f0e"}


def pull_stats(rows):
    """Pull mean with its error, pull rms (about zero) and std, measured/nominal mean and std, the
    largest |pull| and |ratio - 1| (with their samples) of a set of ctau_v2.csv rows."""
    p = np.array([float(x["pull"]) for x in rows]); r = np.array([float(x["ratio"]) for x in rows])
    ip, ir = int(np.argmax(abs(p))), int(np.argmax(abs(r - 1)))
    return dict(n_samples=len(rows), pull_mean=round(p.mean(), 4), pull_mean_err=round(p.std(ddof=1) / np.sqrt(len(p)), 4),
                pull_rms=round(float(np.sqrt((p ** 2).mean())), 4), pull_std=round(p.std(ddof=1), 4),
                ratio_mean=round(r.mean(), 5), ratio_std=round(r.std(ddof=1), 5),
                max_abs_pull=round(float(abs(p[ip])), 3), max_abs_pull_sample=f"{rows[ip]['era']} {rows[ip]['sample']}",
                max_abs_ratio_minus_1=round(float(abs(r[ir] - 1)), 5),
                max_abs_ratio_minus_1_sample=f"{rows[ir]['era']} {rows[ir]['sample']} (n_events {rows[ir]['n_events']})",
                n_abs_pull_gt2=int((abs(p) > 2).sum()), n_abs_pull_gt3=int((abs(p) > 3).sum()))


def figure_pulls(rows):
    fig, ax = plt.subplots(figsize=(10, 7.5), layout="constrained")
    edges = np.linspace(-5, 5, 21)
    for era in ERAS_V2:
        sel = [x for x in rows if x["era"] == era]
        st = pull_stats(sel)
        ax.hist([float(x["pull"]) for x in sel], bins=edges, histtype="step", lw=2.2, color=ECOL_V2[era],
                label=rf"{era}: mean {st['pull_mean']:+.2f} $\pm$ {st['pull_mean_err']:.2f}, rms {st['pull_rms']:.2f}")
    per_era = len(rows) / len(ERAS_V2)
    xx = np.linspace(edges[0], edges[-1], 300)
    ax.plot(xx, per_era * np.diff(edges)[0] * np.exp(-0.5 * xx ** 2) / np.sqrt(2 * np.pi), "k--", lw=1.8,
            label=f"unit Gaussian ({per_era:.0f} samples)")
    ax.set_xlabel(r"pull of the proper $c\tau$: (measured $-$ nominal) / stat. error")
    ax.set_ylabel("samples / 0.5")
    ax.set_xlim(edges[0], edges[-1]); ax.set_ylim(0, ax.get_ylim()[1] * 1.75)
    ax.legend(loc="upper left", fontsize=14, title=f"{len(rows)} produced samples, 180 per era",
              title_fontsize=14, frameon=True, facecolor="white", edgecolor="none", framealpha=0.92)
    an_style.cms_sim_labels([ax], com=13.6)
    return an_style.save(fig, "run3_ctau_all")


def main_v2(summary_path):
    rows = list(csv.DictReader(open(CSV_V2)))
    out = {era: pull_stats([x for x in rows if x["era"] == era]) for era in ERAS_V2}
    out["all"] = pull_stats(rows)
    out["input"] = os.path.relpath(CSV_V2, _DIR)
    out["code_commit"] = V.code_commit()
    for k, v in out.items():
        if isinstance(v, dict):
            print(f"{k:8s} " + ", ".join(f"{a}={b}" for a, b in v.items()))
    if summary_path:
        json.dump(out, open(summary_path, "w"), indent=1)
        print(f"wrote {summary_path}")
    an_style.FIG_DIR = os.path.join(_DIR, "figures_v2")
    print(f"wrote {figure_pulls(rows)}")


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--production", choices=V.PRODUCTIONS, default="v2")
    ap.add_argument("--nfiles", type=int, default=4)
    ap.add_argument("--workers", type=int, default=2)
    ap.add_argument("--replot", action="store_true")
    ap.add_argument("--summary", default=None, help="v2: write the pull and ratio statistics to this JSON")
    args = ap.parse_args()
    an_style.set_style()
    if args.production == "v2":
        return main_v2(args.summary)
    rows = list(csv.DictReader(open(CSV))) if args.replot else run(args.nfiles, args.workers)
    for era in ERAS:
        r = np.array([float(x["ratio"]) for x in rows if x["era"] == era])
        off = [(x["sample"], x["ratio"]) for x in rows if x["era"] == era and abs(float(x["ratio"]) - 1) > 0.1]
        print(f"{era}: {len(r)} samples, ratio mean {r.mean():.4f} std {r.std():.4f}, "
              f"min {r.min():.3f} max {r.max():.3f}; off by > 10%: {off}")
    figure(rows)


if __name__ == "__main__":
    main()
