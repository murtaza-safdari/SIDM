#!/usr/bin/env python3
"""Summary of the committed LJ-efficiency scans (lj_eff_{real,geom}_<era>.json): per mode, era,
channel and efficiency, the per-point (Run 3 - 2018) differences over the 18 grid points as mean,
statistical error of the mean (per-point binomial errors propagated), RMS, [min, max], number
negative, and the chi2 of the 18 differences about their mean (17 dof). A chi2 far above 17 means
the points do not share a common shift, so the mean +- stat is not the uncertainty of one. Also the
in-acceptance fractions. Writes lj_eff_summary<tag>.json (--production v1: no tag, the files of the
first production; v2, the default: _v2, from lj_eff_{real,geom}_v2_<era>.json, five eras). No EOS access."""
import argparse, json, os, sys
import numpy as np

_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _DIR)
import run3_signal_validation as V

def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--production", choices=V.PRODUCTIONS, default="v2")
    args = ap.parse_args()
    eras, tag = V.ERAS_BY_PRODUCTION[args.production], V.output_tag(args.production)
    out = dict(code_commit=V.code_commit(), production=args.production,
               definition=__doc__.split("\n\n")[0].replace("\n", " "))
    for mode in ("geom", "real"):
        out[mode] = {}
        for era in eras:
            rows = json.load(open(os.path.join(_DIR, f"lj_eff_{mode}{tag}_{era}.json")))
            a3 = np.array([r["run3"]["n_acc"] / r["run3"]["n"] for r in rows])
            a2 = np.array([r["v10"]["n_acc"] / r["v10"]["n"] for r in rows])
            e = out[mode][era] = dict(
                acceptance_run3=dict(mean=round(a3.mean(), 4), min=round(a3.min(), 4), max=round(a3.max(), 4)),
                acceptance_2018=dict(mean=round(a2.mean(), 5), min=round(a2.min(), 5), max=round(a2.max(), 5)),
                inputs_commit=sorted({r.get("code_commit") for r in rows}))
            for chan in ("4Mu", "2Mu2E"):
                for fld in ("incl_acc", "excl_acc") if mode == "real" else ("incl_acc",):
                    d, s = [], []
                    for r in rows:
                        if r["chan"] != chan:
                            continue
                        x, y = r["run3"][fld], r["v10"][fld]
                        d.append(100 * (x["p"] - y["p"]))
                        s.append(100 * np.sqrt(x["p"] * (1 - x["p"]) / x["n"] + y["p"] * (1 - y["p"]) / y["n"]))
                    d, s = np.array(d), np.array(s)
                    mean = d.mean()
                    e[f"{chan}_{fld}"] = dict(
                        n_points=len(d), mean_pct=round(mean, 3), stat_of_mean_pct=round(np.sqrt((s ** 2).sum()) / len(d), 3),
                        rms_pct=round(d.std(), 3), min_pct=round(d.min(), 3), max_pct=round(d.max(), 3),
                        n_negative=int((d < 0).sum()),
                        chi2_about_mean=round(float((((d - mean) / s) ** 2).sum()), 1), ndof=len(d) - 1)
    path = os.path.join(_DIR, f"lj_eff_summary{tag}.json")
    json.dump(out, open(path, "w"), indent=1)
    print(f"wrote {path}")
    for mode in ("geom", "real"):
        for era in eras:
            for k, v in out[mode][era].items():
                if isinstance(v, dict) and "mean_pct" in v:
                    print(f"{mode} {era:8s} {k:14s} mean {v['mean_pct']:+.3f} +- {v['stat_of_mean_pct']:.3f}  rms {v['rms_pct']:.3f}"
                          f"  [{v['min_pct']:+.2f}, {v['max_pct']:+.2f}]  neg {v['n_negative']}/18  chi2 {v['chi2_about_mean']}/17")


if __name__ == "__main__":
    main()
