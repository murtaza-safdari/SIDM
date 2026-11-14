#!/usr/bin/env python3
"""Three checks of the veto comparison on the rows of xclean_rows.py.

(a) The rebuilt count on the first production, where the stored count is correct: over all v1 "pre" DSA muons of
    the four eras, the "all" decisions that change when nDTSegments + ceil(trkNumCSCHits / 6) replaces the stored
    nSegments (pass to fail, fail to pass), the fraction with rebuilt = stored (all muons, and those with CSC
    segments), and the number with rebuilt > stored.
(b) Pooled v2 - v1 over the 18 column samples (120 files) per channel, four eras, with the binomial uncertainty
    and with the variance clustered by file inside each sample, var(p) = sum_s sum_f (k_f - p_s n_f)^2 / N^2
    (sample totals fixed, files independent). Eras are combined with inverse-variance weights.
(c) File-level design effect per (production, era, sample): sum_f (k_f - p n_f)^2 / (N p (1 - p)); 1 for
    independent muons.
k_f and n_f are the passing and all "pre" muons of file f (v1 stored count, v2 rebuilt count).

Usage: xclean_checks.py ROWS_DIR [--counts COUNTS.json]
  --counts: per-file (k, n) cache of the column samples; read if it exists, written otherwise.
"""
import argparse
import collections
import glob
import json
import os

import numpy as np

ERAS = ["2022", "2022EE", "2023", "2023BPix"]
CHANS = ["2Mu2E", "4Mu"]
DEC = {"v1": "all_stored", "v2": "all_formula"}


def check_a(rows_dir):
    tot = np.zeros(8, dtype=np.int64)
    for era in ERAS:
        for f in glob.glob(f"{rows_dir}/v1/{era}/*/*.npz"):
            z = np.load(f)
            st, fo = z["all_stored"], z["all_formula"]
            nseg = z["nSegments"].astype(np.int64)
            reb = z["nDTSegments"].astype(np.int64) + np.ceil(z["trkNumCSCHits"] / 6.0).astype(np.int64)
            csc = z["nCSCSegments"] > 0
            tot += [len(st), (st != fo).sum(), (st & ~fo).sum(), (~st & fo).sum(), (reb == nseg).sum(),
                    csc.sum(), (reb[csc] == nseg[csc]).sum(), (reb > nseg).sum()]
    n, ch, pf, fp, eq, ncsc, eqc, over = tot
    print("(a) first production, rebuilt count in place of the stored one, four eras, all samples")
    print(f"    pre muons {n:,}; decisions changed {ch} (pass to fail {pf}, fail to pass {fp}); "
          f"rebuilt = stored: all {eq / n:.4f}, with CSC segments {eqc / ncsc:.4f} (of {ncsc:,}); "
          f"rebuilt > stored: {over}")


def counts(rows_dir, path):
    if path and os.path.exists(path):
        return json.load(open(path))
    kn = {}
    for ver, key in DEC.items():
        for era in ERAS:
            for sdir in sorted(glob.glob(f"{rows_dir}/{ver}/{era}/*")):
                fs = sorted(glob.glob(sdir + "/*.npz"))
                if len(fs) != 120:
                    continue
                kn[f"{ver}|{era}|{os.path.basename(sdir)}"] = [
                    [int(a.sum()), int(len(a))] for a in (np.load(f)[key] for f in fs)]
    if path:
        json.dump(kn, open(path, "w"))
    return kn


def kn_arrays(kn, key):
    a = np.array(kn[key])
    return a[:, 0], a[:, 1]


def check_b(kn):
    print("(b) pooled v2 - v1, 18 column samples per channel, four eras (points)")
    for chan in CHANS:
        ds, sb, sl, nsamp = [], [], [], set()
        for era in ERAS:
            r = {}
            for ver in DEC:
                keys = [k for k in kn if k.startswith(f"{ver}|{era}|{chan}_")]
                nsamp.add(len(keys))
                K = N = 0
                lin = 0.0
                for key in keys:
                    k, n = kn_arrays(kn, key)
                    ps = k.sum() / n.sum()
                    lin += ((k - ps * n) ** 2).sum()
                    K += k.sum()
                    N += n.sum()
                p = K / N
                r[ver] = (p, p * (1 - p) / N, lin / N ** 2)
            ds.append(r["v2"][0] - r["v1"][0])
            sb.append(np.sqrt(r["v1"][1] + r["v2"][1]))
            sl.append(np.sqrt(r["v1"][2] + r["v2"][2]))
        ds = np.array(ds)
        out = []
        for s in (np.array(sb), np.array(sl)):
            w = 1 / s ** 2
            out.append(((w * ds).sum() / w.sum(), 1 / np.sqrt(w.sum())))
        print(f"    {chan} (samples per era {sorted(nsamp)}): d = {100 * out[0][0]:+.3f} +- {100 * out[0][1]:.3f} "
              f"(binomial), +- {100 * out[1][1]:.3f} (file-clustered); d/s = {out[0][0] / out[0][1]:+.2f}, "
              f"{out[1][0] / out[1][1]:+.2f}")


def check_c(kn):
    deff = {}
    for key in kn:
        k, n = kn_arrays(kn, key)
        N = n.sum()
        p = k.sum() / N
        deff[key] = ((k - p * n) ** 2).sum() / (N * p * (1 - p))
    d = np.array(list(deff.values()))
    bys = collections.defaultdict(list)
    for key, v in deff.items():
        bys[key.split("|")[2]].append(v)
    means = {s: np.mean(v) for s, v in bys.items()}
    print("(c) file-level design effect per (production, era, sample)")
    print(f"    cells {len(d)}: mean {d.mean():.2f}, median {np.median(d):.2f}, min {d.min():.2f}, "
          f"max {d.max():.2f}; per-sample means ({len(means)} samples, 8 cells each) "
          f"{min(means.values()):.2f} to {max(means.values()):.2f}")
    for chan in CHANS:
        for ver in DEC:
            dd = [v for key, v in deff.items() if key.startswith(ver + "|") and f"|{chan}_" in key]
            print(f"    {chan} {ver}: {len(dd)} cells, mean {np.mean(dd):.2f}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("rows_dir")
    ap.add_argument("--counts", default=None)
    args = ap.parse_args()
    check_a(args.rows_dir)
    kn = counts(args.rows_dir, args.counts)
    check_b(kn)
    check_c(kn)


if __name__ == "__main__":
    main()
