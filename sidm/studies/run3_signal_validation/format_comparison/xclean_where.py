#!/usr/bin/env python3
"""Where v1 and v2 first differ, for one set of samples: the step-6 tables of the veto comparison.

(a) DSA muons per event (all, "pre"); (b) segment count, v1 stored against v2 rebuilt, barrel (DT hits only) and
endcap (CSC hits only); (c) the five muonMatch slot counts; (d) PF muons per event passing the ljsource muon cuts,
and good matched PF muons per "pre" DSA muon; (e) pass fraction of "all" in bins of DSA abs(eta) and of gen Lxy
(gen-matched, dR < 0.3); (f) v2 pass fraction with one more segment for every DSA muon with CSC hits.
Each line is v1, v2, v2 - v1 and its pull, the uncertainty binomial for fractions and sigma / sqrt(n) for means,
the productions independent.

Usage: xclean_where.py ROWS_DIR OUT.md LABEL SAMPLE[,SAMPLE...] [ERA ...]
"""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from xclean_stats import Rows  # noqa: E402

ERAS = ["2022", "2022EE", "2023", "2023BPix"]


def fr(x):
    p = x.mean()
    return p, np.sqrt(p * (1 - p) / len(x)), len(x)


def mn(x):
    return x.mean(), x.std() / np.sqrt(len(x)), len(x)


def line(label, a, b, scale=1.0, fmt=".4f"):
    (p1, s1, n1), (p2, s2, n2) = a, b
    d, s = p2 - p1, np.hypot(s1, s2)
    pull = d / s if s > 0 else 0.0
    return (f"| {label} | {n1:,} | {n2:,} | {p1 * scale:{fmt}} | {p2 * scale:{fmt}} | {d * scale:+{fmt}} | "
            f"{pull:+.1f} |")


HEAD = "| quantity | n v1 | n v2 | v1 | v2 | v2 - v1 | pull |\n|---|---|---|---|---|---|---|"


def main():
    rows_dir, out, label, samples = sys.argv[1:5]
    samples = samples.split(",")
    eras = sys.argv[5:] or ERAS
    L = [f"## Step 6: {label}\n", f"Samples: {', '.join(samples)}; eras {', '.join(eras)}.\n"]
    w = L.append
    for era in eras:
        r1, r2 = Rows(rows_dir, "v1", era, samples), Rows(rows_dir, "v2", era, samples)
        seg1 = r1["nSegments"].astype(float)
        seg2 = (r2["nDTSegments"] + np.ceil(r2["trkNumCSCHits"] / 6)).astype(float)
        w(f"### {era} ({r1.n_events:,} / {r2.n_events:,} events)\n")
        w(HEAD)
        # (a)
        w(line("(a) DSA muons per event, mean", mn(r1.ev["ev_n_dsa"]), mn(r2.ev["ev_n_dsa"])))
        w(line("(a) \"pre\" DSA muons per event, mean", mn(r1.ev["ev_n_pre"]), mn(r2.ev["ev_n_pre"])))
        for k in range(4):
            lab = f"{k}" if k < 3 else "3 or more"
            sel = (lambda e: e >= 3) if k == 3 else (lambda e, k=k: e == k)
            w(line(f"(a) fraction of events with {lab} \"pre\"", fr(sel(r1.ev["ev_n_pre"])),
                   fr(sel(r2.ev["ev_n_pre"]))))
        # (b)
        regions = {"barrel (DT hits only)": lambda r: (r["trkNumCSCHits"] == 0) & (r["trkNumDTHits"] > 0),
                   "endcap (CSC hits only)": lambda r: (r["trkNumDTHits"] == 0) & (r["trkNumCSCHits"] > 0),
                   "overlap (both)": lambda r: (r["trkNumDTHits"] > 0) & (r["trkNumCSCHits"] > 0)}
        for name, f in regions.items():
            m1, m2 = f(r1), f(r2)
            w(line(f"(b) {name}: fraction of \"pre\"", fr(m1), fr(m2)))
            w(line(f"(b) {name}: segments, mean (v1 stored, v2 rebuilt)", mn(seg1[m1]), mn(seg2[m2]), fmt=".3f"))
            for k in range(1, 7):
                lab = f"= {k}" if k < 6 else ">= 6"
                s1 = seg1[m1] >= k if k == 6 else seg1[m1] == k
                s2 = seg2[m2] >= k if k == 6 else seg2[m2] == k
                w(line(f"(b) {name}: segments {lab}", fr(s1), fr(s2)))
            if "CSC" in name or "both" in name:
                w(line(f"(b) {name}: v1 stored nCSCSegments vs v2 ceil(CSC hits / 6), mean",
                       mn(r1["nCSCSegments"][m1].astype(float)), mn(np.ceil(r2["trkNumCSCHits"][m2] / 6)), fmt=".3f"))
                w(line(f"(b) {name}: CSC hits, mean", mn(r1["trkNumCSCHits"][m1].astype(float)),
                       mn(r2["trkNumCSCHits"][m2].astype(float)), fmt=".3f"))
        # (c)
        for k in range(1, 6):
            a, b = r1[f"slotN{k}"].astype(float), r2[f"slotN{k}"].astype(float)
            w(line(f"(c) muonMatch{k} > 0", fr(a > 0), fr(b > 0)))
            w(line(f"(c) muonMatch{k}, mean over filled slots", mn(a[a >= 0]), mn(b[b >= 0]), fmt=".3f"))
        a, b = r1["slotN1"].astype(float), r2["slotN1"].astype(float)
        for k in range(1, 5):
            lab = f"= {k}" if k < 4 else ">= 4"
            w(line(f"(c) muonMatch1 {lab}", fr(a >= k if k == 4 else a == k), fr(b >= k if k == 4 else b == k)))
        # (d)
        w(line("(d) good PF muons per event, mean", mn(r1.ev["ev_n_good_pf"].astype(float)),
               mn(r2.ev["ev_n_good_pf"].astype(float)), fmt=".3f"))
        w(line("(d) good matched PF muons per \"pre\" DSA muon, mean", mn(r1["n_good_matched"].astype(float)),
               mn(r2["n_good_matched"].astype(float)), fmt=".3f"))
        w(line("(d) \"pre\" DSA muons with >= 1 good matched PF muon", fr(r1["n_good_matched"] > 0),
               fr(r2["n_good_matched"] > 0)))
        # (e)
        p1, p2 = r1["all_stored"], r2["all_formula"]
        w(line("(e) pass \"all\", all \"pre\"", fr(p1), fr(p2)))
        for lo, hi in [(0, 0.9), (0.9, 1.2), (1.2, 2.4)]:
            m1 = (abs(r1["eta"]) >= lo) & (abs(r1["eta"]) < hi)
            m2 = (abs(r2["eta"]) >= lo) & (abs(r2["eta"]) < hi)
            w(line(f"(e) pass \"all\", abs(eta) {lo}-{hi}", fr(p1[m1]), fr(p2[m2])))
        g1, g2 = r1["gen_dr"] < 0.3, r2["gen_dr"] < 0.3
        w(line("(e) gen-matched fraction of \"pre\"", fr(g1), fr(g2)))
        for lo, hi in [(0, 1), (1, 10), (10, 50), (50, 150), (150, np.inf)]:
            m1 = g1 & (r1["gen_lxy"] >= lo) & (r1["gen_lxy"] < hi)
            m2 = g2 & (r2["gen_lxy"] >= lo) & (r2["gen_lxy"] < hi)
            if m1.sum() and m2.sum():
                w(line(f"(e) pass \"all\", gen-matched, gen Lxy {lo}-{hi} cm", fr(p1[m1]), fr(p2[m2])))
                w(line(f"(e) fraction of gen-matched \"pre\" at gen Lxy {lo}-{hi} cm", fr(m1[g1]), fr(m2[g2])))
        # (f)
        w(line("(f) pass \"all\", v2 with rebuilt + 1 for CSC-hit muons (v1 stored)", fr(p1), fr(r2["all_plus1"])))
        w(line("(f) same, v1 also rebuilt + 1 (both shifted)", fr(r1["all_plus1"]), fr(r2["all_plus1"])))
        w("")
    open(out, "w").write("\n".join(L) + "\n")
    print("\n".join(L))


if __name__ == "__main__":
    main()
