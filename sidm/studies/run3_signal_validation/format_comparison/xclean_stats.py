#!/usr/bin/env python3
"""Tables of the v1 / v2 comparison of the DSA-PF segment-match veto from the rows of xclean_rows.py.

Pass fraction = fraction of "pre" DSA muons (pT > 10 GeV, abs(eta) < 2.4, displaced ID) passing the veto; v1 with
the stored nSegments, v2 with the rebuilt one (nDTSegments + ceil(trkNumCSCHits / 6)). Uncertainties are binomial
per muon, the two productions independent: s = sqrt(p1 (1 - p1) / n1 + p2 (1 - p2) / n2). The column s_evt is the
same with the variance clustered by event (muons of one event are not independent), as a check of that
assumption. Combinations over eras or samples are inverse-variance weighted means of the differences; chi2 is
sum (d / s)^2 of the listed differences against zero.

Usage: xclean_stats.py ROWS_DIR OUT.md [--step2-only]
"""
import glob
import os
import sys

import numpy as np
from scipy import stats

ERAS = ["2022", "2022EE", "2023", "2023BPix"]
CHANS = ["2Mu2E", "4Mu"]
DISPLACED = "{chan}_200GeV_1p2GeV_4p8mm"
PROMPT = ["{chan}_100GeV_1p2GeV_0p096mm", "{chan}_150GeV_1p2GeV_0p064mm", "{chan}_200GeV_1p2GeV_0p048mm",
          "{chan}_500GeV_1p2GeV_0p019mm", "{chan}_800GeV_1p2GeV_0p012mm", "{chan}_1000GeV_1p2GeV_0p0096mm"]
HERE = os.path.dirname(os.path.abspath(__file__))
COLUMN = [s.strip() for s in open(os.path.join(HERE, "..", "v2_census/production/reduced_points.txt")) if s.strip()]
DEC = {"v1": "all_stored", "v2": "all_formula"}
DECQ = {"v1": "allq_stored", "v2": "allq_formula"}
N_FILES = {"column": 120, "prompt": 4}


def chunk(path):
    return int(path.rsplit("_j", 1)[1][:3])


class Rows:
    """Concatenated rows of one (version, era, list of samples), optionally restricted in chunk number."""

    def __init__(self, rows_dir, version, era, samples, chunks=None):
        parts, self.n_events, self.n_files, self.uid = [], 0, 0, 0
        for s in samples:
            for f in sorted(glob.glob(f"{rows_dir}/{version}/{era}/{s}/*.npz")):
                if chunks is not None and chunk(f) not in chunks:
                    continue
                z = np.load(f)
                d = {k: z[k] for k in z.files if k not in ("n_events", "url") and not k.startswith("ev_")}
                d["event_uid"] = z["evt"].astype(np.int64) + self.n_events
                d["ev"] = {k: z[k] for k in z.files if k.startswith("ev_")}
                parts.append(d)
                self.n_events += int(z["n_events"])
                self.n_files += 1
        keys = [k for k in parts[0] if k != "ev"]
        self.c = {k: np.concatenate([p[k] for p in parts]) for k in keys}
        self.ev = {k: np.concatenate([p["ev"][k] for p in parts]) for k in parts[0]["ev"]}

    def __getitem__(self, k):
        return self.c[k]

    def __len__(self):
        return len(self.c["pt"])


def frac(passed, uid=None):
    """Pass fraction, binomial variance, and variance clustered by event."""
    n = len(passed)
    p = passed.mean()
    var = p * (1 - p) / n
    if uid is None:
        return p, var, var, n
    r = passed.astype(np.float64) - p
    _, inv = np.unique(uid, return_inverse=True)
    var_evt = np.sum(np.bincount(inv, weights=r) ** 2) / n ** 2
    return p, var, var_evt, n


def compare(r1, r2, sel1=None, sel2=None, dec=DEC):
    a = r1[dec["v1"]] if sel1 is None else r1[dec["v1"]][sel1]
    b = r2[dec["v2"]] if sel2 is None else r2[dec["v2"]][sel2]
    u1 = r1["event_uid"] if sel1 is None else r1["event_uid"][sel1]
    u2 = r2["event_uid"] if sel2 is None else r2["event_uid"][sel2]
    p1, v1, e1, n1 = frac(a, u1)
    p2, v2, e2, n2 = frac(b, u2)
    return dict(p1=p1, p2=p2, n1=n1, n2=n2, d=p2 - p1, s=np.sqrt(v1 + v2), s_evt=np.sqrt(e1 + e2))


def combine(results):
    w = np.array([1 / r["s"] ** 2 for r in results])
    d = np.array([r["d"] for r in results])
    dc = np.sum(w * d) / np.sum(w)
    sc = 1 / np.sqrt(np.sum(w))
    we = np.array([1 / r["s_evt"] ** 2 for r in results])
    se = 1 / np.sqrt(np.sum(we))
    chi2 = float(np.sum(d ** 2 * w))
    return dict(d=dc, s=sc, s_evt=se, chi2=chi2, ndof=len(results), n_neg=int(np.sum(d < 0)),
                n_pos=int(np.sum(d > 0)))


def fmt_row(label, r, extra=""):
    return (f"| {label} | {r['n1']:,} | {r['n2']:,} | {r['p1']:.4f} | {r['p2']:.4f} | {100 * r['d']:+.2f} | "
            f"{100 * r['s']:.2f} | {r['d'] / r['s']:+.1f} | {100 * r['s_evt']:.2f} |{extra}")


HEAD = ("| {what} | pre v1 | pre v2 | v1 | v2 rebuilt | v2 - v1 (points) | s (points) | pull | s_evt (points) |\n"
        "|---|---|---|---|---|---|---|---|---|")


def fmt_comb(label, c):
    p = stats.chi2.sf(c["chi2"], c["ndof"])
    return (f"| {label} | {100 * c['d']:+.2f} +- {100 * c['s']:.2f} | {c['d'] / c['s']:+.1f} | "
            f"{100 * c['s_evt']:.2f} | {c['n_neg']} / {c['n_pos']} | {c['chi2']:.1f} / {c['ndof']} | {p:.3f} |")


COMB_HEAD = ("| combination | d +- s (points) | d / s | s_evt (points) | eras or samples below / above zero | "
             "chi2 / ndof | p(chi2) |\n|---|---|---|---|---|---|---|")


def verdict(c, per_era):
    z = c["d"] / c["s"]
    same = sum(1 for r in per_era if np.sign(r["d"]) == np.sign(c["d"]))
    if abs(z) < 2:
        v = "not confirmed"
    elif abs(z) > 3 and same >= 3:
        v = "confirmed"
    else:
        v = "inconclusive"
    return v, z, same


def main():
    rows_dir, out = sys.argv[1:3]
    step2_only = "--step2-only" in sys.argv[3:]
    L = []
    w = L.append

    # ---- step 2: the eight sets (plus the two new eras) on all events of the same samples
    w("## Step 2: the eight sets on all events, four eras\n")
    w("Displaced = `<chan>_200GeV_1p2GeV_4p8mm`, all 120 files (100,000 events); prompt-like = the six points "
      "`<chan>_<M>GeV_1p2GeV_<ctau>mm` of the 2026-11-12 table, all 4 files each (12,000 events: these are grid-only "
      "samples, so step 1 already used all their events). Veto \"all\".\n")
    step2 = {}
    for chan in CHANS:
        for kind, samples in [("displaced", [DISPLACED.format(chan=chan)]),
                              ("prompt-like", [p.format(chan=chan) for p in PROMPT])]:
            w(f"### {chan} {kind}\n")
            w(HEAD.format(what="era (events v1 / v2)"))
            res, resq, res_new = [], [], []
            for era in ERAS:
                r1, r2 = Rows(rows_dir, "v1", era, samples), Rows(rows_dir, "v2", era, samples)
                r = compare(r1, r2)
                rq = compare(r1, r2, dec=DECQ)
                res.append(r)
                resq.append(rq)
                w(fmt_row(f"{era} ({r1.n_events:,} / {r2.n_events:,}; files {r1.n_files} / {r2.n_files})", r))
                if kind == "displaced":
                    keep = set(range(120)) - set(range(40, 60))
                    res_new.append(compare(Rows(rows_dir, "v1", era, samples, keep),
                                           Rows(rows_dir, "v2", era, samples, keep)))
            w("")
            w(COMB_HEAD)
            c = combine(res)
            w(fmt_comb("four eras", c))
            w(fmt_comb("2022 + 2023", combine([res[0], res[2]])))
            w(fmt_comb("2022EE + 2023BPix", combine([res[1], res[3]])))
            if res_new:
                w(fmt_comb("four eras, without the step-1 chunks j040-j059", combine(res_new)))
            w(fmt_comb("\"all + charge\", four eras", combine(resq)))
            w("")
            step2[(chan, kind)] = (c, res)

    if step2_only:
        c, res = step2[("2Mu2E", "displaced")]
        v, z, same = verdict(c, res)
        w(f"- **step 2, 2Mu2E displaced: {v}.** d = {100 * c['d']:+.3f} +- {100 * c['s']:.3f} points, "
          f"d/s = {z:+.2f} (event-clustered s: {100 * c['s_evt']:.3f} points, d/s_evt = {c['d'] / c['s_evt']:+.2f}), {same} of 4 eras with the sign of d.")
        open(out, "w").write("\n".join(L) + "\n")
        print("\n".join(L))
        return

    # ---- step 3: all column samples
    w("## Step 3: all column samples (18 per channel), four eras\n")
    w("Every column sample has 120 files (100,000 events) per era and production. All 36 have ctau scaled with "
      "the boost (lab decay length of order 40 cm), so all are \"displaced\" in the sense of the 2026-11-12 table; "
      "its prompt-like points are grid-only samples (step 2). Median generated Lxy of the gen-matched (dR < 0.3) "
      "\"pre\" DSA muons is listed per sample (v1, 2022).\n")
    step3 = {}
    for chan in CHANS:
        samples = [s for s in COLUMN if s.startswith(chan + "_")]
        w(f"### {chan}: per sample and era\n")
        w("| sample | median gen Lxy (cm) | " + " | ".join(f"{e}: v2 - v1 (points), pull" for e in ERAS) +
          " | chi2 / 4 |")
        w("|---|---|" + "---|" * len(ERAS) + "---|")
        per_sample = []
        pooled = {e: [] for e in ERAS}
        for s in samples:
            cells, rs = [], []
            for era in ERAS:
                r1, r2 = Rows(rows_dir, "v1", era, [s]), Rows(rows_dir, "v2", era, [s])
                r = compare(r1, r2)
                rs.append(r)
                cells.append(f"{100 * r['d']:+.2f} +- {100 * r['s']:.2f}, {r['d'] / r['s']:+.1f}")
                if era == "2022":
                    g = r1["gen_dr"] < 0.3
                    med = np.median(r1["gen_lxy"][g])
            per_sample += rs
            chi2 = sum((r["d"] / r["s"]) ** 2 for r in rs)
            w(f"| {s} | {med:.1f} | " + " | ".join(cells) + f" | {chi2:.1f} |")
        w("")
        w(f"### {chan}: pooled over the 18 samples\n")
        w(HEAD.format(what="era"))
        res = []
        for era in ERAS:
            r = compare(Rows(rows_dir, "v1", era, samples), Rows(rows_dir, "v2", era, samples))
            res.append(r)
            w(fmt_row(era, r))
        w("")
        w(COMB_HEAD)
        c = combine(res)
        w(fmt_comb("pooled samples, four eras", c))
        w(fmt_comb(f"per-sample differences ({len(per_sample)} = 18 samples x 4 eras)", combine(per_sample)))
        w("")
        step3[chan] = (c, res, combine(per_sample))

    # ---- step 5
    w("## Step 5: verdicts\n")
    w("Rule: |d|/s < 2 not confirmed; |d|/s > 3 with the same sign in at least three of the four eras confirmed; "
      "otherwise inconclusive.\n")
    for label, (c, res) in [("step 2, 2Mu2E displaced", step2[("2Mu2E", "displaced")]),
                            ("step 3, 2Mu2E pooled column samples", step3["2Mu2E"][:2])]:
        v, z, same = verdict(c, res)
        w(f"- **{label}: {v}.** d = {100 * c['d']:+.3f} +- {100 * c['s']:.3f} points, d/s = {z:+.2f} (event-clustered s: {100 * c['s_evt']:.3f} points, d/s_evt = {c['d'] / c['s_evt']:+.2f}), "
          f"{same} of 4 eras with the sign of d.")
    w("")
    open(out, "w").write("\n".join(L) + "\n")
    print("\n".join(L))


if __name__ == "__main__":
    main()
