#!/usr/bin/env python3
"""Every number of the first Run 3 production's outputs (v1, LLPNanoAOD: era_fingerprint.csv,
era_fingerprint_regions.json, gen_comparison_summary.json, lj_eff_summary.json) beside its value in the
second production (v2, EXONanoAOD), with statistical uncertainties, for the four common eras plus the 2024
column (v2 only). Reads the committed v1 and v2 result files of this directory (the tables and JSONs above,
their *_v2 counterparts, the era_fingerprint npz maps and the per-point lj_eff_{geom,real}[_v2]_<era>.json)
and writes v1_vs_v2.md beside this script (--out to change). No EOS access.
Run from this directory with the notebook venv:  python v1_vs_v2.py

Uncertainties (statistical; v1 and v2 are independent productions, the 2018 v10 reference is the same
files in both, so it cancels in v2 - v1):
  * fractions: binomial sqrt(p(1-p)/N) with the N of each number;
  * means: std/sqrt(N); standard deviations: std/sqrt(2N); GenVtx x, y means are rounded to 1e-5 cm in the
    CSV (coarser than their error), so they are compared through the unrounded pulls vtx_pull_x, _y (unit
    error each; VtxSmeared is the same in v1 and v2);
  * medians of stored quantities (sqrt(s), m/sqrt(s_hat)): quantised by nano's stored precision, no error;
  * region averages of efficiencies: binomial with the region's denominator; of the mean number of
    tracker layers: s/sqrt(n) with a per-muon spread s measured on 6 v2 chunks of 4Mu 2023/2023BPix
    (--spread-in 1.4 inside the BPix region, --spread-out 1.9 outside) - an estimate, the v1 maps keep no
    second moment;
  * LJ efficiencies: per point binomial errors of the Run 3 efficiencies only (2018 common).
Differences beyond 2 sigma are listed at the end.
"""
import argparse, csv, json, os, sys
import numpy as np

S = os.path.dirname(os.path.abspath(__file__))
_ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
_ap.add_argument("--out", default=os.path.join(S, "v1_vs_v2.md"))
_ap.add_argument("--spread-in", type=float, default=1.4,
                 help="per-muon spread of the tracker-layer count inside the BPix region")
_ap.add_argument("--spread-out", type=float, default=1.9, help="the same outside it")
_args = _ap.parse_args()
OUT = _args.out
COMMON = ["2022", "2022EE", "2023", "2023BPix"]
S_IN, S_OUT = _args.spread_in, _args.spread_out

rows = []          # (section, quantity, era, v1, s1, v2, s2, note)


def add(sec, q, era, v1, s1, v2, s2, note=""):
    rows.append((sec, q, era, v1, s1, v2, s2, note))


def binom(p, n):
    return float(np.sqrt(p * (1 - p) / n)) if n else float("nan")


# ------------------------------------------------------------------ era_fingerprint.csv
def load_csv(name):
    return {r["era"]: r for r in csv.DictReader(open(os.path.join(S, name)))}


C1, C2 = load_csv("era_fingerprint.csv"), load_csv("era_fingerprint_v2.csv")
H1 = dict(np.load(os.path.join(S, "era_fingerprint.npz"), allow_pickle=True))
H2 = dict(np.load(os.path.join(S, "era_fingerprint_v2.npz"), allow_pickle=True))


def hstd(H, k, era):
    h, e = H[f"vtx{k}_{era}"], H[f"vtx{k}_edges"]
    c = 0.5 * (e[1:] + e[:-1]); m = (h * c).sum() / h.sum()
    return float(np.sqrt((h * (c - m) ** 2).sum() / h.sum()))


def csv_err(r, H, era, col):
    N = int(r["events"]); x = r.get(col, "")
    if x in ("", None):
        return None
    try:
        v = float(x)
    except ValueError:
        return None
    if col in ("sqrt_s_frac_below_tail_cut", "frac_events_pass_2018_req"):
        return binom(v, N)
    if col == "sqrt_s_frac_below_tail_cut_after_2018_req":
        return binom(v, N * float(r["frac_events_pass_2018_req"]))
    if col == "lhe_frac_with_outgoing_parton_tail":
        return binom(v, N * float(r["sqrt_s_frac_below_tail_cut"]))
    if col == "lhe_frac_with_outgoing_parton_core":
        return binom(v, N * (1 - float(r["sqrt_s_frac_below_tail_cut"])))
    if col == "pileup_nTrueInt_mean":
        return float(r["pileup_nTrueInt_std"]) / np.sqrt(N)
    if col == "pileup_nTrueInt_std":
        return float(r["pileup_nTrueInt_std"]) / np.sqrt(2 * N)
    if col in ("genvtx_x_mean_cm", "genvtx_y_mean_cm"):
        return None                   # rounded to 1e-5 cm, coarser than the error (~2e-6): see vtx_pull_x, _y
    if col == "genvtx_z_mean_cm":
        return float(r["genvtx_z_std_cm"]) / np.sqrt(N)
    if col in ("genvtx_z_std_cm", "pv_z_std_cm"):
        return v / np.sqrt(2 * N)
    if col.startswith("vtx_pull_"):
        return 1.0                    # a pull: unit spread per production
    return None


SKIP = {"era", "code_commit", "channels"}
cols = [c for c in C1["2022"] if c not in SKIP and not c.startswith(("ee_", "bp_"))]
for era in COMMON + ["2024", "2018"]:
    r1, r2 = C1.get(era), C2.get(era)
    for c in cols + [c for c in C2.get(era, {}) if c not in cols and c not in SKIP and not c.startswith(("ee_", "bp_"))]:
        v1 = r1.get(c, "") if r1 else ""
        v2 = r2.get(c, "") if r2 else ""
        if v1 == "" and v2 == "":
            continue
        s1 = csv_err(r1, H1, era, c) if r1 and v1 != "" else None
        s2 = csv_err(r2, H2, era, c) if r2 and v2 != "" else None
        note = ("stored precision" if "median" in c else
                "rounded to 1e-5 cm in the CSV; compared through vtx_pull_*" if c in ("genvtx_x_mean_cm", "genvtx_y_mean_cm") else
                "(mean - VtxSmeared) / (std/sqrt(N)); VtxSmeared identical in v1 and v2" if c.startswith("vtx_pull_") else "")
        add("era_fingerprint.csv", c, era, v1, s1, v2, s2, note)

# ------------------------------------------------------------------ era_fingerprint_regions.json
R1 = json.load(open(os.path.join(S, "era_fingerprint_regions.json")))
R2 = json.load(open(os.path.join(S, "era_fingerprint_regions_v2.json")))


def reg_err(key, lab, era, v, R):
    r = R[key]
    n = r.get(f"denominators_{lab}", {}).get(era)
    if lab == "outside_rectangle" and n is None:
        return None
    if n is None:
        return None
    if key.startswith("bp"):
        return (S_OUT if lab == "outside_rectangle" else S_IN) / np.sqrt(n)
    return binom(v, n)


for key, r1 in R1.items():
    if not isinstance(r1, dict):
        continue
    r2 = R2.get(key, {})
    rb = R2.get(f"{key}_in_v1_bins", {})
    for k in ("eta", "phi", "n_flagged_bins", "n_rectangle_bins", "n_rectangle_unflagged_bins"):
        add("regions (own flagged bins)", f"{key}.{k}", "-", r1.get(k), None, r2.get(k), None)
    for lab in ("flagged", "rectangle", "outside_rectangle"):
        for era, v1 in r1[f"{lab}_avg"].items():
            # v1 numbers against v2 in the same (v1) bins: the like-for-like comparison
            v2 = rb.get(f"{lab}_avg", {}).get(era)
            s1 = (binom(v1, r1[f"denominators_{lab}"][era]) if not key.startswith("bp") else
                  S_IN / np.sqrt(r1[f"denominators_{lab}"][era])) if f"denominators_{lab}" in r1 else None
            if lab == "outside_rectangle":     # v1 json has no outside denominators: take them from the npz
                s1 = None
            s2 = reg_err(f"{key}_in_v1_bins", lab, era, v2, R2) if v2 is not None else None
            add("regions (v1 bins)", f"{key}.{lab}_avg", era, v1, s1, v2, s2)
            add("regions (own flagged bins)", f"{key}.{lab}_avg", era, v1, None, r2.get(f"{lab}_avg", {}).get(era), None,
                "v2 over its own flagged bins")
        v2_24 = rb.get(f"{lab}_avg", {}).get("2024")
        if v2_24 is not None:
            add("regions (v1 bins)", f"{key}.{lab}_avg", "2024", "", None, v2_24,
                reg_err(f"{key}_in_v1_bins", lab, "2024", v2_24, R2))
    if key.startswith("bp"):
        for lab in ("flagged", "rectangle"):
            for era, v1 in r1[f"pf_eff_{lab}"].items():
                v2 = rb.get(f"pf_eff_{lab}", {}).get(era)
                n2 = rb.get(f"pf_denominators_{lab}", {}).get(era)
                add("regions (v1 bins)", f"{key}.pf_eff_{lab}", era, v1, None, v2, binom(v2, n2) if n2 else None)
            v2 = rb.get(f"pf_eff_{lab}", {}).get("2024")
            if v2 is not None:
                add("regions (v1 bins)", f"{key}.pf_eff_{lab}", "2024", "", None, v2,
                    binom(v2, rb[f"pf_denominators_{lab}"]["2024"]))
        for lab in ("phi_band_all_eta", "outside_phi_band"):
            for era, v1 in r1[f"pf_eff_{lab}"].items():
                add("regions (own flagged bins)", f"{key}.pf_eff_{lab}", era, v1, None,
                    r2.get(f"pf_eff_{lab}", {}).get(era), None)

# the v1 sigmas of the region averages, from the v1 npz over the v1 bins (same masks as the v2 numbers)
sys.path.insert(0, S)
import era_fingerprint as F                                   # noqa: E402


def mask_sigma(H, Href, kind, a, b, sfx, era, lab):
    sel = F.flagged(Href, kind, a, b, F.THR[kind], sfx)
    ie, ip = np.where(sel)
    rect = np.zeros_like(sel); rect[ie.min():ie.max() + 1, ip.min():ip.max() + 1] = True
    m = {"flagged": sel, "rectangle": rect, "outside_rectangle": ~rect}[lab]
    _, _, n, d, _ = F._maps(H, kind, a, era, sfx)
    p = n[m].sum() / d[m].sum(); N = d[m].sum()
    if kind == "bp":
        return (S_OUT if lab == "outside_rectangle" else S_IN) / np.sqrt(N)
    return binom(p, N)


KMAP = {"eedp_per-dark-photon": ("eedp", "2022", "2022EE", ""), "ee_exclusive": ("ee", "2022", "2022EE", ""),
        "ee_non-exclusive": ("ee", "2022", "2022EE", "_nx"), "bp_exclusive": ("bp", "2023", "2023BPix", ""),
        "bp_non-exclusive": ("bp", "2023", "2023BPix", "_nx")}
for i, (sec, q, era, v1, s1, v2, s2, note) in enumerate(rows):
    if sec == "regions (v1 bins)" and era in COMMON and q.split(".")[1].endswith("_avg") and "pf_eff" not in q:
        key, lab = q.split(".")[0], q.split(".")[1][:-4]
        kind, a, b, sfx = KMAP[key]
        s1 = mask_sigma(H1, H1, kind, a, b, sfx, era, lab)
        if lab == "outside_rectangle" and s2 is None:
            s2 = mask_sigma(H2, H1, kind, a, b, sfx, era, lab)
        rows[i] = (sec, q, era, v1, s1, v2, s2, note)
    if sec == "regions (v1 bins)" and era in COMMON and "pf_eff" in q and s1 is None and v1 != "":
        key, lab = q.split(".")[0], q.split(".")[1][len("pf_eff_"):]
        kind, a, b, sfx = KMAP[key]
        sel = F.flagged(H1, kind, a, b, F.THR[kind], sfx)
        ie, ip = np.where(sel)
        rect = np.zeros_like(sel); rect[ie.min():ie.max() + 1, ip.min():ip.max() + 1] = True
        m = sel if lab == "flagged" else rect
        rows[i] = (sec, q, era, v1, binom(float(v1), H1[f"bp_den_{era}"][m].sum()), v2, s2, note)

# ------------------------------------------------------------------ gen_comparison_summary.json
G1 = json.load(open(os.path.join(S, "gen_comparison_summary.json")))
G2 = json.load(open(os.path.join(S, "gen_comparison_summary_v2.json")))
for y in ("2022", "2018"):
    a, b = G1["pass_fraction"][y], G2["pass_fraction"][y]
    add("gen_comparison_summary.json", f"pass_fraction.{y}", y, a["frac"], binom(a["frac"], a["den"]),
        b["frac"], binom(b["frac"], b["den"]), "2018: same files in v1 and v2" if y == "2018" else "")
for qn, q1 in G1["quantities"].items():
    q2 = G2["quantities"][qn]
    for lab in ("2022_raw", "2022_filter", "2018_raw", "2018_filter"):
        add("gen_comparison_summary.json", f"{qn}.{lab}.mean", lab[:4], q1[lab]["mean"], q1[lab]["stat"],
            q2[lab]["mean"], q2[lab]["stat"])
        add("gen_comparison_summary.json", f"{qn}.{lab}.median", lab[:4], q1[lab]["median"], None,
            q2[lab]["median"], None)
        add("gen_comparison_summary.json", f"{qn}.{lab}.n", lab[:4], q1[lab]["n"], None, q2[lab]["n"], None)
    for rk, lab in (("reldiff_raw", "2022_raw"), ("reldiff_filter", "2022_filter")):
        m18 = q1["2018" + lab[4:]]["mean"]
        add("gen_comparison_summary.json", f"{qn}.{rk}", "2022", q1[rk], q1[lab]["stat"] / m18,
            q2[rk], q2[lab]["stat"] / m18, "2018 part common")

# ------------------------------------------------------------------ lj_eff_summary.json
L1 = json.load(open(os.path.join(S, "lj_eff_summary.json")))
L2p = os.path.join(S, "lj_eff_summary_v2.json")
L2 = json.load(open(L2p)) if os.path.exists(L2p) else None


def run3_mean_sigma(mode, era, chan, fld, tag):
    path = os.path.join(S, f"lj_eff_{mode}{tag}_{era}.json")
    rs = [r for r in json.load(open(path)) if r["chan"] == chan]
    s = [100 * binom(r["run3"][fld]["p"], r["run3"][fld]["n"]) for r in rs]
    return float(np.sqrt(np.sum(np.square(s))) / len(s)), {r["key"]: r["run3"][fld] for r in rs}


if L2:
    for mode in ("geom", "real"):
        if mode not in L2:
            continue
        for era in COMMON + ["2024"]:
            e1, e2 = L1[mode].get(era, {}), L2[mode].get(era, {})
            for k in sorted(set(e1) | set(e2)):
                a, b = e1.get(k), e2.get(k)
                if k.startswith("acceptance_"):
                    for st in ("mean", "min", "max"):
                        add(f"lj_eff_summary.json ({mode})", f"{k}.{st}", era, a[st] if a else "", None,
                            b[st] if b else "", None, "2018: same files" if "2018" in k else "")
                    continue
                if not isinstance(b, dict) or "mean_pct" not in b:
                    continue
                chan, fld = k.split("_", 1)
                s2, p2 = run3_mean_sigma(mode, era, chan, fld, "_v2")
                s1, p1 = (run3_mean_sigma(mode, era, chan, fld, "") if a else (None, None))
                add(f"lj_eff_summary.json ({mode})", f"{k}.mean_pct (Run3-2018)", era,
                    a["mean_pct"] if a else "", s1, b["mean_pct"], s2)
                for st in ("rms_pct", "min_pct", "max_pct", "n_negative", "chi2_about_mean"):
                    add(f"lj_eff_summary.json ({mode})", f"{k}.{st}", era, a[st] if a else "", None, b[st], None)
                if a:          # per-point chi2 of the v2 - v1 Run 3 efficiencies
                    chi2 = sum((p2[x]["p"] - p1[x]["p"]) ** 2 /
                               (binom(p2[x]["p"], p2[x]["n"]) ** 2 + binom(p1[x]["p"], p1[x]["n"]) ** 2)
                               for x in p1)
                    add(f"lj_eff_summary.json ({mode})", f"{k}: chi2 of v2 - v1 Run 3 eff. over {len(p1)} points",
                        era, "", None, round(chi2, 1), None, f"{len(p1)} dof")


# ------------------------------------------------------------------ render
def fmt(x):
    if x is None or x == "":
        return ""
    if isinstance(x, float):
        return f"{x:.6g}"
    return str(x)


def nsig(v1, s1, v2, s2):
    try:
        a, b = float(v1), float(v2)
    except (TypeError, ValueError):
        return None, None
    if s1 is None or s2 is None:
        return b - a, None
    s = float(np.hypot(s1, s2))
    return b - a, ((b - a) / s if s > 0 else None)


lines = ["# v1 (LLPNanoAOD) against v2 (EXONanoAOD): every number of the v1 outputs", "",
         "Written by `v1_vs_v2.py` in this directory; uncertainty definitions in its docstring. "
         "Era `-` = a property of the region, not of one era. 2024 rows: v2 only. "
         "`regions (v1 bins)`: v1 numbers against v2 read over the same (v1) flagged bins; "
         "`regions (own flagged bins)`: v2 over the bins it flags itself (no sigma: the flagged set selects "
         "downward fluctuations). In `regions (v1 bins)` the bins were chosen on v1's own fluctuations (where the "
         "second era reads below the first), so v1's flagged averages of the second era lean low and v2's, read "
         "over the same bins, do not: a positive v2 - v1 there is partly that selection.", ""]
order = list(dict.fromkeys(r[0] for r in rows))
rows.sort(key=lambda r: order.index(r[0]))          # stable: keeps the order within a section
big = []
sec0 = None
count = {}                     # section -> [rows with an n sigma (Run 3), of them > 2, rows on the 2018 files]
for sec, q, era, v1, s1, v2, s2, note in rows:
    if sec != sec0:
        lines += ["", f"## {sec}", "", "| quantity | era | v1 | σ v1 | v2 | σ v2 | v2 − v1 | n σ | note |",
                  "|---|---|---|---|---|---|---|---|---|"]
        sec0 = sec
    d, ns = nsig(v1, s1, v2, s2)
    same = "" if str(v1) == str(v2) or v1 == "" else ("differs" if d is None else "")
    lines.append(f"| {q} | {era} | {fmt(v1)} | {fmt(s1)} | {fmt(v2)} | {fmt(s2)} | "
                 f"{fmt(round(d, 6)) if d is not None else same} | {f'{ns:+.1f}' if ns is not None else ''} | {note} |")
    if ns is not None:
        c = count.setdefault(sec, [0, 0, 0])
        if era == "2018":      # the same 2018 v10 files in v1 and v2: identical by construction
            c[2] += 1
        else:
            c[0] += 1; c[1] += abs(ns) > 2
    if ns is not None and abs(ns) > 2:
        big.append((sec, q, era, v1, v2, ns))
tot = [sum(c[i] for c in count.values()) for i in range(3)]
rb = count.get("regions (v1 bins)", [0, 0, 0])
top = max(big, key=lambda x: abs(x[5])) if big else None
summary = (f"**Summary.** {tot[0]} Run 3 quantities were compared with an n σ value (v1 and v2 both with a "
           f"statistical uncertainty; the {tot[2]} rows on the 2018 reference read the same files in both "
           f"productions and are not counted). {tot[1]} lie beyond 2σ: " +
           "; ".join(f"{q} ({e}{', ' + sec.split('(')[-1].rstrip(')') + ' mode' if 'lj_eff' in sec else ''}) "
                     f"{n:+.1f}σ" for sec, q, e, _, _, n in big) +
           (f". The largest is {abs(top[5]):.1f}σ ({top[1]}, {top[2]})" if top else "") +
           f". About {0.0455 * tot[0]:.0f} would be expected beyond 2σ for Gaussian errors. Of the region "
           f"averages compared over the first production's bins, {rb[1]} of {rb[0]} lie beyond 2σ.")
lines[2:2] = [summary, ""]
lines += ["", "## Differences beyond 2 standard deviations", "",
          "Rows with an n σ value (v1 and v2 both present, both with an uncertainty), Run 3 eras; the 2018 "
          "rows read the same files in both productions and are counted apart: " +
          "; ".join(f"{s}: {c[1]} of {c[0]} beyond 2σ (+{c[2]} 2018 rows)" for s, c in count.items()) +
          f". Total: {tot[1]} of {tot[0]} beyond 2σ (expected about {0.0455 * tot[0]:.0f} for Gaussian errors; "
          f"many rows are correlated, e.g. flagged and rectangle averages of one region), +{tot[2]} 2018 rows.", "",
          "| section | quantity | era | v1 | v2 | n σ |", "|---|---|---|---|---|---|"]
lines += [f"| {s} | {q} | {e} | {fmt(a)} | {fmt(b)} | {n:+.1f} |" for s, q, e, a, b, n in big]
open(OUT, "w").write("\n".join(lines) + "\n")
print(f"wrote {OUT}: {len(rows)} rows, {len(big)} beyond 2 sigma of {tot[0]} Run 3 rows with n sigma, "
      f"largest {abs(top[5]):.1f}" if top else f"wrote {OUT}: {len(rows)} rows, none beyond 2 sigma")
for s_, c in count.items():
    print(f"  {s_}: {c[1]}/{c[0]} (+{c[2]} 2018)")
for s, q, e, a, b, n in big:
    print(f"  {s:32s} {q:60s} {e:8s} {fmt(a):>12s} {fmt(b):>12s} {n:+.1f}")
