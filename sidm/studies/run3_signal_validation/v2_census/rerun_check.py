#!/usr/bin/env python3
"""Re-run the census on n random files of v2_census.csv and compare every census column but 'seconds'.
Writes <work>/rerun_check.csv (the new rows) and prints each difference; exit status 1 if any."""
import argparse, csv, os, random, sys

_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _DIR)
import census_files as C


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--work", required=True)
    ap.add_argument("-n", type=int, default=50)
    ap.add_argument("--seed", type=int, default=20261107)
    ap.add_argument("--tmp", default="/tmp")
    a = ap.parse_args()
    old = list(csv.DictReader(open(os.path.join(a.work, "v2_census.csv"))))
    pick = random.Random(a.seed).sample(old, a.n)
    rb = C.load_readback(os.path.join(_DIR, "..", "era_fingerprint.py"))
    new, ndiff = [], 0
    for o in pick:
        r = C.census(o["url"], rb, a.tmp, {})
        new.append(r)
        d = [(k, o[k], str(r.get(k, ""))) for k in C.COLS if k != "seconds" and o[k] != str(r.get(k, ""))]
        ndiff += bool(d)
        print(f"{os.path.basename(o['url'])}: {'same' if not d else d}", flush=True)
    with open(os.path.join(a.work, "rerun_check.csv"), "w", newline="") as fo:
        w = csv.DictWriter(fo, fieldnames=C.COLS, restval=""); w.writeheader(); w.writerows(new)
    print(f"{a.n} files re-run (seed {a.seed}): {a.n - ndiff} identical, {ndiff} differ")
    sys.exit(1 if ndiff else 0)


if __name__ == "__main__":
    main()
