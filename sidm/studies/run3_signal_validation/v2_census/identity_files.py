#!/usr/bin/env python3
"""Sample identity, one row per file, from the generator branches (file copied to scratch with xrdcp, as in the
census, since the worker's uproot has no xrootd backend): the bound-state (pdgId 35)
and dark-photon (pdgId 32) masses, the number of dark photons per event and the flavours of their daughters,
the sum, sum of squares and count of the dark photons' proper decay lengths, and run / lumi / event numbers.

Usage: identity_files.py --filelist job_NNN.txt --out identity_NNN.csv [--ctau-module ../ctau_all_samples.py] [--tmp DIR]

The proper decay length is the one of ctau_all_samples.py (its proper_ctau and BR, extracted with ast): per dark
photon, |decay vertex - production vertex| / (p/m) in mm, the decay vertex being the production vertex of its
lepton daughter with pdgId > 0. Masses are stored with 8 mantissa bits (study README): the file's median
bound-state mass and median dark-photon mass are recorded, and report.py compares them with the name's values
within one storage step of the nominal value.
"""
import argparse, ast, csv, os, subprocess, time
import numpy as np
import awkward as ak
import uproot

_DIR = os.path.dirname(os.path.abspath(__file__))
COLS = ["url", "id_error", "id_events", "id_m35_median", "id_m35_min", "id_m35_max", "id_n35_first",
        "id_m32_median", "id_m32_min", "id_m32_max", "id_events_2zd", "id_n_zd", "id_events_4mu", "id_events_2mu2e",
        "id_events_other_flavour", "id_zd_daughters_nonlepton", "id_ctau_n", "id_ctau_sum_mm", "id_ctau_sum2_mm2",
        "id_runs", "id_lumis", "id_event_min", "id_event_max", "id_event_unique", "id_seconds"]
EXTRA = ["luminosityBlock", "run", "event"]


def load_ctau(path):
    tree = ast.parse(open(path).read(), path)
    keep = [n for n in tree.body if (isinstance(n, ast.FunctionDef) and n.name == "proper_ctau") or
            (isinstance(n, ast.Assign) and any(getattr(t, "id", "") == "BR" for t in n.targets))]
    if len(keep) != 2:
        raise RuntimeError(f"{path}: proper_ctau and BR not both found")
    ns = dict(np=np, ak=ak, uproot=uproot)
    exec(compile(ast.Module(keep, []), path, "exec"), ns)
    return ns


def identity(url, ns, tmp):
    t0 = time.time()
    row = dict(url=url, id_error="")
    local = os.path.join(tmp, os.path.basename(url))
    try:
        r = subprocess.run(["xrdcp", "-f", "-s", url, local], capture_output=True, text=True, timeout=900)
        if r.returncode:
            raise RuntimeError(f"xrdcp exit {r.returncode}: {r.stderr.strip()[-200:]}")
        f = uproot.open(local)
        a = f["Events"].arrays(ns["BR"] + ["GenPart_status"] + EXTRA)
        pid, mom = a.GenPart_pdgId, a.GenPart_genPartIdxMother
        m = ak.where(mom >= 0, mom, 0)
        mpid = ak.where(mom >= 0, pid[m], 0)
        row["id_events"] = len(a)
        is35 = abs(pid) == 35
        first35 = is35 & (abs(mpid) != 35)                         # bound state as produced (not a copy)
        row["id_n35_first"] = int(ak.sum(first35))
        m35 = ak.to_numpy(ak.flatten(a.GenPart_mass[first35]))
        zd = (abs(pid) == 32) & (abs(mpid) == 35)                  # dark photons from the bound state's decay
        m32 = ak.to_numpy(ak.flatten(a.GenPart_mass[zd]))
        for k, x in (("m35", m35), ("m32", m32)):
            row[f"id_{k}_median"], row[f"id_{k}_min"], row[f"id_{k}_max"] = (
                (float(np.median(x)), float(x.min()), float(x.max())) if len(x) else ("", "", ""))
        nzd = ak.sum(zd, axis=1)
        row["id_n_zd"] = int(ak.sum(nzd))
        row["id_events_2zd"] = int(ak.sum(nzd == 2))
        dau = abs(mpid) == 32                                      # daughters of a dark photon
        nmu = ak.sum(dau & (abs(pid) == 13), axis=1)
        nel = ak.sum(dau & (abs(pid) == 11), axis=1)
        nall = ak.sum(dau & (abs(pid) != 32), axis=1)               # a 32 -> 32 copy is not a decay
        row["id_events_4mu"] = int(ak.sum((nmu == 4) & (nel == 0) & (nall == 4)))
        row["id_events_2mu2e"] = int(ak.sum((nmu == 2) & (nel == 2) & (nall == 4)))
        row["id_events_other_flavour"] = row["id_events"] - row["id_events_4mu"] - row["id_events_2mu2e"]
        row["id_zd_daughters_nonlepton"] = int(ak.sum(nall - nmu - nel))
        x = np.asarray(ns["proper_ctau"](local), dtype=np.float64)
        row.update(id_ctau_n=len(x), id_ctau_sum_mm=repr(float(x.sum())), id_ctau_sum2_mm2=repr(float((x ** 2).sum())))
        row["id_runs"] = ";".join(str(v) for v in np.unique(ak.to_numpy(a.run)))
        row["id_lumis"] = ";".join(str(v) for v in np.unique(ak.to_numpy(a.luminosityBlock)))
        ev = ak.to_numpy(a.event)
        row.update(id_event_min=int(ev.min()), id_event_max=int(ev.max()), id_event_unique=len(np.unique(ev)) == len(ev))
    except Exception as e:
        row["id_error"] = f"{type(e).__name__}: {str(e)[:300]}".replace("\n", " ")
    finally:
        if os.path.exists(local):
            os.remove(local)
    row["id_seconds"] = round(time.time() - t0, 2)
    return row


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--filelist", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--ctau-module", default=os.path.join(_DIR, "..", "ctau_all_samples.py"))
    ap.add_argument("--tmp", default=os.environ.get("_CONDOR_SCRATCH_DIR", "/tmp"))
    a = ap.parse_args()
    ns = load_ctau(a.ctau_module)
    urls = [u.strip() for u in open(a.filelist) if u.strip()]
    with open(a.out, "w", newline="") as fo:
        w = csv.DictWriter(fo, fieldnames=COLS, restval="")
        w.writeheader()
        for i, u in enumerate(urls):
            row = identity(u, ns, a.tmp)
            w.writerow(row); fo.flush()
            print(f"[{i + 1}/{len(urls)}] {row['id_seconds']:6.1f}s {os.path.basename(u)} "
                  f"{row['id_error'] or 'OK'}", flush=True)


if __name__ == "__main__":
    main()
