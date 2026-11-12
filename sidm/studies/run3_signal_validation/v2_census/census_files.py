#!/usr/bin/env python3
"""One census row per file: copy the file to local scratch (xrdcp), checksum it, open it with uproot,
decompress every basket of every Events branch, and read the Runs tree, the branch list and the provenance.

Usage: census_files.py --filelist job_NNN.txt --out census_NNN.csv [--branches branches_NNN.json]
                       [--fingerprint ../era_fingerprint.py] [--tmp DIR]

The provenance read-back (_raw, _blobs, _get, provenance, ACCEPTED_GT) is taken from era_fingerprint.py
by extracting those definitions with ast, so the census runs without that module's plotting imports.
A file that fails any step gets a row with the step and the error in 'error'; the job never stops early.
"""
import argparse, ast, csv, hashlib, json, os, re, subprocess, sys, time, zlib
import uproot

_DIR = os.path.dirname(os.path.abspath(__file__))
COLS = ["era", "sample", "chunk", "url", "bytes", "adler32", "adler32_eos", "open_ok", "full_read_ok", "error",
        "events", "events_baskets", "events_uncompressed_bytes", "lumis", "runs_entries", "run",
        "genEventCount", "genEventSumw", "genEventSumw2", "runs_branches", "n_branches", "branch_hash", "n_hlt_branches",
        "steps", "SIM_release", "SIM_gt", "HLT_release", "HLT_gt", "RECO_release", "RECO_gt", "NANO_release", "NANO_gt",
        "hlt_menu", "com_energy", "vtx_type", "vtx_readDB", "vtx_X0", "vtx_Y0", "vtx_Z0", "vtx_SigmaZ", "vtx_BetaStar",
        "seconds"]


def load_readback(path):
    tree = ast.parse(open(path).read(), path)
    want = {"_raw", "_blobs", "_get", "provenance"}
    keep = [n for n in tree.body if (isinstance(n, ast.FunctionDef) and n.name in want) or
            (isinstance(n, ast.Assign) and any(getattr(t, "id", "") == "ACCEPTED_GT" for t in n.targets))]
    if len(keep) != len(want) + 1:
        raise RuntimeError(f"{path}: found {len(keep)} of the {len(want) + 1} read-back definitions")
    ns = dict(re=re, hashlib=hashlib, uproot=uproot)
    exec(compile(ast.Module(keep, []), path, "exec"), ns)
    return ns


def parse_url(url):
    """.../outputs_v2/<era>/<sample>/<sample>_<era>_<chunk>.root"""
    era, sample, fname = url.split("/")[-3:]
    chunk = fname[:-len(".root")].rsplit("_", 1)[1]
    return era, sample, chunk


def eos_adler32(url):
    if not url.startswith("root://"):          # a local path (tests): no server checksum
        return ""
    host, path = re.match(r"(root://[^/]+)/(/.*)", url).groups()
    r = subprocess.run(["xrdfs", host, "query", "checksum", path], capture_output=True, text=True, timeout=120)
    m = re.match(r"adler32\s+([0-9a-f]+)", r.stdout.strip())
    return m.group(1).zfill(8) if m else f"ERR:{(r.stdout + r.stderr).strip()[:80]}"


def vtx_from_sim(rb, path):
    """Vertex-smearing module of the SIM process: type, readDB and its beam-spot parameters."""
    f = uproot.open(path)
    blobs = rb["_blobs"](f["ParameterSets"]["IdToParameterSetsBlobs"])
    for b in blobs.values():
        if rb["_get"](b, "@process_name") == "SIM" and rb["_get"](b, "VtxSmeared"):
            v = blobs[rb["_get"](b, "VtxSmeared")]
            out = dict(vtx_type=rb["_get"](v, "@module_type"), vtx_readDB=rb["_get"](v, "readDB"))
            for k in ("X0", "Y0", "Z0", "SigmaZ", "BetaStar"):
                x = rb["_get"](v, k)
                out[f"vtx_{k}"] = float(x) if x is not None else ""
            return out
    return {}


def census(url, rb, tmp, branches):
    t0 = time.time()
    era, sample, chunk = parse_url(url)
    row = dict(era=era, sample=sample, chunk=chunk, url=url, open_ok=False, full_read_ok=False, error="")
    local = os.path.join(tmp, os.path.basename(url))
    step = "xrdcp"
    try:
        r = subprocess.run(["xrdcp", "-f", "-s", url, local], capture_output=True, text=True, timeout=900)
        if r.returncode:
            raise RuntimeError(f"exit {r.returncode}: {r.stderr.strip()[-200:]}")
        step = "checksum"
        row["bytes"] = os.path.getsize(local)
        a = 1
        with open(local, "rb") as fi:
            for blk in iter(lambda: fi.read(1 << 22), b""):
                a = zlib.adler32(blk, a)
        row["adler32"] = f"{a:08x}"
        row["adler32_eos"] = eos_adler32(url)
        step = "open"
        f = uproot.open(local)
        ev = f["Events"]
        row["open_ok"] = True
        row["events"] = ev.num_entries
        names = sorted(ev.keys())
        row["n_branches"] = len(names)
        h = hashlib.sha1("\n".join(names).encode()).hexdigest()[:12]
        row["branch_hash"] = h
        branches.setdefault(h, names)
        row["n_hlt_branches"] = sum(n.startswith("HLT_") for n in names)
        step = "runs"
        runs = f["Runs"]
        row["runs_entries"] = runs.num_entries
        row["runs_branches"] = ";".join(runs.keys())
        ra = runs.arrays(["run", "genEventCount", "genEventSumw", "genEventSumw2"], library="np")
        row["run"] = ";".join(str(x) for x in ra["run"])
        row["genEventCount"] = int(ra["genEventCount"].sum())
        row["genEventSumw"] = repr(float(ra["genEventSumw"].sum()))
        row["genEventSumw2"] = repr(float(ra["genEventSumw2"].sum()))
        row["lumis"] = f["LuminosityBlocks"].num_entries
        step = "provenance"
        p = rb["provenance"](local, era)
        row["steps"] = " | ".join(f"{n}:{r}:{g}" for n, r, g in p["steps"])
        for n, r, g in p["steps"]:
            if f"{n}_release" in COLS:
                row[f"{n}_release"], row[f"{n}_gt"] = r, g
        menus = {i["hlt_menu"] for i in p["per_process"].values() if "hlt_menu" in i}
        coms = {i["com"] for i in p["per_process"].values() if "com" in i}
        row["hlt_menu"] = ";".join(sorted(menus))
        row["com_energy"] = ";".join(str(c) for c in sorted(coms))
        row.update(vtx_from_sim(rb, local))
        step = "full read"
        nb = ub = 0
        for b in ev.branches:
            if b.num_entries != ev.num_entries or b.entry_offsets[-1] != ev.num_entries:
                raise RuntimeError(f"branch {b.name}: {b.num_entries} entries, baskets end at {b.entry_offsets[-1]}")
            for i in range(b.num_baskets):
                ub += b.basket(i).data.nbytes; nb += 1
        row.update(events_baskets=nb, events_uncompressed_bytes=ub, full_read_ok=True)
    except Exception as e:
        row["error"] = f"{step}: {type(e).__name__}: {str(e)[:300]}".replace("\n", " ")
    finally:
        if os.path.exists(local):
            os.remove(local)
    row["seconds"] = round(time.time() - t0, 2)
    return row


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--filelist", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--branches", help="JSON {branch_hash: sorted names}, one entry per distinct list seen")
    ap.add_argument("--fingerprint", default=os.path.join(_DIR, "..", "era_fingerprint.py"))
    ap.add_argument("--tmp", default=os.environ.get("_CONDOR_SCRATCH_DIR", "/tmp"))
    a = ap.parse_args()
    rb = load_readback(a.fingerprint)
    urls = [u.strip() for u in open(a.filelist) if u.strip()]
    branches = {}
    with open(a.out, "w", newline="") as fo:
        w = csv.DictWriter(fo, fieldnames=COLS, restval="")
        w.writeheader()
        for i, u in enumerate(urls):
            row = census(u, rb, a.tmp, branches)
            w.writerow(row); fo.flush()
            print(f"[{i + 1}/{len(urls)}] {row['seconds']:6.1f}s {os.path.basename(u)} events={row.get('events', '')} "
                  f"{'OK' if not row['error'] else row['error']}", flush=True)
    if a.branches:
        json.dump(branches, open(a.branches, "w"))


if __name__ == "__main__":
    main()
