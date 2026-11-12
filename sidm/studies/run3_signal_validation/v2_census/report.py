#!/usr/bin/env python3
"""Census report in three stages.

join   (needs the census outputs and the production records): merges the census CSVs with the expected list, the
       EOS listing, the upload records of both submissions and the identity pass, and writes
       <results>/v2_census.csv.gz, branches.json (each distinct branch list), eos_extra.txt (files on EOS not in the
       DAGs) and expected_config.json. Also writes <work>/v2_census.csv (uncompressed copy).
       Inputs: <work>/expected_files.csv, expected_config.json (expected.py), condor/out/census_*.csv and
       branches_*.json (census_files.py), condor_id/out/identity_*.csv (identity_files.py), eos_ls_<era>.txt
       (xrdfs ls -l -R at census time); --condor-v2, the production's condor_v2/ directory at LPC (the LPC
       submission's .cksum records in its ten <era>_{column,grid} folders); --lxplus, the upload records of the CERN
       lxplus submission (one line per .cksum file: '<path> <name>.root adler32=<hex> size=<n>' or
       '<path> <name>.root skip size=<n>').
relist (after the census): compares a fresh xrdfs ls -l -R of every era (<work>/eos_relist_<era>.txt) with the
       census-time listing and queries the EOS server checksum again for the files whose upload record is at both
       sites or at neither; writes <results>/eos_relist.csv.gz (one row per file: size and modification time now,
       adler32 now for the queried files).
render (needs only <results>/, the committed v10/run3 location YAMLs and ../ctau_all_samples.csv): applies the
       checks and writes v2_validation.md, ctau_v2.csv and sidm/configs/ntuples/signal_run3_v2_<era>.yaml.

A file passes when: it is in the EOS listing with its measured size; it opens and every Events basket decompresses
(full_read_ok); Events entries = NEV = genEventCount; the adler32 of the copied file equals the EOS server checksum;
its size and adler32 equal an upload record of either submission, when one exists; its provenance (release and GT per
process, HLT menu, comEnergy, vertex smearing) equals the chain's configuration; and its generator record matches
its name (identity: bound-state and dark-photon masses, two dark photons per event, the channel's daughter
flavours). The branch list is compared between files of an era (one hash expected): a second hash is listed as a
deviation, not a failure.
"""
import argparse, collections, csv, datetime, glob, gzip, io, json, math, os, re, shutil, subprocess, sys

_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(_DIR, "..", "..", "..", ".."))
NTUPLES = os.path.join(ROOT, "sidm", "configs", "ntuples")
ERAS = ["2022", "2022EE", "2023", "2023BPix", "2024"]
EOSDIR = "/store/group/lpcmetx/SIDM/run3_samplegen/outputs_v2"
PROCS = ["SIM", "HLT", "RECO", "NANO"]
SITES = (("lpc", "LPC"), ("lx", "lxplus"))
XRD_HOST = "root://cmseos.fnal.gov/"
V1_CTAU = os.path.join(_DIR, "..", "ctau_all_samples.csv")     # the v1 production's proper decay lengths


def load_csv(path):
    return list(csv.DictReader(open(path)))


# ---------------------------------------------------------------- join
def eos_listing(path, era):
    """{path relative to outputs_v2/<era>/: (size, modification time)} of the .root files in xrdfs ls -l -R
    (the listing prints the modification time in UTC)."""
    out = {}
    for line in open(path):
        p = line.split()
        if line.startswith("-") and p[-1].endswith(".root"):
            out[p[-1].split(f"{EOSDIR}/{era}/", 1)[1]] = (int(p[3]), f"{p[4]} {p[5]}")
    return out


def parse_records(lines):
    """(sub-DAG, record text) pairs -> {file name: [(kind, sub-DAG, adler32 or '', size)]}, kind 'upload' or 'skip'."""
    out = collections.defaultdict(list)
    for subdag, text in lines:
        m = re.fullmatch(r"(\S+\.root) (?:adler32=([0-9a-f]+)|(skip)) size=(\d+)", text.strip())
        if not m:
            raise RuntimeError(f"{subdag}: unrecognised record {text.strip()!r}")
        name, adler, skip, size = m.groups()
        out[name].append(("skip" if skip else "upload", subdag, adler.zfill(8) if adler else "", int(size)))
    return out


def lpc_records(condor_v2):
    """The LPC submission's records: every .cksum file of the ten <era>_{column,grid} DAG folders."""
    def lines():
        for subdag in (f"{e}_{k}" for e in ERAS for k in ("column", "grid")):
            for f in sorted(glob.glob(os.path.join(condor_v2, subdag, "cksum", "*.cksum"))):
                yield subdag, open(f).read()
    return parse_records(lines())


def lxplus_records(path):
    def lines():
        for line in open(path):
            p, text = line.split(" ", 1)
            yield p.split("/")[1], text
    return parse_records(lines())


def record_columns(recs, site):
    """Record summary ('column:upload;grid:skip') and the distinct upload records (size, adler32) of one site."""
    ups = sorted({(s, a) for k, _, a, s in recs if k == "upload"})
    return {f"{site}_records": ";".join(f"{d.split('_', 1)[1]}:{k}" for k, d, _, _ in sorted(recs, key=lambda r: r[1])),
            f"{site}_upload_bytes": ";".join(str(s) for s, _ in ups),
            f"{site}_upload_adler32": ";".join(a for _, a in ups)}


def join(a):
    W, R = a.work, a.results
    os.makedirs(R, exist_ok=True)
    expected = {r["url"]: r for r in load_csv(os.path.join(W, "expected_files.csv"))}
    rows, seen = [], collections.Counter()
    for f in sorted(glob.glob(os.path.join(W, "condor", "out", "census_*.csv"))):
        for r in load_csv(f):
            rows.append(r); seen[r["url"]] += 1
    dup = [u for u, n in seen.items() if n > 1]
    unexpected = sorted(set(seen) - set(expected))
    missing = sorted(set(expected) - set(seen))
    if dup or unexpected:
        sys.exit(f"census rows: {len(dup)} duplicated, {len(unexpected)} not expected")
    for u in missing:                  # an expected file without a census row: a row that says so
        x = expected[u]
        rows.append(dict(era=x["era"], sample=x["sample"], chunk=x["chunk"], url=u, open_ok=False, full_read_ok=False,
                         error="no census row"))
    branches = {}
    for f in sorted(glob.glob(os.path.join(W, "condor", "out", "branches_*.json"))):
        branches.update(json.load(open(f)))
    recs = dict(lpc=lpc_records(a.condor_v2), lx=lxplus_records(a.lxplus))
    for site, label in SITES:
        stray = set(recs[site]) - {os.path.basename(u) for u in expected}
        if stray:
            sys.exit(f"{label} records for {len(stray)} files not in the DAGs, e.g. {sorted(stray)[:3]}")
    listing, extra = {}, []
    for e in ERAS:
        ls = eos_listing(os.path.join(W, f"eos_ls_{e}.txt"), e)
        listing.update({f"{e}/{k}": v for k, v in ls.items()})
        exp_rel = {f"{x['sample']}/{x['file']}" for x in expected.values() if x["era"] == e}
        extra += [f"{e} {p}" for p in sorted(set(ls) - exp_rel)]
    census_cols = list(load_csv(sorted(glob.glob(os.path.join(W, "condor", "out", "census_*.csv")))[0])[0])
    ident = {}
    for f in sorted(glob.glob(os.path.join(W, "condor_id", "out", "identity_*.csv"))):
        for r in load_csv(f):
            if r["url"] in ident:
                sys.exit(f"identity: duplicate row for {r['url']}")
            ident[r["url"]] = r
    id_cols = [c for c in load_csv(sorted(glob.glob(os.path.join(W, "condor_id", "out", "identity_*.csv")))[0])[0]
               if c != "url"] if ident else []
    cols = census_cols + ["nev", "eos_ls_bytes", "eos_ls_mtime"] + [f"{s}_{k}" for s, _ in SITES
                                                    for k in ("records", "upload_bytes", "upload_adler32")] + id_cols
    out = []
    for r in rows:
        x = expected[r["url"]]
        name = os.path.basename(r["url"])
        size, mtime = listing.get(f"{x['era']}/{x['sample']}/{name}", ("", ""))
        row = dict(r, nev=x["nev"], eos_ls_bytes=size, eos_ls_mtime=mtime)
        row.update({k: v for k, v in ident.get(r["url"], {"id_error": "no identity row"}).items() if k != "url"})
        for site, _ in SITES:
            row.update(record_columns(recs[site].get(name, []), site))
        out.append(row)
    out.sort(key=lambda r: (ERAS.index(r["era"]), r["sample"], r["chunk"]))
    buf = io.StringIO()
    w = csv.DictWriter(buf, fieldnames=cols, restval="", lineterminator="\n"); w.writeheader(); w.writerows(out)
    open(os.path.join(W, "v2_census.csv"), "w").write(buf.getvalue())
    with gzip.GzipFile(os.path.join(R, "v2_census.csv.gz"), "wb", compresslevel=9, mtime=0) as fo:
        fo.write(buf.getvalue().encode())
    used = {r["branch_hash"] for r in out if r.get("branch_hash")}
    json.dump({h: branches[h] for h in sorted(used)}, open(os.path.join(R, "branches.json"), "w"), indent=0)
    open(os.path.join(R, "eos_extra.txt"), "w").write("".join(x + "\n" for x in extra))
    shutil.copy(os.path.join(W, "expected_config.json"), os.path.join(R, "expected_config.json"))
    print(f"joined {len(out)} rows ({len(missing)} without census row, {len(set(expected) - set(ident))} without "
          f"identity row), {len(extra)} extra files on EOS, "
          f"records: LPC {sum(map(len, recs['lpc'].values()))}, lxplus {sum(map(len, recs['lx'].values()))} -> {R}")


# ---------------------------------------------------------------- render
def sample_keys():
    """v2 sample directory -> analysis key, from the 2018 v10 YAMLs (their path names the same mass point),
    cross-checked against the v1 Run 3 YAMLs (same directory names as v2)."""
    import yaml
    keys = {}
    for ch in ("4mu", "2mu2e"):
        y = yaml.safe_load(open(os.path.join(NTUPLES, f"signal_{ch}_v10.yaml")))["llpNanoAOD_v2"]["samples"]
        for k, s in y.items():
            d = s["path"].split("/")[0].replace("CutDecayFalse_", "").replace("2Mu2e", "2Mu2E")
            keys[re.sub(r"_v3$", "", d)] = k
        r3 = yaml.safe_load(open(os.path.join(NTUPLES, f"signal_{ch}_run3.yaml")))
        for blk in r3.values():
            for k, s in blk["samples"].items():
                d = s["path"].rstrip("/")
                if keys.get(d) != k:
                    raise RuntimeError(f"{d}: v10 key {keys.get(d)}, run3 key {k}")
    return keys


def expected_prov(cfg):
    r = cfg["resolved"]
    p = {f"{n}_release": cfg["REL"] for n in PROCS[:3]}
    p.update({f"{n}_gt": cfg["COND"] for n in PROCS[:3]})
    p.update(NANO_release=cfg["NANOREL"], NANO_gt=cfg["NANOCOND"], hlt_menu=r["hlt_menu"], com_energy=str(r["com"]),
             vtx_type=r["vtx_type"], vtx_readDB=str(r["vtx_readDB"]).lower())
    for k in ("X0", "Y0", "Z0", "SigmaZ", "BetaStar"):
        if f"vtx_{k}" in r:
            p[f"vtx_{k}"] = str(r[f"vtx_{k}"])
    return p


def _num(k):
    return k.startswith("vtx_") and k not in ("vtx_type", "vtx_readDB")


def uploads(row, site):
    b, a = row[f"{site}_upload_bytes"], row[f"{site}_upload_adler32"]
    return list(zip(map(int, b.split(";")), a.split(";"))) if b else []


def upload_match(row):
    """{site: True/False} for each site with an upload record: does the measured file equal one of its uploads."""
    meas = (int(row["bytes"]), row["adler32"]) if row["bytes"] else None
    return {s: meas in uploads(row, s) for s, _ in SITES if uploads(row, s)}


def upload_class(row):
    has = [s for s, _ in SITES if uploads(row, s)]
    return {(): "neither", ("lpc",): "LPC only", ("lx",): "lxplus only", ("lpc", "lx"): "both"}[tuple(has)]


def nominal(row):
    """Channel, bound-state mass, dark-photon mass (GeV) and proper decay length (mm) from the sample name."""
    m = re.fullmatch(r"SIDM_BsTo2DpTo(4Mu|2Mu2E)_MBs-(\w+)_MDp-(\w+)_ctau-(\w+)", row["sample"])
    chan, mbs, mdp, ctau = m.groups()
    return chan, float(mbs.replace("p", ".")), float(mdp.replace("p", ".")), float(ctau.replace("p", "."))


def storage_step(x):
    """Spacing of GenPart_mass values near x: 8 mantissa bits (study README, 'Stored precision')."""
    return 2.0 ** (math.floor(math.log2(x)) - 8)


ZD_REL_TOL = 0.03     # dark photon: line shape several storage steps wide; per-file median within 3% (or 1 step)
CTAU_PULL_MAX = 5.0   # per-file |mean proper decay length - name's value| / (name's value / sqrt(n))


def file_ctau_pull(row):
    """Per-file pull of the mean proper decay length against the name's value, uncertainty = value / sqrt(n)
    (the exponential's standard deviation equals its mean)."""
    n = int(row["id_ctau_n"]); nom = nominal(row)[3]
    return (float(row["id_ctau_sum_mm"]) / n - nom) / (nom / math.sqrt(n))


def identity_checks(row):
    """Reasons a file's generator record does not match its name; empty when it does."""
    if row["id_error"]:
        return [f"identity: {row['id_error']}"]
    bad = []
    chan, mbs, mdp, _ = nominal(row)
    n = int(row["events"]) if row["events"] else -1
    if int(row["id_events"]) != n:
        bad.append(f"identity read {row['id_events']} events, census {n}")
    if int(row["id_n35_first"]) != n:
        bad.append(f"{row['id_n35_first']} bound states in {n} events")
    if abs(float(row["id_m35_median"]) - mbs) > storage_step(mbs):
        bad.append(f"bound-state mass median {row['id_m35_median']} != {mbs:g} (step {storage_step(mbs):g})")
    if abs(float(row["id_m32_median"]) - mdp) > max(storage_step(mdp), ZD_REL_TOL * mdp):
        bad.append(f"dark-photon mass median {row['id_m32_median']} != {mdp:g} (tolerance "
                   f"{max(storage_step(mdp), ZD_REL_TOL * mdp):g})")
    if int(row["id_events_2zd"]) != n:
        bad.append(f"{row['id_events_2zd']} of {n} events with two dark photons")
    good = int(row["id_events_4mu"] if chan == "4Mu" else row["id_events_2mu2e"])
    if good != n:
        bad.append(f"{good} of {n} events with the {chan} daughters")
    if int(row["id_ctau_n"]) != 2 * n:
        bad.append(f"{row['id_ctau_n']} proper decay lengths for {n} events (2 per event expected)")
    elif abs(file_ctau_pull(row)) >= CTAU_PULL_MAX:
        bad.append(f"mean proper decay length pull {file_ctau_pull(row):+.2f} (|pull| >= {CTAU_PULL_MAX:g})")
    return bad


def checks(row, eprov):
    """Reasons a file fails; empty when it passes."""
    if row["error"]:
        return [row["error"]]
    bad = []
    if row["open_ok"] != "True" or row["full_read_ok"] != "True":
        bad.append(f"open_ok {row['open_ok']}, full_read_ok {row['full_read_ok']}")
    nev = int(row["nev"])
    if not row["eos_ls_bytes"]:
        bad.append("not in the EOS listing")
    elif int(row["eos_ls_bytes"]) != int(row["bytes"]):
        bad.append(f"size {row['bytes']} != EOS listing {row['eos_ls_bytes']}")
    if int(row["events"]) != nev:
        bad.append(f"Events {row['events']} != NEV {nev}")
    if int(row["genEventCount"]) != nev:
        bad.append(f"genEventCount {row['genEventCount']} != NEV {nev}")
    if row["adler32"] != row["adler32_eos"]:
        bad.append(f"adler32 {row['adler32']} != EOS server {row['adler32_eos']}")
    m = upload_match(row)
    if m and not any(m.values()):
        bad.append(f"size/adler32 {row['bytes']}/{row['adler32']} match no upload record (" + "; ".join(
            f"{lab} {row[f'{s}_upload_bytes']}/{row[f'{s}_upload_adler32']}" for s, lab in SITES if s in m) + ")")
    for k, v in eprov.items():
        got = row.get(k, "")
        if not ((float(got) == float(v)) if _num(k) and got else got == v):
            bad.append(f"{k} {got!r} != chain {v!r}")
    return bad + identity_checks(row)


def write_yaml(path, era, rows_by_sample, keys, failed):
    order = {}
    for ch in ("4mu", "2mu2e"):
        for line in open(os.path.join(NTUPLES, f"signal_{ch}_v10.yaml")):
            m = re.match(r"^    (\S+):\s*$", line)
            if m:
                order.setdefault(m.group(1), len(order))
    L = [f"# Run 3 {era} SIDM signal, v2 private production (EXONanoAOD: PAT,NANO:@EXO in CMSSW_16_0_10), both channels.",
         f"# Written by sidm/studies/run3_signal_validation/v2_census/report.py from the v2 census: only files that passed",
         f"# every check are listed; a failing file is commented out with the reason (see that study's README).",
         f"# Consume via make_fileset(samples, 'exoNanoAOD_v2', location_cfg='{os.path.basename(path)}', replace_xcache=True).",
         "exoNanoAOD_v2:",
         f"  path: root://xcache/{EOSDIR}/{era}/",
         "  samples:"]
    for d in sorted(rows_by_sample, key=lambda d: order[keys[d]]):
        L += [f"    {keys[d]}:", f"      path: {d}/", f"      year: '{era}'", "      files:"]
        for r in sorted(rows_by_sample[d], key=lambda r: r["chunk"]):
            fn = os.path.basename(r["url"])
            why = failed.get(r["url"])
            L.append(f"      # - {fn} # {'; '.join(why)}" if why else f"      - {fn}")
    open(path, "w").write("\n".join(L) + "\n")


def md_table(head, rows):
    out = ["| " + " | ".join(head) + " |", "|" + "---|" * len(head)]
    out += ["| " + " | ".join(str(c) for c in r) + " |" for r in rows]
    return out


def relist(a):
    """Fresh listing of every era and a second EOS server checksum for the 'both sites' and 'neither' files."""
    W, R = a.work, a.results
    with gzip.open(os.path.join(R, "v2_census.csv.gz"), "rt") as fi:
        rows = list(csv.DictReader(fi))
    query = {r["url"] for r in rows if upload_class(r) in ("both", "neither")}
    started = datetime.datetime.now(datetime.timezone.utc).strftime("%Y-%m-%d %H:%M:%S")
    out = []
    for e in ERAS:
        path = os.path.join(W, f"eos_relist_{e}.txt")
        with open(path, "w") as fo:
            subprocess.run(["xrdfs", XRD_HOST, "ls", "-l", "-R", f"{EOSDIR}/{e}"], stdout=fo, check=True, timeout=1800)
        for rel, (size, mtime) in sorted(eos_listing(path, e).items()):
            sample, fname = rel.split("/")
            url = f"{XRD_HOST}/{EOSDIR}/{e}/{rel}"
            adler = ""
            if url in query:
                q = subprocess.run(["xrdfs", XRD_HOST, "query", "checksum", f"{EOSDIR}/{e}/{rel}"], capture_output=True,
                                   text=True, timeout=120)
                m = re.match(r"adler32\s+([0-9a-f]+)", q.stdout.strip())
                adler = m.group(1).zfill(8) if m else f"ERR:{(q.stdout + q.stderr).strip()[:60]}"
            out.append(dict(era=e, sample=sample, file=fname, bytes_now=size, mtime_now=mtime, adler32_eos_now=adler,
                            listed_utc=started))
    buf = io.StringIO()
    w = csv.DictWriter(buf, fieldnames=list(out[0]), lineterminator="\n"); w.writeheader(); w.writerows(out)
    with gzip.GzipFile(os.path.join(R, "eos_relist.csv.gz"), "wb", compresslevel=9, mtime=0) as fo:
        fo.write(buf.getvalue().encode())
    print(f"relisted {len(out)} files, {len(query)} checksums queried -> {R}/eos_relist.csv.gz")


def render(a):
    R = a.results
    with gzip.open(os.path.join(R, "v2_census.csv.gz"), "rt") as fi:
        rows = list(csv.DictReader(fi))
    cfg = json.load(open(os.path.join(R, "expected_config.json")))
    branches = json.load(open(os.path.join(R, "branches.json")))
    extra = [l.split() for l in open(os.path.join(R, "eos_extra.txt")) if l.strip()]
    with gzip.open(os.path.join(R, "eos_relist.csv.gz"), "rt") as fi:
        now = {(r["era"], r["sample"], r["file"]): r for r in csv.DictReader(fi)}
    v1 = {(r["era"], r["sample"]): r for r in load_csv(V1_CTAU)}
    keys = sample_keys()
    eprovs = {e: expected_prov(cfg["eras"][e]) for e in ERAS}
    failed, id_failed = {}, {}
    for r in rows:
        why = checks(r, eprovs[r["era"]])
        if why:
            failed[r["url"]] = why
        if identity_checks(r):
            id_failed[r["url"]] = identity_checks(r)
    by_era = collections.defaultdict(list)
    for r in rows:
        by_era[r["era"]].append(r)
    fname = lambda r: f"{r['sample']}/{os.path.basename(r['url'])}"
    key3 = lambda r: (r["era"], r["sample"], os.path.basename(r["url"]))

    # EOS now against the census-time listing
    census_keys = {key3(r) for r in rows}
    relist = {}
    for e in ERAS:
        R_ = by_era[e]
        changed = [r for r in R_ if key3(r) not in now or (now[key3(r)]["bytes_now"], now[key3(r)]["mtime_now"])
                   != (r["eos_ls_bytes"], r["eos_ls_mtime"])]
        new = [k for k in now if k[0] == e and k not in census_keys]
        q = [r for r in R_ if upload_class(r) in ("both", "neither")]       # a failed or missing query counts as changed
        same = sum(key3(r) in now and now[key3(r)]["adler32_eos_now"] == r["adler32_eos"] for r in q)
        relist[e] = dict(changed=changed, new=new, n_q=len(q), same=same,
                         last_census=max(r["eos_ls_mtime"] for r in R_),
                         last_now=max(v["mtime_now"] for k, v in now.items() if k[0] == e))

    # proper decay length per sample (sum over its chunks)
    ctau_rows = []
    groups = collections.defaultdict(list)
    for r in rows:
        if not r["id_error"]:
            groups[(r["era"], r["sample"])].append(r)
    for (e, d), rs in sorted(groups.items(), key=lambda t: (ERAS.index(t[0][0]), t[0][1])):
        n = sum(int(r["id_ctau_n"]) for r in rs)
        s1 = sum(float(r["id_ctau_sum_mm"]) for r in rs); s2 = sum(float(r["id_ctau_sum2_mm2"]) for r in rs)
        mean = s1 / n; err = math.sqrt(max(s2 / n - mean ** 2, 0.0) / n)
        nom = nominal(rs[0])[3]; k = keys[d]; o = v1.get((e, k), {})
        ctau_rows.append(dict(era=e, sample=k, n_files=len(rs), n_decays=n, n_events=sum(int(r["id_events"]) for r in rs),
                              mean_mm=f"{mean:.6g}", stat_err_mm=f"{err:.3g}", nominal_mm=f"{nom:g}",
                              ratio=f"{mean / nom:.5f}", pull=f"{(mean - nom) / err:.3f}",
                              v1_mean_mm=o.get("measured_mm", ""), v1_stat_err_mm=o.get("stat_err_mm", ""),
                              v1_n_decays=o.get("n_decays", "")))
    with open(os.path.join(R, "ctau_v2.csv"), "w", newline="") as fo:
        w = csv.DictWriter(fo, fieldnames=list(ctau_rows[0]), lineterminator="\n"); w.writeheader(); w.writerows(ctau_rows)

    pkeys = ["steps", "hlt_menu", "com_energy", "vtx_type", "vtx_readDB", "vtx_X0", "vtx_Y0", "vtx_Z0", "vtx_SigmaZ",
             "vtx_BetaStar"]
    M = ["# v2 Run 3 SIDM signal census (EXONanoAOD, outputs_v2/)", "",
         f"Population: every file under `{EOSDIR}/<era>/` for the five eras (EOS listing) and every chunk in the "
         "production DAGs (the `VARS` lines of the column and grid DAGs). One census row per file: "
         "`v2_census.csv.gz`; this report is written by `report.py render` from the files in `results/` alone. "
         f"Expected configuration: `expected_config.json`, parsed from `{cfg['chain']}` (md5 {cfg['chain_md5']}; "
         f"identical to the copy in each of the {cfg['n_dag_copies_identical']} era DAG folders of the LPC submission), "
         "with the HLT menu and vertex smearing its `HLT` and `BS` settings select in each era's release on cvmfs.", "",
         "Two-site production: the campaign was submitted twice with the same `chain_job_v2.sh`, DAG node names, "
         "`VARS` lines and output names, writing to the same EOS directory: the LPC submission (LPC HTCondor; its "
         "records are the `.cksum` files in the production's `condor_v2/` directory at LPC) and the CERN lxplus "
         "submission (CERN HTCondor; its `.cksum` records were supplied as one file). Each job writes one record "
         f"(`{cfg['chain']}`): `skip size=…` when the chunk is already on EOS at job start (the check at its lines "
         "59-62, made only when the job starts), otherwise `adler32=… size=…` after its upload (line 234). The census "
         "joins both sites' upload records to every file (columns `lpc_*`, `lx_*`). A chunk with an upload record "
         "at both sites was produced and uploaded at both; the upload overwrites (`xrdcp -f`, line 227), so EOS should "
         "hold the later one, and the other site's record no longer matches it. That the later upload is the one "
         "kept is inferred from the overwrite flag and the matches below, not read from a log.", "",
         "Checks per file: in the EOS listing with the measured size; opens with uproot and every basket of every "
         "Events branch decompresses; Events entries = NEV = Runs genEventCount; adler32 of the copied file = EOS server "
         "checksum; size and adler32 equal an upload record of either site, when one exists; provenance as configured; "
         "identity (generator record against the name, see below).", ""]
    summ, upl, race_rows, devs = [], [], [], []
    era_hash = {}
    for e in ERAS:
        Rr = by_era[e]
        found = sum(1 for r in Rr if r["eos_ls_bytes"])
        ex = [p for era, p in extra if era == e]
        readable = sum(r["open_ok"] == "True" and r["full_read_ok"] == "True" for r in Rr)
        ev = sum(int(r["events"]) for r in Rr if r["events"])
        off = [r for r in Rr if r["events"] and int(r["events"]) != int(r["nev"])]
        pt = collections.Counter(tuple(r.get(k, "") for k in pkeys) for r in Rr if not r["error"])
        bh = collections.Counter(r["branch_hash"] for r in Rr if r["branch_hash"])
        era_hash[e] = bh.most_common(1)[0][0] if bh else None
        sw = sum(1 for r in Rr if r["genEventSumw"] and float(r["genEventSumw"]) != float(r["genEventCount"]))
        sumw = sum(float(r["genEventSumw"]) for r in Rr if r["genEventSumw"])
        gcount = sum(int(r["genEventCount"]) for r in Rr if r["genEventCount"])
        eos_ok = sum(1 for r in Rr if r["adler32"] and r["adler32"] == r["adler32_eos"])
        nfail = sum(1 for r in Rr if r["url"] in failed)
        nid = sum(1 for r in Rr if r["url"] in id_failed)
        rl = relist[e]
        summ.append((e, f"{found} / {len(Rr)}", len(ex), readable, f"{ev:,} / {sum(int(r['nev']) for r in Rr):,}",
                     len(off), len(pt), len(bh), f"{gcount:,} / {sumw:,.1f} ({sw} files differ)",
                     f"{eos_ok} / {len(Rr)}", f"{len(Rr) - nid} / {len(Rr)}", len(rl["changed"]) + len(rl["new"]), nfail))
        cls = collections.Counter(upload_class(r) for r in Rr)
        both = [r for r in Rr if upload_class(r) == "both"]
        bm = collections.Counter(tuple(sorted(s for s, ok in upload_match(r).items() if ok)) for r in both)
        single_ok = {c: sum(1 for r in Rr if upload_class(r) == c and all(upload_match(r).values()))
                     for c in ("LPC only", "lxplus only")}
        upl.append((e, f"{cls['LPC only']} ({single_ok['LPC only']} match)",
                    f"{cls['lxplus only']} ({single_ok['lxplus only']} match)", cls["both"], cls["neither"],
                    sum(cls.values()), bm[("lx",)], bm[("lpc",)], bm[("lpc", "lx")], bm[()]))
        for r in both:
            m = upload_match(r)
            if not all(m.values()):
                race_rows.append((e, fname(r), "lxplus" if m["lx"] else "LPC" if m["lpc"] else "neither",
                                  f"{r['lpc_upload_bytes']} / {r['lpc_upload_adler32']}",
                                  f"{r['lx_upload_bytes']} / {r['lx_upload_adler32']}"))
        M += [f"## {e}", ""]
        M += md_table(["provenance tuple (" + ", ".join(pkeys) + ")", "files"],
                      [(" ; ".join(t), n) for t, n in pt.most_common()]) + [""]
        M += md_table(["branch-name hash (sha1 of sorted names, 12 hex)", "branches", "HLT_ branches", "files"],
                      [(h, len(branches[h]), sum(b.startswith("HLT_") for b in branches[h]), n) for h, n in bh.most_common()]) + [""]
        devs += [(e, p, "on EOS, not in the DAGs") for p in ex]
        devs += [(e, f"{k[1]}/{k[2]}", "on EOS now, not at census time") for k in rl["new"]]
        devs += [(e, fname(r), "size or modification time changed on EOS since the census") for r in rl["changed"]]
        for r in Rr:
            if r["url"] in failed:
                devs.append((e, fname(r), "; ".join(failed[r["url"]])))
            if era_hash[e] and r["branch_hash"] and r["branch_hash"] != era_hash[e]:
                devs.append((e, fname(r), f"branch hash {r['branch_hash']} != era's {era_hash[e]}"))
    head = ["era", "files found / expected", "extra files on EOS", "readable", "events found / expected",
            "chunks with Events != NEV", "provenance tuples", "branch hashes", "genEventCount / genEventSumw",
            "adler32 = EOS server", "identity as named", "changed on EOS since census", "files failing"]
    first = M.index(f"## {ERAS[0]}")
    nneither = sum(u[4] for u in upl)
    top = ["## Summary per era", ""] + md_table(head, summ) + [""]

    top += ["## Upload records of the two submissions", "",
            "Per file, which site holds an upload record for it (a `skip` record is not one), and whether the file on "
            "EOS equals that record (size and adler32). For files with an upload record at both sites, the columns on "
            "the right say which site's record EOS equals.", ""]
    top += md_table(["era", "LPC only", "lxplus only", "both sites", "neither", "sum", "both: EOS = lxplus record",
                     "both: EOS = LPC record", "both: EOS = both records", "both: EOS = neither"], upl) + [""]
    top += [f"'Neither': {nneither} files across the five eras have no upload record at either site (read as a "
            "re-run of their job overwriting its upload record with `skip`: an inference from the record format, not "
            "from a log); they rest on the content checks alone "
            "(readable, NEV events, adler32 = EOS server checksum, provenance, identity).", "",
            "Files uploaded by both sites whose record at one site no longer matches EOS (EOS holds the other site's "
            "upload):", ""]
    top += (md_table(["era", "file", "EOS equals", "LPC upload (bytes / adler32)", "lxplus upload (bytes / adler32)"],
                     race_rows) if race_rows else ["None."]) + [""]

    top += ["## EOS after the census", "",
            "A fresh `xrdfs ls -l -R` of every era (`eos_relist.csv.gz`, started "
            f"{min(v['listed_utc'] for v in now.values())} UTC, after both submissions had stopped) "
            "against the listing taken at census time: size and modification time (UTC) of every file, and the EOS "
            "server checksum queried again for the files of the 'both sites' and 'neither' classes.", ""]
    top += md_table(["era", "latest modification at census", "latest modification now", "files changed or new",
                     "checksums re-queried: unchanged"],
                    [(e, relist[e]["last_census"], relist[e]["last_now"], len(relist[e]["changed"]) + len(relist[e]["new"]),
                      f"{relist[e]['same']} / {relist[e]['n_q']}") for e in ERAS]) + [""]

    # identity
    def steps_off(r, which):
        chan, mbs, mdp, _ = nominal(r)
        nom = mbs if which == "m35" else mdp
        return (float(r[f"id_{which}_median"]) - nom) / storage_step(nom)
    idrows = []
    for e in ERAS:
        ok = [r for r in by_era[e] if not r["id_error"]]
        o35 = [steps_off(r, "m35") for r in ok]; o32 = [steps_off(r, "m32") for r in ok]
        r32 = [abs(float(r["id_m32_median"]) / nominal(r)[2] - 1) for r in ok]
        rng = lambda x: f"{round(min(x)) + 0:+d} .. {round(max(x)) + 0:+d}" if x else "-"
        idrows.append((e, len(ok), sum(1 for r in by_era[e] if r["url"] in id_failed),
                       rng(o35), f"{rng(o32)} (max {max(r32):.2%})" if r32 else "-",
                       sum(int(r["id_events_2zd"]) for r in ok), sum(int(r["id_events_4mu"]) + int(r["id_events_2mu2e"])
                                                                    for r in ok),
                       sum(int(r["id_zd_daughters_nonlepton"]) for r in ok)))
    top += ["## Sample identity", "",
            "From the generator branches of every file (`identity_files.py`): the bound state is the first pdgId 35 "
            "of each event, the dark photons the pdgId 32 whose mother is the bound state, their daughters the "
            "particles whose mother is a dark photon. Masses are stored with 8 mantissa bits; the file's median "
            "bound-state mass must equal the name's value within one storage step, the median dark-photon mass "
            f"within max(one storage step, {ZD_REL_TOL:.0%}) (its line shape is several storage steps wide; the "
            "nearest other dark-photon mass of the grid is four times larger or smaller). Every event must hold "
            "one bound state, two dark photons and the channel's daughters (4Mu: four muons; 2Mu2E: two muons and two "
            "electrons).", ""]
    top += md_table(["era", "files read", "files not as named", "bound-state median - name (storage steps)",
                     "dark-photon median - name (storage steps)", "events with 2 dark photons",
                     "events with the channel's daughters", "non-lepton dark-photon daughters"], idrows) + [""]
    # run / lumi / event
    smp = collections.defaultdict(list)
    for r in rows:
        if not r["id_error"]:
            smp[(r["era"], r["sample"])].append(r)
    same_rl = sum(1 for rs in smp.values() if len({(r["id_runs"], r["id_lumis"]) for r in rs}) == 1)
    from_one = sum(1 for rs in smp.values() if all(r["id_event_min"] == "1" and int(r["id_event_max"]) == int(r["id_events"])
                                                   and r["id_event_unique"] == "True" for r in rs))
    rl_vals = collections.Counter((r["id_runs"], r["id_lumis"]) for r in rows if not r["id_error"])
    multi = sum(1 for rs in smp.values() if len(rs) > 1)
    top += [f"Run, lumi and event numbers: {', '.join(f'run {a} lumi {b} in {n} files' for (a, b), n in rl_vals.most_common())}; "
            f"in {from_one} of {len(smp)} samples every chunk numbers its events 1..N (N its event count) without "
            f"repetition, and in {same_rl} of {len(smp)} all chunks carry the same run and lumi. So (run, lumi, event) "
            f"is unique within a file but not across the chunks of a sample ({multi} samples have more than one chunk): "
            "every chunk restarts at event 1 in run 1, lumi 1. This is a property of the production (each chunk is an "
            "independent cmsRun job with default numbering, `production/chain_job_v2.sh`), not a failure; an event is "
            "identified by (file, event).", ""]
    # proper decay length
    pulls = {e: [float(c["pull"]) for c in ctau_rows if c["era"] == e] for e in ERAS}
    top += ["## Proper decay length per sample", "",
            "Mean proper decay length of the dark photons, summed over every chunk of a sample (100k events for the 36 "
            "column samples, 2000 for the grid-only ones), with the definition of `ctau_all_samples.py` "
            "(|decay vertex - production vertex| / (p/m), the decay vertex being the production vertex of the "
            "pdgId > 0 lepton daughter). Statistical uncertainty = sample standard deviation / sqrt(n); "
            "pull = (mean - name's value) / uncertainty. Full table, with the v1 value (first 4 chunks of the v1 "
            "production, 2022-2023BPix) where one exists: `ctau_v2.csv`.", ""]
    top += md_table(["era", "samples", "pull mean", "pull std", "largest |pull|", "ratio mean", "ratio std"],
                    [(e, "0", "-", "-", "-", "-", "-") if not pulls[e] else (e, len(pulls[e]), f"{sum(pulls[e]) / len(pulls[e]):+.3f}",
                      f"{math.sqrt(sum(p * p for p in pulls[e]) / len(pulls[e]) - (sum(pulls[e]) / len(pulls[e])) ** 2):.3f}",
                      f"{max(map(abs, pulls[e])):.2f}",
                      f"{sum(float(c['ratio']) for c in ctau_rows if c['era'] == e) / len(pulls[e]):.4f}",
                      f"{statistics_std([float(c['ratio']) for c in ctau_rows if c['era'] == e]):.4f}") for e in ERAS]) + [""]
    okr = [r for r in rows if not r["id_error"]]
    wf = max(okr, key=lambda r: abs(file_ctau_pull(r)))
    nsum = len({r["id_ctau_sum_mm"] for r in okr})
    top += [f"Per file: every file holds 2 proper decay lengths per event "
            f"({sum(int(r['id_ctau_n']) == 2 * int(r['events']) for r in okr)} of {len(okr)}); the per-file pull "
            f"(mean against the name's value, uncertainty = value / sqrt(n)) must stay below {CTAU_PULL_MAX:g} in absolute "
            f"value; the largest is {file_ctau_pull(wf):+.2f} ({wf['era']} {sample_keys()[wf['sample']]} {wf['chunk']}). "
            f"Distinct per-file decay-length sums: {nsum} among {len(okr)} files, so no chunk repeats another's "
            "generator record although event numbers repeat across chunks.", ""]
    big = sorted(ctau_rows, key=lambda c: -abs(float(c["pull"])))[:15]
    top += ["Largest pulls:", ""]
    top += md_table(["era", "sample", "decays", "mean (mm)", "stat. unc. (mm)", "name (mm)", "pull", "v1 mean (mm)",
                     "v1 stat. unc. (mm)"],
                    [(c["era"], c["sample"], c["n_decays"], c["mean_mm"], c["stat_err_mm"], c["nominal_mm"], c["pull"],
                      c["v1_mean_mm"] or "-", c["v1_stat_err_mm"] or "-") for c in big]) + [""]
    M[first:first] = top

    M += ["## Provenance against the chain configuration", "",
          "Expected = chain_job_v2.sh (REL/COND for SIM, HLT, RECO; NANOREL/NANOCOND for NANO; HLT and BS resolved in "
          "the release); found = the distinct values over the era's readable files.", ""]
    prow = []
    for e in ERAS:
        for k, v in eprovs[e].items():
            vals = collections.Counter(r.get(k, "") for r in by_era[e] if not r["error"])
            ok = set(vals) == {v} or (_num(k) and all(x and float(x) == float(v) for x in vals))
            prow.append((e, k, v, ", ".join(f"{x} ({n})" for x, n in vals.most_common()), "yes" if ok else "**NO**"))
    M += md_table(["era", "quantity", "chain", "found (files)", "match"], prow) + [""]

    M += ["## Branch-list differences between eras (each era's common list)", "",
          "Counts per pair of eras; names of the non-HLT differences. The HLT_ names are in `branches.json`.", ""]
    brow = []
    for i, e1 in enumerate(ERAS):
        for e2 in ERAS[i + 1:]:
            b1, b2 = set(branches[era_hash[e1]]), set(branches[era_hash[e2]])
            for lab, d in ((f"in {e1}, not {e2}", sorted(b1 - b2)), (f"in {e2}, not {e1}", sorted(b2 - b1))):
                oth = [b for b in d if not b.startswith("HLT_")]
                brow.append((lab, len(d), len(d) - len(oth), len(oth), ", ".join(f"`{b}`" for b in oth) or "none"))
    M += md_table(["pair", "branches", "HLT_", "other", "other names"], brow)
    M += ["", "## Deviations, by file", ""]
    M += md_table(["era", "file", "reason"], devs) if devs else ["None."]
    open(a.md or os.path.join(R, "v2_validation.md"), "w").write("\n".join(M) + "\n")

    for e in ERAS:
        rs = collections.defaultdict(list)
        for r in by_era[e]:
            rs[r["sample"]].append(r)
        write_yaml(os.path.join(a.yaml_dir, f"signal_run3_v2_{e}.yaml"), e, rs, keys, failed)
    worst = max(ctau_rows, key=lambda c: abs(float(c["pull"])))
    print(f"{len(rows)} rows, {len(failed)} failing ({len(id_failed)} identity), {len(devs)} deviations, "
          f"{len(race_rows)} race files, {sum(len(relist[e]['changed']) + len(relist[e]['new']) for e in ERAS)} changed "
          f"on EOS, checksums {sum(relist[e]['same'] for e in ERAS)} / {sum(relist[e]['n_q'] for e in ERAS)} unchanged; "
          f"largest |pull| {worst['pull']} ({worst['era']} {worst['sample']}); wrote "
          f"{a.md or os.path.join(R, 'v2_validation.md')}, ctau_v2.csv and the YAMLs in {a.yaml_dir}")


def statistics_std(x):
    m = sum(x) / len(x)
    return math.sqrt(sum((v - m) ** 2 for v in x) / len(x))


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="stage", required=True)
    j = sub.add_parser("join")
    j.add_argument("--work", required=True)
    j.add_argument("--condor-v2", required=True, help="the production's condor_v2/ directory at LPC (the LPC submission)")
    j.add_argument("--lxplus", required=True, help="the upload records of the CERN lxplus submission, one .cksum per line")
    j.add_argument("--results", default=os.path.join(_DIR, "results"))
    rl = sub.add_parser("relist")
    rl.add_argument("--work", required=True)
    rl.add_argument("--results", default=os.path.join(_DIR, "results"))
    r = sub.add_parser("render")
    r.add_argument("--results", default=os.path.join(_DIR, "results"))
    r.add_argument("--yaml-dir", default=NTUPLES)
    r.add_argument("--md", help="report path (default <results>/v2_validation.md)")
    a = ap.parse_args()
    dict(join=join, relist=relist, render=render)[a.stage](a)


if __name__ == "__main__":
    main()
