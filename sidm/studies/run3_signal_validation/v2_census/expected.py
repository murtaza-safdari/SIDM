#!/usr/bin/env python3
"""Expected content of the v2 Run 3 signal production (outputs_v2/<era>/).

Reads, all read-only:
  <condor_v2>/<era>_{column,grid}/<era>_{column,grid}.dag   one VARS line per chunk (NEV, output name)
  <condor_v2>/<era>_{column,grid}/cksum/<name>.cksum         the LPC submission's record per chunk
  production/chain_job_v2.sh (this directory)                per-era configuration (case block) and comEnergy
  /cvmfs/cms.cern.ch/<arch>/cms/cmssw/<release>/src/...      HLT menu tableName and vertex-smearing parameters
                                                             the chain's settings resolve to
<condor_v2> is the production's condor_v2/ directory at LPC (one folder per era and DAG kind). The copy of
chain_job_v2.sh in each of its ten <era>_{column,grid} folders must be identical to production/chain_job_v2.sh.
Writes <out>/expected_files.csv (one row per expected file), <out>/expected_config.json (per era) and
<out>/joblists/job_NNN.txt (the EOS URLs of each census job; --no-joblists keeps existing ones).

The grid DAG covers all 180 samples j000-j003, the column DAG the 36 column samples j000-j119; the 144 names in
both are one file (the second job found it on EOS and wrote 'skip size=N'). A .cksum carries adler32 only if its
job uploaded the file; 'skip' records keep the size only.
"""
import argparse, csv, glob, hashlib, json, os, re

ERAS = ["2022", "2022EE", "2023", "2023BPix", "2024"]
EOSDIR = "/store/group/lpcmetx/SIDM/run3_samplegen/outputs_v2"
XRD = "root://cmseos.fnal.gov/"
CVMFS = "/cvmfs/cms.cern.ch"
_DIR = os.path.dirname(os.path.abspath(__file__))
CHAIN = os.path.join(_DIR, "production", "chain_job_v2.sh")


def dag_rows(condor_v2, era):
    rows = {}
    for kind in ("column", "grid"):
        dag = os.path.join(condor_v2, f"{era}_{kind}", f"{era}_{kind}.dag")
        for line in open(dag):
            if not line.startswith("VARS "):
                continue
            v = dict(re.findall(r'(\w+)="([^"]*)"', line))
            name = v["CKSUM"][:-len(".cksum")]
            sample = name[:-len(f"_{era}_{v['JOBIDX']}")]
            r = dict(era=era, sample=sample, chunk=v["JOBIDX"], file=name + ".root", nev=int(v["NEV"]),
                     chan=v["CHAN"], mbs=v["MBs"], mdp=v["MDp"], ctau=v["CTAU"], dags=kind)
            if name in rows:
                old = rows[name]
                if old["nev"] != r["nev"]:
                    raise RuntimeError(f"{name}: NEV {old['nev']} in {old['dags']}, {r['nev']} in {kind}")
                old["dags"] += "+" + kind
            else:
                rows[name] = r
            ck = os.path.join(condor_v2, f"{era}_{kind}", "cksum", v["CKSUM"])
            rec = open(ck).read().split() if os.path.exists(ck) else []
            kv = dict(x.split("=", 1) for x in rec[1:] if "=" in x)
            if "size" in kv:
                if rows[name].get("ref_bytes") not in (None, int(kv["size"])):
                    raise RuntimeError(f"{name}: cksum sizes differ between DAGs")
                rows[name]["ref_bytes"] = int(kv["size"])
            if "adler32" in kv:
                rows[name]["ref_adler32"] = kv["adler32"]
            rows[name].setdefault("cksum_status", [])
            rows[name]["cksum_status"].append(f"{kind}:{rec[1].split('=')[0] if len(rec) > 1 else 'missing'}")
    for r in rows.values():
        r["cksum_status"] = ";".join(r["cksum_status"])
        r.setdefault("ref_bytes", ""); r.setdefault("ref_adler32", "")
        r["url"] = f"{XRD}/{EOSDIR}/{era}/{r['sample']}/{r['file']}"
    return sorted(rows.values(), key=lambda r: (r["sample"], r["chunk"]))


def chain_config(chain):
    """The per-era settings of chain_job_v2.sh's case block, plus NANOREL/NANOARCH."""
    src = open(chain).read()
    glob_ = dict(re.findall(r"^(NANOREL|NANOARCH)=(\S+?);?\s", src, re.M))
    block = src[src.index("case $ERA in"):src.index("esac")]
    out = {}
    for m in re.finditer(r"^\s*(\w+)\)\s*(.*?);;", block, re.M | re.S):
        era = m.group(1)
        if era not in ERAS:
            continue
        kv = dict(re.findall(r'(\w+)=("[^"]*"|\S*?)(?:;|\s|$)', m.group(2)))
        out[era] = {k: v.strip('"') for k, v in kv.items()}
        out[era].update(glob_)
    return out


def _pset(text, name):
    m = re.search(rf"^{name}\s*=\s*cms\.PSet\((.*?)^\)", text, re.M | re.S)
    if not m:
        return None
    return {k: float(v) for k, v in re.findall(r"(\w+)\s*=\s*cms\.double\(([-+0-9.eE]+)\)", m.group(1))}


def resolved(cfg):
    """HLT menu tableName and vertex smearing the chain's HLT and BS settings select in the era's release."""
    base = f"{CVMFS}/{cfg['ARCH']}/cms/cmssw/{cfg['REL']}/src"
    hlt = open(f"{base}/HLTrigger/Configuration/python/HLT_{cfg['HLT']}_cff.py").read()
    table = re.search(r"tableName\s*=\s*cms\.string\(\s*['\"]([^'\"]+)", hlt).group(1)
    vs = open(f"{base}/Configuration/StandardSequences/python/VtxSmeared.py").read()
    cfi = re.search(rf"['\"]{re.escape(cfg['BS'])}['\"]\s*:\s*['\"]([\w.]+)['\"]", vs).group(1)
    sub, pkg, mod = cfi.split(".", 2)
    cfitext = open(f"{base}/{sub}/{pkg}/python/{mod.replace('.', '/')}.py").read()
    # 'VtxSmeared = cms.EDProducer("X", ...)' or 'VtxSmeared = X.clone(..., readDB = True)'
    gen = re.search(r'VtxSmeared\s*=\s*(?:cms\.EDProducer\(\s*"(\w+)"|(\w+)\.clone\()', cfitext)
    gen = gen.group(1) or gen.group(2)
    readdb = bool(re.search(r"readDB\s*=\s*(?:cms\.bool\(\s*)?True", cfitext))
    vtx = dict(vtx_type=gen, vtx_readDB=readdb, vtx_cfi=cfi)
    pm = re.search(r"(\w+VtxSmearingParameters)", cfitext)
    if pm and not readdb:
        params = _pset(open(f"{base}/IOMC/EventVertexGenerators/python/VtxSmearedParameters_cfi.py").read(), pm.group(1))
        vtx.update({f"vtx_{k}": params[k] for k in ("X0", "Y0", "Z0", "SigmaZ", "BetaStar")})
    return dict(hlt_menu=table, **vtx)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--condor-v2", required=True, help="the production's condor_v2/ directory at LPC")
    ap.add_argument("--chain", default=CHAIN, help="the production script (default: the committed copy)")
    ap.add_argument("--out", required=True)
    ap.add_argument("--files-per-job", type=int, default=250)
    ap.add_argument("--no-joblists", action="store_true", help="do not rewrite <out>/joblists/")
    a = ap.parse_args()
    rows = [r for era in ERAS for r in dag_rows(a.condor_v2, era)]
    cols = ["era", "sample", "chunk", "file", "nev", "chan", "mbs", "mdp", "ctau", "dags", "cksum_status",
            "ref_bytes", "ref_adler32", "url"]
    os.makedirs(a.out, exist_ok=True)
    with open(os.path.join(a.out, "expected_files.csv"), "w", newline="") as fo:
        w = csv.DictWriter(fo, fieldnames=cols); w.writeheader(); w.writerows(rows)
    text = open(a.chain).read()
    copies = [os.path.join(a.condor_v2, f"{e}_{k}", "chain_job_v2.sh") for e in ERAS for k in ("column", "grid")]
    differ = [c for c in copies if open(c).read() != text]
    if differ:
        raise RuntimeError(f"{len(differ)} of the {len(copies)} DAG folders' chain_job_v2.sh differ from {a.chain}")
    cfg = chain_config(a.chain)
    com = float(re.search(r"comEnergy=cms\.double\(([0-9.]+)\)", text).group(1))   # the GEN fragment
    for era in ERAS:
        cfg[era]["resolved"] = dict(resolved(cfg[era]), com=com)
    chain = os.path.relpath(os.path.abspath(a.chain), _DIR)
    json.dump(dict(chain=chain, chain_md5=hashlib.md5(text.encode()).hexdigest(), n_dag_copies_identical=len(copies),
                   eras=cfg), open(os.path.join(a.out, "expected_config.json"), "w"), indent=1)
    if not a.no_joblists:
        os.makedirs(os.path.join(a.out, "joblists"), exist_ok=True)
        for old in glob.glob(os.path.join(a.out, "joblists", "job_*.txt")):
            os.remove(old)
        for j in range(0, len(rows), a.files_per_job):
            with open(os.path.join(a.out, "joblists", f"job_{j // a.files_per_job:03d}.txt"), "w") as fo:
                fo.writelines(r["url"] + "\n" for r in rows[j:j + a.files_per_job])
    print(f"{len(rows)} expected files, {sum(r['nev'] for r in rows)} events; config of {len(cfg)} eras from {chain} "
          f"(identical to the {len(copies)} DAG folders' copies)")


if __name__ == "__main__":
    main()
