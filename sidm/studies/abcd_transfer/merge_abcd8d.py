"""Merge the per-chunk outputs of the ``abcd8d`` campaigns into one ``.coffea`` file per sample.

Each Condor job writes ``<sample>_<chunk>.coffea`` (payload ``{"out": {sample: {"hists": ...}},
"processed": set, "exception": int}``) to an EOS directory.  This script lists that directory,
checks it against the job list, sums every ``abcd8d*`` histogram per sample (simulation
re-weighted per chunk by its sum of generator weights, see :func:`merge_sample`) and writes one
file per histogram family: the search-channel cubes (``abcd8d_<topology>``) to
``<local>/<kind dir>/<sample>.coffea`` and the proxy-region cubes (``abcd8d_proxy_<topology>``)
to ``<local>/proxy/<kind dir>/<sample>.coffea``, each with the payload layout of the merged
files of the 6D campaign, so that ``abcd6d_io.load_cubes(path, prefix=...)`` reads them.  A
manifest line per sample (chunks merged and expected, input files processed) is written to the
``MANIFEST.txt`` of each family directory (one of the inputs ``transfer_lib.summed_cubes`` keys
its cache on) and the job arguments of the chunks to re-run (missing, or found with an
exception or without output, which are not merged) go to ``<local>/rescue_<kind>.txt``.  Only
chunks in the job list are merged.

Campaign kinds: ``dy`` and ``signal`` (search channels only, the first code version, which
filled the isolation axis before the clip at zero), ``data`` and ``bkg_rest`` (both families),
``dy_proxy`` and ``signal_proxy`` (proxy channels only); the last four ran the fourth code
version (``campaign/``: clipped isolation, proxy channels, and a raw-event pre-filter on the
trigger and the two high-pT muons that every channel requires, which leaves the histograms and
the normalisation unchanged and only shortens the cutflow's first rows).

Runs on the LPC in the notebook environment with a valid proxy:

    python merge_abcd8d.py --kind dy --workers 2
"""
import argparse
import collections
import concurrent.futures
import os
import subprocess
import tempfile

EOS = "root://cmseos.fnal.gov/"
CAMPAIGN = "/uscms_data/d3/murtazas/abcd8d/campaign"
EOS_DIR = "/store/user/murtazas/sidm_condor/ABCD_8D_inclusive_eHEM_skimv1_{tag}"
LOCAL = "/uscms_data/d3/murtazas/abcd8d_local"
# kind -> (EOS directory tag, job-args list, output directory within each family)
KINDS = {
    "dy": ("v1_dy", "dy", "bkg"),
    "signal": ("v1_signal", "signal", "signal"),
    "data": ("v4_data", "data", "data"),
    "bkg_rest": ("v4_bkg_rest", "bkg_rest", "bkg"),
    "dy_proxy": ("v4_dy_proxy", "dy", "bkg"),
    "signal_proxy": ("v4_signal_proxy", "signal", "signal"),
}
FAMILIES = {"abcd8d_proxy": "proxy", "abcd8d": ""}      # histogram prefix -> family directory (longest first)


def family_of(hist_name):
    for prefix, subdir in FAMILIES.items():
        if hist_name.startswith(prefix + "_"):
            return subdir
    return None


def eos_list(path):
    out = subprocess.run(["xrdfs", EOS, "ls", path], check=True, capture_output=True, text=True).stdout
    return [os.path.basename(line.strip()) for line in out.splitlines() if line.strip().endswith(".coffea")]


def expected_chunks(kind):
    """``{sample: {chunk: job-args line}}`` from the campaign's job list."""
    chunks = collections.defaultdict(dict)
    with open(os.path.join(CAMPAIGN, f"job_args_{KINDS[kind][1]}.txt")) as f:
        for line in f:
            sample, chunk, _ = line.split()
            chunks[sample][chunk] = line
    return chunks


def split_name(name):
    stem = name[: -len(".coffea")]
    sample, chunk = stem.rsplit("_", 1)
    return sample, chunk


def flag(metadata, key):
    """A boolean of the chunk metadata (stored as a one-element set accumulator)."""
    value = metadata[key]
    return bool(next(iter(value))) if isinstance(value, (set, frozenset, list)) else bool(value)


def scale_in_place(h, factor):
    """``h *= factor`` on the weighted sums without a copy (value x factor, variance x factor^2)."""
    view = h.view(flow=True)
    view["value"] *= factor
    view["variance"] *= factor * factor


def combine(entries, sample):
    """Accumulate the ``abcd8d*`` histograms of the per-chunk entries (``payload["out"][sample]``)
    of one sample; returns ``(hists, metadata)`` with the summed ``scaled_sum_weights``.

    The processor scales a simulated chunk's histograms to the luminosity with that chunk's own
    sum of generator weights (``metadata["scaled_sum_weights"]``), so a plain sum over chunks
    counts the sample once per chunk.  As in ``sidm/scripts/merge_coffea_chunks_eos.py``, each
    chunk is weighted by its sum of weights and the total divided by the summed sum of weights,
    which equals the histogram scaled once by lumi x xs / sum of weights of the whole sample.
    Data histograms are unscaled and are summed as they are.
    """
    hists, metadata, sumw_total, is_data, names = {}, None, 0.0, None, None
    for entry in entries:
        if metadata is None:
            metadata = dict(entry["metadata"])
            is_data = flag(metadata, "is_data")
            if not is_data and flag(metadata, "unweighted_hist"):
                raise ValueError(f"{sample}: unweighted histograms are not expected in this campaign")
        # a chunk lacking one of the histograms would still add its sum of weights to the
        # normalisation of the others: every chunk must carry the same set
        these = {h for h in entry["hists"] if family_of(h) is not None}
        if names is None:
            names = these
        elif these != names:
            raise ValueError(f"{sample}: the chunks carry different histogram sets ({sorted(these ^ names)})")
        sumw = float(entry["metadata"]["scaled_sum_weights"])
        sumw_total += sumw
        for h, stored in entry["hists"].items():
            if family_of(h) is None:
                continue
            stored = getattr(stored, "hist", stored)
            if not is_data:
                scale_in_place(stored, sumw)
            if h not in hists:
                hists[h] = stored
            else:
                hists[h] += stored          # in place: a 4mu cube is 0.4 GB, the login nodes have 11 GB
        del entry, stored
    if metadata is None:
        raise ValueError(f"{sample}: no chunk carries this sample")
    if not is_data:
        for h in hists.values():
            scale_in_place(h, 1.0 / sumw_total)
    metadata["scaled_sum_weights"] = sumw_total
    return hists, metadata


def merge_sample(args):
    """Fetch the chunks of one sample, :func:`combine` them and write one file per family;
    returns (sample, families written, chunks merged, files processed, bad chunk names).  A
    chunk that reports an exception or carries no output for the sample is not merged and is
    returned as bad (to be re-run, like a missing one); unless ``allow_missing`` the sample is
    then not written at all."""
    import coffea.util      # only here, so that the combination is testable without coffea

    kind, sample, names, allow_missing = args
    eos_dir = EOS_DIR.format(tag=KINDS[kind][0])
    processed, bad, merged = 0, [], 0
    os.makedirs(os.path.join(LOCAL, "tmp"), exist_ok=True)

    def entries():
        nonlocal processed, merged
        with tempfile.TemporaryDirectory(dir=os.path.join(LOCAL, "tmp")) as tmp:
            for name in sorted(names):
                local = os.path.join(tmp, name)
                subprocess.run(["xrdcp", "-f", "-s", f"{EOS}/{eos_dir}/{name}", local], check=True)
                payload = coffea.util.load(local)
                os.remove(local)
                if int(payload.get("exception", 0)) or sample not in payload.get("out", {}):
                    bad.append(name)
                    continue
                processed += len(payload.get("processed", ()))
                merged += 1
                yield payload["out"][sample]

    hists, metadata = combine(entries(), sample)
    if bad and not allow_missing:
        return sample, [], merged, processed, bad
    metadata["n_chunks"] = merged
    families = sorted({family_of(h) for h in hists})
    for family in families:
        out_dir = os.path.join(LOCAL, family, KINDS[kind][2])
        os.makedirs(out_dir, exist_ok=True)
        part = {h: v for h, v in hists.items() if family_of(h) == family}
        payload = {"out": {sample: {"hists": part, "metadata": metadata}},
                   "chunks": merged, "processed": processed, "exception": 0}
        coffea.util.save(payload, os.path.join(out_dir, f"{sample}.coffea"))
    return sample, families, merged, processed, bad


def plan(found, expected, samples, allow_missing):
    """Which chunks to merge per sample and which job lines to re-run: only chunks in the job
    list count (a leftover output under another chunk number would be counted twice); a sample
    with missing chunks is merged only with ``allow_missing``.  Returns (todo, rescue, extras)."""
    rescue, todo, extras = [], [], {}
    for sample in samples:
        present = found.get(sample, set())
        missing = set(expected[sample]) - present
        extra = present - set(expected[sample])
        if extra:
            extras[sample] = sorted(extra, key=int)
        if missing:
            rescue.extend(expected[sample][c] for c in sorted(missing, key=int))
            if not allow_missing:
                continue
        chunks = present & set(expected[sample])
        if chunks:
            todo.append((sample, sorted(chunks, key=int)))
    return todo, rescue, extras


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--kind", required=True, choices=sorted(KINDS))
    parser.add_argument("--workers", type=int, default=2, help="one worker holds a chunk and the running totals, ~1.6 GB")
    parser.add_argument("--samples", nargs="*", help="only these samples (default: all)")
    parser.add_argument("--allow-missing", action="store_true", help="merge samples with missing chunks too")
    args = parser.parse_args()

    os.makedirs(LOCAL, exist_ok=True)
    found = collections.defaultdict(set)
    for name in eos_list(EOS_DIR.format(tag=KINDS[args.kind][0])):
        sample, chunk = split_name(name)
        found[sample].add(chunk)
    expected = expected_chunks(args.kind)
    samples = args.samples or sorted(expected)

    todo, rescue, extras = plan(found, expected, samples, args.allow_missing)
    for sample, chunks in extras.items():
        print(f"{sample}: {len(chunks)} output(s) not in the job list, skipped: {', '.join(chunks[:5])}")
    for sample in samples:
        missing = len(expected[sample]) - len(found.get(sample, set()) & set(expected[sample]))
        if missing:
            print(f"{sample}: {missing} of {len(expected[sample])} chunks missing")
    jobs = [(args.kind, sample, [f"{sample}_{c}.coffea" for c in chunks], args.allow_missing) for sample, chunks in todo]
    print(f"{len(rescue)} missing chunks; merging {len(jobs)} samples")

    # a process pool that raises (BrokenProcessPool) when a worker is killed, instead of hanging
    with concurrent.futures.ProcessPoolExecutor(args.workers) as pool:
        results = list(pool.map(merge_sample, jobs))
    lines = {}
    for sample, families, n_chunks, processed, bad in sorted(results):
        rescue.extend(expected[sample][split_name(name)[1]] for name in bad)
        line = f"{args.kind} {sample} chunks={n_chunks}/{len(expected[sample])} files_processed={processed} exceptions=0"
        if not families:
            print(f"{sample}: {len(bad)} chunk(s) with an exception or no output, not written (rerun them)")
            continue
        for family in families:
            lines.setdefault(family, []).append(line)
        print(line, "->", ", ".join(f or "search" for f in families))
    with open(os.path.join(LOCAL, f"rescue_{args.kind}.txt"), "w") as f:
        f.writelines(rescue)
    print(f"{len(rescue)} chunks to re-run written to rescue_{args.kind}.txt")
    for family, new_lines in lines.items():
        update_manifest(os.path.join(LOCAL, family, "MANIFEST.txt"), args.kind, new_lines)


def update_manifest(path, kind, new_lines):
    """Replace this kind's lines for the merged samples, keep everything else, sorted."""
    merged = {line.split()[1] for line in new_lines}
    kept = []
    if os.path.exists(path):
        with open(path) as f:
            kept = [line.rstrip("\n") for line in f if line.strip()
                    and not (line.split()[0] == kind and line.split()[1] in merged)]
    with open(path, "w") as f:
        f.write("\n".join(sorted(kept + new_lines)) + "\n")


if __name__ == "__main__":
    main()
