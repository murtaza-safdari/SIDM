# v2 census: validation and registration of the v2 Run 3 signal samples

The v2 production (`/store/group/lpcmetx/SIDM/run3_samplegen/outputs_v2/<era>/<sample>/<sample>_<era>_jNNN.root`,
EXONanoAOD made with `PAT,NANO:@EXO` in CMSSW_16_0_10) has 4896 files and 3,888,000 events per era in five eras
(2022, 2022EE, 2023, 2023BPix, 2024). The census measures, for every file, that it is there, that it is readable
to the last basket, that it has the right number of events, that its provenance is the configured one and that
its generator record is the mass point, lifetime and channel its name says, and writes the location YAMLs
`sidm/configs/ntuples/signal_run3_v2_<era>.yaml` from the files that pass.

| script | runs on | does |
|---|---|---|
| `expected.py` | login node, seconds | expected file list and NEV per chunk from the production DAGs (and the LPC submission's `.cksum` record per chunk); expected configuration from `production/chain_job_v2.sh`, with the HLT menu and vertex smearing its settings select in each release on cvmfs; the job lists |
| `census_files.py` | condor, ~6 s per file | per file: `xrdcp` to scratch, adler32 (local and EOS server), uproot open, every basket of every Events branch decompressed, Runs tree, branch list (sha1 of the sorted names), process history (release and global tag per process) and the HLT menu, comEnergy and vertex smearing from the ParameterSets |
| `identity_files.py` | condor, ~3 s per file | per file, from the generator branches: bound-state and dark-photon masses, dark photons per event and their daughters' flavours, the proper decay lengths' count, sum and sum of squares, run / lumi / event numbers |
| `report.py join` | login node, ~2 min | merges the job CSVs with the expected list, the EOS listing and the upload records of both submissions into `results/` |
| `report.py relist` | login node, seconds | lists EOS again and re-queries the server checksum of the files without a unique upload record (`results/eos_relist.csv.gz`) |
| `report.py render` | login node, seconds | from `results/` alone: applies the checks, writes `results/v2_validation.md`, `results/ctau_v2.csv` and the YAMLs |
| `rerun_check.py` | login node, minutes | re-runs the census on random files and compares their rows |

The provenance read-back is the one of `../era_fingerprint.py` (`_raw`, `_blobs`, `_get`, `provenance`,
`ACCEPTED_GT`), loaded by `census_files.py` from that file with `ast` so that the workers do not need its
plotting imports; likewise `identity_files.py` takes `proper_ctau` and `BR` from `../ctau_all_samples.py`, so the
proper decay length is the one of the v1 study (`../ctau_all_samples.csv`).

## The production

`production/` holds copies of the production job: `chain_job_v2.sh` (byte-identical, md5
a3af76b526123b1a6999308a8230b9e0; the same file ran at both submission sites) and its condor submit description as
the DAGs used it, `prod_v2.sub` for 2022 to 2023BPix and `prod_v2_2024.sub` for 2024 (the two differ in
`request_memory` only). The one edit to the submit files is the `x509userproxy` line, where the submitting account's
proxy path is replaced by `/uscms_data/d3/<user>/x509_proxy.pem`; nothing in any of these files is a credential.
`make_dag_v2.py`, `reduced_points.txt` (the 36 column samples) and `points_all.txt` (all 180 samples) are verbatim
copies of the DAG generator and its point lists; the two lists equal the census's column and full populations.
`make_dag_v2.py` fills a `prod_v2.sub` template (`@MEMORY@`, `@DISK@`); the committed submit files are its output.
`expected.py` reads the era settings (releases, global tags, beam spot, HLT menu, nano release and GT) and the
generator's comEnergy from `production/chain_job_v2.sh`, and checks that the copy in each of the ten
`<era>_{column,grid}` DAG folders of the LPC submission is identical to it. The production directory's eleventh
folder, `test/`, is the test-chunk submission (outputs under `outputs_v2/test/`, outside the census); it contributed
no upload records to the census.

Files the copies cite and the repository does not hold: `PROVENANCE.md` (section "v2"), `chain_job.sh` (the v1
chain) and `campaign.sub` (the v1 submit description). In the SampleFactory fork (github murtaza-safdari/SampleFactory),
branch `sidm-run3-lpc` holds `sidm_run3/PROVENANCE.md` without the v2 section, and `sidm_run3_production/chain_job.sh`
and `campaign.sub`; the v2 files (`sidm_run3_production/chain_job_v2.sh`, `make_dag_v2.py`, `prod_v2.sub` template,
`reduced_points.txt`, and the PROVENANCE v2 section) are on a later local branch of that clone that is not on the fork.
The DAG files and the two sites' upload records are not in the repository: `render` needs only committed files,
while `expected.py` and `join` need those records, which the sample producer keeps.

## Command lines

`W` is a work directory, `S` this directory, `CONDOR_V2` the production's `condor_v2/` directory at LPC (one folder
per era and DAG kind, with the DAGs and the LPC submission's `.cksum` records), `LX` the upload records of the CERN
lxplus submission (one line per `.cksum` file: `<path> <name>.root adler32=<hex> size=<n>` or
`<path> <name>.root skip size=<n>`). Python: the SIDM notebook venv for `report.py` and `rerun_check.py`;
`expected.py` needs only the standard library.

```bash
# 1. expected content (reads the DAGs, the .cksum records and production/chain_job_v2.sh, read-only)
python3 $S/expected.py --condor-v2 $CONDOR_V2 --out $W --files-per-job 250

# 2. EOS listing per era (the population of 'files found')
for e in 2022 2022EE 2023 2023BPix 2024; do
  xrdfs root://cmseos.fnal.gov ls -l -R /store/group/lpcmetx/SIDM/run3_samplegen/outputs_v2/$e > $W/eos_ls_$e.txt
done

# 3. census and identity on condor: 98 jobs of 250 files each, 1 CPU, 2 GB (environment, condor/setup_env.sh:
#    CMSSW_14_1_0_pre4 python and a venv with the uproot/awkward/numpy pins of condor/requirements.txt, as
#    condor/run_job.sh builds it)
mkdir -p $W/condor/out $W/condor/logs $W/condor_id/out $W/condor_id/logs
cp $S/condor/{setup_env.sh,run_census.sh,census.sub} $S/census_files.py $S/../era_fingerprint.py condor/requirements.txt $W/condor/
cp $S/condor/{setup_env.sh,run_identity.sh,identity.sub} $S/identity_files.py $S/../ctau_all_samples.py condor/requirements.txt $W/condor_id/
ls $W/joblists | sed 's/job_\(.*\)\.txt/\1/' > $W/condor/jobs.txt; cp $W/condor/jobs.txt $W/condor_id/
(cd $W/condor && condor_submit census.sub); (cd $W/condor_id && condor_submit identity.sub)

# 4. join the census, the identity pass, the EOS listing and both submissions' upload records (jobs done)
python $S/report.py join --work $W --condor-v2 $CONDOR_V2 --lxplus $LX

# 5. after both submissions have stopped: list EOS again, re-query the checksums of 'both sites' and 'neither'
python $S/report.py relist --work $W

# 6. checks, report and YAMLs, from the committed results/ alone
python $S/report.py render

# 7. reproducibility: re-run the census on 50 random files and compare with their rows
python $S/rerun_check.py --work $W -n 50 --seed 20261107
```

## Checks

A file passes when it is on EOS and in the DAGs with its measured size, opens and every Events basket
decompresses (`full_read_ok`), has Events entries = NEV = Runs `genEventCount`, has an adler32 equal to the EOS
server checksum, has the size and adler32 of an upload record of either submission when one exists, has the
configured provenance (SIM, HLT and RECO processes in the era's release and global tag, NANO in CMSSW_16_0_10 with
the era's nano GT, the HLT menu of the era's `HLT` setting, comEnergy 13600 and the vertex smearing of the era's
`BS` setting), and is the sample its name says (identity, below). Branch lists are compared within an era (one hash
expected) and between eras (counts and the non-HLT names in the report; the full lists in `results/branches.json`).

## Two-site production

The campaign was submitted twice, with the same `chain_job_v2.sh`, DAG node names, `VARS` lines and output names,
writing to the same `outputs_v2/`: the LPC submission (its records: the `.cksum` files in the production's
`condor_v2/` directory at LPC) and the CERN lxplus submission (its `.cksum` records, supplied as one file). Every job
writes one record (`production/chain_job_v2.sh`): `skip size=...` if the output is already on EOS when the job
starts (lines 59-62; the job looks only then), otherwise `adler32=... size=...` after its upload (line 234).
`report.py join` attaches both sites' upload records to every file (`lpc_*`, `lx_*`).

A chunk with an upload record at both sites was produced and uploaded at both. The upload overwrites
(`xrdcp -f`, line 227), so EOS holds one of the two, and the census finds that the other site's record no longer
matches EOS. That EOS holds the later upload is inferred from the overwrite flag, not read from a log; for the 12
files whose EOS copy is the lxplus upload, the EOS modification time is 1 to 90 minutes after the time the LPC job's
record was written. A record found as `skip` where the job had uploaded before is read here as a re-run of that job overwriting
its record: that too is an inference, from the record format, not from a log.

Both submissions had stopped before the census: the two lxplus DAGMan runs exited with status 0 on 2026-11-06 at
14:29:31 CET (first batch) and on 2026-11-07 at 12:38:20 CET (second batch, cluster 13139084), as read from their
`dagman.out`; the LPC outer DAG (cluster 4846734) exited with status 0 on 2026-11-07 at 11:58:55 CST
(17:58:55 UTC; its `dagman.out`). The census-time listing was taken on 2026-11-07 at 21:38 UTC. `report.py relist`
listed every era again on 2026-11-10 (time in `results/eos_relist.csv.gz`, column `listed_utc`) and compared size
and modification time of every file, and the EOS server checksum of the files with an upload record at both sites or
at neither; the report's 'EOS after the census' table gives the latest modification time per era and the number of
changed files.

## Sample identity

`identity_files.py` reads the generator branches of every file. The bound state is the first pdgId 35 of an event,
the dark photons are the pdgId 32 whose mother is the bound state, and their daughters the particles whose mother is
a dark photon. GenPart_mass is stored with 8 mantissa bits (`../README.md`, "Stored precision"): the file's median
bound-state mass must equal the name's value within one storage step (2^(floor(log2 m) - 8)); the dark photon has a
line shape several storage steps wide (at 0.25 GeV the per-file median wanders by up to a few steps), so its median
must lie within max(one storage step, 3%) of the name's value, which still separates it from the nearest other
dark-photon mass of the grid (four times larger or smaller). Every event must hold one bound state, two dark photons
and the channel's daughters (4Mu: four muons; 2Mu2E: two muons and two electrons). Each file must hold two proper
decay lengths per event (`id_ctau_n` = 2 x events), and its mean proper decay length must lie within 5 standard
errors of the name's value (per-file pull = (mean - value) / (value / sqrt(n)), the exponential's standard deviation
being its mean); the largest per-file pull is +4.44 (2022 `4Mu_1000GeV_5p0GeV_4p0mm` j016).

The mean proper decay length per sample is formed over all its chunks from the per-file count, sum and sum of
squares, and compared with the name's value (statistical uncertainty: standard deviation / sqrt(n)); the v1 value is
taken from `../ctau_all_samples.csv` where it exists (v1 had no 2024).

Run, lumi and event numbers are a property of the production, not a check: every chunk is an independent cmsRun
job, `production/chain_job_v2.sh` sets no first run, lumi or event number, and the files number their events 1..N
in run 1, lumi 1, so (run, lumi, event) is unique within a file and repeats across
the chunks of a sample. An event is identified by (file, event).

## Results of the 2026-11 census (`results/`)

`v2_validation.md` (the report), `v2_census.csv.gz` (one row per file: the census, NEV, the EOS listing size and
time, both sites' upload records, the identity columns `id_*`), `branches.json` (each distinct branch list),
`eos_extra.txt` (files on EOS not in the DAGs: none), `eos_relist.csv.gz` (the listing after the census),
`ctau_v2.csv` (proper decay length per sample and era), `expected_config.json` (the chain's configuration as
parsed), `rerun_check.csv` (the 50 re-run rows of the first pass).

All five eras: 4896 / 4896 files, all readable to the last basket, 3,888,000 / 3,888,000 events, 0 chunks off NEV,
one provenance tuple and one branch-list hash per era, provenance equal to the chain's configuration, every file
as named (bound-state median mass exact to the storage step; dark-photon median 0 to +2 storage steps from the
name's value, at most 0.78%; every event with one bound state, two dark photons and the channel's daughters),
0 failing files. Upload records: 1307 files have one at the LPC submission only, 23,034 at the lxplus submission
only, 90 at both sites (EOS equals the lxplus upload for 12, the LPC upload for 78), and 49 (2022EE 4, 2023 44,
2023BPix 1) at neither; those 49 rest on the content checks alone. Every file with an upload record equals one.
EOS after the census (listed again on 2026-11-10): 0 of 24,480 files changed in size or modification time, no new
files, 139 / 139 server checksums of the 'both sites' and 'neither' files unchanged; the latest modification time
per era is 2026-11-07 17:58:44 UTC (2022), 2026-11-06 10:25:59 (2022EE), 2026-11-06 09:29:08 (2023),
2026-11-06 06:50:29 (2023BPix), 2026-11-07 10:08:49 (2024).

Proper decay length (900 samples): the pulls against the names' values have mean between -0.06 and +0.07 and
standard deviation between 0.94 and 1.12 per era; 48 samples have |pull| > 2 (41 expected), 3 have |pull| > 3
(2.4 expected), the largest is -3.6 (2024 `2Mu2E_100GeV_5p0GeV_0p4mm`, 4000 decays: 0.378 +- 0.006 mm against
0.4 mm). Against v1 (720 samples, 2022-2023BPix), (v2 - v1) / combined uncertainty has mean +0.02 and standard
deviation 1.02. The re-run of 50 random files (seed 20261107) reproduced every census column.
