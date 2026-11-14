# Format comparison: the DSA-PF segment-match veto in the two Run 3 signal productions

Compares the pass fraction of the DSA-muon cross-cleaning veto (`"all"` of `ljsource_cuts`, and `"all + charge"`)
between the first Run 3 signal production (v1, LLPNanoAOD, `run3_samplegen/outputs/<era>/`) and the second (v2,
EXONanoAOD, `run3_samplegen/outputs_v2/<era>/`). In v2 the stored `DSAMuon_nSegments` is wrong (twice
`nDTSegments`), so v2 uses the rebuilt count `nDTSegments + ceil(trkNumCSCHits / 6)`; v1 uses the stored count.

| script | runs on | does |
|---|---|---|
| `make_tasks.py` | login node, ~30 s | file lists (`VERSION ERA SAMPLE URL` per line) from the location YAMLs: `--stage first` the files of the first comparison (Definitions), `--stage full` every file of the 36 column samples and of the 12 prompt-like points in 2022, 2022EE, 2023, 2023BPix |
| `xclean_rows.py` | login node or condor, ~1 s per file and process | per file, one `.npz` with the "pre" DSA muons (pT > 10 GeV, abs(eta) < 2.4, displaced ID): kinematics, hit and segment counts, the five `muonMatch` slot counts, the number of good matched PF muons, the veto decisions for the stored, rebuilt and rebuilt + 1 segment counts, and the gen Lxy and dR of the nearest gen muon from a dark-photon decay; per event the numbers of DSA, "pre" DSA and good PF muons |
| `xclean_first_table.py` | login node, seconds | the pass-fraction table of the first comparison on its own files (displaced j040-j059, prompt-like j000-j003; 2022, 2023) |
| `xclean_stats.py` | login node, minutes | the tables: per set and era v1, v2, v2 - v1 with binomial and event-clustered uncertainties, the pulls, inverse-variance combinations over eras and samples, chi2 against zero, and the verdicts of the decision rule |
| `xclean_checks.py` | login node, minutes | on the rows: (a) the rebuilt count on v1, where the stored count is correct (decisions changed, agreement, overcounts); (b) the pooled column-sample v2 - v1 with the variance clustered by file inside each sample; (c) the file-level design effect per production, era and sample. Printed; copied into `xclean_results.md` |
| `xclean_where.py` | login node, minutes | per era, v1 against v2 for the inputs of the veto (DSA multiplicity, segment counts by region, match slots, good PF muons, pass fraction in abs(eta) and gen Lxy bins, rebuilt + 1 bound), for locating a difference |
| `condor/` | condor | `run_xclean.sh` and `xclean.sub`: `xclean_rows.py` on a task list per job |

The veto is evaluated with the cut lambdas of `sidm/definitions/cuts.py` themselves, on a `good_matched_muons`
collection built as `sidm/tools/sidm_processor.py` builds it (the PF muons of the five match slots, `ak.drop_none`,
then the `muons` cuts of `ljsource_cuts` in `sidm/configs/selections.yaml`, in order). Files are read with
`root://cmseos.fnal.gov//`, as `make_fileset(..., replace_xcache=True)` gives them.

## Command lines

`W` is a work directory, `S` this directory, `PY` the notebook venv's python (`sidm_venv/bin/python` of the
checkout on LPC); a valid VOMS proxy in `X509_USER_PROXY`.

```bash
# 1. task lists
python3 $S/make_tasks.py --stage first > $W/tasks_first.txt     # 352 files
python3 $S/make_tasks.py --stage full  > $W/tasks_full.txt      # 34,944 files

# 2. rows, interactively (8 processes; files already written are skipped, so runs can overlap or resume)
nice $PY -I $S/xclean_rows.py $W/rows $W/tasks_first.txt 8
grep -E "_200GeV_1p2GeV_4p8mm |_1p2GeV_0p0" $W/tasks_full.txt > $W/tasks_eight_sets.txt # 2,304 files
nice $PY -I $S/xclean_rows.py $W/rows $W/tasks_eight_sets.txt 8

# 3. the other column samples on condor (96 jobs of ~340 files; worker environment as condor/run_job.sh)
grep -vE "_200GeV_1p2GeV_4p8mm |_1p2GeV_0p0" $W/tasks_full.txt > $W/tasks_rest.txt       # 32,640 files
C=$W/condor; mkdir -p $C/lists $C/out $C/logs
cp $S/condor/{run_xclean.sh,xclean.sub} <repo>/condor/{requirements.txt,constraints.txt} $C/
(cd <repo> && tar -czf $C/xclean_code.tar.gz --exclude=__pycache__ sidm/__init__.py sidm/definitions sidm/tools \
    sidm/configs/selections.yaml sidm/studies/run3_signal_validation/format_comparison/xclean_rows.py)
(cd $C && split -n l/96 -d -a 3 ../tasks_rest.txt lists/job_ && for f in lists/job_*; do mv $f $f.txt; done \
    && ls lists | sed 's/job_\(.*\)\.txt/\1/' > jobs.txt && condor_submit xclean.sub)
# when done: unpack, keeping rows already written
for t in $C/out/rows_*.tar; do tar -xf $t -C $W --skip-old-files rows; done

# 4. tables
$PY -I $S/xclean_first_table.py $W/rows > $W/first_table.md           # the first comparison's table
$PY -I $S/xclean_stats.py $W/rows $W/xclean_tables.md                 # all tables and the verdicts of the rule
$PY -I $S/xclean_stats.py $W/rows $W/eight_sets.md --eight-sets-only  # the eight sets only
$PY -I $S/xclean_checks.py $W/rows --counts $W/pooled_counts.json      # checks (a)-(c); the per-file count cache is written if absent
$PY -I $S/xclean_where.py $W/rows $W/where.md "2Mu2E displaced" 2Mu2E_200GeV_1p2GeV_4p8mm   # locates a difference; not run
```

## The results file and the rows

`xclean_results.md` is assembled by hand from the outputs of step 4. Its sections from "The eight sets on all
events" to "Verdicts of the decision rule" are `$W/xclean_tables.md`, unedited; the table under "Reproduction of
the first comparison's table" is `$W/first_table.md`; the block under "Checks on the rows" is the printout of
`xclean_checks.py`. The introduction and the sections "Verdicts", "Not tested" and "Method" are written by hand.

The rows (34,944 `.npz` files, 1.4 GB) are not in the repository; steps 1 to 3 remake them. The copy behind
`xclean_results.md` is one tar file on the group EOS area (1,371,023,360 bytes, adler32 `1f78dff7`). It unpacks
to `$W/rows`, one directory per production:

```bash
xrdcp root://cmseos.fnal.gov//store/group/lpcmetx/SIDM/run3_samplegen/format_comparison/xclean_rows.tar $W/
tar -xf $W/xclean_rows.tar -C $W
```

## Definitions

- First comparison: the eight-set table made on 2026-11-12 with 20,000 displaced and 12,000 prompt-like events
  per production in 2022 and 2023, in which the 2Mu2E displaced set passed the veto 0.92 +- 0.31 points less
  often in v2 than in v1. It is not in the repository as such; `xclean_first_table.py` remakes its 40 values.
- "pre" DSA muon: the `ljsource_cuts` dsaMuons cuts before the veto (pT > 10 GeV, abs(eta) < 2.4, displaced ID).
- Displaced set: `<chan>_200GeV_1p2GeV_4p8mm`; the first comparison used its files j040-j059 (20,000 events),
  this study all 120 (100,000). Prompt-like set: the six grid points `<chan>_{100,150,200,500,800,1000}GeV_1p2GeV_`
  `{0p096,0p064,0p048,0p019,0p012,0p0096}mm`, 4 files of 500 events each, all of their events in both.
- Column samples: the 36 points of `../v2_census/production/reduced_points.txt`, 120 files (100,000 events) per era
  in both productions.
- Uncertainties: binomial per "pre" muon, the productions independent; `s_evt` clusters the variance by event.
  Combinations are inverse-variance weighted with the binomial uncertainties.
