# Format comparison: the DSA-PF segment-match veto in the two Run 3 signal productions

Compares the pass fraction of the DSA-muon cross-cleaning veto (`"all"` of `ljsource_cuts`, and `"all + charge"`)
between the first Run 3 signal production (v1, LLPNanoAOD, `run3_samplegen/outputs/<era>/`) and the second (v2,
EXONanoAOD, `run3_samplegen/outputs_v2/<era>/`). In v2 the stored `DSAMuon_nSegments` is wrong (twice
`nDTSegments`), so v2 uses the rebuilt count `nDTSegments + ceil(trkNumCSCHits / 6)`; v1 uses the stored count.

| script | runs on | does |
|---|---|---|
| `make_tasks.py` | login node, ~30 s | file lists (`VERSION ERA SAMPLE URL` per line) from the location YAMLs: `--stage step1` the files of the 2026-11-12 comparison, `--stage full` every file of the 36 column samples and of the 12 prompt-like points in 2022, 2022EE, 2023, 2023BPix |
| `xclean_rows.py` | login node or condor, ~1 s per file and process | per file, one `.npz` with the "pre" DSA muons (pT > 10 GeV, abs(eta) < 2.4, displaced ID): kinematics, hit and segment counts, the five `muonMatch` slot counts, the number of good matched PF muons, the veto decisions for the stored, rebuilt and rebuilt + 1 segment counts, and the gen Lxy and dR of the nearest gen muon from a dark-photon decay; per event the numbers of DSA, "pre" DSA and good PF muons |
| `xclean_stats.py` | login node, minutes | the tables: per set and era v1, v2, v2 - v1 with binomial and event-clustered uncertainties, the pulls, inverse-variance combinations over eras and samples, chi2 against zero, and the verdicts of the decision rule |
| `condor/` | condor | `run_xclean.sh` and `xclean.sub`: `xclean_rows.py` on a task list per job |

The veto is evaluated with the cut lambdas of `sidm/definitions/cuts.py` themselves, on a `good_matched_muons`
collection built as `sidm/tools/sidm_processor.py` builds it (the PF muons of the five match slots, `ak.drop_none`,
then the `muons` cuts of `ljsource_cuts` in `sidm/configs/selections.yaml`, in order). Files are read with
`root://cmseos.fnal.gov//`, as `make_fileset(..., replace_xcache=True)` gives them.

## Command lines

`W` is a work directory, `S` this directory, `PY` the notebook venv's python
(`/uscms_data/d3/murtazas/SIDM/sidm_venv/bin/python`); a valid VOMS proxy in `X509_USER_PROXY`.

```bash
# 1. task lists
python3 $S/make_tasks.py --stage step1 > $W/tasks_step1.txt     # 352 files
python3 $S/make_tasks.py --stage full  > $W/tasks_full.txt      # 34,944 files

# 2. rows, interactively (8 processes; files already written are skipped, so runs can overlap or resume)
nice $PY -I $S/xclean_rows.py $W/rows $W/tasks_step1.txt 8
grep -E "_200GeV_1p2GeV_4p8mm |_1p2GeV_0p0" $W/tasks_full.txt > $W/tasks_step2.txt      # 2,304 files
nice $PY -I $S/xclean_rows.py $W/rows $W/tasks_step2.txt 8

# 3. the other column samples on condor (96 jobs of ~340 files; worker environment as condor/run_job.sh)
grep -vE "_200GeV_1p2GeV_4p8mm |_1p2GeV_0p0" $W/tasks_full.txt > $W/tasks_step3rest.txt  # 32,640 files
C=$W/condor; mkdir -p $C/lists $C/out $C/logs
cp $S/condor/{run_xclean.sh,xclean.sub} <repo>/condor/{requirements.txt,constraints.txt} $C/
(cd <repo> && tar -czf $C/xclean_code.tar.gz --exclude=__pycache__ sidm/__init__.py sidm/definitions sidm/tools \
    sidm/configs/selections.yaml sidm/studies/run3_signal_validation/format_comparison/xclean_rows.py)
(cd $C && split -n l/96 -d -a 3 ../tasks_step3rest.txt lists/job_ && for f in lists/job_*; do mv $f $f.txt; done \
    && ls lists | sed 's/job_\(.*\)\.txt/\1/' > jobs.txt && condor_submit xclean.sub)
# when done: unpack, keeping rows already written
for t in $C/out/rows_*.tar; do tar -xf $t -C $W --skip-old-files rows; done

# 4. tables
$PY -I $S/xclean_stats.py $W/rows $W/xclean_tables.md                 # all steps
$PY -I $S/xclean_stats.py $W/rows $W/step2_tables.md --step2-only     # the eight sets only
```

## Definitions

- "pre" DSA muon: the `ljsource_cuts` dsaMuons cuts before the veto (pT > 10 GeV, abs(eta) < 2.4, displaced ID).
- Displaced set: `<chan>_200GeV_1p2GeV_4p8mm`; the 2026-11-12 comparison used its chunks j040-j059 (20,000 events),
  this study all 120 (100,000). Prompt-like set: the six grid points `<chan>_{100,150,200,500,800,1000}GeV_1p2GeV_`
  `{0p096,0p064,0p048,0p019,0p012,0p0096}mm`, 4 files of 500 events each, all of their events in both.
- Column samples: the 36 points of `../v2_census/production/reduced_points.txt`, 120 files (100,000 events) per era
  in both productions.
- Uncertainties: binomial per "pre" muon, the productions independent; `s_evt` clusters the variance by event.
  Combinations are inverse-variance weighted with the binomial uncertainties.
