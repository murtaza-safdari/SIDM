# Phase 3 (data sidebands) — safety architecture

Branch `abcd-data-sidebands` (worktree SIDM-wt-abcd-data). Everything data-related
lives here so a grave error (accidental unblinding) is erased by deleting: this
branch + worktree, the dedicated EOS output dir, and the local cache dir. Nothing
merges to the analysis branch until the data round is cleared.

The branch is pushed to the personal fork (`murtaza-safdari/SIDM`) for inspection.
This is blinding-safe: git holds only code and config (the blind-box mask, the
data channels, the interlock, input-file lists) — no histogram values and no event
data. All data yields live on EOS, never in git. The SR box is masked at processor
level before any fill, so even the on-EOS outputs contain no SR data.

## Blinding rules (all enforced BEFORE any histogram fill)
1. SR mask at processor level for data: drop events in the tight-iso x high-mJJ
   box of the declared plane (muiso x mJJ), defined as the UNION of region A over
   every working point that will be examined in data (i.e. A at the loosest ladder
   rung), so no examined rung can expose SR data. MC is never masked.
2. Safe fraction: deterministic 10% of events (event_number % 10 == 0), uniform
   across eras; documented, reproducible, extensible to other deciles later.
3. Data-safe histogram collection (no gen branches: no genPartFlav mothers, no
   gen weights); pipeline runs unweighted for data.
4. Every data notebook asserts masked-region emptiness on load (tripwire) and
   never prints SR-adjacent data counts.

## Gates before the FIRST data condor submission — CLEARED 2026-07-12
- [x] Notebooks 05/06 finalized (veto threshold candidates from MC cost curves)
- [x] Mask verified on MC: region-A bins exactly empty; B/C/D identical to unmasked
      (unit tests 14/14, MC smoke, and one real data file all pass)
- [x] Adversarial review of the mask implementation passed (verdict was
      DO-NOT-SUBMIT; both HIGH leaks fixed — see below — then re-verified)
- [x] Safety checklist shown to analyst; explicit submit confirmation received
      ("Phase 3 is a go")
First campaign: DoubleMuon 2018A, cluster 59757788 (186 chunks), merged to
/store/group/lpcmetx/SIDM/coffea_outputs/murtazas/abcd_data_2018A. Blinding
verified on the merged output: 2mu2e SR A-box = 0.0, 979 sidebands.

## Deletion procedure (if grave error)
ssh lpc:  git -C /uscms_data/d3/murtazas/SIDM-wt-abcd worktree remove --force \
          /uscms_data/d3/murtazas/SIDM-wt-abcd-data && \
          git -C /uscms_data/d3/murtazas/SIDM-wt-abcd branch -D abcd-data-sidebands \
          && git push origin --delete abcd-data-sidebands
EOS:      xrdfs root://cmseos.fnal.gov rm -r <dedicated output dir>
Local:    rm -rf the dedicated cache dir.

## Blinding-review fixes (2026-07-12, adversarial review verdict was DO-NOT-SUBMIT)
Two HIGH potential-SR-leak paths found and fixed before any data run:
1. **in_channel factor removed** from `_abcd_blind_box` (cuts.py). It had used the
   `abcd_mask_*` topology functions — a second copy of the channel-cut logic that,
   if it ever diverged from the "2mu2e"/"4mu" selection cut, could NARROW the veto
   and leak SR. The box is now a pure superset of region A
   (muiso<0.5 & mjj>=50 on the exact fill quantities); topology is enforced by the
   channel cut that already precedes the veto in every *_data selection.
2. **Blinding interlock added** to `run_sidm_chunk.py` (`--is-data`). When is_data,
   it refuses any channel whose resolved evt_cuts lack "ABCD SR blind box veto (data)"
   and any collection not in the data-safe allowlist {abcd_data}, failing loud before
   a single event is read. Closes the "misrouted/typo'd data submission unblinds"
   path. Verified: trips on a non-data channel and on a non-data collection; passes
   on the correct combination.

## OVER-BLINDING CONSTRAINT (MEDIUM finding — bias, not a leak)
The blind box is a full quadrant = region A at the LOOSEST (t=2.0) rung. At tighter
ladder rungs (and at the nominal declared WP muiso<0.25 & mjj>=150) the sidebands
B (muiso in [0.25,0.5), mjj>=150) and C (muiso<0.5, mjj in [50,150)) fall INSIDE the
box and are blinded in data too. Consequence: a DATA ABCD prediction is valid ONLY
at the t=2.0 (SR) rung — its B/C/D are disjoint from the box. A tighter-rung closure
computed on data would be silently biased LOW (no error raised). ENFORCEMENT: the
data analysis notebooks (Phase 3) must restrict to the t=2.0 rung; add a data-mode
guard to staged_points() when those notebooks are built. Documented in the
_abcd_blind_box docstring too.

## Data job-args must pass BOTH --unweighted-hist AND --is-data
(data has no gen weights; --is-data enables the interlock + unweighted running.)
