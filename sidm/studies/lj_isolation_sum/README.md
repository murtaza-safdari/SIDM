# Lepton-jet isolation: cone-sum and soft-jet variants (parked study)

Status: parked, August 2026. The production isolation is unchanged. This
branch keeps the implementation and the MC evaluation so the study does not
have to be redone.

## Why

The production LJ isolation matches each lepton jet to the nearest AK4 PF
jet within dR < 0.4 and takes (E_jet / E_LJ) x (1 - f_lep), where f_lep is
the jet's charged-EM + neutral-EM + muon energy fraction, so the numerator
is the hadronic energy of that one jet. Two worries motivated this study:
LJs with no matched jet get isolation = 0 (the most signal-like value, via
fill_none), and nothing guards against a match to a low-energy jet far below
the LJ energy.

## What was implemented (sidm/tools/sidm_processor.py, build_lepton_jets)

- isolation_sum: sum over ALL AK4 jets with dR(jet, LJ) <= 0.4 of
  E_jet x (1 - f_lep), divided by E_LJ. Anti-kT assigns each PF candidate to
  exactly one jet, so the terms never double count. Per-jet clamp at zero
  (f_lep can exceed 1 because the fractions refer to raw jet energy).
- isolation_soft: the same cone sum over CorrT1METJet (the AK4 jets below
  the NanoAOD Jet threshold, kept for type-1 MET), muon-subtracted via
  rawPt x (1 - muonSubtrFactor), massless four-vectors.
- isolation_sum_soft = isolation_sum + isolation_soft.
- n_cone_jets, n_cone_softjets: jet counts in the cone (explicit
  no-activity category instead of a hidden sentinel).
- CorrT1METJet registered as preLj object "softjets" (objects.py); 35
  histograms (hists.py) and the hist collection "ljiso_study".

## Validation

- Closure: for LJs with exactly one jet in the cone, isolation_sum equals
  the production isolation to 1.5e-7 (max over 1349 LJs).
- P(matched jet) equals P(n_cone_jets >= 1) exactly (0 disagreements in
  2332 LJs); dR <= 0.4 matches coffea nearest's <= threshold.
- Jet and CorrT1METJet are geometrically disjoint: 0 of 14253 soft jets
  within dR < 0.4 of a stored Jet (signal file), 0 of 711 (QCD file).
- No NaN or inf in any new field.

## MC evaluation (run_eval.py, analyze.py; outputs in metrics.json)

17 samples, 220 files, 511k events, unweighted. Signal: MBs = 500 GeV,
MDp in {0.25, 1.2, 5} GeV x {shortest, longest} ctau, both channels, 2
files per point. Backgrounds: QCD MuEnriched Pt80To120 / Pt300To470 /
Pt1000, DYJetsToMuMu_M50, TTJets.

Coverage of the cone (base_ljObjCut channel, mu-type LJs):

| population | P(jet in cone) | P(jet or soft jet) | gain |
|---|---|---|---|
| signal, short ctau | 0.9996 | 0.9996 | 0.000 |
| signal, long ctau  | 0.744  | 0.848  | 0.104 |
| backgrounds        | 0.9998 | 0.9998 | 0.000 |
| egm-type LJs (all) | 1.000  | 1.000  | 0.000 |

Background efficiency at fixed signal efficiency (mu-type LJs, all
backgrounds, base_ljObjCut):

| signal set | variable | eps_s = 0.7 | 0.8 | 0.9 |
|---|---|---|---|---|
| all points | production isolation | 0.064 | 0.086 | 0.124 |
| all points | isolation_sum        | 0.063 | 0.084 | 0.122 |
| all points | isolation_sum_soft   | 0.065 | 0.087 | 0.125 |
| long ctau  | production isolation | 0.079 | 0.111 | 0.155 |
| long ctau  | isolation_sum_soft   | 0.103 | 0.132 | 0.164 |

At the production working points the three variables are equivalent:
mu-LJ iso < 0.10 gives eps_s 0.958 with eps_b 0.165 / 0.164 / 0.166, and
egm-LJ iso < 0.20 gives 0.911 / 0.552 identically.

## Findings

1. At R = 0.4 the cone essentially never holds more than one AK4 jet
   (1 case in 4704 LJs), so isolation_sum is numerically the production
   variable. It is a cleaner formulation (honest zero, no option types,
   sound "activity in the cone" reading), not a different discriminant.
2. The no-matched-jet population is signal-only on MC: backgrounds and
   egm-type LJs are matched at essentially 100%, only long-ctau signal
   mu-LJs drop to 74-83%. The fill_none(..., 0) convention therefore helps
   signal on MC.
3. The soft-jet term fires only on long-ctau signal mu-LJs (pileup and
   underlying-event jets near clean displaced muons), never on backgrounds,
   so including it in the isolation costs about 30% more background at
   eps_s = 0.7 for long-ctau signal. Rejected as a cut variable.
4. A separate, blinded data pass (window-vetoed control channels, not part
   of this branch) found the 4mu no-matched-jet population in data to be
   essentially free of soft-jet activity as well, so the soft-jet term is
   not a handle on it either. The dimuon-vertex fit remains the tool for
   that population.

Decision: keep the production isolation as is. isolation_sum could be
adopted as a definition cleanup at any time (identical selections, explicit
no-activity category); the soft-jet fields stay diagnostic only.

## Files

- run_eval.py, analyze.py: the evaluation run and the analysis that
  produced metrics.json and the plots.
- metrics.json, run_summary.json: numbers behind the tables above.
- Plots are not committed (*.png is ignored repo-wide). Regenerate them with
  analyze.py from the evaluation output ljiso_eval.coffea, kept on LPC at
  /uscms_data/d3/murtazas/ljiso_study/ together with the original plots.
