# Cosmic veto — vertex-consistency (collinearity) follow-up

Design note for a constructive follow-up to the cosmic-veto due diligence
(notebook 09). It proposes one additional diagnostic variable, states what it
should cost and buy, and defers the launch to when the pool eases. No cut is
adopted here; this is the plan for the measurement.

## Where the due diligence left the veto

- **cos(alpha) < -0.98** on the leading mu-LJ (partner pool = event muons not
  inside the two leading LJs) is the recommended veto: it costs essentially no
  signal anywhere (0.00-0.14% across the ctau grid) while tagging the cosmic
  topology (two back-to-back muons).
- **Raw dz / vxy spread** (max-min of the LJ muon reference points) was tested
  and set aside as a standalone or add-on cut: it cuts on spatial *separation
  alone*, and displaced signal muons are *legitimately* separated, so it removes
  6-10% of prompt signal rising to ~90% at ctau ~ 80-96 mm. It is not
  cosmic-specific.
- cos(alpha) has one known blind spot: a two-leg cosmic reconstructed as exactly
  the two leading mu-LJs with no third muon leaves the partner pool empty, so
  min_cosalpha = 1 and the event is not tagged. In 4mu the two signal dark-photon
  LJs are themselves back-to-back, so we cannot simply cut on the two leading LJs
  being back-to-back without self-vetoing signal.

## The idea: separation that is *cosmic-collinear*, not just large

A cosmic is one physical track traversing the detector, mis-reconstructed as two
muons. Its two legs therefore lie on **one straight line**: the vector joining
their points of closest approach is collinear with the (anti-parallel) track
momenta. Two muons from a common (possibly displaced) production vertex instead
**diverge** from that point, so their joining vector is *not* collinear with
either momentum.

Define, for a muon pair with PCA reference points r1, r2 and momenta p1, p2:

    dr   = r1 - r2                         (joining vector)
    coll = | (dr / |dr|) . (p1 / |p1|) |   (collinearity, in [0,1])

- **Cosmic**: |dr| large AND coll -> 1  (separation lies along the track).
- **Signal**: coll < 1  (muons diverge from a vertex), regardless of how large
  |dr| is for a displaced vertex.

The point is that **coll makes "large separation" cosmic-specific**. A cut of the
form (cos(alpha) < -0.98) AND (|dr| large) AND (coll -> 1) targets exactly the
one-track topology while sparing displaced signal, which the raw spread could not
do. Applied to the two leading mu-LJs it also plugs the cos(alpha) blind spot
(two back-to-back mu-LJs, no third muon): signal dark photons fail the collinearity
requirement, a cosmic passes it.

## Why this respects the simplicity principle

- Built entirely from **already-stored quantities** — DSA-muon vx, vy, vz and the
  momentum (pt, eta, phi). No Kalman/adaptive vertex fit, no refitted vertex, no
  new POG-approved object.
- Computed on **DSA constituents**: they carry a full 3D reference point (vx,vy,vz);
  PF muons store innerVx/innerVy but no 3D PCA z, and the blind-spot cosmic
  topology is DSA-dominated (notebook 09: DSA fraction of mu-LJs is 52-68% for
  ctau >= 19 mm). This keeps the variable well defined and cheap.
- It is a **kinematic/geometric tag**, so it needs no data-driven scale factor or
  efficiency to carry into the limit — the same reason cos(alpha) was preferred.

## What to measure (before adopting anything)

Add coll as a per-event diagnostic to the existing cosmic block (alongside
min_cosalpha, dz_spread, vxy_spread) and fill it into the cosmic-diagnostic
histogram collection. Then:

1. **Signal cost** — on signal MC across the ctau grid: the fraction of signal
   removed by the candidate conjunction cut. Must stay near the cos(alpha) level
   (sub-percent), not the spread level, or the variable buys nothing.
2. **Cosmic rejection** — in the cosmic-enriched data control region (the
   cos(alpha)-tagged data sideband already built): does coll -> 1 concentrate
   where cos(alpha) -> -1, confirming the one-track picture, and does it tag the
   blind-spot events cos(alpha) alone misses.
3. **Verdict** — adopt only if it adds rejection at negligible signal cost; if not,
   cos(alpha) alone stands (simplicity default).

## Proposed processor delta (representative)

Inserted after the min_cosalpha block in make_lepton_jets. Between the two
pt-leading mu-LJs, for the most back-to-back DSA-muon pair (one from each LJ),
carry both the joining-vector magnitude and its collinearity:

    d = ljs.dsaMuons                              # event x LJ x constituent
    # leading two mu-LJs -> muon a in LJ0, muon b in LJ1 (broadcast over pairs)
    # dr = r_a - r_b ; coll = |dr_hat . p_a_hat|
    # ljs["lj2_dsep"]  = |dr| of the most back-to-back pair
    # ljs["lj2_coll"]  = coll of that pair  (1 = one-track/cosmic)

(Exact awkward broadcasting finalized at implementation; the leading-LJ pair
bookkeeping mirrors the existing lead2 / extra-muon construction so signal LJ
constituents are handled consistently.)

## Status

Designed, launch **held**. The signal-cost + cosmic-rejection jobs are small
(signal MC + the existing cosmic CR), not a full-Run2018 campaign, but they should
wait until the Full Run2018 drain finishes so they do not compete for slots. On
launch: add the diagnostic, rebuild the code tarball, run signal MC + cosmic CR,
then evaluate against the two thresholds above.
