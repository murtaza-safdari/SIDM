"""Pre-flight the notebook code paths on real merged data (stub bkg sumw_pre).

Exercises every load/plane/stat path the notebooks use so API teething errors are
found BEFORE the real execution. Numbers printed here are NOT results (bkg uses a
stubbed denominator until the condor census lands).
"""
import json
import os
import sys

import numpy as np
import yaml

sys.path.insert(0, os.getcwd())
import abcd_tools as at
import study_setup as ss

failures = []


def check(name, fn):
    try:
        r = fn()
        print(f"PASS {name}" + (f" :: {r}" if isinstance(r, str) else ""))
    except Exception as e:
        print(f"FAIL {name} :: {type(e).__name__}: {e}")
        failures.append(name)


# --- stub bkg sumw_pre from yaml skim factors (code-path exercise only) ---
SUMW_PRE = {}
SUMW_PRE.update(at.census_sumw_pre(ss.CENSUS_SIGNAL_2MU2E))
SUMW_PRE.update(at.census_sumw_pre(ss.CENSUS_SIGNAL_4MU))
bkg_yaml = yaml.safe_load(open("../../configs/ntuples/backgrounds.yaml"))
skim = bkg_yaml["skimmed_llpNanoAOD_v2"]["samples"]
for s in ss.BACKGROUNDS:
    ssw = ss.fetch(s)["metadata"]["scaled_sum_weights"]
    SUMW_PRE[s] = ssw / skim[s].get("skim_factor", 1.0)   # STUB

check("signal census sumw_pre count", lambda: f"{len([k for k in SUMW_PRE if k.startswith(('2Mu2E','4Mu'))])} signal entries")

total_bkg, by_process = ss.accumulate_normalized(list(ss.BACKGROUNDS), SUMW_PRE)
print(f"accumulated {len(ss.BACKGROUNDS)} bkg; scan hists: {len(total_bkg)}")

# --- every plane extracts, both parities, both prescriptions ---
for ch in ss.PLANES:
    for pname in ss.PLANES[ch]:
        for par in (0, 1):
            check(f"plane {ch}/{pname} parity{par}",
                  lambda ch=ch, p=pname, par=par: str(ss.plane_arrays(total_bkg, ch, p, parity=par)[0].shape))
        check(f"plane {ch}/{pname} presc ii",
              lambda ch=ch, p=pname: str(ss.plane_arrays(total_bkg, ch, p, parity=0, prescription="ii")[0].sum().round(2)))

# --- region sums + closure + fit on the incumbent planes ---
for ch, pname in [("2mu2e", "P1_iso_iso"), ("4mu", "Q1_iso_iso")]:
    spec = ss.PLANES[ch][pname]
    vals, var, xe, ye = ss.plane_arrays(total_bkg, ch, pname, parity=0)
    reg = at.region_sums(vals, var, xe, ye, spec["xspec"], spec["yspec"])
    r, vr = at.closure_ratio(reg)
    ne = {k: round(at.n_eff(*reg[k]), 1) for k in "ABCD"}
    print(f"INFO {ch}/{pname}: yields " + " ".join(f"{k}={reg[k][0]:.3g}" for k in "ABCD")
          + f" | R={r:.3f}+-{np.sqrt(max(vr,0)):.3f} | n_eff={ne}   [STUB NORM]")
    check(f"factorization fit {ch}/{pname}",
          lambda v=vals, w=var: f"p={at.factorization_fit(v, w)['pvalue']:.3f}")
    check(f"bootstrap {ch}/{pname}",
          lambda v=vals, w=var, xe=xe, ye=ye, s=spec: str(at.bootstrap_closures(
              v, w, xe, ye, [dict(xspec=s["xspec"], yspec=s["yspec"])], n_boot=50)[0].round(3)))

# --- signal side ---
sig_one = ss.load_normalized("2Mu2E_500GeV_1p2GeV_1p9mm", SUMW_PRE)[0]
svals, svar, sxe, sye = ss.plane_arrays(sig_one, "2mu2e", "P1_iso_iso", parity=0)
sreg = at.region_sums(svals, svar, sxe, sye, ("lt", 0.25), ("lt", 0.10))
print(f"INFO signal 2Mu2E_500GeV_1p2GeV_1p9mm P1 regions: "
      + " ".join(f"{k}={sreg[k][0]:.3g}" for k in "ABCD"))
check("leakage", lambda: str({k: round(v, 4) for k, v in at.leakage_ratios(sreg).items()}))
check("asimov", lambda: f"Z={at.asimov_z(sreg['A'][0], 10.0, 3.0):.3f}")

print(f"\n===== PREFLIGHT: {'ALL PASS' if not failures else str(len(failures)) + ' FAILURES'} =====")
sys.exit(1 if failures else 0)
