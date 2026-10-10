"""Add the proxy-muon validation channels and histograms to the campaign code tree (idempotent)."""
import os
import re
import sys

CODE = sys.argv[1] if len(sys.argv) > 1 else "/uscms_data/d3/murtazas/abcd8d/code"
MARK = "# abcd8d_proxy additions"


def patch(path, addition, anchor=None):
    with open(path) as f:
        s = f.read()
    if MARK in s:
        print(f"{path}: already patched")
        return
    if anchor is None:
        s = s.rstrip("\n") + "\n\n" + MARK + "\n" + addition
    else:
        assert anchor in s, (path, anchor)
        s = s.replace(anchor, anchor + MARK + "\n" + addition, 1)
    with open(path, "w") as f:
        f.write(s)
    print(f"{path}: patched")


patch(os.path.join(CODE, "sidm/definitions/objects.py"),
      'preLj_objs["bjets"] = lambda evts: evts.Jet\n'
      'postLj_objs["proxy_mus"] = lambda objs: objs["ljs"][objs["ljs"].muon_n == 1]\n')

patch(os.path.join(CODE, "sidm/definitions/cuts.py"), '''obj_cut_defs["bjets"] = {
    "btagDeepFlavB_tight": lambda objs: objs["bjets"].btagDeepFlavB >= 0.7100,
    "|eta| < 2.4": lambda objs: abs(objs["bjets"].eta) < 2.4,
    "pT > 30 GeV": lambda objs: objs["bjets"].pt > 30,
}
evt_cut_defs.update({
    "no 2-muon LJs": lambda objs: ak.sum(objs["ljs"].muon_n >= 2, axis=-1) == 0,
    "= 1 2-muon LJs": lambda objs: ak.sum(objs["ljs"].muon_n >= 2, axis=-1) == 1,
    ">=1 proxy muons": lambda objs: ak.num(objs["proxy_mus"]) >= 1,
    "no egm_ljs": lambda objs: ak.num(objs["egm_ljs"]) == 0,
})
''')

sel_path = os.path.join(CODE, "sidm/configs/selections.yaml")
with open(sel_path) as f:
    sel = f.read()
m = re.search(r"abcd6d_inclusive_2mu2e_dxySpread_cosAlpha_mu_veto_eHEM:\n  obj_cuts:\n(.*?)  postLj_obj_cuts:\n", sel, re.S)
assert m, "inclusive 2mu2e channel not found"
obj_cuts = m.group(1)
assert "    dsaMuons:\n" in obj_cuts and "bjets" not in obj_cuts
bjets = "    bjets:\n    - pT > 30 GeV\n    - '|eta| < 2.4'\n    - btagDeepFlavB_tight\n"
post_lj = """  postLj_obj_cuts:
    ljs:
    - pT > 30 GeV
    - '|eta| < 2.4'
    mu_ljs:
    - Mu >= 2
    - dzSpread_mu <= 50
    - dxySpread_mu <= 50
    - dzSpread_pf <= 10
    - dxySpread_pf <= 10
    egm_ljs: []
"""
channels = f"""abcd_proxy_2mu2e_eHEM:
  obj_cuts:
{obj_cuts}{bjets}{post_lj}  evt_cuts:
  - pass triggers
  - PV filter
  - pass flags
  - '>=2 muons (pf or dsa) pT > 26 GeV'
  - no 2-muon LJs
  - '>=1 proxy muons'
  - '>=1 egm_ljs'
  - all cosAlpha(mu, mu) > -0.99
abcd_proxy_4mu_eHEM:
  obj_cuts:
{obj_cuts}{bjets}{post_lj}  evt_cuts:
  - pass triggers
  - PV filter
  - pass flags
  - '>=2 muons (pf or dsa) pT > 26 GeV'
  - = 1 2-muon LJs
  - '>=1 mu_ljs'
  - no egm_ljs
  - '>=1 proxy muons'
  - all cosAlpha(mu, mu) > -0.99
"""
patch(sel_path, channels)

patch(os.path.join(CODE, "sidm/definitions/hists.py"),
      "from sidm.definitions.abcd8d_proxy import register as _register_abcd8d_proxy\n_register_abcd8d_proxy(hist_defs)\n")

patch(os.path.join(CODE, "sidm/configs/hist_collections.yaml"),
      "abcd8d_proxy:\n- abcd8d_proxy_2mu2e\n- abcd8d_proxy_4mu\n")
