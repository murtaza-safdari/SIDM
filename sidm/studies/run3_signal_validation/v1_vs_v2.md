# v1 (LLPNanoAOD) against v2 (EXONanoAOD): every number of the v1 outputs

**Summary.** 135 Run 3 quantities were compared with an n σ value (v1 and v2 both with a statistical uncertainty; the 25 rows on the 2018 reference read the same files in both productions and are not counted). 5 lie beyond 2σ: sqrt_s_frac_below_tail_cut (2022EE) -2.1σ; vtx_pull_x (2023) -2.4σ; pileup_nTrueInt_mean (2023BPix) +2.9σ; vtx_pull_x (2023BPix) -2.2σ; 4Mu_incl_acc.mean_pct (Run3-2018) (2022, geom mode) +2.2σ. The largest is 2.9σ (pileup_nTrueInt_mean, 2023BPix). About 6 would be expected beyond 2σ for Gaussian errors. Of the region averages compared over the first production's bins, 0 of 38 lie beyond 2σ.

Written by `v1_vs_v2.py` in this directory; uncertainty definitions in its docstring. Era `-` = a property of the region, not of one era. 2024 rows: v2 only. `regions (v1 bins)`: v1 numbers against v2 read over the same (v1) flagged bins; `regions (own flagged bins)`: v2 over the bins it flags itself (no sigma: the flagged set selects downward fluctuations). In `regions (v1 bins)` the bins were chosen on v1's own fluctuations (where the second era reads below the first), so v1's flagged averages of the second era lean low and v2's, read over the same bins, do not: a positive v2 - v1 there is partly that selection.


## era_fingerprint.csv

| quantity | era | v1 | σ v1 | v2 | σ v2 | v2 − v1 | n σ | note |
|---|---|---|---|---|---|---|---|---|
| files_per_sample | 2022 | 120 |  | 120 |  | 0 |  |  |
| events | 2022 | 200000 |  | 200000 |  | 0 |  |  |
| sqrt_s_median_GeV | 2022 | 13597.24 |  | 13597.23 |  | -0.01 |  | stored precision |
| sqrt_s_lhe_median_GeV | 2022 | 13599.76 |  | 13599.76 |  | 0 |  | stored precision |
| sqrt_s_tail_cut_GeV | 2022 | 13500.0 |  | 13500.0 |  | 0 |  |  |
| sqrt_s_frac_below_tail_cut | 2022 | 0.02518 | 0.000350328 | 0.025 | 0.000349106 | -0.00018 | -0.4 |  |
| frac_events_pass_2018_req | 2022 | 0.5763 | 0.00110494 | 0.5736 | 0.00110585 | -0.0027 | -1.7 |  |
| sqrt_s_frac_below_tail_cut_after_2018_req | 2022 | 0.0001 | 2.94536e-05 | 9e-05 | 2.8008e-05 | -1e-05 | -0.2 |  |
| lhe_frac_with_outgoing_parton_tail | 2022 | 0.0 | 0 | 0.0 | 0 | 0 |  |  |
| lhe_frac_with_outgoing_parton_core | 2022 | 0.0 | 0 | 0.0 | 0 | 0 |  |  |
| m35_over_lhe_sqrt_shat_median_tail | 2022 | 1.01247 |  | 1.01273 |  | 0.00026 |  | stored precision |
| m35_over_lhe_sqrt_shat_median_core | 2022 | 1.00027 |  | 1.00027 |  | 0 |  | stored precision |
| pileup_nTrueInt_mean | 2022 | 37.46 | 0.0210414 | 37.45 | 0.021019 | -0.01 | -0.3 |  |
| pileup_nTrueInt_std | 2022 | 9.41 | 0.0148785 | 9.4 | 0.0148627 | -0.01 | -0.5 |  |
| genvtx_x_mean_cm | 2022 | 0.10063 |  | 0.10063 |  | 0 |  | rounded to 1e-5 cm in the CSV; compared through vtx_pull_* |
| genvtx_y_mean_cm | 2022 | -0.01499 |  | -0.015 |  | -1e-05 |  | rounded to 1e-5 cm in the CSV; compared through vtx_pull_* |
| genvtx_z_mean_cm | 2022 | 1.2882 | 0.00851271 | 1.2948 | 0.00848275 | 0.0066 | +0.5 |  |
| genvtx_z_std_cm | 2022 | 3.807 | 0.0060194 | 3.7936 | 0.00599821 | -0.0134 | -1.6 |  |
| pv_z_std_cm | 2022 | 3.737 | 0.00590872 | 3.736 | 0.00590713 | -0.001 | -0.1 |  |
| n_hlt_branches | 2022 | 754 |  | 754 |  | 0 |  |  |
| HLT_DoubleL2Mu_L3Mu16NoVtx_VetoL3Mu0DxyMax0p1cm | 2022 | no |  | no |  |  |  |  |
| HLT_DoubleL2Mu10NoVtx_2Cha_CosmicSeed_VetoL3Mu0DxyMax1cm | 2022 | no |  | no |  |  |  |  |
| SIM_release | 2022 | CMSSW_12_4_19 |  | CMSSW_12_4_26 |  | differs |  |  |
| SIM_globaltag | 2022 | 124X_mcRun3_2022_realistic_v12 |  | 124X_mcRun3_2022_realistic_v12 |  |  |  |  |
| HLT_release | 2022 | CMSSW_12_4_16 |  | CMSSW_12_4_26 |  | differs |  |  |
| HLT_globaltag | 2022 | 124X_mcRun3_2022_realistic_v12 |  | 124X_mcRun3_2022_realistic_v12 |  |  |  |  |
| RECO_release | 2022 | CMSSW_12_4_16 |  | CMSSW_12_4_26 |  | differs |  |  |
| RECO_globaltag | 2022 | 124X_mcRun3_2022_realistic_v12 |  | 124X_mcRun3_2022_realistic_v12 |  |  |  |  |
| NANO_release | 2022 | CMSSW_13_0_13 |  | CMSSW_16_0_10 |  | differs |  |  |
| NANO_globaltag | 2022 | 130X_mcRun3_2022_realistic_v5 |  | 130X_mcRun3_2022_realistic_v5 |  |  |  |  |
| SIM_comEnergy_GeV | 2022 | 13600.0 |  | 13600.0 |  | 0 |  |  |
| SIM_vtxsmear_X0_Y0_Z0_SigmaZ_cm | 2022 | 0.100629/-0.014995/1.29815/3.8 |  | 0.100629/-0.014995/1.29815/3.8 |  |  |  |  |
| HLT_hlt_menu | 2022 | /frozen/2022/2e34/v1.2/HLT/V1 |  | /frozen/2022/2e34/v1.2/HLT/V1 |  |  |  |  |
| vtx_pull_x | 2022 | 0.81 | 1 | 0.23 | 1 | -0.58 | -0.4 | (mean - VtxSmeared) / (std/sqrt(N)); VtxSmeared identical in v1 and v2 |
| vtx_pull_y | 2022 | 0.6 | 1 | -0.31 | 1 | -0.91 | -0.6 | (mean - VtxSmeared) / (std/sqrt(N)); VtxSmeared identical in v1 and v2 |
| vtx_pull_z0 | 2022 | -1.17 | 1 | -0.4 | 1 | 0.77 | +0.5 | (mean - VtxSmeared) / (std/sqrt(N)); VtxSmeared identical in v1 and v2 |
| vtx_pull_sigmaz | 2022 | 1.17 | 1 | -1.07 | 1 | -2.24 | -1.6 | (mean - VtxSmeared) / (std/sqrt(N)); VtxSmeared identical in v1 and v2 |
| vtx_check_vs_generator_process | 2022 | SIM: ok |  | SIM: ok |  |  |  |  |
| vtx_match | 2022 | True |  | True |  |  |  |  |
| provenance_first_last_file_match | 2022 | True |  | True |  |  |  |  |
| provenance_same_all_channels | 2022 | True |  | True |  |  |  |  |
| SIM_vtxsmear_readDB | 2022 |  |  | false |  |  |  |  |
| files_per_sample | 2022EE | 120 |  | 120 |  | 0 |  |  |
| events | 2022EE | 200000 |  | 200000 |  | 0 |  |  |
| sqrt_s_median_GeV | 2022EE | 13597.22 |  | 13597.22 |  | 0 |  | stored precision |
| sqrt_s_lhe_median_GeV | 2022EE | 13599.76 |  | 13599.76 |  | 0 |  | stored precision |
| sqrt_s_tail_cut_GeV | 2022EE | 13500.0 |  | 13500.0 |  | 0 |  |  |
| sqrt_s_frac_below_tail_cut | 2022EE | 0.02578 | 0.000354368 | 0.02473 | 0.000347264 | -0.00105 | -2.1 |  |
| frac_events_pass_2018_req | 2022EE | 0.5761 | 0.00110501 | 0.575 | 0.00110538 | -0.0011 | -0.7 |  |
| sqrt_s_frac_below_tail_cut_after_2018_req | 2022EE | 0.0001 | 2.94588e-05 | 8e-05 | 2.63742e-05 | -2e-05 | -0.5 |  |
| lhe_frac_with_outgoing_parton_tail | 2022EE | 0.0 | 0 | 0.0 | 0 | 0 |  |  |
| lhe_frac_with_outgoing_parton_core | 2022EE | 0.0 | 0 | 0.0 | 0 | 0 |  |  |
| m35_over_lhe_sqrt_shat_median_tail | 2022EE | 1.01271 |  | 1.01262 |  | -9e-05 |  | stored precision |
| m35_over_lhe_sqrt_shat_median_core | 2022EE | 1.00027 |  | 1.00027 |  | 0 |  | stored precision |
| pileup_nTrueInt_mean | 2022EE | 37.44 | 0.0210638 | 37.43 | 0.0210638 | -0.01 | -0.3 |  |
| pileup_nTrueInt_std | 2022EE | 9.42 | 0.0148943 | 9.42 | 0.0148943 | 0 | +0.0 |  |
| genvtx_x_mean_cm | 2022EE | 0.10063 |  | 0.10063 |  | 0 |  | rounded to 1e-5 cm in the CSV; compared through vtx_pull_* |
| genvtx_y_mean_cm | 2022EE | -0.015 |  | -0.015 |  | 0 |  | rounded to 1e-5 cm in the CSV; compared through vtx_pull_* |
| genvtx_z_mean_cm | 2022EE | 1.2864 | 0.00850958 | 1.3007 | 0.0084946 | 0.0143 | +1.2 |  |
| genvtx_z_std_cm | 2022EE | 3.8056 | 0.00601718 | 3.7989 | 0.00600659 | -0.0067 | -0.8 |  |
| pv_z_std_cm | 2022EE | 3.746 | 0.00592295 | 3.734 | 0.00590397 | -0.012 | -1.4 |  |
| n_hlt_branches | 2022EE | 616 |  | 616 |  | 0 |  |  |
| HLT_DoubleL2Mu_L3Mu16NoVtx_VetoL3Mu0DxyMax0p1cm | 2022EE | yes |  | yes |  |  |  |  |
| HLT_DoubleL2Mu10NoVtx_2Cha_CosmicSeed_VetoL3Mu0DxyMax1cm | 2022EE | yes |  | yes |  |  |  |  |
| SIM_release | 2022EE | CMSSW_12_4_11_patch3 |  | CMSSW_12_4_26 |  | differs |  |  |
| SIM_globaltag | 2022EE | 124X_mcRun3_2022_realistic_postEE_v1 |  | 124X_mcRun3_2022_realistic_postEE_v1 |  |  |  |  |
| HLT_release | 2022EE | CMSSW_12_4_11_patch3 |  | CMSSW_12_4_26 |  | differs |  |  |
| HLT_globaltag | 2022EE | 124X_mcRun3_2022_realistic_postEE_v1 |  | 124X_mcRun3_2022_realistic_postEE_v1 |  |  |  |  |
| RECO_release | 2022EE | CMSSW_12_4_11_patch3 |  | CMSSW_12_4_26 |  | differs |  |  |
| RECO_globaltag | 2022EE | 124X_mcRun3_2022_realistic_postEE_v1 |  | 124X_mcRun3_2022_realistic_postEE_v1 |  |  |  |  |
| NANO_release | 2022EE | CMSSW_13_0_13 |  | CMSSW_16_0_10 |  | differs |  |  |
| NANO_globaltag | 2022EE | 130X_mcRun3_2022_realistic_postEE_v6 |  | 130X_mcRun3_2022_realistic_postEE_v6 |  |  |  |  |
| SIM_comEnergy_GeV | 2022EE | 13600.0 |  | 13600.0 |  | 0 |  |  |
| SIM_vtxsmear_X0_Y0_Z0_SigmaZ_cm | 2022EE | 0.100629/-0.014995/1.29815/3.8 |  | 0.100629/-0.014995/1.29815/3.8 |  |  |  |  |
| HLT_hlt_menu | 2022EE | /frozen/2022/2e34/v1.4/HLT/V1 |  | /frozen/2022/2e34/v1.4/HLT/V1 |  |  |  |  |
| vtx_pull_x | 2022EE | 1.09 | 1 | -1.3 | 1 | -2.39 | -1.7 | (mean - VtxSmeared) / (std/sqrt(N)); VtxSmeared identical in v1 and v2 |
| vtx_pull_y | 2022EE | -0.85 | 1 | -1.1 | 1 | -0.25 | -0.2 | (mean - VtxSmeared) / (std/sqrt(N)); VtxSmeared identical in v1 and v2 |
| vtx_pull_z0 | 2022EE | -1.39 | 1 | 0.3 | 1 | 1.69 | +1.2 | (mean - VtxSmeared) / (std/sqrt(N)); VtxSmeared identical in v1 and v2 |
| vtx_pull_sigmaz | 2022EE | 0.93 | 1 | -0.18 | 1 | -1.11 | -0.8 | (mean - VtxSmeared) / (std/sqrt(N)); VtxSmeared identical in v1 and v2 |
| vtx_check_vs_generator_process | 2022EE | SIM: ok |  | SIM: ok |  |  |  |  |
| vtx_match | 2022EE | True |  | True |  |  |  |  |
| provenance_first_last_file_match | 2022EE | True |  | True |  |  |  |  |
| provenance_same_all_channels | 2022EE | True |  | True |  |  |  |  |
| SIM_vtxsmear_readDB | 2022EE |  |  | false |  |  |  |  |
| files_per_sample | 2023 | 120 |  | 120 |  | 0 |  |  |
| events | 2023 | 200000 |  | 200000 |  | 0 |  |  |
| sqrt_s_median_GeV | 2023 | 13597.19 |  | 13597.22 |  | 0.03 |  | stored precision |
| sqrt_s_lhe_median_GeV | 2023 | 13599.76 |  | 13599.76 |  | 0 |  | stored precision |
| sqrt_s_tail_cut_GeV | 2023 | 13500.0 |  | 13500.0 |  | 0 |  |  |
| sqrt_s_frac_below_tail_cut | 2023 | 0.02566 | 0.000353564 | 0.02499 | 0.000349038 | -0.00067 | -1.3 |  |
| frac_events_pass_2018_req | 2023 | 0.5738 | 0.00110579 | 0.5764 | 0.00110491 | 0.0026 | +1.7 |  |
| sqrt_s_frac_below_tail_cut_after_2018_req | 2023 | 4e-05 | 1.86692e-05 | 0.00011 | 3.08884e-05 | 7e-05 | +1.9 |  |
| lhe_frac_with_outgoing_parton_tail | 2023 | 0.0 | 0 | 0.0 | 0 | 0 |  |  |
| lhe_frac_with_outgoing_parton_core | 2023 | 0.0 | 0 | 0.0 | 0 | 0 |  |  |
| m35_over_lhe_sqrt_shat_median_tail | 2023 | 1.01255 |  | 1.01279 |  | 0.00024 |  | stored precision |
| m35_over_lhe_sqrt_shat_median_core | 2023 | 1.00027 |  | 1.00027 |  | 0 |  | stored precision |
| pileup_nTrueInt_mean | 2023 | 50.9 | 0.0255583 | 50.9 | 0.0254912 | 0 | +0.0 |  |
| pileup_nTrueInt_std | 2023 | 11.43 | 0.0180724 | 11.4 | 0.018025 | -0.03 | -1.2 |  |
| genvtx_x_mean_cm | 2023 | 0.04586 |  | 0.04585 |  | -1e-05 |  | rounded to 1e-5 cm in the CSV; compared through vtx_pull_* |
| genvtx_y_mean_cm | 2023 | -0.01697 |  | -0.01697 |  | 0 |  | rounded to 1e-5 cm in the CSV; compared through vtx_pull_* |
| genvtx_z_mean_cm | 2023 | -0.0841 | 0.00806662 | -0.0932 | 0.00806639 | -0.0091 | -0.8 |  |
| genvtx_z_std_cm | 2023 | 3.6075 | 0.00570396 | 3.6074 | 0.0057038 | -0.0001 | -0.0 |  |
| pv_z_std_cm | 2023 | 3.518 | 0.00556245 | 3.526 | 0.0055751 | 0.008 | +1.0 |  |
| n_hlt_branches | 2023 | 688 |  | 689 |  | 1 |  |  |
| HLT_DoubleL2Mu_L3Mu16NoVtx_VetoL3Mu0DxyMax0p1cm | 2023 | yes |  | yes |  |  |  |  |
| HLT_DoubleL2Mu10NoVtx_2Cha_CosmicSeed_VetoL3Mu0DxyMax1cm | 2023 | yes |  | yes |  |  |  |  |
| SIM_release | 2023 | CMSSW_13_0_17 |  | CMSSW_13_0_25 |  | differs |  |  |
| SIM_globaltag | 2023 | 130X_mcRun3_2023_realistic_v14 |  | 130X_mcRun3_2023_realistic_v15 |  | differs |  |  |
| HLT_release | 2023 | CMSSW_13_0_14 |  | CMSSW_13_0_25 |  | differs |  |  |
| HLT_globaltag | 2023 | 130X_mcRun3_2023_realistic_v14 |  | 130X_mcRun3_2023_realistic_v15 |  | differs |  |  |
| RECO_release | 2023 | CMSSW_13_0_14 |  | CMSSW_13_0_25 |  | differs |  |  |
| RECO_globaltag | 2023 | 130X_mcRun3_2023_realistic_v14 |  | 130X_mcRun3_2023_realistic_v15 |  | differs |  |  |
| NANO_release | 2023 | CMSSW_13_0_13 |  | CMSSW_16_0_10 |  | differs |  |  |
| NANO_globaltag | 2023 | 130X_mcRun3_2023_realistic_v14 |  | 130X_mcRun3_2023_realistic_v15 |  | differs |  |  |
| SIM_comEnergy_GeV | 2023 | 13600.0 |  | 13600.0 |  | 0 |  |  |
| SIM_vtxsmear_X0_Y0_Z0_SigmaZ_cm | 2023 | 0.0458532/-0.016966/-0.074992/3.6 |  | 0.0458532/-0.016966/-0.074992/3.6 |  |  |  |  |
| HLT_hlt_menu | 2023 | /frozen/2023/2e34/v1.2/HLT/V1 |  | /frozen/2023/2e34/v1.2/HLT/V2 |  | differs |  |  |
| vtx_pull_x | 2023 | 1.25 | 1 | -2.16 | 1 | -3.41 | -2.4 | (mean - VtxSmeared) / (std/sqrt(N)); VtxSmeared identical in v1 and v2 |
| vtx_pull_y | 2023 | -0.68 | 1 | -0.97 | 1 | -0.29 | -0.2 | (mean - VtxSmeared) / (std/sqrt(N)); VtxSmeared identical in v1 and v2 |
| vtx_pull_z0 | 2023 | -1.13 | 1 | -2.25 | 1 | -1.12 | -0.8 | (mean - VtxSmeared) / (std/sqrt(N)); VtxSmeared identical in v1 and v2 |
| vtx_pull_sigmaz | 2023 | 1.31 | 1 | 1.3 | 1 | -0.01 | -0.0 | (mean - VtxSmeared) / (std/sqrt(N)); VtxSmeared identical in v1 and v2 |
| vtx_check_vs_generator_process | 2023 | SIM: ok |  | SIM: ok |  |  |  |  |
| vtx_match | 2023 | True |  | True |  |  |  |  |
| provenance_first_last_file_match | 2023 | True |  | True |  |  |  |  |
| provenance_same_all_channels | 2023 | True |  | True |  |  |  |  |
| SIM_vtxsmear_readDB | 2023 |  |  | false |  |  |  |  |
| files_per_sample | 2023BPix | 120 |  | 120 |  | 0 |  |  |
| events | 2023BPix | 200000 |  | 200000 |  | 0 |  |  |
| sqrt_s_median_GeV | 2023BPix | 13597.21 |  | 13597.18 |  | -0.03 |  | stored precision |
| sqrt_s_lhe_median_GeV | 2023BPix | 13599.76 |  | 13599.76 |  | 0 |  | stored precision |
| sqrt_s_tail_cut_GeV | 2023BPix | 13500.0 |  | 13500.0 |  | 0 |  |  |
| sqrt_s_frac_below_tail_cut | 2023BPix | 0.02518 | 0.000350328 | 0.02493 | 0.000348629 | -0.00025 | -0.5 |  |
| frac_events_pass_2018_req | 2023BPix | 0.5762 | 0.00110497 | 0.5759 | 0.00110508 | -0.0003 | -0.2 |  |
| sqrt_s_frac_below_tail_cut_after_2018_req | 2023BPix | 9e-05 | 2.79447e-05 | 6e-05 | 2.28231e-05 | -3e-05 | -0.8 |  |
| lhe_frac_with_outgoing_parton_tail | 2023BPix | 0.0 | 0 | 0.0 | 0 | 0 |  |  |
| lhe_frac_with_outgoing_parton_core | 2023BPix | 0.0 | 0 | 0.0 | 0 | 0 |  |  |
| m35_over_lhe_sqrt_shat_median_tail | 2023BPix | 1.01277 |  | 1.01291 |  | 0.00014 |  | stored precision |
| m35_over_lhe_sqrt_shat_median_core | 2023BPix | 1.00027 |  | 1.00027 |  | 0 |  | stored precision |
| pileup_nTrueInt_mean | 2023BPix | 51.28 | 0.0272353 | 51.39 | 0.0272129 | 0.11 | +2.9 |  |
| pileup_nTrueInt_std | 2023BPix | 12.18 | 0.0192583 | 12.17 | 0.0192425 | -0.01 | -0.4 |  |
| genvtx_x_mean_cm | 2023BPix | 0.04585 |  | 0.04585 |  | 0 |  | rounded to 1e-5 cm in the CSV; compared through vtx_pull_* |
| genvtx_y_mean_cm | 2023BPix | -0.01697 |  | -0.01696 |  | 1e-05 |  | rounded to 1e-5 cm in the CSV; compared through vtx_pull_* |
| genvtx_z_mean_cm | 2023BPix | -0.0848 | 0.00803598 | -0.0759 | 0.00806103 | 0.0089 | +0.8 |  |
| genvtx_z_std_cm | 2023BPix | 3.5938 | 0.0056823 | 3.605 | 0.00570001 | 0.0112 | +1.4 |  |
| pv_z_std_cm | 2023BPix | 3.521 | 0.00556719 | 3.525 | 0.00557351 | 0.004 | +0.5 |  |
| n_hlt_branches | 2023BPix | 688 |  | 689 |  | 1 |  |  |
| HLT_DoubleL2Mu_L3Mu16NoVtx_VetoL3Mu0DxyMax0p1cm | 2023BPix | yes |  | yes |  |  |  |  |
| HLT_DoubleL2Mu10NoVtx_2Cha_CosmicSeed_VetoL3Mu0DxyMax1cm | 2023BPix | yes |  | yes |  |  |  |  |
| SIM_release | 2023BPix | CMSSW_13_0_14 |  | CMSSW_13_0_25 |  | differs |  |  |
| SIM_globaltag | 2023BPix | 130X_mcRun3_2023_realistic_postBPix_v6 |  | 130X_mcRun3_2023_realistic_postBPix_v6 |  |  |  |  |
| HLT_release | 2023BPix | CMSSW_13_0_14 |  | CMSSW_13_0_25 |  | differs |  |  |
| HLT_globaltag | 2023BPix | 130X_mcRun3_2023_realistic_postBPix_v6 |  | 130X_mcRun3_2023_realistic_postBPix_v6 |  |  |  |  |
| RECO_release | 2023BPix | CMSSW_13_0_14 |  | CMSSW_13_0_25 |  | differs |  |  |
| RECO_globaltag | 2023BPix | 130X_mcRun3_2023_realistic_postBPix_v6 |  | 130X_mcRun3_2023_realistic_postBPix_v6 |  |  |  |  |
| NANO_release | 2023BPix | CMSSW_13_0_13 |  | CMSSW_16_0_10 |  | differs |  |  |
| NANO_globaltag | 2023BPix | 130X_mcRun3_2023_realistic_postBPix_v2 |  | 130X_mcRun3_2023_realistic_postBPix_v6 |  | differs |  |  |
| SIM_comEnergy_GeV | 2023BPix | 13600.0 |  | 13600.0 |  | 0 |  |  |
| SIM_vtxsmear_X0_Y0_Z0_SigmaZ_cm | 2023BPix | 0.0458532/-0.016966/-0.074992/3.6 |  | 0.0458532/-0.016966/-0.074992/3.6 |  |  |  |  |
| HLT_hlt_menu | 2023BPix | /frozen/2023/2e34/v1.2/HLT/V1 |  | /frozen/2023/2e34/v1.2/HLT/V2 |  | differs |  |  |
| vtx_pull_x | 2023BPix | 0.82 | 1 | -2.29 | 1 | -3.11 | -2.2 | (mean - VtxSmeared) / (std/sqrt(N)); VtxSmeared identical in v1 and v2 |
| vtx_pull_y | 2023BPix | 0.29 | 1 | 1.0 | 1 | 0.71 | +0.5 | (mean - VtxSmeared) / (std/sqrt(N)); VtxSmeared identical in v1 and v2 |
| vtx_pull_z0 | 2023BPix | -1.22 | 1 | -0.11 | 1 | 1.11 | +0.8 | (mean - VtxSmeared) / (std/sqrt(N)); VtxSmeared identical in v1 and v2 |
| vtx_pull_sigmaz | 2023BPix | -1.09 | 1 | 0.87 | 1 | 1.96 | +1.4 | (mean - VtxSmeared) / (std/sqrt(N)); VtxSmeared identical in v1 and v2 |
| vtx_check_vs_generator_process | 2023BPix | SIM: ok |  | SIM: ok |  |  |  |  |
| vtx_match | 2023BPix | True |  | True |  |  |  |  |
| provenance_first_last_file_match | 2023BPix | True |  | True |  |  |  |  |
| provenance_same_all_channels | 2023BPix | True |  | True |  |  |  |  |
| SIM_vtxsmear_readDB | 2023BPix |  |  | false |  |  |  |  |
| files_per_sample | 2024 |  |  | 120 |  |  |  |  |
| events | 2024 |  |  | 200000 |  |  |  |  |
| sqrt_s_median_GeV | 2024 |  |  | 13597.23 |  |  |  | stored precision |
| sqrt_s_lhe_median_GeV | 2024 |  |  | 13599.76 |  |  |  | stored precision |
| sqrt_s_tail_cut_GeV | 2024 |  |  | 13500.0 |  |  |  |  |
| sqrt_s_frac_below_tail_cut | 2024 |  |  | 0.02508 | 0.00034965 |  |  |  |
| frac_events_pass_2018_req | 2024 |  |  | 0.5738 | 0.00110579 |  |  |  |
| sqrt_s_frac_below_tail_cut_after_2018_req | 2024 |  |  | 8e-05 | 2.64017e-05 |  |  |  |
| lhe_frac_with_outgoing_parton_tail | 2024 |  |  | 0.0 | 0 |  |  |  |
| lhe_frac_with_outgoing_parton_core | 2024 |  |  | 0.0 | 0 |  |  |  |
| m35_over_lhe_sqrt_shat_median_tail | 2024 |  |  | 1.01258 |  |  |  | stored precision |
| m35_over_lhe_sqrt_shat_median_core | 2024 |  |  | 1.00027 |  |  |  | stored precision |
| pileup_nTrueInt_mean | 2024 |  |  | 45.42 | 0.0213321 |  |  |  |
| pileup_nTrueInt_std | 2024 |  |  | 9.54 | 0.0150841 |  |  |  |
| genvtx_x_mean_cm | 2024 |  |  | 0.02847 |  |  |  | rounded to 1e-5 cm in the CSV; compared through vtx_pull_* |
| genvtx_y_mean_cm | 2024 |  |  | -0.01657 |  |  |  | rounded to 1e-5 cm in the CSV; compared through vtx_pull_* |
| genvtx_z_mean_cm | 2024 |  |  | 0.4131 | 0.00802189 |  |  |  |
| genvtx_z_std_cm | 2024 |  |  | 3.5875 | 0.00567234 |  |  |  |
| pv_z_std_cm | 2024 |  |  | 3.524 | 0.00557193 |  |  |  |
| n_hlt_branches | 2024 |  |  | 716 |  |  |  |  |
| HLT_DoubleL2Mu_L3Mu16NoVtx_VetoL3Mu0DxyMax0p1cm | 2024 |  |  | yes |  |  |  |  |
| HLT_DoubleL2Mu10NoVtx_2Cha_CosmicSeed_VetoL3Mu0DxyMax1cm | 2024 |  |  | yes |  |  |  |  |
| SIM_release | 2024 |  |  | CMSSW_14_0_25 |  |  |  |  |
| SIM_globaltag | 2024 |  |  | 140X_mcRun3_2024_realistic_v26 |  |  |  |  |
| HLT_release | 2024 |  |  | CMSSW_14_0_25 |  |  |  |  |
| HLT_globaltag | 2024 |  |  | 140X_mcRun3_2024_realistic_v26 |  |  |  |  |
| RECO_release | 2024 |  |  | CMSSW_14_0_25 |  |  |  |  |
| RECO_globaltag | 2024 |  |  | 140X_mcRun3_2024_realistic_v26 |  |  |  |  |
| NANO_release | 2024 |  |  | CMSSW_16_0_10 |  |  |  |  |
| NANO_globaltag | 2024 |  |  | 150X_mcRun3_2024_realistic_v2 |  |  |  |  |
| SIM_comEnergy_GeV | 2024 |  |  | 13600.0 |  |  |  |  |
| SIM_vtxsmear_X0_Y0_Z0_SigmaZ_cm | 2024 |  |  | 0/0/0/0 |  |  |  |  |
| HLT_hlt_menu | 2024 |  |  | /frozen/2024/2e34/v1.4/HLT/V1 |  |  |  |  |
| vtx_pull_x | 2024 |  |  | 13622.06 | 1 |  |  | (mean - VtxSmeared) / (std/sqrt(N)); VtxSmeared identical in v1 and v2 |
| vtx_pull_y | 2024 |  |  | -7946.83 | 1 |  |  | (mean - VtxSmeared) / (std/sqrt(N)); VtxSmeared identical in v1 and v2 |
| vtx_pull_z0 | 2024 |  |  | 51.5 | 1 |  |  | (mean - VtxSmeared) / (std/sqrt(N)); VtxSmeared identical in v1 and v2 |
| vtx_pull_sigmaz | 2024 |  |  | 632.46 | 1 |  |  | (mean - VtxSmeared) / (std/sqrt(N)); VtxSmeared identical in v1 and v2 |
| vtx_check_vs_generator_process | 2024 |  |  | SIM: MISMATCH {'x': 13622.1, 'y': -7946.8, 'z0': 51.5, 'sigmaz': 632.5} |  |  |  |  |
| vtx_match | 2024 |  |  | False |  |  |  |  |
| provenance_first_last_file_match | 2024 |  |  | True |  |  |  |  |
| provenance_same_all_channels | 2024 |  |  | True |  |  |  |  |
| vtx_note | 2024 |  |  | VtxSmeared reads the beam spot from the conditions DB (readDB true): its X0/Y0/Z0/SigmaZ parameters are unused, so this check does not apply |  |  |  |  |
| SIM_vtxsmear_readDB | 2024 |  |  | true |  |  |  |  |
| files_per_sample | 2018 | 47 |  | 47 |  | 0 |  |  |
| events | 2018 | 131014 |  | 131014 |  | 0 |  |  |
| sqrt_s_median_GeV | 2018 | 12998.53 |  | 12998.53 |  | 0 |  | stored precision |
| sqrt_s_lhe_median_GeV | 2018 | 12999.7 |  | 12999.7 |  | 0 |  | stored precision |
| sqrt_s_tail_cut_GeV | 2018 | 12904.4 |  | 12904.4 |  | 0 |  |  |
| sqrt_s_frac_below_tail_cut | 2018 | 0.00011 | 2.89743e-05 | 0.00011 | 2.89743e-05 | 0 | +0.0 |  |
| frac_events_pass_2018_req | 2018 | 0.9994 | 6.76529e-05 | 0.9994 | 6.76529e-05 | 0 | +0.0 |  |
| sqrt_s_frac_below_tail_cut_after_2018_req | 2018 | 0.00011 | 2.8983e-05 | 0.00011 | 2.8983e-05 | 0 | +0.0 |  |
| lhe_frac_with_outgoing_parton_tail | 2018 | 0.0 | 0 | 0.0 | 0 | 0 |  |  |
| lhe_frac_with_outgoing_parton_core | 2018 | 0.0 | 0 | 0.0 | 0 | 0 |  |  |
| m35_over_lhe_sqrt_shat_median_tail | 2018 | 1.00824 |  | 1.00824 |  | 0 |  | stored precision |
| m35_over_lhe_sqrt_shat_median_core | 2018 | 1.00019 |  | 1.00019 |  | 0 |  | stored precision |
| pileup_nTrueInt_mean | 2018 | 31.03 | 0.0250581 | 31.03 | 0.0250581 | 0 | +0.0 |  |
| pileup_nTrueInt_std | 2018 | 9.07 | 0.0177188 | 9.07 | 0.0177188 | 0 | +0.0 |  |
| genvtx_x_mean_cm | 2018 | 0.01079 |  | 0.01079 |  | 0 |  | rounded to 1e-5 cm in the CSV; compared through vtx_pull_* |
| genvtx_y_mean_cm | 2018 | 0.04168 |  | 0.04168 |  | 0 |  | rounded to 1e-5 cm in the CSV; compared through vtx_pull_* |
| genvtx_z_mean_cm | 2018 | 0.1507 | 0.00885046 | 0.1507 | 0.00885046 | 0 | +0.0 |  |
| genvtx_z_std_cm | 2018 | 3.2035 | 0.00625822 | 3.2035 | 0.00625822 | 0 | +0.0 |  |
| pv_z_std_cm | 2018 | 3.261 | 0.00637055 | 3.261 | 0.00637055 | 0 | +0.0 |  |
| n_hlt_branches | 2018 | 651 |  | 651 |  | 0 |  |  |
| HLT_DoubleL2Mu_L3Mu16NoVtx_VetoL3Mu0DxyMax0p1cm | 2018 | no |  | no |  |  |  |  |
| HLT_DoubleL2Mu10NoVtx_2Cha_CosmicSeed_VetoL3Mu0DxyMax1cm | 2018 | no |  | no |  |  |  |  |
| SIM_release | 2018 | CMSSW_10_6_17_patch1 |  | CMSSW_10_6_17_patch1 |  |  |  |  |
| SIM_globaltag | 2018 | 106X_upgrade2018_realistic_v11_L1v1 |  | 106X_upgrade2018_realistic_v11_L1v1 |  |  |  |  |
| HLT_release | 2018 | CMSSW_10_2_16_UL |  | CMSSW_10_2_16_UL |  |  |  |  |
| HLT_globaltag | 2018 | 102X_upgrade2018_realistic_v15 |  | 102X_upgrade2018_realistic_v15 |  |  |  |  |
| RECO_release | 2018 | CMSSW_10_6_17_patch1 |  | CMSSW_10_6_17_patch1 |  |  |  |  |
| RECO_globaltag | 2018 | 106X_upgrade2018_realistic_v11_L1v1 |  | 106X_upgrade2018_realistic_v11_L1v1 |  |  |  |  |
| NANO_release | 2018 | CMSSW_10_6_29 |  | CMSSW_10_6_29 |  |  |  |  |
| NANO_globaltag | 2018 | 106X_upgrade2018_realistic_v16_L1v1 |  | 106X_upgrade2018_realistic_v16_L1v1 |  |  |  |  |
| HLT_hlt_menu | 2018 | /frozen/2018/2e34/v3.2/HLT/V1 |  | /frozen/2018/2e34/v3.2/HLT/V1 |  |  |  |  |
| vtx_pull_x | 2018 | 13.38 | 1 | 13.38 | 1 | 0 | +0.0 | (mean - VtxSmeared) / (std/sqrt(N)); VtxSmeared identical in v1 and v2 |
| vtx_pull_y | 2018 | -35.38 | 1 | -35.38 | 1 | 0 | +0.0 | (mean - VtxSmeared) / (std/sqrt(N)); VtxSmeared identical in v1 and v2 |
| vtx_pull_z0 | 2018 | 12.99 | 1 | 12.99 | 1 | 0 | +0.0 | (mean - VtxSmeared) / (std/sqrt(N)); VtxSmeared identical in v1 and v2 |
| vtx_pull_sigmaz | 2018 | -47.38 | 1 | -47.38 | 1 | 0 | +0.0 | (mean - VtxSmeared) / (std/sqrt(N)); VtxSmeared identical in v1 and v2 |
| vtx_check_vs_generator_process | 2018 | GEN: MISMATCH {'x': 13.4, 'y': -35.4, 'z0': 13.0, 'sigmaz': -47.4} |  | GEN: MISMATCH {'x': 13.4, 'y': -35.4, 'z0': 13.0, 'sigmaz': -47.4} |  |  |  |  |
| vtx_match | 2018 | False |  | False |  |  |  |  |
| provenance_first_last_file_match | 2018 | True |  | True |  |  |  |  |
| provenance_same_all_channels | 2018 | True |  | True |  |  |  |  |
| GEN_release | 2018 | CMSSW_10_6_27 |  | CMSSW_10_6_27 |  |  |  |  |
| GEN_globaltag | 2018 | 106X_upgrade2018_realistic_v4 |  | 106X_upgrade2018_realistic_v4 |  |  |  |  |
| DIGI2RAW_release | 2018 | CMSSW_10_6_17_patch1 |  | CMSSW_10_6_17_patch1 |  |  |  |  |
| DIGI2RAW_globaltag | 2018 | 106X_upgrade2018_realistic_v11_L1v1 |  | 106X_upgrade2018_realistic_v11_L1v1 |  |  |  |  |
| PAT_release | 2018 | CMSSW_10_6_29 |  | CMSSW_10_6_29 |  |  |  |  |
| PAT_globaltag | 2018 | 106X_upgrade2018_realistic_v16_L1v1 |  | 106X_upgrade2018_realistic_v16_L1v1 |  |  |  |  |
| GEN_comEnergy_GeV | 2018 | 13000.0 |  | 13000.0 |  | 0 |  |  |
| GEN_vtxsmear_X0_Y0_Z0_SigmaZ_cm | 2018 | 0.0107682/0.041722/0.035748/3.5 |  | 0.0107682/0.041722/0.035748/3.5 |  |  |  |  |
| vtx_note | 2018 | the GEN PSet's VtxSmeared parameters are not those of the stored vertex; the smearing step's configuration is not in the file |  | the GEN PSet's VtxSmeared parameters are not those of the stored vertex; the smearing step's configuration is not in the file |  |  |  |  |
| GEN_vtxsmear_readDB | 2018 |  |  | false |  |  |  |  |

## regions (own flagged bins)

| quantity | era | v1 | σ v1 | v2 | σ v2 | v2 − v1 | n σ | note |
|---|---|---|---|---|---|---|---|---|
| eedp_per-dark-photon.eta | - | [1.5, 2.1] |  | [1.5, 1.9] |  | differs |  |  |
| eedp_per-dark-photon.phi | - | [1.7453, 2.618] |  | [1.7453, 2.618] |  |  |  |  |
| eedp_per-dark-photon.n_flagged_bins | - | 18 |  | 13 |  | -5 |  |  |
| eedp_per-dark-photon.n_rectangle_bins | - | 30 |  | 20 |  | -10 |  |  |
| eedp_per-dark-photon.n_rectangle_unflagged_bins | - | 12 |  | 7 |  | -5 |  |  |
| eedp_per-dark-photon.flagged_avg | 2022 | 0.9432 |  | 0.9707 |  | 0.0275 |  | v2 over its own flagged bins |
| eedp_per-dark-photon.flagged_avg | 2022EE | 0.1693 |  | 0.1462 |  | -0.0231 |  | v2 over its own flagged bins |
| eedp_per-dark-photon.rectangle_avg | 2022 | 0.9619 |  | 0.9493 |  | -0.0126 |  | v2 over its own flagged bins |
| eedp_per-dark-photon.rectangle_avg | 2022EE | 0.4415 |  | 0.3506 |  | -0.0909 |  | v2 over its own flagged bins |
| eedp_per-dark-photon.outside_rectangle_avg | 2022 | 0.9519 |  | 0.9485 |  | -0.0034 |  | v2 over its own flagged bins |
| eedp_per-dark-photon.outside_rectangle_avg | 2022EE | 0.9484 |  | 0.9487 |  | 0.0003 |  | v2 over its own flagged bins |
| ee_exclusive.eta | - | [1.5, 2.0] |  | [1.5, 2.0] |  |  |  |  |
| ee_exclusive.phi | - | [1.7453, 2.4435] |  | [1.7453, 2.4435] |  |  |  |  |
| ee_exclusive.n_flagged_bins | - | 13 |  | 13 |  | 0 |  |  |
| ee_exclusive.n_rectangle_bins | - | 20 |  | 20 |  | 0 |  |  |
| ee_exclusive.n_rectangle_unflagged_bins | - | 7 |  | 7 |  | 0 |  |  |
| ee_exclusive.flagged_avg | 2022 | 0.6105 |  | 0.625 |  | 0.0145 |  | v2 over its own flagged bins |
| ee_exclusive.flagged_avg | 2022EE | 0.0283 |  | 0.0407 |  | 0.0124 |  | v2 over its own flagged bins |
| ee_exclusive.rectangle_avg | 2022 | 0.608 |  | 0.6075 |  | -0.0005 |  | v2 over its own flagged bins |
| ee_exclusive.rectangle_avg | 2022EE | 0.1811 |  | 0.1696 |  | -0.0115 |  | v2 over its own flagged bins |
| ee_exclusive.outside_rectangle_avg | 2022 | 0.5881 |  | 0.591 |  | 0.0029 |  | v2 over its own flagged bins |
| ee_exclusive.outside_rectangle_avg | 2022EE | 0.5885 |  | 0.5877 |  | -0.0008 |  | v2 over its own flagged bins |
| ee_non-exclusive.eta | - | [1.5, 2.1] |  | [1.5, 2.1] |  |  |  |  |
| ee_non-exclusive.phi | - | [1.7453, 2.618] |  | [1.7453, 2.618] |  |  |  |  |
| ee_non-exclusive.n_flagged_bins | - | 19 |  | 18 |  | -1 |  |  |
| ee_non-exclusive.n_rectangle_bins | - | 30 |  | 30 |  | 0 |  |  |
| ee_non-exclusive.n_rectangle_unflagged_bins | - | 11 |  | 12 |  | 1 |  |  |
| ee_non-exclusive.flagged_avg | 2022 | 0.9323 |  | 0.9535 |  | 0.0212 |  | v2 over its own flagged bins |
| ee_non-exclusive.flagged_avg | 2022EE | 0.1987 |  | 0.1963 |  | -0.0024 |  | v2 over its own flagged bins |
| ee_non-exclusive.rectangle_avg | 2022 | 0.9511 |  | 0.9501 |  | -0.001 |  | v2 over its own flagged bins |
| ee_non-exclusive.rectangle_avg | 2022EE | 0.4424 |  | 0.4606 |  | 0.0182 |  | v2 over its own flagged bins |
| ee_non-exclusive.outside_rectangle_avg | 2022 | 0.941 |  | 0.9392 |  | -0.0018 |  | v2 over its own flagged bins |
| ee_non-exclusive.outside_rectangle_avg | 2022EE | 0.9388 |  | 0.9415 |  | 0.0027 |  | v2 over its own flagged bins |
| bp_exclusive.eta | - | [-1.3, 0.1] |  | [-1.6, 0.0] |  | differs |  |  |
| bp_exclusive.phi | - | [-1.2217, -0.8727] |  | [-1.2217, -0.8727] |  |  |  |  |
| bp_exclusive.n_flagged_bins | - | 24 |  | 30 |  | 6 |  |  |
| bp_exclusive.n_rectangle_bins | - | 28 |  | 32 |  | 4 |  |  |
| bp_exclusive.n_rectangle_unflagged_bins | - | 4 |  | 2 |  | -2 |  |  |
| bp_exclusive.flagged_avg | 2023 | 11.435 |  | 11.4082 |  | -0.0268 |  | v2 over its own flagged bins |
| bp_exclusive.flagged_avg | 2023BPix | 9.924 |  | 9.9006 |  | -0.0234 |  | v2 over its own flagged bins |
| bp_exclusive.rectangle_avg | 2023 | 11.4258 |  | 11.5099 |  | 0.0841 |  | v2 over its own flagged bins |
| bp_exclusive.rectangle_avg | 2023BPix | 10.0257 |  | 10.0215 |  | -0.0042 |  | v2 over its own flagged bins |
| bp_exclusive.outside_rectangle_avg | 2023 | 12.2836 |  | 12.2866 |  | 0.003 |  | v2 over its own flagged bins |
| bp_exclusive.outside_rectangle_avg | 2023BPix | 12.2533 |  | 12.2674 |  | 0.0141 |  | v2 over its own flagged bins |
| bp_exclusive.pf_eff_phi_band_all_eta | 2023 | 0.9473 |  | 0.9514 |  | 0.0041 |  |  |
| bp_exclusive.pf_eff_phi_band_all_eta | 2023BPix | 0.9262 |  | 0.9237 |  | -0.0025 |  |  |
| bp_exclusive.pf_eff_outside_phi_band | 2023 | 0.9563 |  | 0.9567 |  | 0.0004 |  |  |
| bp_exclusive.pf_eff_outside_phi_band | 2023BPix | 0.9549 |  | 0.9554 |  | 0.0005 |  |  |
| bp_non-exclusive.eta | - | [-1.3, 0.1] |  | [-1.6, 0.0] |  | differs |  |  |
| bp_non-exclusive.phi | - | [-1.2217, -0.8727] |  | [-1.2217, -0.8727] |  |  |  |  |
| bp_non-exclusive.n_flagged_bins | - | 24 |  | 30 |  | 6 |  |  |
| bp_non-exclusive.n_rectangle_bins | - | 28 |  | 32 |  | 4 |  |  |
| bp_non-exclusive.n_rectangle_unflagged_bins | - | 4 |  | 2 |  | -2 |  |  |
| bp_non-exclusive.flagged_avg | 2023 | 11.4171 |  | 11.3831 |  | -0.034 |  | v2 over its own flagged bins |
| bp_non-exclusive.flagged_avg | 2023BPix | 9.8947 |  | 9.8833 |  | -0.0114 |  | v2 over its own flagged bins |
| bp_non-exclusive.rectangle_avg | 2023 | 11.4063 |  | 11.4836 |  | 0.0773 |  | v2 over its own flagged bins |
| bp_non-exclusive.rectangle_avg | 2023BPix | 9.9958 |  | 9.998 |  | 0.0022 |  | v2 over its own flagged bins |
| bp_non-exclusive.outside_rectangle_avg | 2023 | 12.2647 |  | 12.2683 |  | 0.0036 |  | v2 over its own flagged bins |
| bp_non-exclusive.outside_rectangle_avg | 2023BPix | 12.2345 |  | 12.2468 |  | 0.0123 |  | v2 over its own flagged bins |
| bp_non-exclusive.pf_eff_phi_band_all_eta | 2023 | 0.9715 |  | 0.9714 |  | -0.0001 |  |  |
| bp_non-exclusive.pf_eff_phi_band_all_eta | 2023BPix | 0.9558 |  | 0.9565 |  | 0.0007 |  |  |
| bp_non-exclusive.pf_eff_outside_phi_band | 2023 | 0.975 |  | 0.9751 |  | 0.0001 |  |  |
| bp_non-exclusive.pf_eff_outside_phi_band | 2023BPix | 0.9741 |  | 0.9741 |  | 0 |  |  |

## regions (v1 bins)

| quantity | era | v1 | σ v1 | v2 | σ v2 | v2 − v1 | n σ | note |
|---|---|---|---|---|---|---|---|---|
| eedp_per-dark-photon.flagged_avg | 2022 | 0.9432 | 0.00990365 | 0.9462 | 0.01007 | 0.003 | +0.2 |  |
| eedp_per-dark-photon.flagged_avg | 2022EE | 0.1693 | 0.0167387 | 0.1897 | 0.0174294 | 0.0204 | +0.8 |  |
| eedp_per-dark-photon.flagged_avg | 2024 |  |  | 0.9218 | 0.0120191 |  |  |  |
| eedp_per-dark-photon.rectangle_avg | 2022 | 0.9619 | 0.00660483 | 0.9644 | 0.00649444 | 0.0025 | +0.3 |  |
| eedp_per-dark-photon.rectangle_avg | 2022EE | 0.4415 | 0.0177118 | 0.4779 | 0.0172656 | 0.0364 | +1.5 |  |
| eedp_per-dark-photon.rectangle_avg | 2024 |  |  | 0.949 | 0.00795403 |  |  |  |
| eedp_per-dark-photon.outside_rectangle_avg | 2022 | 0.9519 | 0.00157669 | 0.9479 | 0.00163847 | -0.004 | -1.8 |  |
| eedp_per-dark-photon.outside_rectangle_avg | 2022EE | 0.9484 | 0.0016365 | 0.9506 | 0.00158753 | 0.0022 | +1.0 |  |
| eedp_per-dark-photon.outside_rectangle_avg | 2024 |  |  | 0.9507 | 0.00159562 |  |  |  |
| ee_exclusive.flagged_avg | 2022 | 0.6105 | 0.0166766 | 0.5987 | 0.0175506 | -0.0118 | -0.5 |  |
| ee_exclusive.flagged_avg | 2022EE | 0.0283 | 0.00594296 | 0.0366 | 0.00678468 | 0.0083 | +0.9 |  |
| ee_exclusive.flagged_avg | 2024 |  |  | 0.5785 | 0.0175686 |  |  |  |
| ee_exclusive.rectangle_avg | 2022 | 0.608 | 0.0139372 | 0.6075 | 0.0143744 | -0.0005 | -0.0 |  |
| ee_exclusive.rectangle_avg | 2022EE | 0.1811 | 0.0113908 | 0.1696 | 0.0110377 | -0.0115 | -0.7 |  |
| ee_exclusive.rectangle_avg | 2024 |  |  | 0.5995 | 0.0143192 |  |  |  |
| ee_exclusive.outside_rectangle_avg | 2022 | 0.5881 | 0.00249465 | 0.591 | 0.00249116 | 0.0029 | +0.8 |  |
| ee_exclusive.outside_rectangle_avg | 2022EE | 0.5885 | 0.00250329 | 0.5877 | 0.00247869 | -0.0008 | -0.2 |  |
| ee_exclusive.outside_rectangle_avg | 2024 |  |  | 0.589 | 0.00249616 |  |  |  |
| ee_non-exclusive.flagged_avg | 2022 | 0.9323 | 0.00721427 | 0.9275 | 0.00780797 | -0.0048 | -0.5 |  |
| ee_non-exclusive.flagged_avg | 2022EE | 0.1987 | 0.012103 | 0.2167 | 0.012355 | 0.018 | +1.0 |  |
| ee_non-exclusive.flagged_avg | 2024 |  |  | 0.9159 | 0.00830044 |  |  |  |
| ee_non-exclusive.rectangle_avg | 2022 | 0.9511 | 0.00517334 | 0.9501 | 0.00527783 | -0.001 | -0.1 |  |
| ee_non-exclusive.rectangle_avg | 2022EE | 0.4424 | 0.0123627 | 0.4606 | 0.0121356 | 0.0182 | +1.1 |  |
| ee_non-exclusive.rectangle_avg | 2024 |  |  | 0.9413 | 0.00581154 |  |  |  |
| ee_non-exclusive.outside_rectangle_avg | 2022 | 0.941 | 0.00120262 | 0.9392 | 0.00121942 | -0.0018 | -1.1 |  |
| ee_non-exclusive.outside_rectangle_avg | 2022EE | 0.9388 | 0.00122672 | 0.9415 | 0.00118979 | 0.0027 | +1.6 |  |
| ee_non-exclusive.outside_rectangle_avg | 2024 |  |  | 0.9425 | 0.00118818 |  |  |  |
| bp_exclusive.flagged_avg | 2023 | 11.435 | 0.0300676 | 11.4645 | 0.0293908 | 0.0295 | +0.7 |  |
| bp_exclusive.flagged_avg | 2023BPix | 9.924 | 0.0320424 | 9.9631 | 0.0316956 | 0.0391 | +0.9 |  |
| bp_exclusive.flagged_avg | 2024 |  |  | 9.9466 | 0.0318676 |  |  |  |
| bp_exclusive.rectangle_avg | 2023 | 11.4258 | 0.0279164 | 11.4459 | 0.0274352 | 0.0201 | +0.5 |  |
| bp_exclusive.rectangle_avg | 2023BPix | 10.0257 | 0.0294427 | 10.0387 | 0.0290409 | 0.013 | +0.3 |  |
| bp_exclusive.rectangle_avg | 2024 |  |  | 10.0312 | 0.0293584 |  |  |  |
| bp_exclusive.outside_rectangle_avg | 2023 | 12.2836 | 0.00541023 | 12.2862 | 0.00539777 | 0.0026 | +0.3 |  |
| bp_exclusive.outside_rectangle_avg | 2023BPix | 12.2533 | 0.00539834 | 12.2654 | 0.00540637 | 0.0121 | +1.6 |  |
| bp_exclusive.outside_rectangle_avg | 2024 |  |  | 12.2599 | 0.00538073 |  |  |  |
| bp_exclusive.pf_eff_flagged | 2023 | 0.9488 | 0.00460881 | 0.9545 | 0.00427534 | 0.0057 | +0.9 |  |
| bp_exclusive.pf_eff_flagged | 2023BPix | 0.8895 | 0.00676925 | 0.8968 | 0.00651567 | 0.0073 | +0.8 |  |
| bp_exclusive.pf_eff_flagged | 2024 |  |  | 0.9028 | 0.00641858 |  |  |  |
| bp_exclusive.pf_eff_rectangle | 2023 | 0.9491 | 0.00426643 | 0.9566 | 0.0039054 | 0.0075 | +1.3 |  |
| bp_exclusive.pf_eff_rectangle | 2023BPix | 0.8961 | 0.00607594 | 0.8976 | 0.00594916 | 0.0015 | +0.2 |  |
| bp_exclusive.pf_eff_rectangle | 2024 |  |  | 0.9005 | 0.00596047 |  |  |  |
| bp_non-exclusive.flagged_avg | 2023 | 11.4171 | 0.0296799 | 11.4451 | 0.0290597 | 0.028 | +0.7 |  |
| bp_non-exclusive.flagged_avg | 2023BPix | 9.8947 | 0.031196 | 9.9418 | 0.0308307 | 0.0471 | +1.1 |  |
| bp_non-exclusive.flagged_avg | 2024 |  |  | 9.9182 | 0.0310804 |  |  |  |
| bp_non-exclusive.rectangle_avg | 2023 | 11.4063 | 0.0275518 | 11.4302 | 0.0271499 | 0.0239 | +0.6 |  |
| bp_non-exclusive.rectangle_avg | 2023BPix | 9.9958 | 0.0287093 | 10.0139 | 0.0282843 | 0.0181 | +0.4 |  |
| bp_non-exclusive.rectangle_avg | 2024 |  |  | 10.0017 | 0.0286132 |  |  |  |
| bp_non-exclusive.outside_rectangle_avg | 2023 | 12.2647 | 0.00535711 | 12.2676 | 0.00534626 | 0.0029 | +0.4 |  |
| bp_non-exclusive.outside_rectangle_avg | 2023BPix | 12.2345 | 0.00534343 | 12.2446 | 0.0053526 | 0.0101 | +1.3 |  |
| bp_non-exclusive.outside_rectangle_avg | 2024 |  |  | 12.2417 | 0.00533065 |  |  |  |
| bp_non-exclusive.pf_eff_flagged | 2023 | 0.9733 | 0.0033709 | 0.9756 | 0.00316525 | 0.0023 | +0.5 |  |
| bp_non-exclusive.pf_eff_flagged | 2023BPix | 0.9375 | 0.00522651 | 0.944 | 0.00492438 | 0.0065 | +0.9 |  |
| bp_non-exclusive.pf_eff_flagged | 2024 |  |  | 0.9488 | 0.00477565 |  |  |  |
| bp_non-exclusive.pf_eff_rectangle | 2023 | 0.974 | 0.00308898 | 0.9765 | 0.00290353 | 0.0025 | +0.6 |  |
| bp_non-exclusive.pf_eff_rectangle | 2023BPix | 0.9417 | 0.00466571 | 0.943 | 0.00454943 | 0.0013 | +0.2 |  |
| bp_non-exclusive.pf_eff_rectangle | 2024 |  |  | 0.9477 | 0.00443316 |  |  |  |

## gen_comparison_summary.json

| quantity | era | v1 | σ v1 | v2 | σ v2 | v2 − v1 | n σ | note |
|---|---|---|---|---|---|---|---|---|
| pass_fraction.2022 | 2022 | 0.58142 | 0.00450343 | 0.58292 | 0.00450115 | 0.0015 | +0.2 |  |
| pass_fraction.2018 | 2018 | 0.99945 | 0.000137505 | 0.99945 | 0.000137505 | 0 | +0.0 | 2018: same files in v1 and v2 |
| lep_pt.2022_raw.mean | 2022 | 40.3755 | 0.13468 | 40.3633 | 0.13525 | -0.0122 | -0.1 |  |
| lep_pt.2022_raw.median | 2022 | 35.5 |  | 35.375 |  | -0.125 |  |  |
| lep_pt.2022_raw.n | 2022 | 48000 |  | 48000 |  | 0 |  |  |
| lep_pt.2022_filter.mean | 2022 | 43.9926 | 0.16829 | 43.8696 | 0.16919 | -0.123 | -0.5 |  |
| lep_pt.2022_filter.median | 2022 | 40.5 |  | 40.375 |  | -0.125 |  |  |
| lep_pt.2022_filter.n | 2022 | 27908 |  | 27980 |  | 72 |  |  |
| lep_pt.2018_raw.mean | 2018 | 43.9077 | 0.08225 | 43.9077 | 0.08225 | 0 | +0.0 |  |
| lep_pt.2018_raw.median | 2018 | 40.625 |  | 40.625 |  | 0 |  |  |
| lep_pt.2018_raw.n | 2018 | 116292 |  | 116292 |  | 0 |  |  |
| lep_pt.2018_filter.mean | 2018 | 43.9119 | 0.08226 | 43.9119 | 0.08226 | 0 | +0.0 |  |
| lep_pt.2018_filter.median | 2018 | 40.625 |  | 40.625 |  | 0 |  |  |
| lep_pt.2018_filter.n | 2018 | 116228 |  | 116228 |  | 0 |  |  |
| lep_pt.reldiff_raw | 2022 | -0.08045 | 0.00306734 | -0.08072 | 0.00308032 | -0.00027 | -0.1 | 2018 part common |
| lep_pt.reldiff_filter | 2022 | 0.00184 | 0.00383245 | -0.00096 | 0.00385294 | -0.0028 | -0.5 | 2018 part common |
| lep_abseta.2022_raw.mean | 2022 | 1.27954 | 0.00439 | 1.28377 | 0.00441 | 0.00423 | +0.7 |  |
| lep_abseta.2022_raw.median | 2022 | 1.08203 |  | 1.08984 |  | 0.00781 |  |  |
| lep_abseta.2022_raw.n | 2022 | 48000 |  | 48000 |  | 0 |  |  |
| lep_abseta.2022_filter.mean | 2022 | 0.954 | 0.0038 | 0.96108 | 0.00381 | 0.00708 | +1.3 |  |
| lep_abseta.2022_filter.median | 2022 | 0.86719 |  | 0.87207 |  | 0.00488 |  |  |
| lep_abseta.2022_filter.n | 2022 | 27908 |  | 27980 |  | 72 |  |  |
| lep_abseta.2018_raw.mean | 2018 | 0.95439 | 0.00185 | 0.95439 | 0.00185 | 0 | +0.0 |  |
| lep_abseta.2018_raw.median | 2018 | 0.86914 |  | 0.86914 |  | 0 |  |  |
| lep_abseta.2018_raw.n | 2018 | 116292 |  | 116292 |  | 0 |  |  |
| lep_abseta.2018_filter.mean | 2018 | 0.95427 | 0.00185 | 0.95427 | 0.00185 | 0 | +0.0 |  |
| lep_abseta.2018_filter.median | 2018 | 0.86914 |  | 0.86914 |  | 0 |  |  |
| lep_abseta.2018_filter.n | 2018 | 116228 |  | 116228 |  | 0 |  |  |
| lep_abseta.reldiff_raw | 2022 | 0.34069 | 0.0045998 | 0.34512 | 0.00462075 | 0.00443 | +0.7 | 2018 part common |
| lep_abseta.reldiff_filter | 2022 | -0.00028 | 0.0039821 | 0.00714 | 0.00399258 | 0.00742 | +1.3 | 2018 part common |
| dp_dR.2022_raw.mean | 2022 | 0.05208 | 0.00039 | 0.05168 | 0.00036 | -0.0004 | -0.8 |  |
| dp_dR.2022_raw.median | 2022 | 0.03852 |  | 0.03906 |  | 0.00054 |  |  |
| dp_dR.2022_raw.n | 2022 | 24000 |  | 24000 |  | 0 |  |  |
| dp_dR.2022_filter.mean | 2022 | 0.03683 | 0.00012 | 0.03701 | 0.00012 | 0.00018 | +1.1 |  |
| dp_dR.2022_filter.median | 2022 | 0.03315 |  | 0.03338 |  | 0.00023 |  |  |
| dp_dR.2022_filter.n | 2022 | 13954 |  | 13990 |  | 36 |  |  |
| dp_dR.2018_raw.mean | 2018 | 0.03695 | 6e-05 | 0.03695 | 6e-05 | 0 | +0.0 |  |
| dp_dR.2018_raw.median | 2018 | 0.0332 |  | 0.0332 |  | 0 |  |  |
| dp_dR.2018_raw.n | 2018 | 58146 |  | 58146 |  | 0 |  |  |
| dp_dR.2018_filter.mean | 2018 | 0.03694 | 6e-05 | 0.03694 | 6e-05 | 0 | +0.0 |  |
| dp_dR.2018_filter.median | 2018 | 0.0332 |  | 0.0332 |  | 0 |  |  |
| dp_dR.2018_filter.n | 2018 | 58114 |  | 58114 |  | 0 |  |  |
| dp_dR.reldiff_raw | 2022 | 0.40947 | 0.0105548 | 0.39865 | 0.0097429 | -0.01082 | -0.8 | 2018 part common |
| dp_dR.reldiff_filter | 2022 | -0.00298 | 0.00324851 | 0.00189 | 0.00324851 | 0.00487 | +1.1 | 2018 part common |
| dp_pt.2022_raw.mean | 2022 | 80.7458 | 0.17321 | 80.7217 | 0.17377 | -0.02413 | -0.1 |  |
| dp_pt.2022_raw.median | 2022 | 85.75 |  | 85.25 |  | -0.5 |  |  |
| dp_pt.2022_raw.n | 2022 | 24000 |  | 24000 |  | 0 |  |  |
| dp_pt.2022_filter.mean | 2022 | 87.9808 | 0.18832 | 87.735 | 0.19094 | -0.24582 | -0.9 |  |
| dp_pt.2022_filter.median | 2022 | 90.75 |  | 90.5 |  | -0.25 |  |  |
| dp_pt.2022_filter.n | 2022 | 13954 |  | 13990 |  | 36 |  |  |
| dp_pt.2018_raw.mean | 2018 | 87.8118 | 0.08942 | 87.8118 | 0.08942 | 0 | +0.0 |  |
| dp_pt.2018_raw.median | 2018 | 90.5 |  | 90.5 |  | 0 |  |  |
| dp_pt.2018_raw.n | 2018 | 58146 |  | 58146 |  | 0 |  |  |
| dp_pt.2018_filter.mean | 2018 | 87.8202 | 0.08941 | 87.8202 | 0.08941 | 0 | +0.0 |  |
| dp_pt.2018_filter.median | 2018 | 90.5 |  | 90.5 |  | 0 |  |  |
| dp_pt.2018_filter.n | 2018 | 58114 |  | 58114 |  | 0 |  |  |
| dp_pt.reldiff_raw | 2022 | -0.08047 | 0.00197251 | -0.08074 | 0.00197889 | -0.00027 | -0.1 | 2018 part common |
| dp_pt.reldiff_filter | 2022 | 0.00183 | 0.00214438 | -0.00097 | 0.00217422 | -0.0028 | -0.9 | 2018 part common |
| dp_mass.2022_raw.mean | 2022 | 1.2006 | 0.00014 | 1.20055 | 0.00014 | -5e-05 | -0.3 |  |
| dp_mass.2022_raw.median | 2022 | 1.19922 |  | 1.19922 |  | 0 |  |  |
| dp_mass.2022_raw.n | 2022 | 24000 |  | 24000 |  | 0 |  |  |
| dp_mass.2022_filter.mean | 2022 | 1.20077 | 0.00019 | 1.20051 | 0.00018 | -0.00026 | -1.0 |  |
| dp_mass.2022_filter.median | 2022 | 1.19922 |  | 1.19922 |  | 0 |  |  |
| dp_mass.2022_filter.n | 2022 | 13954 |  | 13990 |  | 36 |  |  |
| dp_mass.2018_raw.mean | 2018 | 1.20076 | 9e-05 | 1.20076 | 9e-05 | 0 | +0.0 |  |
| dp_mass.2018_raw.median | 2018 | 1.19922 |  | 1.19922 |  | 0 |  |  |
| dp_mass.2018_raw.n | 2018 | 58146 |  | 58146 |  | 0 |  |  |
| dp_mass.2018_filter.mean | 2018 | 1.20076 | 9e-05 | 1.20076 | 9e-05 | 0 | +0.0 |  |
| dp_mass.2018_filter.median | 2018 | 1.19922 |  | 1.19922 |  | 0 |  |  |
| dp_mass.2018_filter.n | 2018 | 58114 |  | 58114 |  | 0 |  |  |
| dp_mass.reldiff_raw | 2022 | -0.00013 | 0.000116593 | -0.00017 | 0.000116593 | -4e-05 | -0.2 | 2018 part common |
| dp_mass.reldiff_filter | 2022 | 1e-05 | 0.000158233 | -0.00021 | 0.000149905 | -0.00022 | -1.0 | 2018 part common |
| dp_proper_mm.2022_raw.mean | 2022 | 4.86327 | 0.0311 | 4.82851 | 0.03124 | -0.03476 | -0.8 |  |
| dp_proper_mm.2022_raw.median | 2022 | 3.38258 |  | 3.33517 |  | -0.04741 |  |  |
| dp_proper_mm.2022_raw.n | 2022 | 24000 |  | 24000 |  | 0 |  |  |
| dp_proper_mm.2022_filter.mean | 2022 | 4.87454 | 0.04088 | 4.81964 | 0.04092 | -0.0549 | -0.9 |  |
| dp_proper_mm.2022_filter.median | 2022 | 3.39004 |  | 3.33391 |  | -0.05613 |  |  |
| dp_proper_mm.2022_filter.n | 2022 | 13954 |  | 13990 |  | 36 |  |  |
| dp_proper_mm.2018_raw.mean | 2018 | 4.80207 | 0.0198 | 4.80207 | 0.0198 | 0 | +0.0 |  |
| dp_proper_mm.2018_raw.median | 2018 | 3.33475 |  | 3.33475 |  | 0 |  |  |
| dp_proper_mm.2018_raw.n | 2018 | 58146 |  | 58146 |  | 0 |  |  |
| dp_proper_mm.2018_filter.mean | 2018 | 4.80229 | 0.01981 | 4.80229 | 0.01981 | 0 | +0.0 |  |
| dp_proper_mm.2018_filter.median | 2018 | 3.33341 |  | 3.33341 |  | 0 |  |  |
| dp_proper_mm.2018_filter.n | 2018 | 58114 |  | 58114 |  | 0 |  |  |
| dp_proper_mm.reldiff_raw | 2022 | 0.01274 | 0.00647637 | 0.00551 | 0.00650553 | -0.00723 | -0.8 | 2018 part common |
| dp_proper_mm.reldiff_filter | 2022 | 0.01504 | 0.00851261 | 0.00361 | 0.00852093 | -0.01143 | -0.9 | 2018 part common |

## lj_eff_summary.json (geom)

| quantity | era | v1 | σ v1 | v2 | σ v2 | v2 − v1 | n σ | note |
|---|---|---|---|---|---|---|---|---|
| 2Mu2E_incl_acc.mean_pct (Run3-2018) | 2022 | 0.163 | 0.023172 | 0.181 | 0.0231565 | 0.018 | +0.5 |  |
| 2Mu2E_incl_acc.rms_pct | 2022 | 0.171 |  | 0.152 |  | -0.019 |  |  |
| 2Mu2E_incl_acc.min_pct | 2022 | -0.232 |  | -0.081 |  | 0.151 |  |  |
| 2Mu2E_incl_acc.max_pct | 2022 | 0.542 |  | 0.499 |  | -0.043 |  |  |
| 2Mu2E_incl_acc.n_negative | 2022 | 1 |  | 1 |  | 0 |  |  |
| 2Mu2E_incl_acc.chi2_about_mean | 2022 | 21.5 |  | 21.3 |  | -0.2 |  |  |
| 2Mu2E_incl_acc: chi2 of v2 - v1 Run 3 eff. over 18 points | 2022 |  |  | 27.4 |  |  |  | 18 dof |
| 4Mu_incl_acc.mean_pct (Run3-2018) | 2022 | -0.039 | 0.00613238 | -0.02 | 0.00601922 | 0.019 | +2.2 |  |
| 4Mu_incl_acc.rms_pct | 2022 | 0.038 |  | 0.036 |  | -0.002 |  |  |
| 4Mu_incl_acc.min_pct | 2022 | -0.11 |  | -0.074 |  | 0.036 |  |  |
| 4Mu_incl_acc.max_pct | 2022 | 0.03 |  | 0.041 |  | 0.011 |  |  |
| 4Mu_incl_acc.n_negative | 2022 | 14 |  | 12 |  | -2 |  |  |
| 4Mu_incl_acc.chi2_about_mean | 2022 | 21.7 |  | 20.1 |  | -1.6 |  |  |
| 4Mu_incl_acc: chi2 of v2 - v1 Run 3 eff. over 18 points | 2022 |  |  | 18.8 |  |  |  | 18 dof |
| acceptance_2018.mean | 2022 | 0.99958 |  | 0.99958 |  | 0 |  | 2018: same files |
| acceptance_2018.min | 2022 | 0.99845 |  | 0.99845 |  | 0 |  | 2018: same files |
| acceptance_2018.max | 2022 | 1 |  | 1 |  | 0 |  | 2018: same files |
| acceptance_run3.mean | 2022 | 0.6745 |  | 0.6744 |  | -0.0001 |  |  |
| acceptance_run3.min | 2022 | 0.354 |  | 0.3525 |  | -0.0015 |  |  |
| acceptance_run3.max | 2022 | 0.8846 |  | 0.8859 |  | 0.0013 |  |  |
| 2Mu2E_incl_acc.mean_pct (Run3-2018) | 2022EE | -0.38 | 0.0241216 | -0.352 | 0.0240407 | 0.028 | +0.8 |  |
| 2Mu2E_incl_acc.rms_pct | 2022EE | 0.18 |  | 0.156 |  | -0.024 |  |  |
| 2Mu2E_incl_acc.min_pct | 2022EE | -0.825 |  | -0.62 |  | 0.205 |  |  |
| 2Mu2E_incl_acc.max_pct | 2022EE | -0.108 |  | 0.015 |  | 0.123 |  |  |
| 2Mu2E_incl_acc.n_negative | 2022EE | 18 |  | 17 |  | -1 |  |  |
| 2Mu2E_incl_acc.chi2_about_mean | 2022EE | 26.9 |  | 19.6 |  | -7.3 |  |  |
| 2Mu2E_incl_acc: chi2 of v2 - v1 Run 3 eff. over 18 points | 2022EE |  |  | 18.2 |  |  |  | 18 dof |
| 4Mu_incl_acc.mean_pct (Run3-2018) | 2022EE | -0.037 | 0.00616456 | -0.032 | 0.00611788 | 0.005 | +0.6 |  |
| 4Mu_incl_acc.rms_pct | 2022EE | 0.043 |  | 0.046 |  | 0.003 |  |  |
| 4Mu_incl_acc.min_pct | 2022EE | -0.115 |  | -0.116 |  | -0.001 |  |  |
| 4Mu_incl_acc.max_pct | 2022EE | 0.052 |  | 0.055 |  | 0.003 |  |  |
| 4Mu_incl_acc.n_negative | 2022EE | 14 |  | 14 |  | 0 |  |  |
| 4Mu_incl_acc.chi2_about_mean | 2022EE | 29 |  | 30.4 |  | 1.4 |  |  |
| 4Mu_incl_acc: chi2 of v2 - v1 Run 3 eff. over 18 points | 2022EE |  |  | 32.5 |  |  |  | 18 dof |
| acceptance_2018.mean | 2022EE | 0.99958 |  | 0.99958 |  | 0 |  | 2018: same files |
| acceptance_2018.min | 2022EE | 0.99845 |  | 0.99845 |  | 0 |  | 2018: same files |
| acceptance_2018.max | 2022EE | 1 |  | 1 |  | 0 |  | 2018: same files |
| acceptance_run3.mean | 2022EE | 0.6747 |  | 0.6747 |  | 0 |  |  |
| acceptance_run3.min | 2022EE | 0.3538 |  | 0.3568 |  | 0.003 |  |  |
| acceptance_run3.max | 2022EE | 0.8848 |  | 0.886 |  | 0.0012 |  |  |
| 2Mu2E_incl_acc.mean_pct (Run3-2018) | 2023 | -0.153 | 0.0237342 | -0.138 | 0.0237073 | 0.015 | +0.4 |  |
| 2Mu2E_incl_acc.rms_pct | 2023 | 0.151 |  | 0.147 |  | -0.004 |  |  |
| 2Mu2E_incl_acc.min_pct | 2023 | -0.337 |  | -0.524 |  | -0.187 |  |  |
| 2Mu2E_incl_acc.max_pct | 2023 | 0.184 |  | 0.115 |  | -0.069 |  |  |
| 2Mu2E_incl_acc.n_negative | 2023 | 15 |  | 17 |  | 2 |  |  |
| 2Mu2E_incl_acc.chi2_about_mean | 2023 | 21.6 |  | 18.2 |  | -3.4 |  |  |
| 2Mu2E_incl_acc: chi2 of v2 - v1 Run 3 eff. over 18 points | 2023 |  |  | 32.2 |  |  |  | 18 dof |
| 4Mu_incl_acc.mean_pct (Run3-2018) | 2023 | 0.008 | 0.00584924 | 0.013 | 0.00586384 | 0.005 | +0.6 |  |
| 4Mu_incl_acc.rms_pct | 2023 | 0.042 |  | 0.046 |  | 0.004 |  |  |
| 4Mu_incl_acc.min_pct | 2023 | -0.07 |  | -0.038 |  | 0.032 |  |  |
| 4Mu_incl_acc.max_pct | 2023 | 0.103 |  | 0.134 |  | 0.031 |  |  |
| 4Mu_incl_acc.n_negative | 2023 | 9 |  | 7 |  | -2 |  |  |
| 4Mu_incl_acc.chi2_about_mean | 2023 | 27.8 |  | 31.7 |  | 3.9 |  |  |
| 4Mu_incl_acc: chi2 of v2 - v1 Run 3 eff. over 18 points | 2023 |  |  | 18.3 |  |  |  | 18 dof |
| acceptance_2018.mean | 2023 | 0.99958 |  | 0.99958 |  | 0 |  | 2018: same files |
| acceptance_2018.min | 2023 | 0.99845 |  | 0.99845 |  | 0 |  | 2018: same files |
| acceptance_2018.max | 2023 | 1 |  | 1 |  | 0 |  | 2018: same files |
| acceptance_run3.mean | 2023 | 0.6741 |  | 0.6746 |  | 0.0005 |  |  |
| acceptance_run3.min | 2023 | 0.3545 |  | 0.3537 |  | -0.0008 |  |  |
| acceptance_run3.max | 2023 | 0.8829 |  | 0.8852 |  | 0.0023 |  |  |
| 2Mu2E_incl_acc.mean_pct (Run3-2018) | 2023BPix | -0.161 | 0.02377 | -0.14 | 0.0236967 | 0.021 | +0.6 |  |
| 2Mu2E_incl_acc.rms_pct | 2023BPix | 0.158 |  | 0.142 |  | -0.016 |  |  |
| 2Mu2E_incl_acc.min_pct | 2023BPix | -0.466 |  | -0.481 |  | -0.015 |  |  |
| 2Mu2E_incl_acc.max_pct | 2023BPix | 0.096 |  | 0.134 |  | 0.038 |  |  |
| 2Mu2E_incl_acc.n_negative | 2023BPix | 15 |  | 15 |  | 0 |  |  |
| 2Mu2E_incl_acc.chi2_about_mean | 2023BPix | 22.4 |  | 19.2 |  | -3.2 |  |  |
| 2Mu2E_incl_acc: chi2 of v2 - v1 Run 3 eff. over 18 points | 2023BPix |  |  | 19.1 |  |  |  | 18 dof |
| 4Mu_incl_acc.mean_pct (Run3-2018) | 2023BPix | 0.015 | 0.00578336 | 0.008 | 0.00586653 | -0.007 | -0.8 |  |
| 4Mu_incl_acc.rms_pct | 2023BPix | 0.052 |  | 0.048 |  | -0.004 |  |  |
| 4Mu_incl_acc.min_pct | 2023BPix | -0.068 |  | -0.078 |  | -0.01 |  |  |
| 4Mu_incl_acc.max_pct | 2023BPix | 0.122 |  | 0.085 |  | -0.037 |  |  |
| 4Mu_incl_acc.n_negative | 2023BPix | 10 |  | 9 |  | -1 |  |  |
| 4Mu_incl_acc.chi2_about_mean | 2023BPix | 40.9 |  | 35.7 |  | -5.2 |  |  |
| 4Mu_incl_acc: chi2 of v2 - v1 Run 3 eff. over 18 points | 2023BPix |  |  | 9.5 |  |  |  | 18 dof |
| acceptance_2018.mean | 2023BPix | 0.99958 |  | 0.99958 |  | 0 |  | 2018: same files |
| acceptance_2018.min | 2023BPix | 0.99845 |  | 0.99845 |  | 0 |  | 2018: same files |
| acceptance_2018.max | 2023BPix | 1 |  | 1 |  | 0 |  | 2018: same files |
| acceptance_run3.mean | 2023BPix | 0.6743 |  | 0.6746 |  | 0.0003 |  |  |
| acceptance_run3.min | 2023BPix | 0.3534 |  | 0.3551 |  | 0.0017 |  |  |
| acceptance_run3.max | 2023BPix | 0.8864 |  | 0.8844 |  | -0.002 |  |  |
| 2Mu2E_incl_acc.mean_pct (Run3-2018) | 2024 |  |  | -0.025 | 0.0235183 |  |  |  |
| 2Mu2E_incl_acc.rms_pct | 2024 |  |  | 0.154 |  |  |  |  |
| 2Mu2E_incl_acc.min_pct | 2024 |  |  | -0.212 |  |  |  |  |
| 2Mu2E_incl_acc.max_pct | 2024 |  |  | 0.426 |  |  |  |  |
| 2Mu2E_incl_acc.n_negative | 2024 |  |  | 12 |  |  |  |  |
| 2Mu2E_incl_acc.chi2_about_mean | 2024 |  |  | 20.4 |  |  |  |  |
| 4Mu_incl_acc.mean_pct (Run3-2018) | 2024 |  |  | -0.027 | 0.00607992 |  |  |  |
| 4Mu_incl_acc.rms_pct | 2024 |  |  | 0.043 |  |  |  |  |
| 4Mu_incl_acc.min_pct | 2024 |  |  | -0.091 |  |  |  |  |
| 4Mu_incl_acc.max_pct | 2024 |  |  | 0.068 |  |  |  |  |
| 4Mu_incl_acc.n_negative | 2024 |  |  | 13 |  |  |  |  |
| 4Mu_incl_acc.chi2_about_mean | 2024 |  |  | 26.4 |  |  |  |  |
| acceptance_2018.mean | 2024 |  |  | 0.99958 |  |  |  | 2018: same files |
| acceptance_2018.min | 2024 |  |  | 0.99845 |  |  |  | 2018: same files |
| acceptance_2018.max | 2024 |  |  | 1 |  |  |  | 2018: same files |
| acceptance_run3.mean | 2024 |  |  | 0.6745 |  |  |  |  |
| acceptance_run3.min | 2024 |  |  | 0.3545 |  |  |  |  |
| acceptance_run3.max | 2024 |  |  | 0.8854 |  |  |  |  |

## lj_eff_summary.json (real)

| quantity | era | v1 | σ v1 | v2 | σ v2 | v2 − v1 | n σ | note |
|---|---|---|---|---|---|---|---|---|
| 2Mu2E_excl_acc.mean_pct (Run3-2018) | 2022 | 6.953 | 0.0455077 | 6.968 | 0.0455176 | 0.015 | +0.2 |  |
| 2Mu2E_excl_acc.rms_pct | 2022 | 7.513 |  | 7.429 |  | -0.084 |  |  |
| 2Mu2E_excl_acc.min_pct | 2022 | 0.337 |  | 0.74 |  | 0.403 |  |  |
| 2Mu2E_excl_acc.max_pct | 2022 | 23.532 |  | 23.621 |  | 0.089 |  |  |
| 2Mu2E_excl_acc.n_negative | 2022 | 0 |  | 0 |  | 0 |  |  |
| 2Mu2E_excl_acc.chi2_about_mean | 2022 | 12557.6 |  | 12289.7 |  | -267.9 |  |  |
| 2Mu2E_excl_acc: chi2 of v2 - v1 Run 3 eff. over 18 points | 2022 |  |  | 13.2 |  |  |  | 18 dof |
| 2Mu2E_incl_acc.mean_pct (Run3-2018) | 2022 | 7.415 | 0.0434884 | 7.37 | 0.0435253 | -0.045 | -0.7 |  |
| 2Mu2E_incl_acc.rms_pct | 2022 | 8.026 |  | 7.961 |  | -0.065 |  |  |
| 2Mu2E_incl_acc.min_pct | 2022 | 0.366 |  | 0.591 |  | 0.225 |  |  |
| 2Mu2E_incl_acc.max_pct | 2022 | 25.086 |  | 24.901 |  | -0.185 |  |  |
| 2Mu2E_incl_acc.n_negative | 2022 | 0 |  | 0 |  | 0 |  |  |
| 2Mu2E_incl_acc.chi2_about_mean | 2022 | 15642.2 |  | 15394.2 |  | -248 |  |  |
| 2Mu2E_incl_acc: chi2 of v2 - v1 Run 3 eff. over 18 points | 2022 |  |  | 15.5 |  |  |  | 18 dof |
| 4Mu_excl_acc.mean_pct (Run3-2018) | 2022 | 0.752 | 0.0426083 | 0.728 | 0.042607 | -0.024 | -0.4 |  |
| 4Mu_excl_acc.rms_pct | 2022 | 0.361 |  | 0.345 |  | -0.016 |  |  |
| 4Mu_excl_acc.min_pct | 2022 | 0.1 |  | 0.239 |  | 0.139 |  |  |
| 4Mu_excl_acc.max_pct | 2022 | 1.439 |  | 1.469 |  | 0.03 |  |  |
| 4Mu_excl_acc.n_negative | 2022 | 0 |  | 0 |  | 0 |  |  |
| 4Mu_excl_acc.chi2_about_mean | 2022 | 45.7 |  | 41.3 |  | -4.4 |  |  |
| 4Mu_excl_acc: chi2 of v2 - v1 Run 3 eff. over 18 points | 2022 |  |  | 23.1 |  |  |  | 18 dof |
| 4Mu_incl_acc.mean_pct (Run3-2018) | 2022 | 0.719 | 0.0395737 | 0.691 | 0.0395896 | -0.028 | -0.5 |  |
| 4Mu_incl_acc.rms_pct | 2022 | 0.435 |  | 0.403 |  | -0.032 |  |  |
| 4Mu_incl_acc.min_pct | 2022 | -0.058 |  | -0.051 |  | 0.007 |  |  |
| 4Mu_incl_acc.max_pct | 2022 | 1.429 |  | 1.477 |  | 0.048 |  |  |
| 4Mu_incl_acc.n_negative | 2022 | 1 |  | 1 |  | 0 |  |  |
| 4Mu_incl_acc.chi2_about_mean | 2022 | 72.7 |  | 65.8 |  | -6.9 |  |  |
| 4Mu_incl_acc: chi2 of v2 - v1 Run 3 eff. over 18 points | 2022 |  |  | 17.3 |  |  |  | 18 dof |
| acceptance_2018.mean | 2022 | 0.99958 |  | 0.99958 |  | 0 |  | 2018: same files |
| acceptance_2018.min | 2022 | 0.99845 |  | 0.99845 |  | 0 |  | 2018: same files |
| acceptance_2018.max | 2022 | 1 |  | 1 |  | 0 |  | 2018: same files |
| acceptance_run3.mean | 2022 | 0.6745 |  | 0.6744 |  | -0.0001 |  |  |
| acceptance_run3.min | 2022 | 0.354 |  | 0.3525 |  | -0.0015 |  |  |
| acceptance_run3.max | 2022 | 0.8846 |  | 0.8859 |  | 0.0013 |  |  |
| 2Mu2E_excl_acc.mean_pct (Run3-2018) | 2022EE | 6.642 | 0.0456145 | 6.695 | 0.0455805 | 0.053 | +0.8 |  |
| 2Mu2E_excl_acc.rms_pct | 2022EE | 7.459 |  | 7.429 |  | -0.03 |  |  |
| 2Mu2E_excl_acc.min_pct | 2022EE | 0.386 |  | 0.438 |  | 0.052 |  |  |
| 2Mu2E_excl_acc.max_pct | 2022EE | 23.081 |  | 23.272 |  | 0.191 |  |  |
| 2Mu2E_excl_acc.n_negative | 2022EE | 0 |  | 0 |  | 0 |  |  |
| 2Mu2E_excl_acc.chi2_about_mean | 2022EE | 12308.9 |  | 12288.5 |  | -20.4 |  |  |
| 2Mu2E_excl_acc: chi2 of v2 - v1 Run 3 eff. over 18 points | 2022EE |  |  | 10.3 |  |  |  | 18 dof |
| 2Mu2E_incl_acc.mean_pct (Run3-2018) | 2022EE | 6.99 | 0.0436804 | 7.024 | 0.0436483 | 0.034 | +0.6 |  |
| 2Mu2E_incl_acc.rms_pct | 2022EE | 8.002 |  | 7.968 |  | -0.034 |  |  |
| 2Mu2E_incl_acc.min_pct | 2022EE | 0.17 |  | 0.181 |  | 0.011 |  |  |
| 2Mu2E_incl_acc.max_pct | 2022EE | 24.677 |  | 24.547 |  | -0.13 |  |  |
| 2Mu2E_incl_acc.n_negative | 2022EE | 0 |  | 0 |  | 0 |  |  |
| 2Mu2E_incl_acc.chi2_about_mean | 2022EE | 15419.1 |  | 15446.2 |  | 27.1 |  |  |
| 2Mu2E_incl_acc: chi2 of v2 - v1 Run 3 eff. over 18 points | 2022EE |  |  | 13.9 |  |  |  | 18 dof |
| 4Mu_excl_acc.mean_pct (Run3-2018) | 2022EE | 0.949 | 0.0424865 | 0.874 | 0.0425171 | -0.075 | -1.2 |  |
| 4Mu_excl_acc.rms_pct | 2022EE | 0.396 |  | 0.326 |  | -0.07 |  |  |
| 4Mu_excl_acc.min_pct | 2022EE | 0.209 |  | 0.411 |  | 0.202 |  |  |
| 4Mu_excl_acc.max_pct | 2022EE | 1.539 |  | 1.485 |  | -0.054 |  |  |
| 4Mu_excl_acc.n_negative | 2022EE | 0 |  | 0 |  | 0 |  |  |
| 4Mu_excl_acc.chi2_about_mean | 2022EE | 50.2 |  | 36.2 |  | -14 |  |  |
| 4Mu_excl_acc: chi2 of v2 - v1 Run 3 eff. over 18 points | 2022EE |  |  | 11 |  |  |  | 18 dof |
| 4Mu_incl_acc.mean_pct (Run3-2018) | 2022EE | 0.833 | 0.0394677 | 0.842 | 0.039466 | 0.009 | +0.2 |  |
| 4Mu_incl_acc.rms_pct | 2022EE | 0.448 |  | 0.375 |  | -0.073 |  |  |
| 4Mu_incl_acc.min_pct | 2022EE | 0.012 |  | 0.34 |  | 0.328 |  |  |
| 4Mu_incl_acc.max_pct | 2022EE | 1.437 |  | 1.482 |  | 0.045 |  |  |
| 4Mu_incl_acc.n_negative | 2022EE | 0 |  | 0 |  | 0 |  |  |
| 4Mu_incl_acc.chi2_about_mean | 2022EE | 71.5 |  | 53.1 |  | -18.4 |  |  |
| 4Mu_incl_acc: chi2 of v2 - v1 Run 3 eff. over 18 points | 2022EE |  |  | 14.4 |  |  |  | 18 dof |
| acceptance_2018.mean | 2022EE | 0.99958 |  | 0.99958 |  | 0 |  | 2018: same files |
| acceptance_2018.min | 2022EE | 0.99845 |  | 0.99845 |  | 0 |  | 2018: same files |
| acceptance_2018.max | 2022EE | 1 |  | 1 |  | 0 |  | 2018: same files |
| acceptance_run3.mean | 2022EE | 0.6747 |  | 0.6747 |  | 0 |  |  |
| acceptance_run3.min | 2022EE | 0.3538 |  | 0.3568 |  | 0.003 |  |  |
| acceptance_run3.max | 2022EE | 0.8848 |  | 0.886 |  | 0.0012 |  |  |
| 2Mu2E_excl_acc.mean_pct (Run3-2018) | 2023 | 6.945 | 0.0455309 | 6.984 | 0.0454912 | 0.039 | +0.6 |  |
| 2Mu2E_excl_acc.rms_pct | 2023 | 7.499 |  | 7.502 |  | 0.003 |  |  |
| 2Mu2E_excl_acc.min_pct | 2023 | 0.358 |  | 0.474 |  | 0.116 |  |  |
| 2Mu2E_excl_acc.max_pct | 2023 | 23.648 |  | 23.897 |  | 0.249 |  |  |
| 2Mu2E_excl_acc.n_negative | 2023 | 0 |  | 0 |  | 0 |  |  |
| 2Mu2E_excl_acc.chi2_about_mean | 2023 | 12590.2 |  | 12428.3 |  | -161.9 |  |  |
| 2Mu2E_excl_acc: chi2 of v2 - v1 Run 3 eff. over 18 points | 2023 |  |  | 20.6 |  |  |  | 18 dof |
| 2Mu2E_incl_acc.mean_pct (Run3-2018) | 2023 | 7.873 | 0.043308 | 7.903 | 0.0432652 | 0.03 | +0.5 |  |
| 2Mu2E_incl_acc.rms_pct | 2023 | 8.026 |  | 8.049 |  | 0.023 |  |  |
| 2Mu2E_incl_acc.min_pct | 2023 | 0.749 |  | 0.785 |  | 0.036 |  |  |
| 2Mu2E_incl_acc.max_pct | 2023 | 25.464 |  | 25.995 |  | 0.531 |  |  |
| 2Mu2E_incl_acc.n_negative | 2023 | 0 |  | 0 |  | 0 |  |  |
| 2Mu2E_incl_acc.chi2_about_mean | 2023 | 15820.3 |  | 15712.9 |  | -107.4 |  |  |
| 2Mu2E_incl_acc: chi2 of v2 - v1 Run 3 eff. over 18 points | 2023 |  |  | 22.3 |  |  |  | 18 dof |
| 4Mu_excl_acc.mean_pct (Run3-2018) | 2023 | 1.424 | 0.0422534 | 1.492 | 0.0421814 | 0.068 | +1.1 |  |
| 4Mu_excl_acc.rms_pct | 2023 | 0.736 |  | 0.783 |  | 0.047 |  |  |
| 4Mu_excl_acc.min_pct | 2023 | 0.068 |  | 0.067 |  | -0.001 |  |  |
| 4Mu_excl_acc.max_pct | 2023 | 2.445 |  | 2.622 |  | 0.177 |  |  |
| 4Mu_excl_acc.n_negative | 2023 | 0 |  | 0 |  | 0 |  |  |
| 4Mu_excl_acc.chi2_about_mean | 2023 | 179.5 |  | 201.7 |  | 22.2 |  |  |
| 4Mu_excl_acc: chi2 of v2 - v1 Run 3 eff. over 18 points | 2023 |  |  | 23.8 |  |  |  | 18 dof |
| 4Mu_incl_acc.mean_pct (Run3-2018) | 2023 | 2.019 | 0.0387054 | 2.053 | 0.0386446 | 0.034 | +0.6 |  |
| 4Mu_incl_acc.rms_pct | 2023 | 0.9 |  | 0.908 |  | 0.008 |  |  |
| 4Mu_incl_acc.min_pct | 2023 | 0.421 |  | 0.43 |  | 0.009 |  |  |
| 4Mu_incl_acc.max_pct | 2023 | 3.252 |  | 3.393 |  | 0.141 |  |  |
| 4Mu_incl_acc.n_negative | 2023 | 0 |  | 0 |  | 0 |  |  |
| 4Mu_incl_acc.chi2_about_mean | 2023 | 307.9 |  | 315.7 |  | 7.8 |  |  |
| 4Mu_incl_acc: chi2 of v2 - v1 Run 3 eff. over 18 points | 2023 |  |  | 19.9 |  |  |  | 18 dof |
| acceptance_2018.mean | 2023 | 0.99958 |  | 0.99958 |  | 0 |  | 2018: same files |
| acceptance_2018.min | 2023 | 0.99845 |  | 0.99845 |  | 0 |  | 2018: same files |
| acceptance_2018.max | 2023 | 1 |  | 1 |  | 0 |  | 2018: same files |
| acceptance_run3.mean | 2023 | 0.6741 |  | 0.6746 |  | 0.0005 |  |  |
| acceptance_run3.min | 2023 | 0.3545 |  | 0.3537 |  | -0.0008 |  |  |
| acceptance_run3.max | 2023 | 0.8829 |  | 0.8852 |  | 0.0023 |  |  |
| 2Mu2E_excl_acc.mean_pct (Run3-2018) | 2023BPix | 7.012 | 0.0454975 | 7.027 | 0.0454561 | 0.015 | +0.2 |  |
| 2Mu2E_excl_acc.rms_pct | 2023BPix | 7.532 |  | 7.614 |  | 0.082 |  |  |
| 2Mu2E_excl_acc.min_pct | 2023BPix | 0.409 |  | 0.507 |  | 0.098 |  |  |
| 2Mu2E_excl_acc.max_pct | 2023BPix | 23.639 |  | 24 |  | 0.361 |  |  |
| 2Mu2E_excl_acc.n_negative | 2023BPix | 0 |  | 0 |  | 0 |  |  |
| 2Mu2E_excl_acc.chi2_about_mean | 2023BPix | 12580.4 |  | 12838.7 |  | 258.3 |  |  |
| 2Mu2E_excl_acc: chi2 of v2 - v1 Run 3 eff. over 18 points | 2023BPix |  |  | 13.4 |  |  |  | 18 dof |
| 2Mu2E_incl_acc.mean_pct (Run3-2018) | 2023BPix | 7.9 | 0.0432768 | 7.921 | 0.0432276 | 0.021 | +0.3 |  |
| 2Mu2E_incl_acc.rms_pct | 2023BPix | 8.067 |  | 8.155 |  | 0.088 |  |  |
| 2Mu2E_incl_acc.min_pct | 2023BPix | 0.769 |  | 0.755 |  | -0.014 |  |  |
| 2Mu2E_incl_acc.max_pct | 2023BPix | 25.797 |  | 25.937 |  | 0.14 |  |  |
| 2Mu2E_incl_acc.n_negative | 2023BPix | 0 |  | 0 |  | 0 |  |  |
| 2Mu2E_incl_acc.chi2_about_mean | 2023BPix | 15801 |  | 16170.7 |  | 369.7 |  |  |
| 2Mu2E_incl_acc: chi2 of v2 - v1 Run 3 eff. over 18 points | 2023BPix |  |  | 16.4 |  |  |  | 18 dof |
| 4Mu_excl_acc.mean_pct (Run3-2018) | 2023BPix | 1.571 | 0.0421475 | 1.591 | 0.0421531 | 0.02 | +0.3 |  |
| 4Mu_excl_acc.rms_pct | 2023BPix | 0.785 |  | 0.824 |  | 0.039 |  |  |
| 4Mu_excl_acc.min_pct | 2023BPix | 0.161 |  | 0.184 |  | 0.023 |  |  |
| 4Mu_excl_acc.max_pct | 2023BPix | 2.806 |  | 2.716 |  | -0.09 |  |  |
| 4Mu_excl_acc.n_negative | 2023BPix | 0 |  | 0 |  | 0 |  |  |
| 4Mu_excl_acc.chi2_about_mean | 2023BPix | 197.9 |  | 221 |  | 23.1 |  |  |
| 4Mu_excl_acc: chi2 of v2 - v1 Run 3 eff. over 18 points | 2023BPix |  |  | 13.1 |  |  |  | 18 dof |
| 4Mu_incl_acc.mean_pct (Run3-2018) | 2023BPix | 2.136 | 0.0385875 | 2.138 | 0.038598 | 0.002 | +0.0 |  |
| 4Mu_incl_acc.rms_pct | 2023BPix | 0.948 |  | 0.996 |  | 0.048 |  |  |
| 4Mu_incl_acc.min_pct | 2023BPix | 0.206 |  | 0.344 |  | 0.138 |  |  |
| 4Mu_incl_acc.max_pct | 2023BPix | 3.422 |  | 3.475 |  | 0.053 |  |  |
| 4Mu_incl_acc.n_negative | 2023BPix | 0 |  | 0 |  | 0 |  |  |
| 4Mu_incl_acc.chi2_about_mean | 2023BPix | 333 |  | 375.4 |  | 42.4 |  |  |
| 4Mu_incl_acc: chi2 of v2 - v1 Run 3 eff. over 18 points | 2023BPix |  |  | 13.3 |  |  |  | 18 dof |
| acceptance_2018.mean | 2023BPix | 0.99958 |  | 0.99958 |  | 0 |  | 2018: same files |
| acceptance_2018.min | 2023BPix | 0.99845 |  | 0.99845 |  | 0 |  | 2018: same files |
| acceptance_2018.max | 2023BPix | 1 |  | 1 |  | 0 |  | 2018: same files |
| acceptance_run3.mean | 2023BPix | 0.6743 |  | 0.6746 |  | 0.0003 |  |  |
| acceptance_run3.min | 2023BPix | 0.3534 |  | 0.3551 |  | 0.0017 |  |  |
| acceptance_run3.max | 2023BPix | 0.8864 |  | 0.8844 |  | -0.002 |  |  |
| 2Mu2E_excl_acc.mean_pct (Run3-2018) | 2024 |  |  | 6.631 | 0.0455628 |  |  |  |
| 2Mu2E_excl_acc.rms_pct | 2024 |  |  | 7.597 |  |  |  |  |
| 2Mu2E_excl_acc.min_pct | 2024 |  |  | 0.003 |  |  |  |  |
| 2Mu2E_excl_acc.max_pct | 2024 |  |  | 23.446 |  |  |  |  |
| 2Mu2E_excl_acc.n_negative | 2024 |  |  | 0 |  |  |  |  |
| 2Mu2E_excl_acc.chi2_about_mean | 2024 |  |  | 12794 |  |  |  |  |
| 2Mu2E_incl_acc.mean_pct (Run3-2018) | 2024 |  |  | 7.513 | 0.043413 |  |  |  |
| 2Mu2E_incl_acc.rms_pct | 2024 |  |  | 8.07 |  |  |  |  |
| 2Mu2E_incl_acc.min_pct | 2024 |  |  | 0.26 |  |  |  |  |
| 2Mu2E_incl_acc.max_pct | 2024 |  |  | 25.155 |  |  |  |  |
| 2Mu2E_incl_acc.n_negative | 2024 |  |  | 0 |  |  |  |  |
| 2Mu2E_incl_acc.chi2_about_mean | 2024 |  |  | 15777.4 |  |  |  |  |
| 4Mu_excl_acc.mean_pct (Run3-2018) | 2024 |  |  | 0.643 | 0.0425557 |  |  |  |
| 4Mu_excl_acc.rms_pct | 2024 |  |  | 0.673 |  |  |  |  |
| 4Mu_excl_acc.min_pct | 2024 |  |  | -0.546 |  |  |  |  |
| 4Mu_excl_acc.max_pct | 2024 |  |  | 1.708 |  |  |  |  |
| 4Mu_excl_acc.n_negative | 2024 |  |  | 4 |  |  |  |  |
| 4Mu_excl_acc.chi2_about_mean | 2024 |  |  | 142 |  |  |  |  |
| 4Mu_incl_acc.mean_pct (Run3-2018) | 2024 |  |  | 1.083 | 0.0392759 |  |  |  |
| 4Mu_incl_acc.rms_pct | 2024 |  |  | 0.759 |  |  |  |  |
| 4Mu_incl_acc.min_pct | 2024 |  |  | -0.309 |  |  |  |  |
| 4Mu_incl_acc.max_pct | 2024 |  |  | 2.294 |  |  |  |  |
| 4Mu_incl_acc.n_negative | 2024 |  |  | 2 |  |  |  |  |
| 4Mu_incl_acc.chi2_about_mean | 2024 |  |  | 209.1 |  |  |  |  |
| acceptance_2018.mean | 2024 |  |  | 0.99958 |  |  |  | 2018: same files |
| acceptance_2018.min | 2024 |  |  | 0.99845 |  |  |  | 2018: same files |
| acceptance_2018.max | 2024 |  |  | 1 |  |  |  | 2018: same files |
| acceptance_run3.mean | 2024 |  |  | 0.6745 |  |  |  |  |
| acceptance_run3.min | 2024 |  |  | 0.3545 |  |  |  |  |
| acceptance_run3.max | 2024 |  |  | 0.8854 |  |  |  |  |

## Differences beyond 2 standard deviations

Rows with an n σ value (v1 and v2 both present, both with an uncertainty), Run 3 eras; the 2018 rows read the same files in both productions and are counted apart: era_fingerprint.csv: 4 of 48 beyond 2σ (+12 2018 rows); regions (v1 bins): 0 of 38 beyond 2σ (+0 2018 rows); gen_comparison_summary.json: 0 of 25 beyond 2σ (+13 2018 rows); lj_eff_summary.json (geom): 1 of 8 beyond 2σ (+0 2018 rows); lj_eff_summary.json (real): 0 of 16 beyond 2σ (+0 2018 rows). Total: 5 of 135 beyond 2σ (expected about 6 for Gaussian errors; many rows are correlated, e.g. flagged and rectangle averages of one region), +25 2018 rows.

| section | quantity | era | v1 | v2 | n σ |
|---|---|---|---|---|---|
| era_fingerprint.csv | sqrt_s_frac_below_tail_cut | 2022EE | 0.02578 | 0.02473 | -2.1 |
| era_fingerprint.csv | vtx_pull_x | 2023 | 1.25 | -2.16 | -2.4 |
| era_fingerprint.csv | pileup_nTrueInt_mean | 2023BPix | 51.28 | 51.39 | +2.9 |
| era_fingerprint.csv | vtx_pull_x | 2023BPix | 0.82 | -2.29 | -2.2 |
| lj_eff_summary.json (geom) | 4Mu_incl_acc.mean_pct (Run3-2018) | 2022 | -0.039 | -0.02 | +2.2 |
