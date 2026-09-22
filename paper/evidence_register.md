# Evidence register for the manuscript

Last updated: 2026-09-22  
Purpose: every load-bearing statement in the paper must map to one of the records below.

## Evidence classes

| ID | Evidence class | Allowed use |
|---|---|---|
| E1 | V15 frozen public-evidence-constrained synthetic simulation | Main dispatch comparison; report as simulation and descriptive holdout evidence |
| E2 | V17 user-designated 72 h replay, original provenance `synthetic_surrogate` | Target-domain replay comparison; do not call field data |
| E3 | Official public WITSML real-world drilling-process example without electrical active-power channels | Supports process/data-interface context only; cannot validate power forecasting or target-rig energy control |
| E4 | V16 engineering diagnostics | Limitations and robustness discussion only; not frozen V15 acceptance evidence |
| E5 | Software regression test | Reproducibility and implementation integrity only; not evidence of field performance |
| E6 | Frozen V15 resolved dispatch configuration and solver lock | Method parameters, cost coefficients, solver version, time limit, and MIP gap |
| E7 | V17 run metadata and preparation audit | Chronological split, train-only scaling, validation-only ensemble/error-envelope selection, and test-target exclusion |
| E8 | Standalone independent full-horizon MILP oracle on V15 trajectories | Perfect-foresight reference value; not a causal online competitor or formal lower bound |
| E9 | Registered ten-seed paired stability summary | Seed-level exact tests, bootstrap intervals, and Holm correction within the synthetic-holdout population |
| E10 | Standalone independent causal rolling full-MILP baseline on the same V15 windows | Fairer information-set comparator using the causal point forecast; it omits the proposed risk/SOC supervisory layer |
| E11 | Locked no-tuning V15 holdout extensions | Adds two plus six eligible independent seeds through result-blind queues; original V15 artifacts remain unchanged |
| E12 | History-matched standalone causal full-MILP extension | Causal comparator with exact shared warm-up startup/shutdown histories and explicit physical audits |
| E13 | Ten-holdout orthogonal risk-by-supervisor intervention | Mechanism attribution at seed level; the risk-on/off supervisory contrast is the clean marginal risk-channel test |
| E14 | Registered V15 extension-v3 protocol and freeze manifest | Predetermined 12-seed queue, six-eligible stopping rule, seed-level inference, and multiplicity policy |
| E15 | Ten-holdout forecast-envelope intervention | Paired mechanism attribution for envelope diversity with scenario/window/plant controls held fixed |
| E16 | Registered-extension evidence audit | Deterministic protocol, stopping-rule, seed, physical-audit, multiplicity, and SHA-256 manifest check |
| E17 | Frozen V19 evidential-risk candidate and prospective protocol | Candidate-space declaration, development-only selection, causal-input boundary, fixed queue, six-eligible stopping rule, and no-retuning constraint |
| E18 | Six-holdout prospective evidential-interface audit | Paired recognition, calibration, action, cost, shortage, physical-audit, and SHA-256 evidence checks; synthetic population only |
| E19 | V20 conflict-rule/calibration development audit | Retained fail-closed result: 12 candidates screened and none passed the original classification/recall gates; no V20 holdout opened |
| E20 | Frozen V21 six-holdout probability-calibration audit | Non-overlapping first-six-eligible queue; Dempster-rule and scalar-calibration decomposition; independent probability, action, cost, shortage, physical, and SHA-256 checks |
| E21 | Frozen V24 12-seed confirmatory controller study | Primary RSS EENS contrasts, operating-cost trade-off, seed-cluster intervals, exact sign-flip tests, Holm correction, technical gates, and post-freeze audit; synthetic active-power population only |
| E22 | V28 public real-data readiness ledger | Eight public dataset families, ten prepared one-minute electrical series, source/prepared hashes, quality rules, license/provenance, and explicit evidence boundaries |
| E23 | V28 frozen public electrical forecasting experiment | Eight real electrical series at 1/15/60-min horizons, test-open-once policy, daily cluster bootstrap, and Holm-adjusted 1-min family |
| E24 | V28 public real BESS observational audit | M5BAT Pb1 and Tsukuba power/SOC consistency, efficiency, tracking, and quality metrics; not a causal RSS controller test |
| E25 | Public real drilling-process external-validity audit | Energistics Well B and Utah FORGE state distributions and dwell times compared with the simulator; no synchronized rig-bus active power |
| E26 | V27 heterogeneous-simulation eligibility record | Eleven technically passing units plus one prespecified unit with zero eligible stable-drilling windows before controller execution; protocol-level efficacy conclusion incomplete |

## Claim-to-artifact map

| Claim ID | Manuscript claim | Evidence | Artifact |
|---|---|---|---|
| C1 | Target-domain iTransformer is the strongest available predictor in the current replay | E2 | `/home/zsx/桌面/QZ/rig-energy-forecasting/artifacts/v17_target_domain_adaptation/forecast/benchmark_metrics.csv` |
| C2 | iTransformer obtains MAE 56.512 kW, RMSE 115.623 kW, and R² 0.9395 on the target-domain retrospective split | E2 | same as C1 |
| C3 | iTransformer improves MAE by 52.55% relative to Persistence in that replay | E2 | same as C1; compute against Persistence row |
| C4 | RSS has 813.43 kWh total unserved energy across ten independent eligible holdouts | E1 + E11 + E14 | original `seed_20261011`/`seed_20261012`, v2 `seed_20261014`/`seed_20261015`, and v3 `seed_20261016`--`seed_20261021` method summaries |
| C5 | RSS is 3.27%, 10.22%, and 30.23% below RB, MLR, and history-matched CR in aggregate unserved energy | E11 + E12 + E14 | `/home/zsx/桌面/QZ/paper/experiments/results/extended_holdout_analysis.json` |
| C6 | RSS has 95,366.92 yuan aggregate risk-adjusted cost versus 105,192.33 yuan for MLR over ten holdouts | E1 + E11 + E14 | ten method-summary files listed for C4 |
| C7 | All 210 main, 30 risk-neutralized, 210 envelope-ablation, and 30 history-matched CR trajectories pass their recorded physical audits | E11 + E12 + E13 + E15 | ten main results; ten risk-ablation results; ten uncertainty-ablation results; ten history-matched CR JSON files |
| C8 | The current risk classifier does not meet the preregistered Macro-F1 and high-risk recall gates | E2 | `/home/zsx/桌面/QZ/rig-energy-forecasting/artifacts/v17_user_designated_external_acceptance/acceptance_result.json` |
| C9 | The combined extreme diagnostic reduces critical-load unserved energy but leaves high total unserved energy | E4 | `/home/zsx/桌面/QZ/rig-energy-forecasting/artifacts/v16_expert_remediation_surrogate/final_summary.json` |
| C10 | The official WITSML example cannot validate total active-power forecasting | E3 | V18 status record and WITSML audit artifacts in the project handoff |
| C11 | The current code regression suites pass the counts recorded in `paper/progress.md` | E5 | project and `paper/experiments` pytest commands in the declared environment |
| C12 | In the ten-holdout 2×2 audit, the reserve–SOC supervisor changes mean seed-level unserved energy by −2.86 kWh without adaptive risk and −4.11 kWh with adaptive risk; neither contrast is adverse on any seed | E13 | `/home/zsx/桌面/QZ/paper/experiments/results/extended_holdout_analysis.json`; `.csv` |
| C13 | Removing forecast-envelope diversity increases aggregate shortage from 813.43 to 834.61 kWh; the paired mean effect is −2.12 kWh, exact p = 0.03, with 6 improvements and 4 ties | E15 | `/home/zsx/桌面/QZ/paper/experiments/results/uncertainty_envelope_ablation/`; `extended_holdout_analysis.json` |
| C14 | Under five V16 surrogate stress scenarios, priority allocation keeps total unserved energy unchanged while reducing critical-load unserved energy by 208.162 kWh in the base case and 803.748 kWh under joint extreme stress | E4 diagnostic | `/home/zsx/桌面/QZ/paper/experiments/results/emergency_priority_ablation/emergency_priority_ablation.csv`; manifest records source hash and approval boundary |
| C15 | Frozen dispatch uses the declared cost coefficients and HiGHS/SciPy solver lock | E6 | `artifacts/v15_generator_first_reserve/seed_20261011/resolved_dispatch_config.yaml`; `artifacts/v17_user_designated_external_acceptance/dispatch_v15_frozen_data_aligned/multi_period_highs_solver_lock.json` |
| C16 | Target-domain preprocessing and model selection exclude test targets from fitting and validation selection | E7 | `artifacts/v17_target_domain_adaptation/forecast/run_metadata.json`; `artifacts/v17_target_domain_adaptation/data/preparation_audit.json`; source code `src/rig_energy/experiment.py` |
| C17 | Independent deterministic full-MILP oracle gives 201.833 kWh total unserved energy on the two frozen holdout windows, with optimal status for all six unique scenario windows and physical residual checks passing | E8 | `/home/zsx/桌面/QZ/paper/experiments/run_independent_full_milp.py`; `/home/zsx/桌面/QZ/paper/experiments/results/independent_full_milp/seed_20261011.json`; `seed_20261012.json` |
| C18 | RSS minus RB is −2.75 ± 5.39 kWh and RSS minus MLR is −9.26 ± 11.31 kWh; Holm-adjusted exact p-values are 0.02 and 0.01 | E9 + E14 | `/home/zsx/桌面/QZ/paper/experiments/results/extended_holdout_analysis.json`; `.csv` |
| C19 | The clean risk-on minus risk-off shortage contrast is 0.00 kWh on every seed; risk changes internal mode on 8.78% of steps and physical actions on 1.72% | E13 | `/home/zsx/桌面/QZ/paper/experiments/results/risk_signal_ablation/`; `risk_action_overlap.json`; `risk_action_overlap.png` |
| C20 | History-matched CR gives 1,165.81 kWh across ten holdouts; all 30 trajectories pass, mean solve time rounds to 0.01 s per decision, and the maximum is 0.19 s | E12 | `/home/zsx/桌面/QZ/paper/experiments/run_independent_causal_full_milp.py`; `/home/zsx/桌面/QZ/paper/experiments/results/independent_causal_full_milp_history_matched/seed_*.json` |
| C21 | Seed 20261013 was excluded before any controller execution because it failed the locked overall-load calibration gate; thresholds were not relaxed | E11 | `/home/zsx/桌面/QZ/rig-energy-forecasting/artifacts/paper_v15_extension_holdouts/seed_20261013/eligibility.json`; locked replacement config and extension manifest |
| C22 | The v3 primary extension stopped after the first six eligible seeds (20261016--20261021), and later seeds were not run under that protocol | E14 | `/home/zsx/桌面/QZ/rig-energy-forecasting/configs/paper_v15_extension_holdouts_v3.yaml`; `/home/zsx/桌面/QZ/rig-energy-forecasting/artifacts/paper_v15_extension_holdouts_v3/freeze_manifest.json` |
| C23 | The three primary RSS contrasts have Holm-adjusted exact p-values 0.02, 0.01, and 0.01 at manuscript display precision | E9 + E14 | `/home/zsx/桌面/QZ/paper/experiments/results/extended_holdout_analysis.json`, field `primary_multiplicity` |
| C24 | The extension audit passes all registered checks and hashes 73 protocol/code/result files | E16 | `/home/zsx/桌面/QZ/paper/experiments/results/registered_extension_v3_audit.json` |
| C25 | V19 candidate 59 was selected on development seeds 20261008/20261009 and frozen before target-domain diagnosis and prospective holdouts | E17 | `/home/zsx/桌面/QZ/rig-energy-forecasting/artifacts/v19_evidential_risk/risk_candidate_freeze.json`; `prospective_run_freeze.json` |
| C26 | The V19 prospective set is the first six upstream-eligible queue members (20261022, 20261024, 20261025, 20261027, 20261028, 20261029); 20261023 and 20261026 remain recorded as ineligible before controller execution | E17 + E18 | `/home/zsx/桌面/QZ/rig-energy-forecasting/artifacts/v19_evidential_risk/prospective_source/seed_*/eligibility.json`; `prospective_summary.json` |
| C27 | E-RSS increases mean dispatch Macro-F1 from 0.73 to 0.76 on all six prospective seeds (mean delta 0.03, bootstrap 95% interval 0.02--0.04, exact p = 0.03) | E18 | `/home/zsx/桌面/QZ/paper/experiments/results/v19_evidential_risk/v19_paper_audit.json`; `v19_seed_metrics.csv` |
| C28 | E-RSS worsens mean multiclass Brier score from 0.05 to 0.11 on all six prospective seeds (mean delta 0.06, exact p = 0.03) | E18 | same as C27; raw probability rows independently audited for normalization |
| C29 | E-RSS changes 53/6,480 physical action rows but changes unserved energy by 0.00 kWh on all six seeds; aggregate cost difference is not significant (p = 0.44) | E18 | project V19 per-seed `result.json`; `prospective_summary.json`; paper-side V19 audit |
| C30 | None of 12 V20 conflict-rule/temperature candidates passed all original classification-superiority and recall gates, so V20 stopped before holdout evaluation | E19 | `/home/zsx/桌面/QZ/rig-energy-forecasting/artifacts/v20_evidential_calibration/development_audit.json`; `development_search/candidate_results.json` |
| C31 | V21 candidate 9 (normalized Dempster, T = 0.50) and queue 20261040–20261051 were frozen before outcomes; the first six eligible seeds are 20261040, 20261041, 20261042, 20261045, 20261046, and 20261047 | E20 | `/home/zsx/桌面/QZ/rig-energy-forecasting/artifacts/v21_probability_calibration/run_freeze.json`; `execution_clarification_1.json`; `independent_audit.json` |
| C32 | DC-RSS reduces mean Brier score from 0.07 for raw D-RSS to 0.03 after calibration on all six V21 seeds (mean −0.04; 56.73%; exact p = 0.03); the raw Dempster-versus-Yager rule effect is −0.03 | E20 | `/home/zsx/桌面/QZ/paper/experiments/results/v21_probability_calibration/v21_paper_audit.json`; `v21_seed_metrics.csv` |
| C33 | DC-RSS and RSS both have mean Brier 0.03 at display precision; three of six seeds favor each and exact p = 0.69, so V21 does not establish probability superiority over RSS | E20 | same as C32; project `independent_audit.json` |
| C34 | DC-RSS raises mean Macro-F1 from 0.75 to 0.77, retains mean high/severe recall 0.99/0.98, changes 57/6,480 actions versus RSS, and leaves aggregate unserved energy unchanged at 466.05 kWh; scalar calibration changes no argmax and does not enter dispatch | E20 | same as C32; project V21 per-seed `result.json` and `independent_audit_manifest.json` |
| C35 | Across 12 frozen V24 seeds, RSS mean EENS is 56.35 kWh versus 61.59 for RB, 71.94 for point-forecast MILP, and 61.85 for risk-adaptive residual-CVaR MILP | E21 | `/home/zsx/桌面/QZ/rig-energy-forecasting/artifacts/V24_submission_revision/confirmatory/analysis/method_summary.csv` |
| C36 | The three V24 primary RSS EENS differences are −5.24, −15.59, and −5.49 kWh, with bootstrap 95% intervals excluding zero and Holm-adjusted exact p-values 0.0396, 0.00293, and 0.00586 | E21 | `/home/zsx/桌面/QZ/rig-energy-forecasting/artifacts/V24_submission_revision/confirmatory/analysis/primary_contrasts.csv` |
| C37 | V24 RSS realized operating cost is higher by 195.98, 280.17, and 153.83 yuan in the three primary contrasts | E21 | same as C36; `mean_operating_cost_difference_yuan` |
| C38 | The V24 post-freeze audit has zero freeze-hash mismatches, zero seed-manifest mismatches, and zero technical failures; one retained seed uses the declared always-severe risk fallback | E21 | `/home/zsx/桌面/QZ/rig-energy-forecasting/artifacts/V24_submission_revision/confirmatory/postfreeze_audit/audit_manifest.json` |
| C39 | V24 leave-one-component-out interventions support generator-first execution and residual scenarios, while the isolated adaptive-CVaR effect has an interval spanning zero | E21 | `/home/zsx/桌面/QZ/rig-energy-forecasting/artifacts/V24_submission_revision/preregistered_ablations/component_effect_summary.csv` |
| C40 | V28 prepared 10 one-minute electrical series with 13,571,131 rows, of which 12,351,203 (91.01%) pass frozen source-specific quality rules and none has an imputed target | E22 | `/home/zsx/桌面/QZ/rig-energy-forecasting/artifacts/v28_public_real_validation/data_readiness/dataset_readiness_manifest.json`; `DATASET_READINESS.md` |
| C41 | The frozen V28 forecast experiment retains all 24 dataset--horizon combinations; selected models beat persistence on 5/8 series at 1 min and 8/8 at both 15 and 60 min | E23 | `/home/zsx/桌面/QZ/rig-energy-forecasting/artifacts/v28_public_real_validation/forecast_confirmatory/forecast_confirmatory_analysis.json` |
| C42 | At 1 min, three of eight forecast effects are Holm-significant improvements; two are significant degradations; universal improvement is not supported | E23 | `/home/zsx/桌面/QZ/rig-energy-forecasting/artifacts/v28_public_real_validation/forecast_confirmatory/forecast_confirmatory_summary.csv` |
| C43 | M5BAT and Tsukuba SOC-direction consistency is 97.98% and 95.28%, respectively; their power--SOC-change correlations are −0.6568 and −0.8223 | E24 | `/home/zsx/桌面/QZ/rig-energy-forecasting/artifacts/v28_public_real_validation/bess_observational/bess_observational_report.json` |
| C44 | Energistics Well B and Utah FORGE have 2-min median state dwell versus 43 min in simulation; simulator temporal persistence is not externally supported | E25 | `/home/zsx/桌面/QZ/rig-energy-forecasting/artifacts/public_real_multi_dataset_validation/temporal_external_validity_assessment.json` |
| C45 | V27 is incomplete because one of 12 prespecified heterogeneous units had no eligible stable-drilling window before dispatch; the other 11 are not used as a replacement efficacy population | E26 | `/home/zsx/桌面/QZ/rig-energy-forecasting/artifacts/V27_heterogeneous_validation/confirmatory/seed_20261304/eligibility.json`; `configs/v27_heterogeneous_validation.yaml` |

Claims C4--C24 describe the earlier V15/V19--V21 manuscript cycle and remain
for traceability. They are not the current confirmatory controller results and
must not be substituted for C35--C39 in the revised abstract, main table,
discussion, or conclusion.

## Forbidden statements

The following statements are not supported by the current evidence package:

- “field validated”, “industrial deployment completed”, or “validated on real target-rig SCADA”;
- “statistically significant SOTA” or “universally superior to all advanced methods”;
- “all metrics are optimal”;
- “the emergency layer eliminates unserved energy under combined failures”;
- “the three-tier load proportions are approved for field operation”;
- “iTransformer is our methodological innovation”;
- “the current discrete risk classifier independently improves dispatch outcomes” or “the current risk channel is a validated control contribution”.
- “raw E-RSS pignistic scores are calibrated probabilities”, “E-RSS or DC-RSS reduces shortage”, “DC-RSS is probabilistically superior to RSS”, or “evidence theory improves every risk metric on every seed”.
- “validated on a real drilling rig”, “real-world closed-loop controller superiority”, or “66.7% field validated”.
- “RSS lowers both EENS and realized operating cost in V24”.
- “the forecasting method improves every real series at 1 min”.

## Author verification before submission

1. Reopen the source CSV/JSON files and verify every number copied into the manuscript.
2. Replace any placeholder marked `TO VERIFY` only after the corresponding artifact exists.
3. Add the final figure/table file names and data provenance after the author creates them.
4. Re-run the manuscript consistency check after every result change.
