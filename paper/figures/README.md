# Figure manifest

The files in this directory are paper-use copies of existing experiment outputs. Their source artifacts remain unchanged in the project directory.

To regenerate the English paper-side renderings for Figs. 1–5:

```bash
conda run -n qz-rig-energy-ml python ./paper/figures/regenerate_english_figures.py
```

The source hashes and output list are recorded in `english_figures_manifest.json`.

| Paper figure | Local file | Source artifact | Evidence | Intended use | Current QA status |
|---|---|---|---|---|---|
| Fig. 1 | `fig02_load_feature_summary_en.png` | frozen calibrated load and state metrics | E1 | motivate state-dependent load variation and persistent peak behavior | visually checked; original retained |
| Fig. 2 | `fig03_typical_state_curves_en.png` | frozen calibrated load record | E1 | show representative operating-state profiles | visually checked; original retained |
| Fig. 3 | `fig04_target_domain_forecast_traces_en.png` | V17 forecast prediction archive | E2 | compare four prespecified representative forecasts over an overall segment and its largest target transition | rerendered with Target, iTransformer, SA-PT, CEF, and Persistence only; not field data |
| Fig. 4 | `fig05_target_domain_forecast_metrics_en.png` | V17 benchmark metrics | E2 | compare all candidates on MAE/RMSE/peak/transition MAE | abbreviations and two-decimal value labels used |
| Fig. 5 | `fig06_dispatch_connection_transition_en.png` | V17 aligned connection-transition trajectories | E2 + E1 protocol context | compare source power and SOC for RB, MLR, and RSS | reduced from seven to three prespecified main-comparison methods; illustrative replay only |
| Fig. 6 | `../experiments/results/extended_holdout_analysis.png` | `paper/experiments/summarize_extended_holdouts.py` from registered ten-holdout evidence | E9 + E13 + E15 | show seed-level controller differences and mean mechanism effects with bootstrap intervals | zero risk effect shown explicitly as a point and annotation, not an empty panel |
| Fig. 7 | `../experiments/results/risk_action_overlap.png` | `paper/experiments/summarize_risk_action_overlap.py` from registered ten-holdout trajectories | E13 | explain the gap between risk triggers, mode changes, and physical actions | visually checked; negative risk-channel result retained |
| Fig. 8 | `../experiments/results/v19_evidential_risk/v19_evidential_risk_audit.png` | frozen V19 six-holdout results plus independent paper-side audit | E17 + E18 | separate paired recognition changes from physical-action and shortage effects | visually checked at 2197×979 px; two panels; null shortage result printed on panel |
| Fig. 9 | `../experiments/results/emergency_priority_ablation/emergency_priority_ablation.png` | `paper/experiments/run_emergency_priority_ablation_summary.py` from V16 `emergency_load_metrics.csv` | E4 | compare critical-load protection with total shortage under stress | paper-side diagnostic figure; surrogate only; priority fractions require drilling/electrical/HSE approval |
| Fig. 10 | `fig07_joint_stress_critical_protection.png` | `artifacts/v16_expert_remediation_surrogate/joint_extreme_critical_load_protection.png` | E4 | show critical-load protection under joint stress | suitable limitations figure; label surrogate SIL and report total shortage |

## Figures intentionally not copied

The existing aggregate dispatch comparison plots contain overlapping method–scenario labels and are difficult to read at manuscript size. They remain in the source artifacts but are not used as paper figures. A clean English aggregate plot should be generated from the two frozen holdout CSV files if a main dispatch-results figure is needed.

## Final figure QA checklist

- Figures 1–8 now have paper-side renderings tied to their source artifacts; retain the original source images for provenance.
- Keep units on every axis and define unserved energy and SOC in captions.
- Add evidence ID, dataset identity, seed/scenario, and whether the result is a holdout or diagnostic.
- Use line style and markers in addition to color.
- Check readability at final two-column width.
- Do not crop away transition intervals, legends, or negative values.
