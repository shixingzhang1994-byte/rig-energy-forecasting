# Evidence chain

All paths under `rig-energy-forecasting/` resolve at code snapshot `3499ed37cb93684b90a30926be95400ac3f0a72c`. The manuscript and claim matrix resolve at project baseline `a636430956d75e7000ddd09fab4c0469e1253eb5`.

| Evidence block | Source and scripts | Frozen configuration / data identity | Retained result | Manuscript mapping | Status |
|---|---|---|---|---|---|
| V24 controller comparison | `src/rig_energy/optimization/dispatch.py`; `scripts/run_v24_protocol_seed.py`; `scripts/audit_v24_evidence.py` | `configs/v24_submission_revision.yaml`; `configs/v24_dispatch_matched.yaml`; per-seed evidence manifests | `artifacts/V24_submission_revision/confirmatory/analysis/analysis_manifest.json`; post-freeze audit | `paper/submission/body.tex`; `paper/review/claims-matrix-v28.md` | Implemented and validated within the frozen 12-seed simulator protocol |
| V24 component ablations | `scripts/run_v24_preregistered_ablations.py`; `scripts/audit_v24_ablations.py` | V24 frozen protocol and retained seeds | `artifacts/V24_submission_revision/preregistered_ablations/analysis_manifest.json`; `component_effect_summary.csv` | `paper/submission/body.tex`; claims matrix | Completed; risk-adaptive CVaR has no independently established stable benefit |
| V27 heterogeneous branch | `scripts/run_v27_protocol_seed.py`; `scripts/analyze_v27_heterogeneous.py` | `configs/v27_heterogeneous_validation.yaml`; frozen parameter sets | `artifacts/V27_heterogeneous_validation/confirmatory/seed_20261304/eligibility_failure_record.json` plus 11 retained unit results | `paper/submission/body.tex`; claims matrix | Executed but validation incomplete; no 12-unit efficacy conclusion |
| V28 public-data forecasting | `src/rig_energy/validation/public_electrical_forecast.py`; `scripts/run_v28_joint_public_forecast.py`; `scripts/analyze_v28_joint_public_forecast.py` | `configs/v28_public_real_forecast_frozen.yaml`; public dataset registry and preparation audits | `artifacts/v28_public_real_validation/forecast_confirmatory/forecast_confirmatory_analysis.json`; metrics and summary CSV files | `paper/submission/body.tex`; claims matrix | 24 retained dataset–horizon results; 1-minute result is mixed |
| V28 BESS and temporal checks | `scripts/run_v28_bess_observational.py`; `scripts/analyze_public_real_datasets.py`; `scripts/build_v28_experiment_evidence.py` | `configs/v28_bess_observational_r1_frozen.yaml`; `configs/public_real_dataset_registry.yaml` | BESS observational report; temporal external-validity assessment; final experiment evidence manifest | `paper/submission/body.tex`; claims matrix | Observational/component evidence only; not controller field efficacy |

## Integrity limits

- The V24 frozen manifests bind the relevant source and configuration files and passed the retained post-freeze audit.
- V28 result ledgers bind frozen configuration and data identities, but the original result JSON files do not contain a native source-commit field. This package closes that gap externally by pinning the entire code snapshot; it does not rewrite the frozen result files.
- Raw and prepared third-party data are intentionally absent. Their source metadata and retained preparation audits are included, but full byte-for-byte recomputation requires reacquisition.
