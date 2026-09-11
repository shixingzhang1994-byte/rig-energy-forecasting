# Comparison experiment matrix

Version: 2026-09-03  
Scope: paper preparation for *Energy Reports*  
Evidence rule: only numbers linked to an existing artifact are allowed in the manuscript. A planned or not-yet-run experiment must not be described as a result.

## 1. What is already available

The current evidence package contains two complementary comparison tracks:

1. **Forecasting track**: Persistence, XGBoost, LSTM, TCN, PatchTST, iTransformer, and state-aware/ensemble variants on the user-designated 72 h target-domain replay and the earlier public-evidence-constrained synthetic protocol.
2. **Dispatch track**: Rule-Based, Persistence-MPC, ML-Robust-MPC, Scenario-CVaR-MPC, Risk-Adaptive-CVaR-MPC, Risk-Aware-Ensemble-MPC, and Risk-SOC-Supervisory-MPC under the frozen V15 protocol.

The paper should present the second track as the main system comparison and the first track as the upstream forecast module comparison. The strongest upstream model on the target-domain replay is iTransformer; therefore the paper must not claim that the project forecasting model is the best predictor.

## 2. Forecasting comparison available for the manuscript

Source artifact: `./rig-energy-forecasting/artifacts/v17_target_domain_adaptation/forecast/benchmark_metrics.csv`  
Protocol: retrospective target-domain adaptation split; this is not a blind field test, and the underlying 72 h file remains classified as `synthetic_surrogate`.

| Model | MAE (kW) | RMSE (kW) | R² | Peak MAE (kW) | Transition MAE (kW) | Status |
|---|---:|---:|---:|---:|---:|---|
| iTransformer | 56.512 | 115.623 | 0.9395 | 46.421 | 361.817 | strongest target-domain predictor |
| PatchTST | 59.474 | 118.295 | 0.9367 | 55.852 | 388.972 | strong baseline |
| StateAware-Patch-Transformer | 60.957 | 120.836 | 0.9339 | 58.757 | 359.878 | state-aware baseline |
| Causal-ErrorFeedback-Ensemble | 61.894 | 120.218 | 0.9346 | 63.700 | 394.725 | project ensemble candidate |
| TCN | 66.434 | 125.618 | 0.9286 | 51.485 | 377.375 | neural baseline |
| Validation-Weighted-Ensemble | 68.271 | 126.150 | 0.9280 | 57.850 | 388.129 | ensemble baseline |
| StateAware-DualBranch-Patch-Transformer | 72.412 | 129.552 | 0.9240 | 68.868 | 370.530 | state-aware baseline |
| StateAware-TCN-Attention | 74.753 | 144.994 | 0.9049 | 63.508 | 370.950 | state-aware baseline |
| XGBoost | 87.089 | 156.913 | 0.8886 | 63.395 | 371.289 | tree baseline |
| LSTM | 98.011 | 162.779 | 0.8801 | 88.275 | 450.368 | recurrent baseline |
| Persistence | 119.096 | 238.764 | 0.7420 | 66.125 | 460.857 | naive baseline |

The manuscript may state that target-domain iTransformer reduces MAE by 52.55% relative to Persistence, but should also state that its R² is 0.9395, below the internal 0.95 development gate, and that it is a strong engineering candidate rather than a project-specific novelty claim.

The paper-side staged mechanism ablation is now available at `paper/experiments/results/ablation_summary.csv`. It reuses the two frozen holdout rows and does not rerun or modify the original artifacts. The four comparable rows are ML-Robust-MPC, Scenario-CVaR-MPC, Risk-Adaptive-CVaR-MPC, and Risk-SOC-Supervisory-MPC. Full emergency-disable and uncertainty-envelope-disable arms remain unavailable.

The uncertainty-envelope arm is now available as a separate single-seed diagnostic at `paper/experiments/results/no_uncertainty_envelope_v15_seed_20261011/`. It passed the independent physical audit, but it is not a new frozen acceptance and must be reported as a one-seed diagnostic. The emergency-load protection arm is available as a separate V16 proportional-versus-priority diagnostic at `paper/experiments/results/emergency_priority_ablation/`; it is not unified with the rolling dispatch benchmark and does not validate an operational protection scheme.

## 3. Dispatch comparison available for the manuscript

Source artifacts:  
- `./rig-energy-forecasting/artifacts/v15_generator_first_reserve/seed_20261011/method_summary.csv`
- `./rig-energy-forecasting/artifacts/v15_generator_first_reserve/seed_20261012/method_summary.csv`
- `./rig-energy-forecasting/artifacts/v15_generator_first_reserve/seed_20261011/result.json`
- `./rig-energy-forecasting/artifacts/v15_generator_first_reserve/seed_20261012/result.json`

The two frozen holdout seeds contain 21 method–scenario trajectories per seed. The proposed controller is `Risk-SOC-Supervisory-MPC`.

| Method | Total unserved energy (kWh, two holdouts) | Risk-adjusted cost (yuan, two holdouts) | Diesel fuel (L, two holdouts) | Worst max decision time (s) |
|---|---:|---:|---:|---:|
| Risk-SOC-Supervisory-MPC | 209.052 | 23,723.794 | 142.608 | 0.212 |
| ML-Robust-MPC | 210.593 | 23,990.944 | 142.349 | 0.107 |
| Rule-Based | 211.837 | 24,355.986 | 158.421 | 0.0001 |

The individual holdout results are:

| Seed | Proposed unserved (kWh) | Rule-Based (kWh) | ML-Robust-MPC (kWh) | Proposed/ML risk cost | Worst scenario regret (kWh) | Physical audit |
|---:|---:|---:|---:|---:|---:|---|
| 20261011 | 104.558 | 104.811 | 104.777 | 0.9914 | 0.000 | pass |
| 20261012 | 104.493 | 107.026 | 105.816 | 0.9863 | 0.000 | pass |

These results support the restrained claim that the controller achieved descriptive small improvements in a frozen public-evidence-constrained synthetic simulation: 1.315% lower unserved energy than Rule-Based and 0.732% lower than ML-Robust-MPC, with lower risk-adjusted cost than ML-Robust-MPC. They do not support a universal SOTA, statistical significance, or field-effectiveness claim.

## 4. Comparisons that should be added before submission if time permits

These are the highest-value missing comparisons, in order of reviewer impact:

| Priority | Experiment | Why it matters | Current state | Acceptance rule |
|---|---|---|---|---|
| P0 | Independent full MILP reference using locked SciPy/HiGHS | Distinguishes controller logic from solver implementation and catches objective/constraint mismatches | Complete: standalone causal rolling comparator plus perfect-foresight reference on four seeds | same causal point forecast/plant trace for the rolling arm; report information boundary and physical audit |
| P0 | Component ablation: no risk envelope, no SOC supervisor, no emergency protection | Identifies which mechanism produces the small improvement | Not available as a clean frozen manuscript table | predeclare variants; no test-set tuning; report shortage, cost, fuel, SOC violations, and runtime |
| P0 | Same forecast injected into every dispatcher | Separates forecast quality from dispatch policy | Partly available across V15 methods | hold forecast predictions and seed fixed; compare only dispatch policy |
| P1 | Combined stress test: grid derating + generator outage + storage power limit + SOC bias + telemetry delay | Tests the reviewer’s likely robustness question | Diagnostic exists, but total unserved remains high | report both critical-load and total unserved; do not claim elimination of risk |
| P1 | Missing-data block injection with retraining and no-retraining arms | Tests realistic SCADA degradation | Diagnostic only, not frozen V15 acceptance | predeclare missingness blocks and report degradation with confidence intervals |
| P1 | Three or more independent seeds for the final dispatch candidate | Prevents two-seed overinterpretation | Complete exploratory extension: four fixed seeds, two development and two frozen holdout | retain the two frozen holdouts as primary evidence; report seed and seed×scenario paired summaries without inferential claims |
| P2 | Runtime and scalability sweep over horizon, number of generators, and scenario count | Supports deployment feasibility | Basic runtime is available | report median/p95 solve time and failure-to-solve count |

The independent MILP and four-seed extensions are now recorded. The remaining component-ablation and deployment items are still future validation; the manuscript labels them as staged/diagnostic evidence rather than causal or field claims.

## 5. Fairness protocol for any new run

- Freeze the data file, split, prediction horizon, control horizon, initial SOC, terminal SOC handling, prices, penalties, and scenario seeds before execution.
- Use the same forecast output, timestamp alignment, warm-up, and missing-value treatment for every dispatcher in one comparison.
- Do not use future true load, future operation state, or future-derived transition flags as online features.
- Report both aggregate and per-scenario results; never allow one easy scenario to hide a safety loss in a transition scenario.
- Include physical audits: power balance, grid capacity, generator capacity, storage power, SOC bounds, charge/discharge exclusivity, integer commitments, and decision latency.
- Treat target-domain replay as replay of a synthetic surrogate unless the data provenance is independently verified.

## 6. Reproduction commands

The project environment is `qz-rig-energy-ml`.

```bash
cd ./rig-energy-forecasting
conda run -n qz-rig-energy-ml pytest -q
```

Observed on 2026-09-03: `85 passed, 5 warnings`.

The current artifacts are already generated. Do not overwrite them. Any new experiment must be written under a new versioned directory in the project and then referenced from `paper/`.
