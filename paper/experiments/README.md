# Paper experiment records

本目录只保存论文侧的实验矩阵、协议说明和结果解释，不复制或覆盖项目原始实验产物。

## Current source artifacts

- Forecast comparison: `./rig-energy-forecasting/artifacts/v17_target_domain_adaptation/forecast/benchmark_metrics.csv`
- Frozen dispatch holdout 1: `./rig-energy-forecasting/artifacts/v15_generator_first_reserve/seed_20261011/`
- Frozen dispatch holdout 2: `./rig-energy-forecasting/artifacts/v15_generator_first_reserve/seed_20261012/`
- Locked extension holdout 3: `./rig-energy-forecasting/artifacts/paper_v15_extension_holdouts_v2/seed_20261014/`
- Locked extension holdout 4: `./rig-energy-forecasting/artifacts/paper_v15_extension_holdouts_v2/seed_20261015/`
- Registered extension-v3 holdouts 5--10: `./rig-energy-forecasting/artifacts/paper_v15_extension_holdouts_v3/seed_20261016/` through `seed_20261021/`
- Extension-v3 preregistration: `./paper/experiments/preregistered_extension_v3.yaml`
- Upstream-ineligible candidate record: `./rig-energy-forecasting/artifacts/paper_v15_extension_holdouts/seed_20261013/eligibility.json`
- Risk/target-domain acceptance: `./rig-energy-forecasting/artifacts/v17_user_designated_external_acceptance/acceptance_result.json`
- Frozen V19 evidential candidate and prospective holdouts: `./rig-energy-forecasting/artifacts/v19_evidential_risk/`
- V20 failed development search: `./rig-energy-forecasting/artifacts/v20_evidential_calibration/`
- Frozen V21 conflict-rule/calibration holdouts and independent audit: `./rig-energy-forecasting/artifacts/v21_probability_calibration/`
- Stress/missingness diagnostics: `./rig-energy-forecasting/artifacts/v16_expert_remediation_surrogate/`

## Reproduction check

```bash
cd ./rig-energy-forecasting
conda run -n qz-rig-energy-ml pytest -q
```

Observed on 2026-09-09: 108 passed, 5 warnings. The paper-side suite under
`paper/experiments` separately reports 14 passed.

## V20/V21 conflict-rule and probability-calibration audit

V20 screened 12 predeclared conflict-rule/temperature candidates on four
development seeds. None passed all original classification-superiority and
recall gates, so no V20 holdout was opened. The retained machine decision is
`artifacts/v20_evidential_calibration/development_audit.json`.

V21 asks a distinct calibration-only question. Candidate 9 (normalized
Dempster fusion, T=0.50), the ordered queue 20261040--20261051, first-six-
eligible stopping rule, gates, and action-attribution boundary were frozen
before outcomes. The six eligible seeds are 20261040, 20261041, 20261042,
20261045, 20261046, and 20261047.

Re-run the read-only project audit and paper reconstruction with:

```bash
cd ./rig-energy-forecasting
python scripts/audit_v21_probability_holdouts.py
cd .
python paper/experiments/summarize_v21_probability_calibration.py
```

The independently reconstructed calibrated-minus-raw-Dempster Brier effect is
−0.0386623 (six of six favorable; exact p=0.03125). The calibrated candidate
does not establish Brier superiority over RSS (three favorable, three adverse;
p=0.6875), while its Macro-F1 effect versus RSS is +0.0203198 on all six seeds.
Temperature scaling changes no argmax and does not enter dispatch. Aggregate
unserved energy is 466.0477 kWh for both RSS and the candidate; all 252
method--scenario trajectories pass the physical audit.

## Prospective evidential-risk interface audit

V19 replaces only the frozen RSS discrete-risk input with a causal Yager-rule evidence fusion interface. Candidate 59 was selected on development seeds 20261008 and 20261009 and frozen before the target-domain diagnostic and prospective holdouts. The prospective queue and six-eligible stopping rule were frozen in `rig-energy-forecasting/configs/v19_evidential_risk_holdouts.yaml`. The resulting eligible seeds are 20261022, 20261024, 20261025, 20261027, 20261028, and 20261029; seeds 20261023 and 20261026 remain recorded as upstream-ineligible before controller execution.

Regenerate the project-side evidence with:

```bash
cd ./rig-energy-forecasting
python scripts/run_v19_evidential_holdouts.py
```

Independently re-audit the frozen result files and regenerate the compact paper figure with:

```bash
cd .
python paper/experiments/summarize_v19_evidential_risk.py
python paper/experiments/plot_v19_evidential_risk.py
```

The paper-side audit records a dispatch Macro-F1 increase from 0.73 to 0.76 on all six seeds (exact two-sided sign-flip p = 0.03125), a Brier-score deterioration from 0.05 to 0.11, 53 changed physical action rows out of 6,480, and a 0.00 kWh shortage difference on every seed. The evidence therefore supports recognition and interpretability improvements, not calibrated probabilities or a control-outcome benefit.

## Citation structure check

The 2026-09-07 no-contact, offline structural check reports exactly:

`VERDICT-LINE: PASS: 0/49 verified, 0 errors, 0 warnings (0 checks skipped)`

This is parse/duplicate validation only: `0/49 verified` means that no live
index round-trip was attempted. No author email is requested or required for
this local check. The three 2025–2026 additions were separately resolved from
their DOI records before inclusion.

## Legacy staged ablation summary

Run:

```bash
conda run -n qz-rig-energy-ml python ./paper/experiments/run_ablation_summary.py
```

Outputs:

- `results/ablation_summary.csv`
- `results/ablation_summary.png`
- `results/ablation_summary_manifest.json`

This is an aggregation of the original two frozen V15 holdout rows and is retained only for provenance. The current manuscript uses the ten-holdout orthogonal risk-by-supervisor audit documented below.

## Independent full-MILP oracle reference

The standalone `./paper/experiments/run_independent_full_milp.py` re-encodes a deterministic full-horizon MILP without importing the project dispatch solver. It uses the realized load trajectory over each window, so it is a perfect-foresight oracle/reference value rather than a causal online competitor or formal lower bound. Regenerate the four recorded seed artifacts with:

```bash
for seed in 20261008 20261009 20261011 20261012; do
  conda run -n qz-rig-energy-ml python ./paper/experiments/run_independent_full_milp.py \
    --artifact-root ./rig-energy-forecasting/artifacts/v15_generator_first_reserve/seed_${seed} \
    --config ./rig-energy-forecasting/artifacts/v15_generator_first_reserve/seed_${seed}/resolved_dispatch_config.yaml \
    --output ./paper/experiments/results/independent_full_milp/seed_${seed}.json
done
```

The two frozen holdouts give 201.833 kWh total oracle unserved energy, with optimal status for all six unique scenario windows and physical checks passing.

## History-matched independent causal rolling full-MILP comparator

The standalone `./paper/experiments/run_independent_causal_full_milp.py` re-encodes a 12-step clustered unit-commitment/storage MILP without importing the project dispatch solver. It uses only the causal `Causal-ErrorFeedback-Ensemble` point forecast at each source index, holds currently measured capacity limits over the forecast window, solves every 5 s, and applies the first action. `./paper/experiments/export_shared_initial_histories.py` reconstructs the exact shared-warm-up startup/shutdown histories and verifies terminal SOC, generator power, and committed-unit count against the saved snapshot. The comparator intentionally omits the risk signal, forecast-error envelope, and SOC supervisory switching.

```bash
for seed in 20261011 20261012; do
  conda run -n qz-rig-energy-ml python ./paper/experiments/export_shared_initial_histories.py \
    --artifact-root ./rig-energy-forecasting/artifacts/v15_generator_first_reserve/seed_${seed} \
    --output ./paper/experiments/results/shared_initial_histories/seed_${seed}.json
  conda run -n qz-rig-energy-ml python ./paper/experiments/run_independent_causal_full_milp.py \
    --artifact-root ./rig-energy-forecasting/artifacts/v15_generator_first_reserve/seed_${seed} \
    --config ./rig-energy-forecasting/artifacts/v15_generator_first_reserve/seed_${seed}/resolved_dispatch_config.yaml \
    --initial-history ./paper/experiments/results/shared_initial_histories/seed_${seed}.json \
    --output ./paper/experiments/results/independent_causal_full_milp_history_matched/seed_${seed}.json
done
```

The same procedure is applied to all extension seeds using their extension artifact roots. Across ten holdouts, CR gives 1,165.81 kWh total unserved energy; all 30 trajectories pass the power-balance and charge/discharge audits. The unit tests in `tests/test_independent_causal_full_milp.py` verify binding minimum-up histories, empty histories, forced-capacity lock release, and netting of historical events to the units actually online.

## Ten independent holdouts and orthogonal mechanism audit

The primary statistical summary excludes development seeds and treats the seed, not the three scenarios, as the independent unit. The single command below discovers the ten registered eligible roots, reuses completed artifacts, creates any missing CR/risk/envelope interventions, and regenerates the summaries:

```bash
conda run -n qz-rig-energy-ml python ./paper/experiments/run_registered_extension_pipeline.py --workers 4
```

The recorded output is `results/extended_holdout_analysis.json`/`.csv`/`.png`. Exact two-sided sign-flip tests and seed-bootstrap intervals are reported at n = 10, with Holm correction across the three primary controller contrasts. The adjusted values are 0.015625, 0.0078125, and 0.005859375 before manuscript rounding; the inference boundary is the registered synthetic-holdout population.

`run_risk_signal_ablation.py` neutralizes only the predicted/dispatch risk level and reserve adder while holding the SOC thresholds, reserve actions, forecast envelope, capacities, windows, plant, costs, and solver fixed. Its ten output directories are under `results/risk_signal_ablation/`. The risk-on minus risk-off shortage is 0.00 kWh on every seed. `summarize_risk_action_overlap.py` produces `results/risk_action_overlap.json` and `.png`; it records 8.78% internal mode changes and 1.72% physical-action changes.

`run_no_uncertainty_ablation.py` removes forecast-envelope diversity while preserving the causal point forecast and every other declared control layer. Its ten paired outputs are under `results/uncertainty_envelope_ablation/`. Envelope-on minus envelope-off shortage is −2.12 ± 4.34 kWh per seed, exact p = 0.03125; all 210 ablation trajectories pass.

The emergency priority-load diagnostic is available under `results/emergency_priority_ablation/`. Run `conda run -n qz-rig-energy-ml python ./paper/experiments/run_emergency_priority_ablation_summary.py` to regenerate its CSV, manifest, and figure. It compares proportional curtailment with priority-greedy allocation over five V16 surrogate stress scenarios. Total unserved energy is unchanged by the allocation policy; the reported benefit is reduced critical-load shortage. The result is diagnostic only, and the priority fractions remain not operationally approved.

The former two-holdout aggregate dispatch figure under `results/dispatch_aggregate_comparison/` is retained for provenance but is no longer used in the manuscript.

For any new P0/P1 experiment, create a new versioned artifact directory in the project first, freeze its protocol, then add its path and claim mapping here and in `../evidence_register.md`. Never overwrite the original two V15 holdouts or replace an eligible extension seed after inspecting controller outcomes.
