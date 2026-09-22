# V28 final experiment evidence report

Overall audit status: **pass**

## What is complete

- Public-data preparation: 8 dataset families, 10 prepared electrical series, 13,571,131 one-minute rows, and 12,351,203 quality-passing rows (91.01%).
- Real electrical forecasting: 8 series × 3 horizons = 24 retained confirmatory results. At 1 minute, 5/8 improve directionally and 3/8 are Holm-significant; universal improvement is **not** supported. At 15 and 60 minutes, 8/8 and 8/8 beat persistence, respectively.
- Real BESS observation: M5BAT SOC-direction consistency is 97.98%; Tsukuba is 95.28%. These are physical/behavioral checks, not causal controller tests.
- Real drilling-process audit: both public records have a median state dwell of 2 minutes, versus 43 minutes in simulation. The simulator temporal-persistence assumption therefore fails this external check.
- Simulated controller evaluation: V24 retains 12 frozen seeds, all adverse/null results, and the reliability-cost tradeoff.

## Defensible real-data proportion

Measured public data primarily support **4/6 = 66.7%** of the six load-bearing evidence blocks. This clears the requested 60% threshold only under the explicitly defined evidence-block accounting. It is not a percentage of rows, not a percentage of every analysis in the repository, and not evidence that 66.7% of controller validation occurred in the field. Direct target-rig closed-loop field validation remains **0%**.

## Manuscript-safe conclusions

- Claim cross-domain real-data support at 15- and 60-minute forecasting horizons; report the mixed 1-minute result without a universal-improvement claim.
- Claim observational consistency of real BESS power/SOC behavior, not real-world efficacy of the proposed controller.
- Report the drilling temporal mismatch as a limitation and use it to motivate simulator recalibration or a robustness study.
- Report V24 as frozen same-simulator evidence of lower EENS with higher operating cost; do not describe it as field validation.
- Retain the invalid first M5BAT run and disclose the R1 unit correction: the released DC-power field is numerically kW despite its W suffix, as verified against annual current×voltage calculations.

## Integrity result

All 11 artifact-integrity checks pass. The frozen forecast and BESS config hashes match their reports, all 24 forecast prediction files are present, and the V24 post-freeze audit reports zero hash mismatches and zero technical failures.
