# V24/V28 claim--experiment matrix

Audit date: 2026-09-22. This matrix governs the current CAS submission draft;
the older `claims-matrix.md` is retained only as a V15--V21 historical record.

| Load-bearing claim | Manuscript location | Evidence | Status and boundary |
|---|---|---|---|
| RSS lowers mean EENS relative to RB, point-forecast MILP, and risk-adaptive residual-CVaR by 5.24, 15.59, and 5.49 kWh | Abstract; Results; Table `tab:primary-contrasts`; Conclusion | C35--C36 / E21 | Supported in the frozen 12-seed simulator population only |
| All three V24 primary contrasts have bootstrap intervals below zero and remain significant after Holm correction | Abstract; Results; Table `tab:primary-contrasts` | C36 / E21 | Supported; seed is the independent unit and three scenarios are clustered |
| RSS has higher realized operating cost in all three primary contrasts | Abstract; Results; Discussion; Conclusion | C37 / E21 | Supported adverse trade-off; no joint reliability/economic, equal-cost, or Pareto-superiority claim because no cost-weight sweep was run |
| V24 completed with no technical failure or hash mismatch; one retained seed used the declared fallback | Results | C38 / E21 | Supported; fallback is not certified learned-risk performance. Post hoc exclusion retains negative mean EENS differences for all three references, but does not replace the primary analysis |
| Generator-first execution and residual scenarios contribute to simulated reliability; adaptive CVaR is unresolved | Results; Discussion | C39 / E21 | Exploratory component evidence except the confirmatory supervisor contrast |
| The prepared public suite contains eight families and ten electrical series with 13,571,131 one-minute rows; 91.01% pass frozen quality rules | Data | C40 / E22 | Supported readiness statement; not a controller outcome |
| Selected forecasts beat persistence on 5/8 real series at 1 min and 8/8 at 15 and 60 min | Abstract; Results; Tables `tab:real-forecast` and `tab:real-forecast-detail`; Conclusion | C41--C42 / E23 | Supported across eight series from three dataset families; dataset-level table exposes three improvements, two degradations, and three inconclusive 1-min results |
| The selected model beats the 60-min rolling mean in 8/8 series at each horizon | Results | V28 `forecast_confirmatory_metrics.csv` | Supported descriptive secondary comparison; no multiplicity-controlled inference claimed |
| M5BAT and Tsukuba show 97.98% and 95.28% SOC-direction consistency | Abstract; Results; Table `tab:real-bess` | C43 / E24 | Supported observational BESS behavior under frozen power and SOC-change eligibility thresholds; no causal RSS credit or threshold-sensitivity claim |
| Pooled real drilling-state median dwell is 2 min in both records versus 43 min in simulation | Abstract; Results; Discussion; Limitations; Conclusion | C44 / E25 | Supported adverse assumption diagnostic; Energistics is 85.39% unknown and state mappings/mixes differ, so this is not state-conditional calibration |
| RSS decision latency has median/90th/95th/99th/max 0.009/0.828/1.560/2.413/4.802 s, with 0/12,960 over 5 s | Results; Limitations | V24 trajectory CSVs | Supported on the development platform only; not industrial real-time certification |
| V27 heterogeneous validation is incomplete because 1/12 units was ineligible before dispatch | Limitations | C45 / E26 | Supported protocol status; the other 11 units are not a replacement efficacy population |
| Priority allocation reduces critical-load but not total shortage | Results; Discussion | C9, C14 / E4 | Supported diagnostic only; class shares require site approval |

## Claim boundary used in the revision

The manuscript may emphasize that multiple public real-world datasets were
used for bounded component and assumption checks. It may not describe RSS as
field validated or as superior in real closed-loop operation. Real data support
drilling-process diagnostics, cross-domain forecasting, and BESS physical
behavior; RSS efficacy is still estimated in simulation.

The post hoc 4/6 evidence-block share (66.7%) is retained in the project ledger
for planning but is intentionally omitted from the manuscript because it is not
a standard or preregistered scientific endpoint.
