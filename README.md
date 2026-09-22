# V24/V28 evidence-synchronized reproducibility package

This branch is the compact public audit package for the manuscript **“Forecast-informed reserve–SOC supervision for drilling-rig power systems: confirmatory simulation and public real-data checks.”** It binds the manuscript, frozen configurations, source code, tests, and retained compact evidence to immutable Git commits. It does not rerun or replace any experiment.

## Frozen identifiers

- Project baseline: `a636430956d75e7000ddd09fab4c0469e1253eb5`
- Project baseline tag: `baseline-v24-v28-20260922`
- Code and evidence snapshot: `3499ed37cb93684b90a30926be95400ac3f0a72c`
- Package branch: `reproducibility-package-v24-v28`

The `rig-energy-forecasting` Git submodule is pinned to the code and evidence snapshot above. The historical `reproducibility-package-v1` branch and `v1.0.0` tag remain the separate V21 package.

## Obtain and verify

```bash
git clone --branch reproducibility-package-v24-v28 --recurse-submodules \
  git@github.com:shixingzhang1994-byte/rig-energy-forecasting.git
cd rig-energy-forecasting
git submodule status
sha256sum -c MANIFEST.sha256
```

The expected submodule commit is `3499ed37cb93684b90a30926be95400ac3f0a72c`. `MANIFEST.sha256` covers all regular files in this package except the manifest itself; Git records the submodule commit separately.

## Included

- current CAS manuscript source, bibliography, required figures, and compiled PDF;
- all source, configuration, script, and test files in the pinned code snapshot;
- compact V24 confirmatory and ablation evidence;
- the V27 incomplete-validation record, including the ineligible unit;
- compact V28 dataset audits, forecast summaries, BESS observations, and final evidence ledger;
- claim and citation audit files used by the current manuscript.

See `EVIDENCE_CHAIN.md` for the code → configuration → data → result → manuscript mapping.

## Excluded

Raw public datasets, prepared Parquet files, prediction exports, model weights, trajectory-level bulk outputs, caches, and historical development runs are excluded because of size, third-party licensing, or non-authoritative status. Dataset identities and retrieval metadata remain in `rig-energy-forecasting/configs/public_real_dataset_registry.yaml` and the retained audit manifests.

This means the package supports source-level and compact-evidence audit immediately. Full recomputation of V28 requires reacquiring the third-party public datasets under their own terms. No target-rig synchronized measurements are included because none were used.

## Evidence boundary

- V24 supports a reliability–cost trade-off only within the frozen same-simulator, 12-seed protocol.
- V27 is incomplete by protocol and is not a successful validation version.
- V28 supports bounded public real-data forecasting, BESS-observation, and temporal-validity checks; it does not establish target-rig closed-loop efficacy, HIL validation, deployment safety, or cross-rig generalization.
- The package does not support claims of universal forecasting improvement, simultaneous EENS and cost reduction, or state-of-the-art performance.

## Licensing and citation

Code, scripts, tests, and configurations are licensed under MIT. Author-generated documentation and evidence artifacts are licensed under CC BY 4.0. Manuscript and third-party materials follow the exclusions and notices in `LICENSE` and `RIGHTS_NOTICE.md`. Citation metadata mirrors the current manuscript and should be rechecked before a DOI-bearing archive is finalized.
