# V24/V28 DOI archive candidate record

Created: 2026-09-22

## Immutable sources

- Public archive tag: `v24-v28-reproducibility-20260922`
- Tag object: `0b3b501788df1136988242eb9b7abcb54a792795`
- Reproducibility-package commit: `0f9bee6ed535724afefa31483a54cb2d351062e7`
- Code and evidence submodule commit: `3499ed37cb93684b90a30926be95400ac3f0a72c`
- Preflighted manuscript source commit: `9efbe4125b4ebd118e25be2065f72ed7d60a2cea`

## Archive

- File: `reserve_soc_v24_v28_reproducibility_20260922.zip`
- Size: `5,561,974` bytes
- SHA-256: `64b5793787997abbc4b1e230ec56e08a11c5d52f527b6c311e5c08e495edf157`
- Sidecar: `reserve_soc_v24_v28_reproducibility_20260922.zip.sha256`

The ZIP was assembled only from the tagged package tree and the pinned code
submodule tree. Git metadata were excluded. The internal `MANIFEST.sha256`
passed in the assembled tree, and `unzip -t` reported no compressed-data
errors. The ZIP itself remains an ignored local release payload; this record
and its checksum sidecar are version-controlled.

## Scope and deposit status

The archive includes the technically preflighted manuscript, source,
configurations, tests, compact evidence, audit records, `CITATION.cff`, and
`.zenodo.json`. It excludes third-party raw data, prepared Parquet files,
prediction exports, model weights, bulk trajectories, caches, and historical
development runs.

Metadata are internally consistent across the manuscript, CRediT block,
`CITATION.cff`, and `.zenodo.json`. This does not replace factual approval by
all authors. As of this record, the archive has not been uploaded to a DOI
repository and no new DOI has been minted.
