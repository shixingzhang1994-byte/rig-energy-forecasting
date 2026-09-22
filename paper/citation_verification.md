# Citation verification record

Date: 2026-09-10  
Bibliography: `latex/refs.bib`  
Entries: 49

The final online multi-index check returned:

`VERDICT-LINE: FAIL: 46/49 verified, 3 errors, 0 warnings (0 checks skipped)`

The three remaining flags were manually reconciled against publisher or
institutional records and are metadata-matching false positives rather than
errors in the retained BibTeX entries:

1. `tereza2023riskaware`: Crossref reverses the first author's given and family
   names. ScienceDirect, Zenodo, the Slovak University of Technology author
   record, and CORDIS confirm the author as Tereza Ábelová.
2. `wu2021autoformer`: after DBLP TLS access failed, Crossref fuzzy matching
   returned an unrelated 2024 article. The official NeurIPS record confirms
   the retained Autoformer entry (Wu, Xu, Wang, and Long; NeurIPS 34; 2021;
   pp. 22419--22430). The unrelated DOI was not added.
3. `zhou2022fedformer`: after DBLP TLS access failed, Crossref fuzzy matching
   returned an unrelated 2026 book chapter. The official PMLR record confirms
   the retained FEDformer entry (ICML 2022; PMLR 162; pp. 27268--27286). The
   unrelated DOI was not added.

No unresolved warning or retraction flag was reported. The machine verdict is
preserved verbatim above and must not be represented as a clean automated
pass. Re-run the online check only if the bibliography changes or DBLP access
becomes available.

Primary reconciliation records:

- Ábelová paper: https://www.sciencedirect.com/science/article/abs/pii/S2405896323014441
- Autoformer: https://proceedings.neurips.cc/paper_files/paper/2021/hash/bcc0d400288793e8bdcd7c19a8ac0c2b-Abstract.html
- FEDformer: https://proceedings.mlr.press/v162/zhou22g.html
