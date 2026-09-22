# Energy Reports submission preflight — 2026-09-22

## Verdict

**Technical files pass the local preflight with manual conditions.** The project owner confirmed the revised six-author order, both affiliations, removal of shared first authorship, and the revised CRediT allocation. The manuscript compiles, the rendered PDF is legible, all fonts are embedded, PDF author metadata is empty, and the required CRediT, competing-interest, funding, data-availability, and generative-AI sections are present. Signed or email attestations from each added, retained, and removed author were not supplied for independent verification; retain those records before upload and recheck the live journal form for graphical-abstract and separate declaration-file requirements.

## Deterministic checks

- Target: Elsevier *Energy Reports*, original research, single-blind author-visible submission.
- Source: `paper/submission/main.tex` using `cas-sc` 2.4 with `a4paper,fleqn`; no manuscript-side margin or line-spacing compression was found.
- Abstract: 246 words; no citation or URL in the abstract.
- PDF: 21 physical pages comprising 1 Highlights page and 20 numbered manuscript pages; no certified hard page limit was available from the journal page.
- PDF metadata: title present; author field empty; no JavaScript; not encrypted.
- Fonts: all listed fonts are embedded and subset; no Type 3 fonts.
- Visual QA: all 21 rendered pages inspected; no clipping, overlap, missing figure, or unreadable glyph was found.
- Compilation: successful. The known CAS front-matter 117 pt internal overfull-box and empty-anchor warnings remain; no visible title-page clipping was found.

The bundled linter reports one `ERROR` and one `WARN`, both false positives caused by its generic section-title detector: the official multiline AI declaration exists at `paper/submission/main.tex:161`, and the Elsevier singular heading “Declaration of competing interest” exists at line 120. The raw JSON/text reports are retained without alteration.

## Author and metadata consistency

- The author order is identical in the manuscript, CRediT block, `CITATION.cff`, and `.zenodo.json`: Shixing Zhang; Pengchong Wei; Lei Luo; Yilei Wang; Yaoze Sun; Zhiming Liu.
- All six manuscript authors carry affiliations 1 and 2. The corresponding-author email is present for Shixing Zhang.
- Shixing Zhang is the sole first and corresponding author; the former shared-first-authorship statement has been removed.
- Every listed CRediT role uses a standard taxonomy label and every author has at least one role.
- Competing-interest text discloses employment by CNPC-affiliated companies; funding identifies project `2026DQ03150` and states the funder role.
- The Elsevier generative-AI disclosure names OpenAI Codex, describes its use, excludes authorship/method/result determination, and assigns responsibility to the authors.
- ORCID count is zero; add only verified identifiers if required.
- Data availability points to the current V24/V28 package branch and distinguishes it from the V21 DOI package. The pre-author-change tag and ZIP are superseded and must not be deposited.

## Required human confirmation before upload

1. Retain explicit approval from every retained, added, and removed author covering name spelling/order, both affiliations, removal of shared first authorship, CRediT roles, corresponding email, employment disclosure, funding statement, and approval of the final manuscript.
2. The corresponding author must confirm no concurrent submission and verify in the live Editorial Manager workflow whether a graphical abstract or separate competing-interest `.docx` is required. `Highlights.docx` is present; no graphical-abstract file is currently included.

## Policy sources checked

- Energy Reports Guide for Authors: https://www.sciencedirect.com/journal/energy-reports/publish/guide-for-authors
- Elsevier generative-AI policy: https://www.elsevier.com/about/policies-and-standards/generative-ai-policies-for-journals
- Elsevier publishing ethics: https://www.elsevier.com/about/policies-and-standards/publishing-ethics

No manuscript was submitted and no DOI deposit was created during this preflight.
