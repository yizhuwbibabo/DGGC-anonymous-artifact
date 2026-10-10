# Formatting-only change log

This file records every change made while converting the supplied source to the COLING 2027 ARR/*ACL review format.

## Files whose prose is unchanged

The following supplied files were copied byte-for-byte (the Related Work filename was changed only because spaces are inconvenient in `\input` paths):

- `sections/Introduction.tex`
- `sections/Related_Work.tex` (source: `Related Work.tex`)
- `sections/GaussianCL.tex`
- `sections/Experiment.tex`
- `appendix/Preliminary.tex`

## Required structural changes

- `main.tex` is a new ACL/ARR review wrapper. It selects the official `acl` review style, anonymous author metadata, the required packages, the existing abstract, the original section order, the mandatory `Limitations` section, the original Supplemental Material Statement, the bibliography hook, and the appendix hook.
- The original `Conclusion.tex` ended with a future-work sentence. That sentence is copied verbatim to `sections/Limitations.tex` so the required COLING `Limitations` section appears after the conclusion and before the references. No words were added to or removed from that sentence.
- `appendix/Appendix.tex` receives one structural `\end{itemize}` after the first inactive `\iffalse ... \fi` block. The supplied closing command was inside a later inactive block, which left the active list unclosed and prevented LaTeX compilation. No prose or list item was changed.
- `Related Work.tex` is stored as `sections/Related_Work.tex`; only the filename and corresponding `\input` path changed.

## Added project files and assets

- Official ACL style files: `acl.sty`, `acl_natbib.bst`, and copies under `acl-template/`.
- `Makefile`, `README.md`, `.gitignore`, this audit file, and `REFERENCE_CROSSCHECK.md`.
- The five supplied figures, copied without editing, under `figure/`.
- `references.bib`, containing exactly the 55 active citation entries selected from the supplied `sample-base.bib` after bidirectional cross-checking.

No placeholder bibliography or invented references were added. The 118 supplied but uncited entries were excluded from `references.bib` and listed in `REFERENCE_CROSSCHECK.md` for auditability.
