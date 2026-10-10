# COLING 2027 ARR submission scaffold

This directory is a formatting-only submission copy. The scientific method, results, and body wording are retained from the supplied files.

## Build

Place the official ARR/ACL style files (`acl.sty`, `acl_natbib.bst`) beside `main.tex` or keep the copies under `acl-template/` and add that directory to `TEXINPUTS`. The cross-checked bibliography is now provided as `references.bib`; the referenced figures are under `figure/` (the directory name follows the supplied section files).

Example:

```bash
make
```

Equivalent explicit commands:

```bash
TEXINPUTS=./acl-template:$TEXINPUTS pdflatex -interaction=nonstopmode main.tex
TEXINPUTS=./acl-template:$TEXINPUTS bibtex main
TEXINPUTS=./acl-template:$TEXINPUTS pdflatex -interaction=nonstopmode main.tex
TEXINPUTS=./acl-template:$TEXINPUTS pdflatex -interaction=nonstopmode main.tex
```

Use the ARR review template and `review` option for the ARR submission. Do not use `final` for review.

## Required author action before upload

- `references.bib` contains the 55 active cited entries extracted from the supplied `sample-base.bib`; see `REFERENCE_CROSSCHECK.md` for the bidirectional audit.
- The five supplied paper figures are already under `figure/`: `intro_example.pdf`, `overview.pdf`, `ablation_study.pdf`, `case_study.pdf`, and `case_study_epoch.pdf`.
- Confirm the anonymous title/author metadata in `main.tex` matches the ARR form.
- Confirm the main-body PDF is at most eight pages. References and the appendix are outside the main-body limit under the COLING 2027 CFP.
- Keep the mandatory Limitations section after Conclusion and before References.
- Upload supplementary material separately if requested by ARR; it must remain anonymous and the paper must remain self-contained.

`appendix/Preliminary.tex` is preserved as supplied but is not included by the original paper wrapper; it is retained for source completeness. `sections/Limitations.tex` contains the existing future-work sentence moved verbatim into the mandatory COLING section. The appendix receives one structural `\end{itemize}` closure because the supplied source placed that command inside an inactive `\iffalse` block; no prose or scientific content was changed.
