# Figure and table layout audit

The supplied PDFs under the figure directory were inspected without changing
their contents. The inclusion width and float type were changed in the LaTeX
sources so table-like composites remain within one column and diagrams and
plots use both columns.

| PDF | Role | Inclusion |
| --- | --- | --- |
| intro_example.pdf | table example | one-column figure at column width |
| case_study.pdf | table example | one-column figure at column width |
| overview.pdf | framework diagram | two-column figure at full text width |
| ablation_study.pdf | six-panel chart | two-column figure at 99% text width |
| case_study_epoch.pdf | training plot | centered two-column figure at 90% text width |

The active tabular environments in sections/Experiment.tex (dataset
statistics, overall performance, parameter analysis, loss parameter, and
efficiency) are wrapped in a resize operation constrained to the column
width. Their text, numbers, captions, labels, and ordering are unchanged;
the wrapper only prevents the original single-column tables from crossing
into the adjacent column.

The same width classification is reflected in the inactive appendix figure
blocks and in appendix/Preliminary.tex for source consistency.

Figure 4 (`case_study.pdf`) and Figure 5 (`case_study_epoch.pdf`) remain in
their original numerical order, but their float definitions are queued before
the Efficiency subsection. This changes only float scheduling: Figure 4 is
placed on page 7 and Figure 5 on page 8, while the captions, labels, image
files, data, and surrounding prose are unchanged. Figure 5 is centered across
the two columns at 90% of the text width so the complete main text, including
Limitations, ends on page 8 without an overrun.

## Validation

With the supplied five PDFs and the cross-checked references.bib, the project
builds with make. The resulting PDF has 11 pages total; the main text,
including the required Limitations section, occupies pages 1--8. The
supplemental-material statement and references continue on page 9, followed by
the appendix. The build has no overfull horizontal or vertical box
diagnostics. The only remaining layout note is the ACL float warning that page
7 is a float-only page, which is a consequence of placing the requested
full-width ablation chart.
