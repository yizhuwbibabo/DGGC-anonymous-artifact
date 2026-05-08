# Dataset structure

The public datasets are not redistributed in this anonymous package. Place them
under a local data directory and set `DATA_ROOT`.

Expected structure:

```text
${DATA_ROOT}/
├── santos_small_benchmark/
│   ├── datalake/
│   ├── query/
│   └── gt_santos_small.json
├── tus_small_benchmark/
│   ├── datalake/
│   ├── query/
│   └── gt_tus_small.json
└── tus_large_benchmark/
    ├── datalake/
    ├── query/
    └── gt_tus_large.json
```

The relation-map construction scripts read only the datalake tables. The
ground-truth files are used only for evaluation metrics.
