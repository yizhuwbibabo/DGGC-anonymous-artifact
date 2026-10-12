# Anonymous Supplemental Material for DGGC

This package contains the code, configurations, scripts, and selected result
files used to verify the DGGC experiments reported in the paper.

## Directory

```text
DGGC/
├── README.md
├── SUPPLEMENTAL_MATERIAL_STATEMENT.txt
├── ANONYMIZATION_CHECKLIST.md
├── MANIFEST.csv
├── requirements.txt
├── environment.yml
├── configs/
│   ├── santos_small.yaml
│   ├── tus_small.yaml
│   └── tus_large.yaml
├── data/
│   └── README_dataset_structure.md
├── results/
│   ├── overall.csv
│   ├── relation_map.csv
│   ├── lambda.csv
│   ├── efficiency.csv
│   └── README.md
├── scripts/
│   └── run_reproduce.sh
└── src/
    ├── dggc/
    │   ├── model.py
    │   ├── train.py
    │   ├── evaluate.py
    │   └── README.md
    ├── relation_map/
    │   ├── build_map.py
    │   └── filter_map.py
    └── utils/
        ├── summarize_ablation.py
        └── summarize_sweeps.py
```

## Setup

```bash
conda env create -f environment.yml
conda activate dggc-anonymous
```

or

```bash
pip install -r requirements.txt
```

Set paths:

```bash
export DATA_ROOT=/path/to/datasets
export DGGC_WORK_DIR=/path/to/outputs
export MODEL_NAME=/path/to/roberta-base
```

## Run

```bash
MODE=build_map DATASET=tus_small bash scripts/run_reproduce.sh
MODE=train DATASET=tus_small bash scripts/run_reproduce.sh
MODE=evaluate DATASET=tus_small CHECKPOINT=/path/to/epoch_10.pth bash scripts/run_reproduce.sh
MODE=sweep_relmap DATASET=tus_small bash scripts/run_reproduce.sh
```

## Notes

The relation-map construction in this package does not use ground-truth files.
Ground-truth files are only used by the evaluation step to compute MAP and
Recall. Large checkpoints, embedding caches, debug logs, optional rerankers,
and public benchmark datasets are not included.
