# DGGC source code

This folder contains the core DGGC implementation.

```text
model.py     Gaussian table and column encoder.
train.py     DGGC training with relation-map positives.
evaluate.py  FAISS-based evaluation with cached embeddings.
README.md    This file.
```

Example training command:

```bash
python src/dggc/train.py \
  --data_dir ${DATA_ROOT}/tus_small_benchmark/datalake \
  --relation_map ${DGGC_WORK_DIR}/tus_small/relation_map_semantic.json \
  --model_name ${MODEL_NAME} \
  --output_dir ${DGGC_WORK_DIR}/tus_small/checkpoints \
  --epochs 10 \
  --batch_size 64 \
  --lr 8e-6
```

Example evaluation command:

```bash
python src/dggc/evaluate.py \
  --model_dir ${DGGC_WORK_DIR}/tus_small/checkpoints/epoch_10.pth \
  --candidate_dir ${DATA_ROOT}/tus_small_benchmark/datalake \
  --query_dir ${DATA_ROOT}/tus_small_benchmark/query \
  --gt_path ${DATA_ROOT}/tus_small_benchmark/gt_tus_small.json \
  --k 10 \
  --top_n 300 \
  --output_csv ${DGGC_WORK_DIR}/tus_small/eval_k10.csv
```
