#!/usr/bin/env bash
set -euo pipefail

# Anonymous DGGC reproduction entry point.
#
# Usage:
#   MODE=build_map  DATASET=tus_small  bash scripts/run_reproduce.sh
#   MODE=train      DATASET=tus_small  bash scripts/run_reproduce.sh
#   MODE=evaluate   DATASET=tus_small  CHECKPOINT=/path/to/epoch_10.pth bash scripts/run_reproduce.sh
#   MODE=sweep_relmap DATASET=tus_small bash scripts/run_reproduce.sh

MODE="${MODE:-all}"
DATASET="${DATASET:-tus_small}"
DATA_ROOT="${DATA_ROOT:-./data}"
DGGC_WORK_DIR="${DGGC_WORK_DIR:-./outputs}"
MODEL_NAME="${MODEL_NAME:-bert-base-uncased}"

BATCH_SIZE="${BATCH_SIZE:-64}"
LR="${LR:-8e-6}"
EPOCHS="${EPOCHS:-10}"
TOP_N="${TOP_N:-300}"

case "${DATASET}" in
  santos_small)
    DATA_DIR="${DATA_ROOT}/santos_small_benchmark/datalake"
    QUERY_DIR="${DATA_ROOT}/santos_small_benchmark/query"
    GT_PATH="${DATA_ROOT}/santos_small_benchmark/gt_santos_small.json"
    EVAL_K="${EVAL_K:-5 10}"
    ;;
  tus_small)
    DATA_DIR="${DATA_ROOT}/tus_small_benchmark/datalake"
    QUERY_DIR="${DATA_ROOT}/tus_small_benchmark/query"
    GT_PATH="${DATA_ROOT}/tus_small_benchmark/gt_tus_small.json"
    EVAL_K="${EVAL_K:-10 60}"
    ;;
  tus_large)
    DATA_DIR="${DATA_ROOT}/tus_large_benchmark/datalake"
    QUERY_DIR="${DATA_ROOT}/tus_large_benchmark/query"
    GT_PATH="${DATA_ROOT}/tus_large_benchmark/gt_tus_large.json"
    EVAL_K="${EVAL_K:-10 60}"
    ;;
  *)
    echo "Unknown DATASET=${DATASET}" >&2
    exit 1
    ;;
esac

OUT_DIR="${DGGC_WORK_DIR}/${DATASET}"
REL_MAP="${OUT_DIR}/relation_map_semantic.json"
mkdir -p "${OUT_DIR}"

run_build_map() {
  echo "[build_map] DATASET=${DATASET}"
  python src/relation_map/build_map.py \
    --data_dir "${DATA_DIR}" \
    --output_file "${REL_MAP}"
}

run_train() {
  echo "[train] DATASET=${DATASET}"
  python src/dggc/train.py \
    --data_dir "${DATA_DIR}" \
    --relation_map "${REL_MAP}" \
    --model_name "${MODEL_NAME}" \
    --output_dir "${OUT_DIR}/checkpoints" \
    --epochs "${EPOCHS}" \
    --batch_size "${BATCH_SIZE}" \
    --lr "${LR}"
}

run_evaluate() {
  echo "[evaluate] DATASET=${DATASET}"
  CKPT="${CHECKPOINT:-${OUT_DIR}/checkpoints/epoch_${EPOCHS}.pth}"
  for K in ${EVAL_K}; do
    python src/dggc/evaluate.py \
      --model_dir "${CKPT}" \
      --candidate_dir "${DATA_DIR}" \
      --query_dir "${QUERY_DIR}" \
      --gt_path "${GT_PATH}" \
      --k "${K}" \
      --top_n "${TOP_N}" \
      --output_csv "${OUT_DIR}/eval_k${K}.csv"
  done
}

run_sweep_relmap() {
  echo "[sweep_relmap] DATASET=${DATASET}"
  for TAU in 0.60 0.65 0.70 0.75 0.80 0.85; do
    python src/relation_map/filter_map.py \
      --input_relation_map "${REL_MAP}" \
      --output_relation_map "${OUT_DIR}/relation_map_tau_${TAU}.json" \
      --semantic_score "${TAU}"
  done
}

case "${MODE}" in
  build_map) run_build_map ;;
  train) run_train ;;
  evaluate) run_evaluate ;;
  sweep_relmap) run_sweep_relmap ;;
  all) run_build_map; run_train; run_evaluate ;;
  *)
    echo "Unknown MODE=${MODE}" >&2
    exit 1
    ;;
esac
