#!/usr/bin/env bash
set -euo pipefail

DATASET="fb15k237"    # fb15k237, WN18, WN18RR, codexsmall, codexmedium, UMLS
MODELS=("TransE" "DistMult" "ConvE" "RGCN" "RotatE")
MODELS=("RGCN") # "TransE" "DistMult" "ConvE" "RGCN" "RotatE"
DIM=512
BATCH=10
LOG_DIR="4c_baseline/logs/${DATASET}"
CUDA=("4" "5" "3" "2" "1")

mkdir -p "$LOG_DIR"

num_models=${#MODELS[@]}
num_gpus=${#CUDA[@]}
if (( num_gpus == 0 )); then
  echo "ERROR: CUDA array is empty." >&2
  exit 1
fi

for i in "${!MODELS[@]}"; do
  MODEL="${MODELS[$i]}"
  GPU_INDEX=$(( i % num_gpus ))
  GPU_ID="${CUDA[$GPU_INDEX]}"

  LOG_FILE="${MODEL}-D${DIM}_allpred.txt"
  echo "Running model: $MODEL (DIM=$DIM, BATCH=$BATCH) on CUDA:$GPU_ID"

  CUDA_VISIBLE_DEVICES="$GPU_ID" nohup python baseline/kge_pred.py \
      --DATASET "$DATASET" \
      --MODEL "$MODEL" \
      --DIM "$DIM" \
      --BATCH "$BATCH" \
      > "${LOG_DIR}/${LOG_FILE}" 2>&1 &

  echo "Log saved to: ${LOG_DIR}/${LOG_FILE}"
  sleep 1
done
