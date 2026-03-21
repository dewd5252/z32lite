#!/usr/bin/env bash
set -euo pipefail

# Colab-first one-click pipeline:
# 1) build/validate/export dataset
# 2) train selected profile
# 3) export GGUF

PROFILE="${1:-balanced}"
OUTPUT_ROOT="${2:-/content/z32lite_runs}"
TRAIN_FILE="${3:-dataset/processed/qwen_jsonl/train.jsonl}"
EVAL_FILE="${4:-dataset/processed/qwen_jsonl/holdout.jsonl}"

echo "[1/6] Installing training dependencies"
pip install -q -r finetune/requirements-colab.txt

echo "[2/6] Building dataset"
python3 dataset/build_dataset.py

echo "[3/6] Validating dataset"
python3 dataset/validate_dataset.py

echo "[4/6] Exporting Qwen JSONL"
python3 dataset/export_qwen_jsonl.py

echo "[5/6] Training profile=${PROFILE}"
python3 finetune/train_qlora.py \
  --train-file "${TRAIN_FILE}" \
  --eval-file "${EVAL_FILE}" \
  --profile "${PROFILE}" \
  --output-root "${OUTPUT_ROOT}"

echo "[6/6] Exporting GGUF"
python3 finetune/export_gguf.py \
  --model-dir "${OUTPUT_ROOT}/${PROFILE}/final_merged" \
  --output-dir "${OUTPUT_ROOT}/gguf"

echo "Pipeline complete."
echo "GGUF outputs:"
echo "  ${OUTPUT_ROOT}/gguf/z32lite_f16.gguf"
echo "  ${OUTPUT_ROOT}/gguf/z32lite_Q4_K_M.gguf"
