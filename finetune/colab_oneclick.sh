#!/usr/bin/env bash
set -Eeuo pipefail
trap 'echo "ERROR: command failed at line ${LINENO}: ${BASH_COMMAND}" >&2' ERR

# Colab-first one-click pipeline:
# 1) build/validate/export dataset
# 2) train selected profile
# 3) export GGUF

PROFILE="${1:-balanced}"
OUTPUT_ROOT="${2:-/content/z32lite_runs}"
TRAIN_FILE="${3:-dataset/processed/qwen_jsonl/train.jsonl}"
EVAL_FILE="${4:-dataset/processed/qwen_jsonl/holdout.jsonl}"
mkdir -p "${OUTPUT_ROOT}"
RUN_LOG="${OUTPUT_ROOT}/pipeline.log"
STATUS_JSON="${OUTPUT_ROOT}/pipeline_status.json"
exec > >(tee -a "${RUN_LOG}") 2>&1

if [[ ! -d "/content" ]]; then
  echo "This script must run on Google Colab runtime."
  echo "Current environment does not have /content."
  echo "Open the notebook with Colab kernel, then run all cells again."
  exit 2
fi

echo "Log file: ${RUN_LOG}"
python3 - <<'PY'
import os
try:
    import torch
    print("torch:", torch.__version__)
    print("cuda_available:", torch.cuda.is_available())
    if torch.cuda.is_available():
        print("gpu_name:", torch.cuda.get_device_name(0))
except Exception as exc:
    print("torch check failed:", exc)
print("COLAB_RELEASE_TAG:", os.environ.get("COLAB_RELEASE_TAG"))
PY

if ! python3 - <<'PY'
import torch
raise SystemExit(0 if torch.cuda.is_available() else 1)
PY
then
  cat <<'EOF'
[fatal] GPU runtime is not attached.
Fix in Colab web:
1) Runtime -> Change runtime type
2) Hardware accelerator -> GPU
3) Save, then Runtime -> Restart and run all
EOF
  exit 1
fi

echo "[1/6] Installing training dependencies"
pip install -q -r finetune/requirements-colab.txt

echo "[2/6] Building dataset"
python3 dataset/build_dataset.py

echo "[3/6] Validating dataset"
python3 dataset/validate_dataset.py

echo "[4/6] Exporting Qwen JSONL"
python3 dataset/export_qwen_jsonl.py

echo "[5/6] Training profile=${PROFILE}"
TRAINED_PROFILE="${PROFILE}"
if ! python3 finetune/train_qlora.py \
  --train-file "${TRAIN_FILE}" \
  --eval-file "${EVAL_FILE}" \
  --profile "${PROFILE}" \
  --output-root "${OUTPUT_ROOT}"; then
  if [[ "${PROFILE}" != "colab_safe" ]]; then
    echo "Primary profile failed. Retrying once with profile=colab_safe ..."
    TRAINED_PROFILE="colab_safe"
    python3 finetune/train_qlora.py \
      --train-file "${TRAIN_FILE}" \
      --eval-file "${EVAL_FILE}" \
      --profile "${TRAINED_PROFILE}" \
      --max-seq-length 1024 \
      --save-steps 200 \
      --output-root "${OUTPUT_ROOT}"
  else
    exit 1
  fi
fi

echo "[6/6] Exporting GGUF"
python3 finetune/export_gguf.py \
  --model-dir "${OUTPUT_ROOT}/${TRAINED_PROFILE}/final_merged" \
  --output-dir "${OUTPUT_ROOT}/gguf"

python3 - <<PY
import json
from pathlib import Path

status = {
    "requested_profile": "${PROFILE}",
    "trained_profile": "${TRAINED_PROFILE}",
    "output_root": "${OUTPUT_ROOT}",
    "merged_model_dir": f"${OUTPUT_ROOT}/${TRAINED_PROFILE}/final_merged",
    "gguf_fp16": f"${OUTPUT_ROOT}/gguf/z32lite_f16.gguf",
    "gguf_q4": f"${OUTPUT_ROOT}/gguf/z32lite_Q4_K_M.gguf",
}
Path("${STATUS_JSON}").write_text(
    json.dumps(status, ensure_ascii=False, indent=2),
    encoding="utf-8",
)
print(f"✅ wrote status file: ${STATUS_JSON}")
PY

echo "Pipeline complete."
echo "Trained profile: ${TRAINED_PROFILE}"
echo "GGUF outputs:"
echo "  ${OUTPUT_ROOT}/gguf/z32lite_f16.gguf"
echo "  ${OUTPUT_ROOT}/gguf/z32lite_Q4_K_M.gguf"
