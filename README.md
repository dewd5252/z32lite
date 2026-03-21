# Z32LITE - Training-First On-Device Assistant

Z32LITE is a privacy-first Android assistant project built around `Qwen/Qwen2.5-1.5B-Instruct`.

Current project direction is explicit:
1. Build and evaluate the model first.
2. Export GGUF artifacts.
3. Integrate into the Android runtime after passing eval gates.

## Current Reality
- Dataset + eval pipeline is implemented and versioned in `dataset/`.
- Fine-tuning workflow is `Colab-first` (including VS Code extension connected to Colab runtime).
- Android app exists as prototype UI + system tools, but local inference runtime is not yet final production integration.

## Repository Structure

```text
dataset/
  build_dataset.py           # Builds train/holdout/eval artifacts
  validate_dataset.py        # Schema + leakage validation
  export_qwen_jsonl.py       # JSON -> Qwen JSONL export
  score_outputs.py           # Eval scoring against eval.json
  processed/                 # Generated artifacts
finetune/
  MODEL_SPEC.md              # Model behavior contract
  train_qlora.py             # Colab training entrypoint
  export_gguf.py             # Colab GGUF conversion/quantization
  requirements-colab.txt
android/
  z32lite_app/               # Flutter + Kotlin/JNI app prototype
```

## Model Contract
Structured outputs are fixed:

```text
SYSTEM_ACTION:{"action":"set_volume","direction":"up"}
NOTIFY_USER:{"message":"...", "action_pending":"..."}
```

No schema expansion should happen before eval stability.

## Colab-First Workflow
All training/export steps are expected to run on Google Colab runtime (or VS Code Colab extension).

### 1) Build and validate dataset

```bash
python3 dataset/build_dataset.py
python3 dataset/validate_dataset.py
python3 dataset/export_qwen_jsonl.py
```

### 2) Upload artifacts to Colab
- `dataset/processed/qwen_jsonl/train.jsonl`
- `dataset/processed/qwen_jsonl/holdout.jsonl`
- `dataset/processed/eval.json`
- `finetune/train_qlora.py`
- `finetune/export_gguf.py`
- `finetune/requirements-colab.txt`

### 3) Train in Colab

```bash
pip install -r finetune/requirements-colab.txt
python finetune/train_qlora.py \
  --train-file dataset/processed/qwen_jsonl/train.jsonl \
  --eval-file dataset/processed/qwen_jsonl/holdout.jsonl \
  --profile balanced
```

### 4) Export GGUF in Colab

```bash
python finetune/export_gguf.py \
  --model-dir /content/z32lite_runs/balanced/final_merged \
  --output-dir /content/z32lite_runs/gguf
```

### 5) Score model outputs

```bash
python dataset/score_outputs.py --predictions predictions.jsonl
```

## Acceptance Gates Before App Integration
- Tool-calling accuracy on clear actions: `>= 90%`
- Mixed/ambiguous tool scenarios: `>= 80%`
- Reduced hallucination on time-sensitive questions
- No severe regression after `Q4_K_M` quantization

## License
- Base model: Apache 2.0 (Qwen 2.5)
- Project code: MIT
