# Project Z32LITE - Executive Summary (Updated)

## Strategic Direction
Z32LITE now follows a strict `training-first` execution path:
1. Train and evaluate model quality.
2. Export validated GGUF artifacts.
3. Integrate only validated model outputs into Android runtime.

## Core Objective
Deliver a lightweight on-device assistant (`Qwen 2.5 1.5B`) that is:
- strong in tool-calling
- natural in Egyptian Arabic, Arabic, and English
- useful for common general questions
- low-hallucination on time-sensitive topics
- practical for `3-4GB RAM` class devices

## Current Status
### Completed
- Dataset pipeline with explicit splits:
  - `train` / `holdout` / `eval`
- Validation tooling:
  - schema checks
  - duplicate checks
  - split leakage checks
- Qwen JSONL exporter for SFT
- Eval scorer for model output benchmarking
- Colab-first fine-tuning and GGUF export scripts

### In Progress
- Running full QLoRA cycles on Colab profiles (`balanced`, `tool_heavy`, `light_regularization`)
- Benchmarking against eval acceptance thresholds

### Not Yet Declared Complete
- Final production-grade on-device inference integration in Android app
- Pilot readiness claim

## Execution Environment Policy
- Training and model export are designed for **Google Colab runtime**.
- VS Code is supported only as a client connected to Colab runtime.
- No local training path is required by this workflow.

## Risk Controls
- Prevent over-tool-calling with split distribution + eval gates.
- Prevent overclaiming up-to-date facts via boundary/freshness dataset category.
- Keep schema stable (`SYSTEM_ACTION`, `NOTIFY_USER`) until eval stability is reached.

## Success Criteria for Integration Gate
- `>= 90%` clear tool-call accuracy
- `>= 80%` mixed/ambiguous tool scenarios
- measurable hallucination reduction on fresh/current questions
- acceptable quality retention after `Q4_K_M` quantization
