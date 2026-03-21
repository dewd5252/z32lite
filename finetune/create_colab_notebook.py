"""Generate a Colab-ready notebook that runs the full Z32LITE pipeline."""

from __future__ import annotations

import json
from pathlib import Path


NOTEBOOK_PATH = Path(__file__).resolve().parent / "z32lite_colab_oneclick.ipynb"


def make_code_cell(source: str) -> dict:
    return {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": [line + "\n" for line in source.strip("\n").splitlines()],
    }


def make_markdown_cell(source: str) -> dict:
    return {
        "cell_type": "markdown",
        "metadata": {},
        "source": [line + "\n" for line in source.strip("\n").splitlines()],
    }


def build_notebook() -> dict:
    cells = [
        make_markdown_cell(
            """
# Z32LITE Colab One-Click Pipeline

Notebook ده بيشغّل كامل مسار:
1. تجهيز الداتا
2. التحقق
3. التصدير لصيغة Qwen JSONL
4. تدريب QLoRA
5. تصدير GGUF

شغل `Runtime > Run all`.
            """
        ),
        make_code_cell(
            """
# ---- config ----
REPO_URL = "https://github.com/<YOUR_USER>/<YOUR_REPO>.git"
REPO_DIR = "/content/z32lite"
PROFILE = "balanced"  # balanced | tool_heavy | light_regularization
OUTPUT_ROOT = "/content/z32lite_runs"
            """
        ),
        make_code_cell(
            """
import os
import subprocess
from pathlib import Path

def run(cmd, cwd=None):
    print("$", cmd)
    subprocess.run(cmd, shell=True, check=True, cwd=cwd)

if not Path(REPO_DIR).exists():
    run(f"git clone {REPO_URL} {REPO_DIR}")
else:
    print("Repo already exists, skipping clone.")

os.chdir(REPO_DIR)
run("git pull --ff-only || true")
            """
        ),
        make_code_cell(
            """
run("chmod +x finetune/colab_oneclick.sh")
run(f"./finetune/colab_oneclick.sh {PROFILE} {OUTPUT_ROOT}")
            """
        ),
        make_code_cell(
            """
print("Artifacts:")
print(f"- merged model: {OUTPUT_ROOT}/{PROFILE}/final_merged")
print(f"- fp16 gguf:    {OUTPUT_ROOT}/gguf/z32lite_f16.gguf")
print(f"- q4 gguf:      {OUTPUT_ROOT}/gguf/z32lite_Q4_K_M.gguf")
            """
        ),
    ]

    return {
        "cells": cells,
        "metadata": {
            "colab": {"name": "z32lite_colab_oneclick.ipynb", "provenance": []},
            "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
            "language_info": {"name": "python"},
        },
        "nbformat": 4,
        "nbformat_minor": 5,
    }


def main() -> int:
    notebook = build_notebook()
    NOTEBOOK_PATH.write_text(json.dumps(notebook, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"✅ Wrote {NOTEBOOK_PATH}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
