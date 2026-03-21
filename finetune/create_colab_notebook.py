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
REPO_URL = "https://github.com/dewd5252/z32lite.git"
REPO_DIR = "/content/z32lite"
PROFILE = "balanced"  # balanced | tool_heavy | light_regularization
OUTPUT_ROOT = "/content/z32lite_runs"
            """
        ),
        make_code_cell(
            """
import os
import sys
import subprocess

print("Python:", sys.version)
print("Has /content:", os.path.exists("/content"))
print("COLAB_RELEASE_TAG:", os.environ.get("COLAB_RELEASE_TAG"))

if not os.path.exists("/content"):
    raise RuntimeError(
        "Not running on Colab runtime. "
        "In VS Code choose: Select Kernel > Colab > Auto Connect, then Run All again."
    )

try:
    import torch
    cuda_ok = torch.cuda.is_available()
    print("cuda_available:", cuda_ok)
    if cuda_ok:
        print("gpu_name:", torch.cuda.get_device_name(0))
except Exception:
    cuda_ok = False
    print("cuda_available: False (torch unavailable)")

if not cuda_ok:
    # fall back to nvidia-smi visibility for a clearer hint in Colab UI
    smi = subprocess.run("nvidia-smi -L", shell=True, capture_output=True, text=True)
    print("nvidia-smi:", smi.stdout.strip() or smi.stderr.strip() or "not available")
    raise RuntimeError(
        "GPU runtime is not attached. In Colab Web: Runtime > Change runtime type > GPU, then reconnect."
    )
            """
        ),
        make_code_cell(
            """
import os
import shutil
import subprocess
from pathlib import Path

def _tail_log(path: Path, n: int = 200):
    if not path.exists():
        print(f"[debug] log file not found: {path}")
        return
    print(f"\\n--- last {n} lines of {path} ---")
    lines = path.read_text(encoding="utf-8", errors="ignore").splitlines()
    for line in lines[-n:]:
        print(line)
    print("--- end log ---\\n")

def run(cmd, cwd=None, debug_log: Path | None = None):
    print("$", cmd)
    result = subprocess.run(cmd, shell=True, cwd=cwd)
    if result.returncode != 0:
        print(f"[error] command failed with code={result.returncode}")
        if debug_log is not None:
            _tail_log(debug_log, n=220)
        raise RuntimeError(f"Command failed: {cmd}")

repo_path = Path(REPO_DIR)
if not repo_path.exists():
    run(f"git clone {REPO_URL} {REPO_DIR}")
else:
    # لو المجلد موجود لكن مش git repo صحيح، نحذفه ونعيد clone تلقائياً
    if not (repo_path / ".git").exists():
        print("Existing folder is not a git repo. Re-cloning...")
        shutil.rmtree(repo_path)
        run(f"git clone {REPO_URL} {REPO_DIR}")
    else:
        print("Repo already exists, pulling latest...")
        os.chdir(REPO_DIR)
        run("git fetch --all")
        run("git reset --hard origin/main")

os.chdir(REPO_DIR)
            """
        ),
        make_code_cell(
            """
run("chmod +x finetune/colab_oneclick.sh")
run(
    f"bash ./finetune/colab_oneclick.sh {PROFILE} {OUTPUT_ROOT}",
    debug_log=Path(f"{OUTPUT_ROOT}/pipeline.log"),
)
            """
        ),
        make_code_cell(
            """
print("Artifacts:")
from pathlib import Path
status_file = Path(f"{OUTPUT_ROOT}/pipeline_status.json")
if status_file.exists():
    import json
    status = json.loads(status_file.read_text(encoding="utf-8"))
    print(f"- requested profile: {status['requested_profile']}")
    print(f"- trained profile:   {status['trained_profile']}")
    print(f"- merged model:      {status['merged_model_dir']}")
    print(f"- fp16 gguf:         {status['gguf_fp16']}")
    print(f"- q4 gguf:           {status['gguf_q4']}")
else:
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
