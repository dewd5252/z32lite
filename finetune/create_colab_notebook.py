"""Generate Colab notebooks for Z32LITE training pipeline."""

from __future__ import annotations

import json
from pathlib import Path


FINETUNE_DIR = Path(__file__).resolve().parent
NOTEBOOKS = {
    "z32lite_colab_oneclick.ipynb": "Z32LITE Colab One-Click Pipeline",
    "z32lite_finetune.ipynb": "Z32LITE Finetune (Unified One-Click Runtime)",
}


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


def build_notebook(title: str, notebook_name: str) -> dict:
    cells = [
        make_markdown_cell(
            f"""
# {title}

Notebook ده بيشغّل كامل مسار:
1. تجهيز الداتا
2. التحقق
3. التصدير لصيغة Qwen JSONL
4. تدريب QLoRA
5. تصدير GGUF

الـ runtime الرسمي للتدريب هو **fp16 على T4**.
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
print("git_commit:", subprocess.check_output("git rev-parse --short HEAD", shell=True).decode().strip())
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
import json
import subprocess

try:
    print("git_commit:", subprocess.check_output("git rev-parse --short HEAD", shell=True).decode().strip())
except Exception:
    pass

status_file = Path(f"{OUTPUT_ROOT}/pipeline_status.json")
if status_file.exists():
    status = json.loads(status_file.read_text(encoding="utf-8"))
    print(f"- requested profile:   {status.get('requested_profile')}")
    print(f"- trained profile:     {status.get('trained_profile')}")
    print(f"- requested precision: {status.get('requested_precision')}")
    print(f"- resolved precision:  {status.get('resolved_precision')}")
    print(f"- cuda name:           {status.get('cuda_name')}")
    print(f"- bf16 supported:      {status.get('bf16_supported')}")
    print(f"- preflight json:      {status.get('preflight_json')}")
    print(f"- merged model:        {status.get('merged_model_dir')}")
    print(f"- fp16 gguf:           {status.get('gguf_fp16')}")
    print(f"- q4 gguf:             {status.get('gguf_q4')}")
else:
    print(f"- merged model: {OUTPUT_ROOT}/{PROFILE}/final_merged")
    print(f"- fp16 gguf:    {OUTPUT_ROOT}/gguf/z32lite_f16.gguf")
    print(f"- q4 gguf:      {OUTPUT_ROOT}/gguf/z32lite_Q4_K_M.gguf")

preflight = Path(f"{OUTPUT_ROOT}/preflight.json")
if preflight.exists():
    print("\\nPreflight:")
    print(preflight.read_text(encoding="utf-8"))
            """
        ),
    ]

    return {
        "cells": cells,
        "metadata": {
            "colab": {"name": notebook_name, "provenance": []},
            "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
            "language_info": {"name": "python"},
        },
        "nbformat": 4,
        "nbformat_minor": 5,
    }


def main() -> int:
    for notebook_name, title in NOTEBOOKS.items():
        notebook = build_notebook(title=title, notebook_name=notebook_name)
        path = FINETUNE_DIR / notebook_name
        path.write_text(json.dumps(notebook, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"✅ Wrote {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
