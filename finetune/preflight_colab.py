"""Preflight checks for Colab GPU training."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-root", default="/content/z32lite_runs")
    parser.add_argument("--require-gpu", dest="require_gpu", action="store_true")
    parser.add_argument("--allow-cpu", dest="require_gpu", action="store_false")
    parser.set_defaults(require_gpu=True)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    output_root = Path(args.output_root)
    output_root.mkdir(parents=True, exist_ok=True)
    preflight_path = output_root / "preflight.json"

    try:
        import torch
    except Exception as exc:  # pragma: no cover - defensive guard for runtime envs
        payload = {
            "ok": False,
            "error": f"torch import failed: {exc}",
            "cuda_available": False,
        }
        preflight_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        print(json.dumps(payload, indent=2))
        return 1

    cuda_available = torch.cuda.is_available()
    gpu_name = torch.cuda.get_device_name(0) if cuda_available else None
    bf16_supported = bool(cuda_available and torch.cuda.is_bf16_supported())
    payload = {
        "ok": bool(cuda_available or not args.require_gpu),
        "cuda_available": cuda_available,
        "gpu_name": gpu_name,
        "device_count": int(torch.cuda.device_count()) if cuda_available else 0,
        "bf16_supported": bf16_supported,
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
        "colab_release_tag": os.environ.get("COLAB_RELEASE_TAG"),
        "torch_version": torch.__version__,
    }

    preflight_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(json.dumps(payload, indent=2))

    if args.require_gpu and not cuda_available:
        print("[fatal] GPU runtime is required. Please attach a Colab GPU runtime.")
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
