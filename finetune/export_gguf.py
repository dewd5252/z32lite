"""Export a merged Hugging Face checkpoint to GGUF inside Colab."""

from __future__ import annotations

import argparse
import subprocess
from pathlib import Path


def run(cmd: list[str], cwd: Path | None = None) -> None:
    print("$", " ".join(cmd))
    subprocess.run(cmd, cwd=str(cwd) if cwd else None, check=True)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-dir", required=True, help="Merged HF model directory")
    parser.add_argument("--output-dir", required=True, help="Where GGUF files will be written")
    parser.add_argument("--llama-cpp-dir", default="/content/llama.cpp")
    parser.add_argument("--quantization", default="Q4_K_M")
    parser.add_argument("--skip-clone", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    model_dir = Path(args.model_dir)
    output_dir = Path(args.output_dir)
    llama_cpp_dir = Path(args.llama_cpp_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if not llama_cpp_dir.exists() and not args.skip_clone:
        run(["git", "clone", "https://github.com/ggerganov/llama.cpp.git", str(llama_cpp_dir)])

    run(["pip", "install", "-r", "requirements.txt"], cwd=llama_cpp_dir)
    run(["make", "quantize", "-j4"], cwd=llama_cpp_dir)

    fp16_path = output_dir / "z32lite_f16.gguf"
    quantized_path = output_dir / f"z32lite_{args.quantization}.gguf"

    run(
        [
            "python",
            "convert_hf_to_gguf.py",
            str(model_dir),
            "--outtype",
            "f16",
            "--outfile",
            str(fp16_path),
        ],
        cwd=llama_cpp_dir,
    )
    run(
        [
            str(llama_cpp_dir / "quantize"),
            str(fp16_path),
            str(quantized_path),
            args.quantization,
        ]
    )

    print(f"✅ fp16 gguf: {fp16_path}")
    print(f"✅ quantized gguf: {quantized_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
