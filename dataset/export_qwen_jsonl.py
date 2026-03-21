"""Export processed JSON splits into Qwen chat-format JSONL for Colab training."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from dataset.build_dataset import SYSTEM_PROMPT
from dataset.schema import build_qwen_chat_text


def export_split(input_path: Path, output_path: Path, system_prompt: str) -> None:
    examples = json.loads(input_path.read_text(encoding="utf-8"))
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as handle:
        for example in examples:
            record = {
                "id": example["id"],
                "category": example["category"],
                "language": example["language"],
                "source": example["source"],
                "text": build_qwen_chat_text(example, system_prompt),
            }
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input-dir",
        default=str(Path(__file__).resolve().parent / "processed"),
        help="Directory containing train.json / holdout.json / eval.json",
    )
    parser.add_argument(
        "--output-dir",
        default=str(Path(__file__).resolve().parent / "processed" / "qwen_jsonl"),
        help="Where to write JSONL exports",
    )
    parser.add_argument(
        "--system-prompt",
        default=SYSTEM_PROMPT,
        help="System prompt inserted into the Qwen chat template",
    )
    args = parser.parse_args()

    input_dir = Path(args.input_dir)
    output_dir = Path(args.output_dir)
    for split in ["train", "holdout", "eval"]:
        export_split(
            input_dir / f"{split}.json",
            output_dir / f"{split}.jsonl",
            args.system_prompt,
        )
        print(f"✅ exported {split}.jsonl")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
