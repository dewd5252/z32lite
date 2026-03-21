"""Validate generated dataset artifacts for schema, duplicates, and split leakage."""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from dataset.schema import conversation_signature, validate_example


def load_examples(path: Path) -> list[dict]:
    return json.loads(path.read_text(encoding="utf-8"))


def validate_files(paths: list[Path]) -> int:
    seen_by_signature: dict[str, str] = {}
    exit_code = 0

    for path in paths:
        examples = load_examples(path)
        errors: list[str] = []
        for example in examples:
            item_errors = validate_example(example)
            if item_errors:
                errors.append(f"{example.get('id')}: {', '.join(item_errors)}")

            signature = conversation_signature(example)
            previous = seen_by_signature.get(signature)
            if previous and previous != example["split"]:
                errors.append(
                    f"{example['id']}: conversation overlaps with split {previous}"
                )
            else:
                seen_by_signature[signature] = example["split"]

        counter = Counter(example["category"] for example in examples)
        print(f"[{path.name}] {len(examples)} examples | categories={dict(counter)}")
        if errors:
            exit_code = 1
            print("  errors:")
            for error in errors[:30]:
                print(f"   - {error}")
    return exit_code


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "paths",
        nargs="*",
        help="Dataset JSON files to validate. Defaults to processed train/holdout/eval.",
    )
    args = parser.parse_args()

    if args.paths:
        paths = [Path(path) for path in args.paths]
    else:
        root = Path(__file__).resolve().parent / "processed"
        paths = [root / "train.json", root / "holdout.json", root / "eval.json"]

    return validate_files(paths)


if __name__ == "__main__":
    raise SystemExit(main())
