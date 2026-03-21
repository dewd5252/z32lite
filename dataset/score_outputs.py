"""Score model outputs against the Z32LITE eval suite."""

from __future__ import annotations

import argparse
import json
import re
import sys
from collections import Counter
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from dataset.schema import parse_structured_response

STOPWORDS = {
    "the",
    "a",
    "an",
    "and",
    "or",
    "is",
    "are",
    "to",
    "of",
    "في",
    "من",
    "على",
    "عن",
    "هو",
    "هي",
    "ده",
    "دي",
    "إن",
    "أن",
    "this",
    "that",
}


def tokenize(text: str) -> set[str]:
    return {
        token
        for token in re.findall(r"[\w\u0600-\u06FF]+", text.lower())
        if token not in STOPWORDS and len(token) > 1
    }


def normalize_structured(text: str) -> dict[str, Any]:
    parsed = parse_structured_response(text)
    return {"kind": parsed["kind"], "payload": parsed["payload"]}


def load_eval(path: Path) -> list[dict[str, Any]]:
    return json.loads(path.read_text(encoding="utf-8"))


def load_predictions(path: Path) -> dict[str, str]:
    if path.suffix == ".jsonl":
        predictions: dict[str, str] = {}
        with path.open(encoding="utf-8") as handle:
            for line in handle:
                if not line.strip():
                    continue
                record = json.loads(line)
                predictions[record["id"]] = record["output"]
        return predictions

    records = json.loads(path.read_text(encoding="utf-8"))
    if isinstance(records, dict):
        return {key: value for key, value in records.items()}
    return {record["id"]: record["output"] for record in records}


def score_text(prediction: str, reference: str, must_not_start_with: list[str]) -> tuple[bool, float]:
    if any(prediction.strip().startswith(prefix) for prefix in must_not_start_with):
        return False, 0.0
    pred_tokens = tokenize(prediction)
    ref_tokens = tokenize(reference)
    if not ref_tokens:
        return False, 0.0
    overlap = len(pred_tokens & ref_tokens) / len(ref_tokens)
    return overlap >= 0.22, overlap


def score_predictions(eval_examples: list[dict[str, Any]], predictions: dict[str, str]) -> dict[str, Any]:
    totals = Counter()
    passes = Counter()
    details: list[dict[str, Any]] = []
    missing: list[str] = []

    for example in eval_examples:
        example_id = example["id"]
        category = example["category"]
        totals[category] += 1
        prediction = predictions.get(example_id)
        if prediction is None:
            missing.append(example_id)
            details.append({"id": example_id, "category": category, "passed": False, "reason": "missing prediction"})
            continue

        evaluation = example["evaluation"]
        passed = False
        score = 0.0
        if evaluation["mode"] == "structured_exact":
            expected = normalize_structured(evaluation["expected"])
            got = normalize_structured(prediction)
            passed = expected == got
            score = 1.0 if passed else 0.0
        else:
            passed, score = score_text(
                prediction,
                evaluation["reference"],
                evaluation.get("must_not_start_with", []),
            )

        if passed:
            passes[category] += 1
        details.append(
            {
                "id": example_id,
                "category": category,
                "passed": passed,
                "score": round(score, 4),
                "prediction": prediction,
                "reference": example["conversations"][1]["value"],
            }
        )

    aggregate = {
        "overall_accuracy": round(sum(passes.values()) / max(sum(totals.values()), 1), 4),
        "per_category": {
            category: {
                "passed": passes[category],
                "total": totals[category],
                "accuracy": round(passes[category] / totals[category], 4),
            }
            for category in sorted(totals)
        },
        "missing_predictions": missing,
        "details": details,
    }
    return aggregate


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--eval",
        default=str(Path(__file__).resolve().parent / "processed" / "eval.json"),
        help="Path to eval.json",
    )
    parser.add_argument(
        "--predictions",
        required=True,
        help="JSON or JSONL file containing {id, output}",
    )
    parser.add_argument(
        "--report",
        help="Optional path to write a full JSON report",
    )
    args = parser.parse_args()

    report = score_predictions(
        load_eval(Path(args.eval)),
        load_predictions(Path(args.predictions)),
    )
    print(f"overall_accuracy={report['overall_accuracy']}")
    for category, summary in report["per_category"].items():
        print(
            f"{category}: {summary['passed']}/{summary['total']} "
            f"({summary['accuracy']:.2%})"
        )
    if report["missing_predictions"]:
        print(f"missing_predictions={len(report['missing_predictions'])}")

    if args.report:
        Path(args.report).write_text(
            json.dumps(report, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
