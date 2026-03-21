from __future__ import annotations

import json
import unittest
from pathlib import Path

from dataset.build_dataset import (
    EVAL_TARGETS,
    TARGET_COUNTS,
    ExampleFactory,
    build_dataset,
    generate_boundary_pool,
    generate_conversational_pool,
    generate_safety_pool,
    generate_tool_pool,
    system_action,
)
from dataset.schema import parse_structured_response, validate_example
from dataset.score_outputs import score_predictions


class DatasetPipelineTests(unittest.TestCase):
    def test_system_action_payload_is_valid_json(self) -> None:
        value = system_action("set_volume", direction="up", stream="media")
        parsed = parse_structured_response(value)
        self.assertEqual(parsed["kind"], "system_action")
        self.assertEqual(parsed["payload"]["action"], "set_volume")
        self.assertEqual(parsed["payload"]["direction"], "up")

    def test_generation_pools_are_non_empty(self) -> None:
        factory = ExampleFactory()
        self.assertGreater(len(generate_conversational_pool(factory)), 100)
        self.assertGreater(len(generate_tool_pool(factory)), 100)
        self.assertGreater(len(generate_safety_pool(factory)), 100)
        self.assertGreater(len(generate_boundary_pool(factory)), 100)

    def test_build_dataset_artifacts_validate(self) -> None:
        manifest = build_dataset()
        self.assertGreater(manifest["train"]["count"], 0)
        total_pool = sum(TARGET_COUNTS.values())
        self.assertEqual(
            manifest["train"]["count"] + manifest["holdout"]["count"],
            total_pool,
        )
        self.assertEqual(manifest["eval"]["count"], sum(EVAL_TARGETS.values()))
        train_path = Path("dataset/processed/train.json")
        train = json.loads(train_path.read_text(encoding="utf-8"))
        for example in train[:25]:
            self.assertEqual(validate_example(example), [])

    def test_scoring_accepts_perfect_predictions(self) -> None:
        eval_examples = [
            {
                "id": "z32-test",
                "split": "eval",
                "category": "tool_calling",
                "language": "ar-eg",
                "source": "curated",
                "tags": ["tool"],
                "conversations": [
                    {"from": "user", "value": "علي الصوت"},
                    {"from": "assistant", "value": system_action("set_volume", direction="up", stream="ring")},
                ],
                "evaluation": {
                    "mode": "structured_exact",
                    "expected": system_action("set_volume", direction="up", stream="ring"),
                },
            }
        ]
        report = score_predictions(
            eval_examples,
            {"z32-test": system_action("set_volume", direction="up", stream="ring")},
        )
        self.assertEqual(report["overall_accuracy"], 1.0)


if __name__ == "__main__":
    unittest.main()
